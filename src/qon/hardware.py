"""
hardware.py  --  RUNNING THE CIRCUIT ON A REAL IBM QUANTUM COMPUTER (SET D)
==========================================================================

Everything else in this project is simulated on a laptop.  Feature set D -
"HSBC's condition" - needs a real chip.  This module contains every step:

1. `find_chain`          IBM chips are not a straight line of qubits; they are
                         a "heavy-hex" honeycomb.  Our circuit needs 109
                         qubits in a row, so we look for a snake-like path of
                         109 connected qubits, avoiding the worst couplers.
2. `prepare`             translate the circuit into the chip's own gates
                         ("transpile") and lay it on that path.
3. `build_run_plan`      decide which events to run and IN WHAT ORDER.
                         Two controls the critics asked for are built in:
                           * the order is RANDOM (seeded), not by date, so
                             slow drift of the chip cannot act as a clock;
                           * every few circuits a fixed "sentinel" circuit is
                             repeated; identical inputs should give identical
                             outputs, so their scatter measures the drift.
4. `run_plan`            submit the jobs through IBM's Estimator with the same
                         error handling HSBC used (Pauli twirling + TREX) and
                         collect the 327 measured averages per event.

The same code runs against a local simulated chip ("fake backend"), which is
how it is tested without spending any real quantum time; see
scripts/06_hardware_run.py.

`chip_snapshot` is also used by the noisy simulator (set C): it reads the
per-qubit error rates of the chosen path from IBM's published calibration
data, so the simulated chip has the same good and bad qubits as a real one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .circuit import CircuitSpec, build_observables, build_qiskit_circuit, output_names


# ---------------------------------------------------------------------------
# 1. a line of qubits on a honeycomb chip
# ---------------------------------------------------------------------------
def two_qubit_gate_name(backend) -> str:
    for name in ("cz", "ecr", "cx"):
        if name in backend.target.operation_names:
            return name
    raise ValueError("backend has no known two-qubit gate")


def edge_errors(backend) -> dict[tuple[int, int], float]:
    """Error rate of the entangling gate on every coupler (pair of
    neighbouring qubits).  Couplers with no reported error count as broken."""
    gate = two_qubit_gate_name(backend)
    edges: dict[tuple[int, int], float] = {}
    for (q1, q2), props in backend.target[gate].items():
        err = props.error if (props is not None and props.error is not None) else 1.0
        key = (min(q1, q2), max(q1, q2))
        edges[key] = min(edges.get(key, 1.0), float(err))
    return edges


def _search(edges, n_nodes: int, length: int, max_error: float, tries: int,
            step_cap: int, seed: int):
    rng = np.random.default_rng(seed)
    adj: dict[int, list[int]] = {q: [] for q in range(n_nodes)}
    for (a, b), e in edges.items():
        if e <= max_error:
            adj[a].append(b)
            adj[b].append(a)
    starts = [q for q in sorted(adj, key=lambda q: (len(adj[q]), q)) if adj[q]]
    best = None
    for attempt in range(tries):
        start = starts[attempt % len(starts)]
        path, seen, steps = [start], {start}, 0

        def order(q):   # visit dead-end-prone neighbours first (Warnsdorff's rule)
            return sorted(adj[q], key=lambda x: (sum(y not in seen for y in adj[x]), rng.random()))

        stack = [iter(order(start))]
        while stack and steps < step_cap and len(path) < length:
            steps += 1
            nxt = next(stack[-1], None)
            if nxt is None:                      # dead end: step back
                stack.pop()
                seen.discard(path.pop())
            elif nxt not in seen:
                path.append(nxt)
                seen.add(nxt)
                stack.append(iter(order(nxt)))
        if len(path) == length:
            cost = sum(edges[(min(a, b), max(a, b))] for a, b in zip(path[:-1], path[1:]))
            if best is None or cost < best[0]:
                best = (cost, list(path))
    return best


def find_chain(backend, length: int = C.N_QUBITS, seed: int = 0) -> list[int]:
    """A path of `length` connected physical qubits with low total gate error.
    First tries to avoid every coupler worse than 5%, then relaxes."""
    edges = edge_errors(backend)
    for max_error in (0.05, 0.2, 1.0):
        best = _search(edges, backend.num_qubits, length, max_error,
                       tries=60, step_cap=300_000, seed=seed)
        if best is not None:
            return best[1]
    raise RuntimeError(f"no chain of {length} qubits found on {backend.name}; "
                       "try a 156-qubit Heron r2 chip (e.g. ibm_fez)")


def chip_snapshot(backend, chain: list[int]) -> dict:
    """Error rates along the chain, from the backend's calibration data."""
    from qiskit import QuantumCircuit, transpile

    t = backend.target
    gate = two_qubit_gate_name(backend)
    edges = edge_errors(backend)
    # how many native gates does one Heisenberg gate cost?
    qc = QuantumCircuit(2)
    qc.rxx(0.37, 0, 1)
    qc.ryy(0.37, 0, 1)
    qc.rzz(0.37, 0, 1)
    native = transpile(qc, basis_gates=[gate, "sx", "rz", "x"], optimization_level=3,
                       seed_transpiler=0)
    counts = native.count_ops()
    n_1q = int(np.ceil((counts.get("sx", 0) + counts.get("x", 0)) / 2))
    return {
        "backend": backend.name,
        "chain": [int(q) for q in chain],
        "two_qubit_gate": gate,
        "two_qubit_gates_per_heisenberg_gate": int(counts.get(gate, 0)),
        "one_qubit_gates_per_qubit_per_heisenberg_gate": n_1q,
        "two_qubit_error": [edges[(min(a, b), max(a, b))] for a, b in zip(chain[:-1], chain[1:])],
        "one_qubit_error": [float(t["sx"][(q,)].error) for q in chain],
        "readout_error": [float(t["measure"][(q,)].error) for q in chain],
    }


def save_snapshot(path=C.CHIP_SNAPSHOT) -> dict:
    """Write the calibration snapshot used by the noisy simulator (set C).
    Uses Qiskit's bundled snapshot of ibm_fez, so no IBM account is needed."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    backend = FakeFez()
    snap = chip_snapshot(backend, find_chain(backend))
    path.write_text(json.dumps(snap, indent=1))
    return snap


# ---------------------------------------------------------------------------
# 2. the circuit in the chip's own language
# ---------------------------------------------------------------------------
@dataclass
class Prepared:
    isa_circuit: object          # the transpiled, still-parameterised circuit
    parameters: object           # its 216 gate-angle parameters, in slot order
    observables: list            # the 327 observables mapped to physical qubits
    chain: list[int]


def prepare(spec: CircuitSpec, backend, chain: list[int] | None = None,
            optimization_level: int = 1) -> Prepared:
    from qiskit.transpiler import generate_preset_pass_manager

    chain = chain if chain is not None else find_chain(backend, spec.n_qubits)
    qc, params = build_qiskit_circuit(spec)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      initial_layout=chain, seed_transpiler=0)
    isa = pm.run(qc)
    obs = [o.apply_layout(isa.layout) for o in build_observables(spec.n_qubits)]
    return Prepared(isa_circuit=isa, parameters=params, observables=obs, chain=list(chain))


# ---------------------------------------------------------------------------
# 3. what to run, and in what order
# ---------------------------------------------------------------------------
def build_run_plan(events: pd.DataFrame, n_events: int, sentinel_every: int = 20,
                   seed: int = 2026) -> pd.DataFrame:
    """Choose `n_events` test-window events (the same number from every day)
    and put them in RANDOM order, with a sentinel after every
    `sentinel_every` events.  Returns one row per circuit to execute:
    position | row (event row, or -1 for a sentinel) | is_sentinel."""
    rng = np.random.default_rng(seed)
    main = events[events["period"] == "main"]
    days = np.sort(main["day"].unique())
    per_day = np.full(len(days), n_events // len(days))
    per_day[: n_events % len(days)] += 1
    chosen = np.concatenate([rng.choice(main.loc[main["day"] == d, "row"].to_numpy(),
                                        size=k, replace=False)
                             for d, k in zip(days, per_day) if k > 0])
    order = rng.permutation(chosen)
    rows, flags = [], []
    for k, r in enumerate(order):
        if k % sentinel_every == 0:
            rows.append(-1)
            flags.append(True)
        rows.append(int(r))
        flags.append(False)
    rows.append(-1)
    flags.append(True)
    return pd.DataFrame({"position": np.arange(len(rows)), "row": rows, "is_sentinel": flags})


# ---------------------------------------------------------------------------
# 4. execution
# ---------------------------------------------------------------------------
def run_plan(prepared: Prepared, plan: pd.DataFrame, phi: np.ndarray, backend,
             shots: int = 1024, events_per_job: int = 50, mitigation: bool = True,
             verbose: bool = True) -> pd.DataFrame:
    """Execute the plan and return one row per executed circuit with the 327
    measured averages (columns X0..X108, Y0.., Z0..).

    phi : (n_all_events, 216) gate angles of every event (row-aligned with
          the event table).  Sentinels use the angles of the plan's first event.
    """
    from qiskit_ibm_runtime import EstimatorV2

    estimator = EstimatorV2(mode=backend)
    estimator.options.default_shots = shots
    if mitigation:
        # the settings HSBC describes: Pauli twirling + TREX readout mitigation
        estimator.options.resilience_level = 1
        estimator.options.twirling.enable_gates = True

    sentinel_row = int(plan.loc[~plan["is_sentinel"], "row"].iloc[0])
    angle_rows = np.where(plan["row"].to_numpy() >= 0, plan["row"].to_numpy(), sentinel_row)
    names = output_names(len(prepared.chain))
    obs_column = [[o] for o in prepared.observables]          # shape (327, 1): broadcast over events
    frames = []
    for start in range(0, len(plan), events_per_job):
        part = plan.iloc[start: start + events_per_job]
        values = phi[angle_rows[start: start + events_per_job]]      # (n, 216)
        job = estimator.run([(prepared.isa_circuit, obs_column, values)])
        result = job.result()[0]
        evs = np.asarray(result.data.evs)                      # (327, n)
        frame = part.copy()
        frame["job_id"] = job.job_id() if hasattr(job, "job_id") else "local"
        frame["finished_at"] = pd.Timestamp.now(tz="UTC")
        frame = pd.concat([frame.reset_index(drop=True),
                           pd.DataFrame(evs.T, columns=names)], axis=1)
        frames.append(frame)
        if verbose:
            print(f"  job {len(frames)}: circuits {start}..{start + len(part) - 1} done", flush=True)
    return pd.concat(frames, ignore_index=True)


def sentinel_drift(results: pd.DataFrame, shots: int) -> dict:
    """How much do repeated identical circuits disagree, compared with what
    the randomness of `shots` measurements alone would explain?"""
    names = [c for c in results.columns if c[0] in "XYZ" and c[1:].isdigit()]
    s = results.loc[results["is_sentinel"], names].to_numpy()
    observed = s.std(axis=0, ddof=1)
    expected = np.sqrt(np.clip(1.0 - s.mean(axis=0) ** 2, 0.0, None) / shots)
    half = len(s) // 2
    shift = np.abs(s[:half].mean(axis=0) - s[half:].mean(axis=0))
    return {"n_sentinels": int(len(s)),
            "median_observed_std": float(np.median(observed)),
            "median_shot_noise_std": float(np.median(expected)),
            "excess_ratio": float(np.median(observed) / np.median(expected)),
            "median_shift_first_vs_second_half": float(np.median(shift))}
