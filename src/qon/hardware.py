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
4. `estimate_quantum_seconds`
                         IBM's free plan gives 10 minutes of quantum time per
                         28 days.  This works out, before anything is sent,
                         roughly how much a run will use.
5. `run_plan`            submit the jobs through IBM's Estimator with the same
                         error handling HSBC used (Pauli twirling + TREX) and
                         collect the 327 measured averages per event.  It
                         reads what every job really cost, STOPS BY ITSELF
                         before a budget of quantum seconds is crossed, and
                         hands the results over after every job so nothing
                         that was paid for can be lost.
6. `save_run`/`load_run` write those results to disk safely and read them
                         back, so an interrupted run continues where it
                         stopped instead of starting again.

The same code runs against a local simulated chip ("fake backend"), which is
how it is tested without spending any real quantum time; see
scripts/06_hardware_run.py.

`chip_snapshot` is also used by the noisy simulator (set C): it reads the
per-qubit error rates of the chosen path from IBM's published calibration
data, so the simulated chip has the same good and bad qubits as a real one.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

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
# 4. how much quantum time will it cost?
# ---------------------------------------------------------------------------
# IBM charges "quantum seconds": the time the chip is reserved for a job.  The
# free Open Plan gives 600 of them per 28 days.  IBM's published rule of thumb
# (quantum.cloud.ibm.com/docs/guides/estimate-job-run-time) is
#
#     about 2 seconds to load each batch of circuits into the electronics
#   + (pause between shots + length of the circuit) x number of shots
#
# The pause between shots (250 millionths of a second) dominates: our whole
# circuit takes only about 3 millionths of a second.
BATCH_LOADING_SECONDS = 2.0        # IBM's figure for loading one batch
DEFAULT_REP_DELAY = 250e-6         # pause between shots on most IBM chips
DEFAULT_BATCH_SIZE = 300           # most circuits one batch can hold
READOUT_CALIBRATION_ROUNDS = 32    # extra calibration circuits TREX runs per job (IBM default)
MEASUREMENT_BASES = 3              # X, Y and Z need separate measurements
BUDGET_SAFETY = 1.2                # assume the next job costs 20% more than the average so far
MIN_JOB_LIMIT_SECONDS = 1          # smallest per-job limit we ask IBM to enforce


def circuit_seconds(prepared: "Prepared", backend) -> float:
    """How long one run of the circuit takes on the chip, measurement included."""
    try:
        qc = prepared.isa_circuit.copy()
        qc.measure_active()
        return float(qc.estimate_duration(backend.target, unit="s"))
    except Exception:
        return 5e-6          # a few millionths of a second: negligible next to the pause


def rep_delay_seconds(backend) -> float:
    """The pause the chip leaves between two shots so the qubits can reset."""
    for read in (lambda: backend.default_rep_delay,
                 lambda: backend.configuration().default_rep_delay):
        try:
            value = read()
            if value:
                return float(value)
        except Exception:
            pass
    return DEFAULT_REP_DELAY


def twirl_copies(shots: int) -> tuple[int, int]:
    """IBM's "auto" rule for Pauli twirling: the shots of one circuit are
    spread over several randomly re-dressed copies of it.  Returns
    (number of copies, shots per copy).  The total stays equal to `shots`
    as long as `shots` is a multiple of 64."""
    per_copy = max(64, math.ceil(shots / 32))
    return math.ceil(shots / per_copy), per_copy


def estimate_quantum_seconds(n_circuits: int, shots: int, circuit_s: float,
                             rep_delay: float = DEFAULT_REP_DELAY, n_jobs: int = 1,
                             mitigation: bool = True,
                             batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[float, float]:
    """(low, high) estimate of the quantum seconds a run will use.

    The measuring time is firm.  What is uncertain is how many batches IBM
    splits a job into when twirling is on, because every twirled copy is a
    circuit of its own: "low" assumes one batch per job, "high" assumes the
    copies fill batches of `batch_size`.  It is an estimate; the only exact
    figure is what IBM reports after a job has run."""
    per_shot = rep_delay + circuit_s
    measuring = n_circuits * MEASUREMENT_BASES * shots * per_shot
    copies, calibration = 1, 0.0
    if mitigation:
        copies, per_copy = twirl_copies(shots)
        calibration = n_jobs * (BATCH_LOADING_SECONDS
                                + READOUT_CALIBRATION_ROUNDS * per_copy * per_shot)
    many_batches = max(n_jobs, math.ceil(n_circuits * MEASUREMENT_BASES * copies / batch_size))
    return (measuring + calibration + n_jobs * BATCH_LOADING_SECONDS,
            measuring + calibration + many_batches * BATCH_LOADING_SECONDS)


def billed_seconds(job, wait: float = 90.0, poll: float = 3.0) -> float | None:
    """Quantum seconds IBM charged for a finished job, or None if it will not
    say.  IBM reports 0 until its own bookkeeping has caught up, so this asks
    again for up to `wait` seconds before giving up."""
    deadline = time.monotonic() + wait
    while True:
        try:
            value = job.usage()
        except Exception:
            value = None
        if value:
            return float(value)
        if time.monotonic() >= deadline:
            break
        time.sleep(poll)
    try:
        value = job.usage(partial=True)       # last resort: the running total so far
    except Exception:
        value = None
    return float(value) if value else None


def _is_real_chip(backend) -> bool:
    try:
        from qiskit_ibm_runtime import IBMBackend
        return isinstance(backend, IBMBackend)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 5. execution, with a budget
# ---------------------------------------------------------------------------
@dataclass
class RunOutcome:
    """Everything a run has produced so far."""
    results: pd.DataFrame            # one row per executed circuit, 327 measured averages each
    jobs: list = field(default_factory=list)   # one record per job: id, circuits, cost, status
    completed: bool = False          # True when every circuit of the plan has been run
    stopped_because: str | None = None         # plain-English reason if the run ended early

    @property
    def used_seconds(self) -> float:
        """Quantum seconds spent so far (measured where IBM reported it)."""
        return float(sum(j["quantum_seconds"] for j in self.jobs))


def _outcome(frames: list, jobs: list, plan: pd.DataFrame, stopped: str | None) -> RunOutcome:
    if frames:
        results = pd.concat(frames, ignore_index=True).sort_values("position", ignore_index=True)
    else:
        results = pd.DataFrame(columns=list(plan.columns))
    return RunOutcome(results=results, jobs=list(jobs), completed=len(results) == len(plan),
                      stopped_because=stopped)


def run_plan(prepared: Prepared, plan: pd.DataFrame, phi: np.ndarray, backend,
             shots: int = 1024, events_per_job: int = 50, mitigation: bool = True,
             verbose: bool = True, budget_seconds: float | None = None,
             earlier: RunOutcome | None = None, checkpoint=None,
             usage_reader=None) -> RunOutcome:
    """Execute the plan, one job of `events_per_job` circuits at a time.

    phi            : (n_all_events, 216) gate angles of every event (row-aligned
                     with the event table).  Sentinels use the angles of the
                     plan's first event.
    budget_seconds : the most quantum seconds this run may use IN TOTAL,
                     counting `earlier`.  Before each job the cost of the next
                     one is predicted from what the jobs so far really cost;
                     if it would cross the budget the run stops there.  As a
                     second line of defence IBM is told to cancel any job that
                     overruns what is left.  None = no limit (simulated chip).
    earlier        : what a previous, interrupted run already produced.  Those
                     circuits are not run again.
    checkpoint     : called with the RunOutcome after EVERY job, so the caller
                     can save it.  A crash then loses nothing already paid for.
    usage_reader   : function(job) -> quantum seconds charged, or None.  By
                     default IBM is asked; a simulated chip costs nothing.

    Because the plan is in random order, a run that stops early still holds a
    fair random sample of the planned events.
    """
    from qiskit_ibm_runtime import EstimatorV2

    def new_estimator():
        est = EstimatorV2(mode=backend)
        est.options.default_shots = shots
        if mitigation:
            # the settings HSBC describes: Pauli twirling + TREX readout mitigation
            est.options.resilience_level = 1
            est.options.twirling.enable_gates = True
        return est

    estimator = new_estimator()
    use_job_limit = True          # switched off if IBM refuses a job that carries the limit

    say = print if verbose else (lambda *a, **k: None)
    if usage_reader is None:
        usage_reader = billed_seconds if _is_real_chip(backend) else (lambda job: None)
    metered = budget_seconds is not None

    sentinel_row = int(plan.loc[~plan["is_sentinel"], "row"].iloc[0])
    plan_rows = plan["row"].to_numpy()
    angle_rows = np.where(plan_rows >= 0, plan_rows, sentinel_row)
    names = output_names(len(prepared.chain))
    obs_column = [[o] for o in prepared.observables]          # shape (327, 1): broadcast over events

    frames, jobs = [], []
    if earlier is not None:
        jobs = list(earlier.jobs)
        if len(earlier.results):
            frames = [earlier.results]
    done = set(int(p) for p in frames[0]["position"]) if frames else set()
    todo = np.flatnonzero(~plan["position"].isin(done).to_numpy())    # circuits still to run
    if done:
        say(f"  continuing an earlier run: {len(done)} circuits already done, {len(todo)} to go")

    before = sum(j["quantum_seconds"] for j in jobs)          # spent by the earlier run, if any
    circuit_s, delay = circuit_seconds(prepared, backend), rep_delay_seconds(backend)
    batch_size = getattr(backend, "max_circuits", None) or DEFAULT_BATCH_SIZE
    stopped = None
    for start in range(0, len(todo), events_per_job):
        idx = todo[start: start + events_per_job]
        part = plan.iloc[idx]
        used = sum(j["quantum_seconds"] for j in jobs)

        # ---- what will the next job cost?  Prefer what the jobs so far really cost.
        measured = [j for j in jobs if j["usage_source"] == "measured" and j["status"] == "done"]
        if measured:
            per_circuit = (sum(j["quantum_seconds"] for j in measured)
                           / sum(j["n_circuits"] for j in measured))
            predicted = BUDGET_SAFETY * per_circuit * len(part)
        else:
            predicted = estimate_quantum_seconds(len(part), shots, circuit_s, delay, 1,
                                                 mitigation, batch_size)[1]
        if metered:
            if used + predicted > budget_seconds:
                stopped = (f"the next job is expected to cost about {predicted:.0f} quantum seconds "
                           f"and {used - before:.0f} of the {budget_seconds - before:.0f}-second "
                           "budget are used")
                break
            # second line of defence: IBM cancels the job if it overruns what is left
            if use_job_limit:
                estimator.options.max_execution_time = max(MIN_JOB_LIMIT_SECONDS,
                                                           math.ceil(budget_seconds - used))

        # ---- run it
        record = {"job_id": None, "n_circuits": int(len(part)),
                  "first_position": int(part["position"].iloc[0]),
                  "quantum_seconds": 0.0, "usage_source": "none", "status": "submitted"}
        job, evs = None, None
        pub = [(prepared.isa_circuit, obs_column, phi[angle_rows[idx]])]
        try:
            try:
                job = estimator.run(pub)
            except Exception as refusal:
                if not (metered and use_job_limit):
                    raise
                # IBM did not accept the job as sent.  The per-job limit is the one
                # unusual thing on it, so try once more without; the budget check
                # before every job still applies.  (Nothing has been charged.)
                say(f"  job not accepted with a per-job limit ({type(refusal).__name__}); "
                    "retrying once without it", flush=True)
                use_job_limit, estimator = False, new_estimator()
                job = estimator.run(pub)
            record["job_id"] = str(job.job_id()) if hasattr(job, "job_id") else "local"
            say(f"  job {len(jobs) + 1}: {len(part)} circuits sent (id {record['job_id']})", flush=True)
            evs = np.asarray(job.result()[0].data.evs)         # (327, n)
            record["status"] = "done"
        except KeyboardInterrupt:
            record["status"] = "interrupted"
            stopped = "the run was interrupted from the keyboard"
            try:                                               # stop the chip working on it
                job.cancel()
            except Exception:
                pass
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            stopped = f"a job failed ({record['error'][:160]})"

        # ---- what did it cost?  (a job that was never accepted costs nothing)
        seconds = usage_reader(job) if job is not None else None
        if seconds is not None:
            record.update(quantum_seconds=float(seconds), usage_source="measured")
        elif metered and job is not None:
            record.update(quantum_seconds=float(predicted), usage_source="estimated")
        jobs.append(record)

        if record["status"] == "done":
            frame = part.copy().reset_index(drop=True)
            frame["job_id"] = record["job_id"]
            frame["finished_at"] = pd.Timestamp.now(tz="UTC")
            frame["shots"] = shots
            frames.append(pd.concat([frame, pd.DataFrame(evs.T, columns=names)], axis=1))
        if checkpoint is not None:
            checkpoint(_outcome(frames, jobs, plan, stopped))
        if metered:
            total = sum(j["quantum_seconds"] for j in jobs)
            say(f"         {record['status']}; cost {record['quantum_seconds']:.1f} quantum seconds "
                f"({record['usage_source']}); used {total - before:.0f} of "
                f"{budget_seconds - before:.0f}", flush=True)
        elif record["status"] == "done":
            say(f"         done ({sum(len(f) for f in frames)} of {len(plan)} circuits)", flush=True)
        if stopped:
            break
    final = _outcome(frames, jobs, plan, stopped)
    if checkpoint is not None:
        checkpoint(final)                 # so the reason for stopping is on record too
    return final


# ---------------------------------------------------------------------------
# 6. saving a run, and picking it up again
# ---------------------------------------------------------------------------
def _write_atomically(path: Path, write) -> None:
    """Write to a temporary file, then swap it in.  If the computer dies half
    way through, the previous complete file is still there."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_run(path, outcome: RunOutcome, manifest: dict) -> None:
    """Save the results (`path`, parquet) and the record of the run
    (`path` with .json: settings, every job's id and cost, why it stopped)."""
    path = Path(path)
    if len(outcome.results):
        _write_atomically(path, lambda tmp: outcome.results.to_parquet(tmp, index=False))
    record = dict(manifest)
    record.update(n_circuits_done=int(len(outcome.results)), complete=bool(outcome.completed),
                  quantum_seconds_used=round(outcome.used_seconds, 2),
                  stopped_because=outcome.stopped_because, jobs=outcome.jobs)
    _write_atomically(path.with_suffix(".json"),
                      lambda tmp: tmp.write_text(json.dumps(record, indent=1, default=str)))


def load_run(path) -> tuple[RunOutcome, dict] | None:
    """Read back what `save_run` wrote, or None if there is no saved run."""
    path = Path(path)
    record_file = path.with_suffix(".json")
    if not record_file.exists():
        return None
    record = json.loads(record_file.read_text())
    if "jobs" not in record:                 # a record written before runs could be resumed
        return None
    results = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    outcome = RunOutcome(results=results, jobs=list(record["jobs"]),
                         completed=bool(record.get("complete", False)),
                         stopped_because=record.get("stopped_because"))
    return outcome, record


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
