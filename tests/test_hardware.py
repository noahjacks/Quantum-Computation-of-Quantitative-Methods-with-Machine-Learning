"""
CHECK: the real-hardware pipeline is sound, without using any quantum time.

* the 109 qubits chosen on the chip really are connected in a row;
* the circuit translated for the chip uses exactly 3 entangling gates per
  Heisenberg gate and only the chosen qubits;
* the run plan sends events in random order with regular sentinel repeats;
* executing a short version of the circuit on a simulated IBM chip returns
  values that agree with the exact simulation (this catches any mix-up of
  which measurement belongs to which qubit).
"""
import numpy as np
import pandas as pd
import pytest

from qon.circuit import make_spec
from qon.hardware import (build_run_plan, chip_snapshot, edge_errors, find_chain, prepare,
                          run_plan, sentinel_drift)
from qon.simulator import simulate_noiseless


@pytest.fixture(scope="module")
def fake():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def test_chain_is_a_connected_line_of_distinct_qubits(fake):
    chain = find_chain(fake, 109)
    edges = edge_errors(fake)
    assert len(chain) == 109 and len(set(chain)) == 109
    for a, b in zip(chain[:-1], chain[1:]):
        assert (min(a, b), max(a, b)) in edges                # neighbours are really coupled
    snap = chip_snapshot(fake, chain)
    assert snap["two_qubit_gates_per_heisenberg_gate"] == 3
    assert max(snap["two_qubit_error"]) < 0.05                # no broken coupler on the path
    assert len(snap["two_qubit_error"]) == 108 and len(snap["readout_error"]) == 109


def test_circuit_for_the_chip_has_three_entangling_gates_per_heisenberg_gate(fake):
    spec = make_spec(98)
    chain = find_chain(fake, 109)
    prepared = prepare(spec, fake, chain)
    circuit = prepared.isa_circuit
    assert circuit.count_ops()["cz"] == 3 * spec.n_slots == 648
    used = {circuit.find_bit(q).index for inst in circuit.data for q in inst.qubits
            if inst.operation.name == "cz"}
    assert used == set(chain)
    assert len(prepared.observables) == 327
    assert len(circuit.parameters) == spec.n_slots


def test_run_plan_is_random_in_time_with_regular_sentinels():
    rng = np.random.default_rng(0)
    events = pd.DataFrame({"row": np.arange(3000), "day": np.repeat(np.arange(30), 100),
                           "period": ["calibration"] * 1000 + ["main"] * 2000})
    plan = build_run_plan(events, n_events=400, sentinel_every=20)
    ev = plan[~plan["is_sentinel"]]
    assert len(ev) == 400 and ev["row"].is_unique and ev["row"].min() >= 1000    # test window only
    per_day = events.set_index("row").loc[ev["row"], "day"].value_counts()
    assert per_day.min() == per_day.max() == 20                                   # same number every day
    days_in_run_order = events.set_index("row").loc[ev["row"], "day"].to_numpy()
    assert abs(np.corrcoef(np.arange(400), days_in_run_order)[0, 1]) < 0.15       # order unrelated to date
    assert plan["is_sentinel"].sum() == 21 and plan["is_sentinel"].iloc[0] and plan["is_sentinel"].iloc[-1]
    gaps = np.diff(np.where(plan["is_sentinel"])[0])
    assert set(gaps) == {21}                                                      # 20 events between sentinels
    assert build_run_plan(events, 400).equals(plan)                               # reproducible


def test_sentinel_statistics_detect_drift():
    rng = np.random.default_rng(1)
    names = [f"{p}{q}" for p in "XYZ" for q in range(4)]
    shots = 1000
    steady = 0.3 + rng.standard_normal((40, 12)) * np.sqrt((1 - 0.09) / shots)
    drifting = steady + np.linspace(0, 0.2, 40)[:, None]
    for values, lo, hi in ((steady, 0.7, 1.3), (drifting, 1.8, 10)):
        df = pd.DataFrame(values, columns=names)
        df["is_sentinel"] = True
        assert lo < sentinel_drift(df, shots)["excess_ratio"] < hi


def test_short_circuit_on_a_simulated_ibm_chip_matches_the_exact_simulation(fake):
    """End to end: choose qubits, translate, run through IBM's Estimator on a
    simulated noisy chip, and compare with the exact noiseless values."""
    from qiskit_aer import AerSimulator
    spec = make_spec(98, n_qubits=8)
    rng = np.random.default_rng(2)
    phi = spec.gate_angles(2 * np.pi * np.tanh(rng.normal(size=(40, 98)) / 3))
    backend = AerSimulator.from_backend(fake)
    prepared = prepare(spec, backend, find_chain(fake, 8))
    plan = pd.DataFrame({"position": [0, 1, 2, 3], "row": [-1, 5, 17, 5],
                         "is_sentinel": [True, False, False, False]})
    res = run_plan(prepared, plan, phi, backend, shots=2000, mitigation=False, verbose=False)
    names = [c for c in res.columns if c[0] in "XYZ" and c[1:].isdigit()]
    assert len(names) == 24
    exact = simulate_noiseless(spec, phi[[5, 5, 17, 5]])          # the sentinel repeats event 5
    measured = res[names].to_numpy()
    assert np.corrcoef(measured.ravel(), exact.ravel())[0, 1] > 0.97
    slope = (measured * exact).sum() / (exact ** 2).sum()
    assert 0.85 < slope < 1.0                                     # noise can only shrink
