"""
CHECK: the real-hardware pipeline is sound, without using any quantum time.

* the 109 qubits chosen on the chip really are connected in a row;
* the circuit translated for the chip uses exactly 3 entangling gates per
  Heisenberg gate and only the chosen qubits;
* the run plan sends events in random order with regular sentinel repeats;
* executing a short version of the circuit on a simulated IBM chip returns
  values that agree with the exact simulation (this catches any mix-up of
  which measurement belongs to which qubit);
* the run cannot overspend: it stops before a budget of quantum seconds is
  crossed, keeps everything already measured, continues later without
  repeating a circuit, and survives a job that fails.

The budget checks charge a pretend price per job on a simulated chip.  What
IBM's own servers report for a real job cannot be tested without an account.
"""
import json

import numpy as np
import pandas as pd
import pytest

from qon.circuit import make_spec
from qon.hardware import (build_run_plan, chip_snapshot, edge_errors, estimate_quantum_seconds,
                          find_chain, load_run, prepare, run_plan, save_run, sentinel_drift,
                          twirl_copies)
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
    outcome = run_plan(prepared, plan, phi, backend, shots=2000, mitigation=False, verbose=False)
    assert outcome.completed and outcome.stopped_because is None and outcome.used_seconds == 0
    res = outcome.results
    names = [c for c in res.columns if c[0] in "XYZ" and c[1:].isdigit()]
    assert len(names) == 24
    exact = simulate_noiseless(spec, phi[[5, 5, 17, 5]])          # the sentinel repeats event 5
    measured = res[names].to_numpy()
    assert np.corrcoef(measured.ravel(), exact.ravel())[0, 1] > 0.97
    slope = (measured * exact).sum() / (exact ** 2).sum()
    assert 0.85 < slope < 1.0                                     # noise can only shrink


# ---------------------------------------------------------------------------
# the budget: a run must not be able to overspend, or to lose what it measured
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small(fake):
    """A short (8-qubit) version of the pipeline on a simulated IBM chip:
    9 circuits = 7 events with a sentinel at each end."""
    from qiskit_aer import AerSimulator
    spec = make_spec(98, n_qubits=8)
    rng = np.random.default_rng(3)
    phi = spec.gate_angles(2 * np.pi * np.tanh(rng.normal(size=(40, 98)) / 3))
    backend = AerSimulator.from_backend(fake)
    prepared = prepare(spec, backend, find_chain(fake, 8))
    plan = pd.DataFrame({"position": np.arange(9), "row": [-1, 4, 9, 14, 19, 24, 29, 34, -1],
                         "is_sentinel": [True] + [False] * 7 + [True]})
    return prepared, plan, phi, backend


def test_cost_estimate_follows_ibms_published_formula():
    assert twirl_copies(1024) == (16, 64) and twirl_copies(4096) == (32, 128)
    # the default run: 421 circuits, 1024 shots, 9 jobs, a circuit of 3.432 millionths of a second
    per_shot = 250e-6 + 3.432e-6
    low, high = estimate_quantum_seconds(421, 1024, 3.432e-6, 250e-6, n_jobs=9)
    measuring = 421 * 3 * 1024 * per_shot
    calibration = 9 * (2 + 32 * 64 * per_shot)
    assert low == pytest.approx(measuring + calibration + 9 * 2)
    assert high == pytest.approx(measuring + calibration + 68 * 2)     # 421*3*16 copies, 300 per batch
    assert 360 < low < 375 and 480 < high < 495                         # about 6 to 8 minutes
    # HSBC's 4096 shots would not fit into the free 10 minutes
    assert estimate_quantum_seconds(421, 4096, 3.432e-6, 250e-6, n_jobs=9)[0] > 600
    # without error handling: measuring plus one loading per job, nothing else
    plain = estimate_quantum_seconds(10, 100, 0.0, 250e-6, n_jobs=2, mitigation=False)
    assert plain == pytest.approx((10 * 3 * 100 * 250e-6 + 4,) * 2)


def test_run_stops_before_the_budget_keeps_results_and_continues_later(small, tmp_path):
    prepared, plan, phi, backend = small
    path = tmp_path / "hardware_features.parquet"
    settings = {"backend": "pretend chip", "shots": 200}
    saved, charged = [], []

    def price(job):                       # every job "costs" 30 quantum seconds
        charged.append(1)
        return 30.0

    def checkpoint(outcome):
        save_run(path, outcome, settings)
        saved.append(len(outcome.results))

    run = dict(shots=200, events_per_job=3, mitigation=False, verbose=False,
               usage_reader=price, checkpoint=checkpoint)

    # a budget too small for even one job: nothing is sent, nothing is charged
    nothing = run_plan(prepared, plan, phi, backend, budget_seconds=1, **run)
    assert len(nothing.results) == 0 and nothing.jobs == [] and charged == []
    assert "budget" in nothing.stopped_because
    assert not path.exists() and load_run(path)[1]["n_circuits_done"] == 0   # a record, but no results file
    saved.clear()

    # budget 70: job 1 costs 30, job 2 costs 30 (total 60); a third at the
    # measured price plus the 20% margin (36) would reach 96, so it is not sent
    first = run_plan(prepared, plan, phi, backend, budget_seconds=70, **run)
    assert not first.completed and "budget" in first.stopped_because
    assert first.used_seconds == 60 <= 70 and len(charged) == 2
    assert list(first.results["position"]) == [0, 1, 2, 3, 4, 5]
    assert saved == [3, 6, 6]                    # saved after EVERY job, and once more on stopping

    # what is on disk is the complete record of the interrupted run
    earlier, record = load_run(path)
    assert len(earlier.results) == 6 and record["complete"] is False
    assert record["quantum_seconds_used"] == 60 and record["backend"] == "pretend chip"
    assert [j["usage_source"] for j in record["jobs"]] == ["measured", "measured"]
    assert json.loads(path.with_suffix(".json").read_text())["stopped_because"] == first.stopped_because

    # continuing later runs only the missing job and never repeats a circuit
    second = run_plan(prepared, plan, phi, backend, budget_seconds=200, earlier=earlier, **run)
    assert second.completed and second.stopped_because is None
    assert len(charged) == 3 and second.used_seconds == 90
    assert list(second.results["position"]) == list(range(9))
    names = [c for c in second.results.columns if c[0] in "XYZ" and c[1:].isdigit()]
    np.testing.assert_array_equal(second.results[names].to_numpy()[:6],
                                  first.results[names].to_numpy())  # earlier measurements kept as they were
    assert load_run(path)[1]["complete"] is True


def test_a_failed_job_does_not_lose_the_results_before_it(small, monkeypatch):
    import qiskit_ibm_runtime
    prepared, plan, phi, backend = small
    real_run, calls = qiskit_ibm_runtime.EstimatorV2.run, []

    def flaky(self, *args, **kwargs):     # the second submission fails
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("chip went offline")
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(qiskit_ibm_runtime.EstimatorV2, "run", flaky)
    run = dict(shots=200, events_per_job=3, mitigation=False, verbose=False)
    out = run_plan(prepared, plan, phi, backend, **run)
    assert len(out.results) == 3 and not out.completed               # job 1 survives
    assert [j["status"] for j in out.jobs] == ["done", "failed"]
    assert "chip went offline" in out.stopped_because
    assert out.jobs[1]["quantum_seconds"] == 0                       # never accepted, so never charged

    # with a budget, a job refused at the door is tried once more without the
    # per-job limit; here the retry succeeds and the run completes
    calls.clear()
    out = run_plan(prepared, plan, phi, backend, budget_seconds=1000,
                   usage_reader=lambda job: 5.0, **run)
    assert out.completed and len(calls) == 4 and out.used_seconds == 15
