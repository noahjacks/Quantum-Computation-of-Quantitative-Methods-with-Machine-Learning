"""
STEP 6 - Run the circuit on a REAL IBM quantum computer (feature set D).

THIS IS THE ONLY STEP THAT NEEDS AN IBM QUANTUM ACCOUNT AND SPENDS REAL
QUANTUM TIME.  The free Open Plan gives 10 minutes (600 "quantum seconds")
per 28 days, so the script is built not to waste them:

  * BUDGET.  It may spend at most --budget-seconds (default 480 = 8 minutes).
    After every job it reads what IBM really charged, and it does not send
    the next job if that would cross the budget.  IBM is also told to cancel
    any single job that overruns what is left.
  * NOTHING IS LOST.  Results are saved after every job.  If the run stops
    (budget reached, a failed job, a closed laptop), give the same command
    again later: it continues where it stopped and repeats nothing.
  * A SHORT RUN IS STILL VALID.  Events are sent in random order, so the
    events measured before a stop are a fair random sample of the plan.

Do the three modes in this order:

1) Dry run on a simulated chip - free, no account, takes a couple of minutes:

       .venv/bin/python scripts/06_hardware_run.py --mode local

   Checks that the full 109-qubit circuit fits on the chip, prints what the
   real run is expected to cost, then executes a 12-qubit version of exactly
   the same pipeline (find qubits in a row, translate the circuit, submit,
   collect the measured averages) and compares the answers with the exact
   simulation.

2) Timing job on the real chip - costs roughly 10 to 15 quantum seconds:

       .venv/bin/python scripts/06_hardware_run.py --mode timing --backend ibm_fez

   Runs 5 circuits and prints what IBM charged for them, which turns the
   estimate into a measurement.

3) The real run:

       .venv/bin/python scripts/06_hardware_run.py --mode run --backend ibm_fez \
              --events 400 --shots 1024 --budget-seconds 480

   Then analyse it with scripts/07_analyse_hardware.py.

   Estimated cost of these settings: 6 to 8 minutes of quantum time (IBM's
   published formula; see estimate_quantum_seconds in src/qon/hardware.py).
   Do not raise --shots to HSBC's 4096: that needs about 22 minutes.

Before modes 2 and 3 you must save your IBM Quantum API key ONCE, yourself,
in a Python prompt (never paste it into a script or a chat):

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(token="<your key>", instance="<your instance CRN>")

Design choices that are FROZEN before the run (see PREREGISTRATION.md):
  * events are sent in RANDOM order (seed 2026), not by date
  * a fixed "sentinel" circuit is repeated every 20 events to measure drift
  * Pauli twirling + TREX readout mitigation, exactly as in the HSBC paper

Honest note: the budget logic is tested on a simulated chip with pretend
prices (tests/test_hardware.py).  The three things only IBM's servers can
confirm - the usage figure they report, their per-job limit, and the
remaining-allowance figure - have not been exercised, because that needs an
account.  The timing job is the first real test of them.
"""
import argparse
import math
import time

import _path  # noqa: F401
import numpy as np

from qon import config as C
from qon.circuit import make_spec
from qon.feature_sets import HARDWARE_FILE, SETS_FILE, load_feature_sets
from qon.dataset import FEATURE_COLUMNS
from qon.hardware import (BUDGET_SAFETY, build_run_plan, chip_snapshot, circuit_seconds,
                          estimate_quantum_seconds, find_chain, load_run, prepare,
                          rep_delay_seconds, run_plan, save_run, sentinel_drift)

N_INPUTS = len(FEATURE_COLUMNS)          # 98 inputs per quote request
DEFAULT_BUDGET = {"run": 480.0, "timing": 60.0}     # quantum seconds; the free plan gives 600
ALLOWANCE_MARGIN = 15.0                  # never plan to use the last seconds of the allowance
EVENTS_PER_JOB = 50                      # circuits per job in the real run
TIMING_FILE = C.DATA_PROCESSED / "hardware_features_TIMING.parquet"
DRY_RUN_FILE = C.DATA_PROCESSED / "hardware_features_LOCAL_DRY_RUN.parquet"


def minutes(seconds: float) -> str:
    return f"{seconds:.0f} quantum seconds ({seconds / 60:.1f} min)"


def remaining_allowance(service) -> float | None:
    """Quantum seconds IBM says are left in this account's allowance, or None
    if it does not say."""
    try:
        value = service.usage().get("usage_remaining_seconds")
        return None if value is None else float(value)
    except Exception:
        return None


def cost_estimate(prepared, backend, n_circuits: int, shots: int, events_per_job: int,
                  mitigation: bool) -> tuple[float, float]:
    n_jobs = max(1, math.ceil(n_circuits / events_per_job))
    batch = getattr(backend, "max_circuits", None) or 300
    return estimate_quantum_seconds(n_circuits, shots, circuit_seconds(prepared, backend),
                                    rep_delay_seconds(backend), n_jobs, mitigation, batch)


def set_aside(path) -> None:
    """Rename an old results file instead of deleting it."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for f in (path, path.with_suffix(".json")):
        if f.exists():
            f.rename(f.with_name(f"{f.stem}.old-{stamp}{f.suffix}"))
            print(f"  kept the previous file as {f.stem}.old-{stamp}{f.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", choices=["local", "timing", "run"], default="local")
    ap.add_argument("--backend", default="ibm_fez", help="IBM backend name (modes timing/run)")
    ap.add_argument("--events", type=int, default=None, help="number of events to run")
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--events-per-job", type=int, default=EVENTS_PER_JOB,
                    help="circuits sent to the chip at a time; results are saved after each "
                         "lot (smaller lots cost more: every job has fixed overheads)")
    ap.add_argument("--budget-seconds", type=float, default=None,
                    help="most quantum seconds to spend in this session "
                         "(default 480 for --mode run, 60 for --mode timing)")
    ap.add_argument("--fresh", action="store_true",
                    help="--mode run: start again instead of continuing a saved run "
                         "(the old files are renamed, not deleted)")
    ap.add_argument("--dry-run-qubits", type=int, default=12,
                    help="length of the short circuit executed in --mode local")
    args = ap.parse_args()

    fs = load_feature_sets()
    events = fs["events"]
    with np.load(SETS_FILE) as data:
        phi = data["phi"].astype(float)                  # gate angles of every event
    spec = make_spec(n_features=N_INPUTS)                # the real 109-qubit circuit
    assert spec.n_slots == phi.shape[1]
    exact = fs["sets"]["B"]                              # perfect-chip values, for a sanity check
    earlier, chain, budget, session_budget = None, None, None, None

    if args.mode == "local":
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime.fake_provider import FakeFez
        from qon.kernel_backtest import scaled_angles
        from qon.simulator import simulate_noiseless
        fake = FakeFez()
        print("DRY RUN on a simulated copy of ibm_fez (no account, no quantum time).")
        # (a) the full 109-qubit circuit: check that it fits on the chip
        full = prepare(spec, fake, find_chain(fake, spec.n_qubits))
        print(f"  full circuit fits: {spec.n_qubits} qubits in a row, depth "
              f"{full.isa_circuit.depth()}, {full.isa_circuit.count_ops().get('cz', 0)} entangling gates")
        # (b) what the real run is expected to cost (IBM's published formula)
        real_plan = build_run_plan(events, 400)
        low, high = cost_estimate(full, fake, len(real_plan), 1024, EVENTS_PER_JOB, True)
        print(f"  the real run (400 events = {len(real_plan)} circuits, 1024 shots) is estimated at "
              f"{low / 60:.1f} to {high / 60:.1f} minutes of quantum time;")
        print(f"  the free plan gives 10 minutes per 28 days, and the script stops by itself at "
              f"{DEFAULT_BUDGET['run'] / 60:.0f}.")
        # (c) simulating a noisy 109-qubit chip gate by gate takes hours, so the
        #     rehearsal executes a short version of the same circuit instead
        spec = make_spec(n_features=N_INPUTS, n_qubits=args.dry_run_qubits)
        phi = spec.gate_angles(scaled_angles())
        exact = simulate_noiseless(spec, phi)
        backend = AerSimulator.from_backend(fake)
        chain = find_chain(fake, spec.n_qubits)
        n_events, shots, mitigation = args.events or 12, args.shots or 256, False
        out_file = DRY_RUN_FILE
        print(f"  executing the {spec.n_qubits}-qubit version on the simulated chip ...")
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            service = QiskitRuntimeService()
        except Exception as exc:
            raise SystemExit("No saved IBM Quantum account found. Save your key once as "
                             "described at the top of this file.\n" + str(exc))
        backend = service.backend(args.backend)
        mitigation = True
        if args.mode == "timing":
            n_events, shots, out_file = 5, args.shots or 1024, TIMING_FILE
        else:
            n_events, shots, out_file = args.events or 400, args.shots or 1024, HARDWARE_FILE
        session_budget = (args.budget_seconds if args.budget_seconds is not None
                          else DEFAULT_BUDGET[args.mode])

        # ---- continue a saved run, or refuse to overwrite one
        if args.mode == "run" and args.fresh:
            set_aside(out_file)
        elif args.mode == "run" and (out_file.exists() or out_file.with_suffix(".json").exists()):
            saved = load_run(out_file)
            if saved is None:
                raise SystemExit(f"{out_file.name} exists but has no run record, so it cannot be "
                                 "continued. Add --fresh to start again (the old file is kept "
                                 "under a new name).")
            earlier, record = saved
            wanted = {"backend": backend.name, "shots": shots, "n_events": n_events,
                      "n_qubits": spec.n_qubits}
            differs = {k: (record.get(k), v) for k, v in wanted.items() if record.get(k) != v}
            if differs:
                raise SystemExit(
                    "A saved hardware run exists with different settings "
                    f"(saved, asked for): {differs}.\nTo continue it, use --backend "
                    f"{record.get('backend')} --events {record.get('n_events')} --shots "
                    f"{record.get('shots')}. To start again, add --fresh (the old files are kept "
                    "under a new name).")
            chain = [int(q) for q in record["chain"]]          # the same qubits as before
            if earlier.completed:
                print(f"The saved run is already complete ({len(earlier.results)} circuits). "
                      "Nothing to do; no quantum time spent.")
                return

        # ---- how much may be spent
        left = remaining_allowance(service)
        if left is None:
            print("IBM did not report how much of the allowance is left; check it on "
                  "quantum.cloud.ibm.com before a long run.")
        else:
            print(f"IBM reports {minutes(left)} left in this account's allowance.")
            if left - ALLOWANCE_MARGIN < session_budget:
                session_budget = max(0.0, left - ALLOWANCE_MARGIN)
                print(f"  budget for this session lowered to {minutes(session_budget)}")
        if session_budget <= 0:
            print("No quantum time is available for this session, so nothing was sent. The "
                  "allowance refills over a rolling 28 days.")
            return
        budget = session_budget + (earlier.used_seconds if earlier is not None else 0.0)
        if chain is None:
            chain = find_chain(backend, spec.n_qubits)

    snap = chip_snapshot(backend if args.mode != "local" else fake, chain)
    print(f"backend: {snap['backend']}   {len(chain)}-qubit chain starts {chain[:8]} ...")
    print(f"  median two-qubit gate error on the chain: {np.median(snap['two_qubit_error']):.4f}"
          f"   worst: {max(snap['two_qubit_error']):.4f}")
    prepared = prepare(spec, backend, chain)
    ops = prepared.isa_circuit.count_ops()
    print(f"  transpiled circuit: depth {prepared.isa_circuit.depth()}, gates {dict(ops)}")

    plan = build_run_plan(events, n_events)
    print(f"  plan: {int((~plan['is_sentinel']).sum())} events + "
          f"{int(plan['is_sentinel'].sum())} sentinel repeats = {len(plan)} circuits, "
          f"{shots} shots x 3 measurement bases each")
    if budget is not None:
        to_run = len(plan) - (len(earlier.results) if earlier is not None else 0)
        low, high = cost_estimate(prepared, backend, to_run, shots, args.events_per_job, mitigation)
        print(f"  estimated cost of the {to_run} circuits still to run: {low:.0f} to {high:.0f} "
              f"quantum seconds ({low / 60:.1f} to {high / 60:.1f} min)")
        print(f"  budget for this session: {minutes(session_budget)}")
        if high > session_budget:
            print("  NOTE: this may not fit. The run stops by itself before the budget is crossed "
                  "and keeps what it has;\n        give the same command again later to continue.")

    manifest = {"mode": args.mode, "backend": snap["backend"], "chain": [int(q) for q in chain],
                "n_qubits": spec.n_qubits, "shots": shots, "n_events": n_events,
                "events_per_job": args.events_per_job, "n_circuits": int(len(plan)),
                "mitigation": mitigation, "budget_seconds": budget}
    outcome = run_plan(prepared, plan, phi, backend, shots=shots,
                       events_per_job=args.events_per_job, mitigation=mitigation,
                       budget_seconds=budget, earlier=earlier,
                       checkpoint=lambda o: save_run(out_file, o, manifest))
    results = outcome.results
    if len(results) and int(results["is_sentinel"].sum()) >= 2:
        manifest["sentinels"] = sentinel_drift(results, shots)
    save_run(out_file, outcome, manifest)

    # ---- what happened
    if outcome.completed:
        print(f"run complete: {len(results)} circuits saved in {out_file.name}")
    else:
        print(f"RUN STOPPED EARLY: {outcome.stopped_because}.")
        print(f"  {len(results)} of {len(plan)} circuits are saved in {out_file.name}.")
        if args.mode == "run":
            print("  To continue, give the same command again; circuits already done are not repeated.")
    if budget is not None:
        before = earlier.used_seconds if earlier is not None else 0.0
        print(f"quantum time used in this session: {minutes(outcome.used_seconds - before)}"
              + (f"; by this run in total: {minutes(outcome.used_seconds)}" if before else ""))
    if "sentinels" in manifest:
        print("sentinel check (identical circuits repeated through the run):")
        for key, value in manifest["sentinels"].items():
            print(f"  {key}: {value:.4g}")

    # sanity check against the exact simulation of a perfect chip
    done_events = results[~results["is_sentinel"].astype(bool)] if len(results) else results
    if len(done_events):
        names = [c for c in results.columns if c[0] in "XYZ" and c[1:].isdigit()]
        measured = done_events[names].to_numpy(dtype=float)
        reference = exact[done_events["row"].to_numpy(dtype=int)]
        corr = np.corrcoef(measured.ravel(), reference.ravel())[0, 1]
        slope = (measured * reference).sum() / (reference ** 2).sum()
        print(f"agreement with the perfect-chip simulation: correlation {corr:.3f}, "
              f"average shrink factor {slope:.3f}")

    if args.mode == "timing":
        paid = [j for j in outcome.jobs if j["usage_source"] == "measured" and j["status"] == "done"]
        if paid:
            seconds = sum(j["quantum_seconds"] for j in paid)
            per_circuit = seconds / sum(j["n_circuits"] for j in paid)
            fit = int(DEFAULT_BUDGET["run"] / (per_circuit * 1.05 * BUDGET_SAFETY))
            print(f"IBM charged {seconds:.1f} quantum seconds for these circuits = "
                  f"{per_circuit:.2f} per circuit.")
            print("  (A small job carries more loading time per circuit than a large one, so the "
                  "real run should cost a little less per circuit.)")
            print(f"  At this price about {fit} events fit in {DEFAULT_BUDGET['run']:.0f} quantum "
                  "seconds. Pass that number as --events, or leave it: the run stops by itself.")
        else:
            print("IBM has not reported the cost of this job yet. Look up its 'usage' on "
                  f"quantum.cloud.ibm.com and divide by {len(plan)} circuits.")


if __name__ == "__main__":
    main()
