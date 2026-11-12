"""
STEP 6 - Run the circuit on a REAL IBM quantum computer (feature set D).

THIS IS THE ONLY STEP THAT NEEDS AN IBM QUANTUM ACCOUNT AND SPENDS REAL
QUANTUM TIME.  Do the three modes in this order:

1) Dry run on a simulated chip - free, no account, takes a couple of minutes:

       .venv/bin/python scripts/06_hardware_run.py --mode local

   Checks that the full 109-qubit circuit fits on the chip, then executes a
   12-qubit version of exactly the same pipeline (find qubits in a row,
   translate the circuit, submit, collect the measured averages) and
   compares the answers with the exact simulation.

2) Timing job on the real chip - costs a few seconds of quantum time:

       .venv/bin/python scripts/06_hardware_run.py --mode timing --backend ibm_fez

   Runs 5 circuits and reports how many quantum-seconds they used, so you can
   work out how many events fit in your budget (the Open Plan gives 10
   minutes per 28 days).

3) The real run:

       .venv/bin/python scripts/06_hardware_run.py --mode run --backend ibm_fez \
              --events 400 --shots 1024

   Then analyse it with scripts/07_analyse_hardware.py.

Before modes 2 and 3 you must save your IBM Quantum API key ONCE, yourself,
in a Python prompt (never paste it into a script or a chat):

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(token="<your key>", instance="<your instance CRN>")

Design choices that are FROZEN before the run (see PREREGISTRATION.md):
  * events are sent in RANDOM order (seed 2026), not by date
  * a fixed "sentinel" circuit is repeated every 20 events to measure drift
  * Pauli twirling + TREX readout mitigation, exactly as in the HSBC paper
"""
import argparse
import json

import _path  # noqa: F401
import numpy as np

from qon import config as C
from qon.circuit import make_spec
from qon.feature_sets import HARDWARE_FILE, SETS_FILE, load_feature_sets
from qon.dataset import FEATURE_COLUMNS
from qon.hardware import (build_run_plan, chip_snapshot, find_chain, prepare, run_plan,
                          sentinel_drift)

N_INPUTS = len(FEATURE_COLUMNS)          # 98 inputs per quote request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--mode", choices=["local", "timing", "run"], default="local")
    ap.add_argument("--backend", default="ibm_fez", help="IBM backend name (modes timing/run)")
    ap.add_argument("--events", type=int, default=None, help="number of events to run")
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--events-per-job", type=int, default=50)
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
        # (b) simulating a noisy 109-qubit chip gate by gate takes hours, so the
        #     rehearsal executes a short version of the same circuit instead
        spec = make_spec(n_features=N_INPUTS, n_qubits=args.dry_run_qubits)
        phi = spec.gate_angles(scaled_angles())
        exact = simulate_noiseless(spec, phi)
        backend = AerSimulator.from_backend(fake)
        chain = find_chain(fake, spec.n_qubits)
        n_events, shots, mitigation = args.events or 12, args.shots or 256, False
        out_file = C.DATA_PROCESSED / "hardware_features_LOCAL_DRY_RUN.parquet"
        print(f"  executing the {spec.n_qubits}-qubit version on the simulated chip ...")
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            service = QiskitRuntimeService()
        except Exception as exc:
            raise SystemExit("No saved IBM Quantum account found. Save your key once as "
                             "described at the top of this file.\n" + str(exc))
        backend = service.backend(args.backend)
        chain = find_chain(backend, spec.n_qubits)
        mitigation = True
        if args.mode == "timing":
            n_events, shots = 5, args.shots or 1024
            out_file = C.DATA_PROCESSED / "hardware_features_TIMING.parquet"
        else:
            n_events, shots = args.events or 400, args.shots or 1024
            out_file = HARDWARE_FILE

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
    results = run_plan(prepared, plan, phi, backend, shots=shots,
                       events_per_job=args.events_per_job, mitigation=mitigation)
    results["shots"] = shots
    results.to_parquet(out_file, index=False)
    manifest = {"mode": args.mode, "backend": snap["backend"], "chain": chain, "shots": shots,
                "n_circuits": int(len(plan)), "mitigation": mitigation,
                "sentinels": sentinel_drift(results, shots)}
    out_file.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    print(f"saved {out_file.name}")
    print("sentinel check (identical circuits repeated through the run):")
    print(json.dumps(manifest["sentinels"], indent=2))

    # sanity check against the exact simulation of a perfect chip
    names = [c for c in results.columns if c[0] in "XYZ" and c[1:].isdigit()]
    ev_rows = results.loc[~results["is_sentinel"], "row"].to_numpy()
    measured = results.loc[~results["is_sentinel"], names].to_numpy()
    exact = exact[ev_rows]
    corr = np.corrcoef(measured.ravel(), exact.ravel())[0, 1]
    slope = (measured * exact).sum() / (exact ** 2).sum()
    print(f"agreement with the perfect-chip simulation: correlation {corr:.3f}, "
          f"average shrink factor {slope:.3f}")
    if args.mode == "timing":
        print("Now look up this job's 'usage' (quantum seconds) on quantum.cloud.ibm.com, "
              f"divide by {len(plan)} circuits, and size --events to your budget.")


if __name__ == "__main__":
    main()
