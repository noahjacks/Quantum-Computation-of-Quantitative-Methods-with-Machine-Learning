"""
noise.py  --  WHAT A REAL QUANTUM COMPUTER DOES TO THE NUMBERS
==============================================================

A real quantum chip never returns the exact expectation values that
`simulate_noiseless` computes.  Four separate things go wrong, and this file
models each of them so that we can switch them on one at a time:

1. GATE ERRORS.  Every gate is slightly imperfect.  We model each Heisenberg
   gate as "perfect gate, then with small probability the two qubits are
   scrambled" (depolarizing noise).  Net effect: every measured average
   shrinks towards zero.  [handled inside simulator.simulate_noisy]

2. READOUT ERRORS.  Sometimes a 0 is read as a 1 or vice versa.  This also
   shrinks the averages.  IBM's "TREX" method (which HSBC used) measures the
   shrink factor with calibration circuits and divides it back out, which
   removes the bias but amplifies the randomness.

3. SHOT NOISE.  An "expectation value" is an average over a finite number of
   repeated runs ("shots"; HSBC used 4096).  An average of 4096 coin flips
   is not exact: it wobbles by up to ~1.6%.

4. DRIFT.  The chip's error rates are not constant; they wander over minutes
   and hours.  Circuits run in the same job share the same error rates, so a
   measured value carries a faint fingerprint of WHEN it was run.  If the
   data is sent to the chip in time order, that fingerprint is a hidden
   clock - "the third suspect" in our project.

Error sizes come from IBM's published calibration snapshot of a real Heron
chip (see config.py and hardware.py::save_snapshot).  When the team runs on
real hardware, `fit_classical_surrogate` measures the real shrink and jitter
directly from the data instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config as C
from .circuit import CircuitSpec


@dataclass
class NoiseParams:
    lam_slot: np.ndarray      # (n_slots,) depolarizing probability of each gate
    readout: np.ndarray       # (n_qubits,) probability of misreading each qubit
    shots: int


def make_noise_params(spec: CircuitSpec, scale: float = 1.0, shots: int = C.SHOTS,
                      snapshot_path=C.CHIP_SNAPSHOT) -> NoiseParams:
    """Build the simulated chip from a real calibration snapshot: every qubit
    pair and every qubit gets the error rate IBM reported for it.
    `scale` multiplies all gate and readout errors (scale=0: perfect chip).

    One Heisenberg gate is compiled into several native gates (3 two-qubit
    gates plus a few one-qubit gates per qubit), each of which can fail."""
    snap = json.loads(snapshot_path.read_text())
    n_pairs = spec.n_qubits - 1
    if len(snap["chain"]) < spec.n_qubits:
        raise ValueError("calibration snapshot has fewer qubits than the circuit")
    e2 = np.minimum(np.array(snap["two_qubit_error"][:n_pairs]), 0.5)
    e1 = np.minimum(np.array(snap["one_qubit_error"][:spec.n_qubits]), 0.25)
    readout = np.array(snap["readout_error"][:spec.n_qubits])
    # Convert "average gate error e" to a depolarizing probability:
    # two-qubit gate: lam = 4e/3 ; one-qubit gate: lam = 2e.
    survive_2q = (1.0 - 4.0 * e2 / 3.0) ** snap["two_qubit_gates_per_heisenberg_gate"]
    survive_1q = (1.0 - 2.0 * e1) ** snap["one_qubit_gates_per_qubit_per_heisenberg_gate"]
    lam_pair = 1.0 - survive_2q * survive_1q[:-1] * survive_1q[1:]
    # slot s of the circuit acts on pair (s mod n_pairs)
    lam_slot = lam_pair[np.arange(spec.n_slots) % n_pairs]
    return NoiseParams(lam_slot=np.clip(scale * lam_slot, 0.0, 1.0),
                       readout=np.clip(scale * readout, 0.0, 0.45), shots=shots)


def batch_of_events(exec_order: np.ndarray, batch_size: int = C.DRIFT_BATCH_SIZE) -> np.ndarray:
    """exec_order[k] = index of the event that is run k-th on the chip.
    Returns, for every event, the number of the job ('batch') it runs in."""
    position = np.empty(len(exec_order), dtype=int)
    position[exec_order] = np.arange(len(exec_order))
    return position // batch_size


def drifting_gate_errors(params: NoiseParams, batch: np.ndarray, n_pairs: int,
                         rel_sigma: float = C.DRIFT_REL_SIGMA, ar1: float = C.DRIFT_AR1,
                         seed: int = C.DRIFT_SEED) -> np.ndarray:
    """Per-event gate errors when the chip drifts.

    Each qubit pair's error rate is multiplied by a factor that wanders from
    job to job (a slowly-varying random walk that keeps returning to 1;
    technically a log-normal AR(1) process).  Returns (n_events, n_slots)."""
    rng = np.random.default_rng(seed)
    n_batches = int(batch.max()) + 1
    z = np.empty((n_batches, n_pairs))
    z[0] = rng.standard_normal(n_pairs)
    for b in range(1, n_batches):
        z[b] = ar1 * z[b - 1] + np.sqrt(1.0 - ar1 ** 2) * rng.standard_normal(n_pairs)
    factor = np.exp(rel_sigma * z - 0.5 * rel_sigma ** 2)          # mean 1
    pair_of_slot = np.arange(len(params.lam_slot)) % n_pairs
    lam = params.lam_slot[None, :] * factor[batch][:, pair_of_slot]
    return np.clip(lam, 0.0, 1.0)


def measure(expectations: np.ndarray, params: NoiseParams, batch: np.ndarray,
            seed: int, trex: bool = True) -> np.ndarray:
    """Turn exact expectation values into what a finite experiment returns.

    expectations : (n_events, 3 * n_qubits) exact values (after gate noise).
    batch        : job number of each event (TREX is calibrated once per job).
    Steps: readout error shrinks each value -> estimate it from `shots`
    random outcomes -> (optionally) TREX divides by the measured shrink."""
    rng = np.random.default_rng(seed)
    n, d = expectations.shape
    n_qubits = d // 3
    shrink = np.tile(1.0 - 2.0 * params.readout, 3)               # per output column
    raw = np.clip(shrink[None, :] * expectations, -1.0, 1.0)
    if params.shots and params.shots > 0:
        ones = rng.binomial(params.shots, (1.0 + raw) / 2.0)
        est = 2.0 * ones / params.shots - 1.0
    else:
        est = raw.copy()
    if trex:
        # Calibration circuits estimate the shrink factor, once per job and
        # qubit, from the same number of shots, so the estimate is itself noisy.
        n_batches = int(batch.max()) + 1
        f = 1.0 - 2.0 * params.readout                              # (n_qubits,)
        if params.shots and params.shots > 0:
            f_hat = f[None, :] + rng.standard_normal((n_batches, n_qubits)) * np.sqrt(
                np.maximum(1.0 - f ** 2, 0.0) / params.shots)[None, :]
        else:
            f_hat = np.broadcast_to(f, (n_batches, n_qubits))
        est = est / np.tile(f_hat, (1, 3))[batch]
    return est


# ---------------------------------------------------------------------------
# Feature set E: "is hardware noise just classical noise?"
# ---------------------------------------------------------------------------
@dataclass
class ClassicalSurrogate:
    """For every output column j:  target_j ~ intercept_j + slope_j * clean_j,
    with leftover scatter of standard deviation sigma_j."""
    intercept: np.ndarray
    slope: np.ndarray
    sigma: np.ndarray

    def apply(self, clean: np.ndarray, seed: int) -> np.ndarray:
        """Shrink and jitter the noiseless features with ordinary random
        numbers - no quantum computer involved."""
        rng = np.random.default_rng(seed)
        return (self.intercept + self.slope * clean
                + self.sigma * rng.standard_normal(clean.shape))


def fit_classical_surrogate(clean: np.ndarray, target: np.ndarray) -> ClassicalSurrogate:
    """Measure, column by column, how the noisy features (`target`: hardware
    set D, or the noisy simulator C) relate to the noiseless ones (`clean`:
    set B): by how much they are shrunk and how much random scatter is left."""
    cm, tm = clean.mean(axis=0), target.mean(axis=0)
    var = ((clean - cm) ** 2).mean(axis=0)
    cov = ((clean - cm) * (target - tm)).mean(axis=0)
    slope = np.where(var > 1e-14, cov / np.where(var > 1e-14, var, 1.0), 0.0)
    intercept = tm - slope * cm
    resid = target - (intercept + slope * clean)
    return ClassicalSurrogate(intercept=intercept, slope=slope, sigma=resid.std(axis=0))


# ---------------------------------------------------------------------------
# Feature set F: "is it just more non-linear features?"
# ---------------------------------------------------------------------------
def random_fourier_features(z: np.ndarray, z_calibration: np.ndarray, n_out: int,
                            seed: int) -> np.ndarray:
    """A classic, purely classical way to make many non-linear features:
    cos(w . z + b) with random w and b (Rahimi & Recht 2007).  The length
    scale is the median distance between calibration points (the standard
    'median heuristic'), so nothing is tuned on the test window."""
    rng = np.random.default_rng(seed)
    sub = z_calibration[rng.choice(len(z_calibration), size=min(2000, len(z_calibration)),
                                   replace=False)]
    sq = (sub ** 2).sum(axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * sub @ sub.T, 0.0)
    length = np.sqrt(np.median(d2[np.triu_indices(len(sub), k=1)]))
    w = rng.standard_normal((n_out, z.shape[1])) / length
    b = rng.uniform(0.0, 2.0 * np.pi, n_out)
    return np.cos(z @ w.T + b)
