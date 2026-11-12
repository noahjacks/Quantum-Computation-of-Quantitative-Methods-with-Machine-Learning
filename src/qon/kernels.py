"""
kernels.py  --  PART 2 (THE EXTENSION): KERNELS, CLASSICAL AND QUANTUM
=====================================================================

What changes in Part 2
----------------------
Part 1 kept the learning algorithm fixed and changed its INPUTS (classical
vs quantum features).  Part 2 does the opposite: the inputs are always the
classical features, and the LEARNING ALGORITHM is what changes.  In
particular we ask whether a *quantum kernel* can beat classical algorithms at
estimating p(fill).

What is a kernel?
-----------------
A kernel k(x, x') is a similarity score between two quote requests x and x':
close to 1 if they are "alike", close to 0 if not.  A kernel machine (here a
support vector machine, SVM) predicts the outcome of a new request from the
outcomes of the past requests most similar to it.  Everything depends on how
"similar" is defined:

  linear      k = x . x'                        (plain dot product)
  RBF         k = exp(-gamma |x - x'|^2)         (the classical workhorse)

  FQK  "fidelity quantum kernel"   k = |<psi(x)|psi(x')>|^2
       Load x and x' into two quantum states with our circuit and ask how
       much the two states overlap.  This is the textbook quantum kernel.

  PQK  "projected quantum kernel"  k = exp(-gamma * sum_q |rho_q(x) - rho_q(x')|^2)
       Load x and x' into quantum states, MEASURE each qubit (exactly the
       327 numbers of feature set B), and apply an RBF kernel to those
       measurements.  Proposed by Huang et al. (2021); it is the kernel
       version of HSBC's "projected quantum feature map".

Two practical facts decide which of these could ever run on real hardware:
* PQK needs one circuit per request   (16,000 circuits: feasible)
* FQK needs one circuit per PAIR      (~10^8 circuits: not feasible)
and FQK suffers from "exponential concentration": with many qubits two
different inputs almost never overlap, every similarity is ~0, and the
kernel carries no information unless the rotation scale alpha is shrunk
(alpha plays the role of the kernel's bandwidth).

How the 109-qubit overlaps are computed exactly
-----------------------------------------------
A shallow circuit on a line produces a state with very little entanglement
across any cut, which can be stored exactly as a *matrix product state*
(MPS): one small tensor per qubit instead of 2^109 amplitudes.  The overlap
of two MPS is a cheap chain of small matrix products.  tests/test_kernels.py
checks the result against Qiskit.
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from .circuit import CircuitSpec


# ---------------------------------------------------------------------------
# classical-style kernels
# ---------------------------------------------------------------------------
def squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = (X ** 2).sum(axis=1)[:, None]
    yy = (Y ** 2).sum(axis=1)[None, :]
    return np.maximum(xx + yy - 2.0 * X @ Y.T, 0.0)


def median_gamma(X: np.ndarray, seed: int = 0, max_points: int = 800) -> float:
    """The usual default bandwidth: 1 / (median squared distance)."""
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(max_points, len(X)), replace=False)]
    d2 = squared_distances(sub, sub)
    return 1.0 / float(np.median(d2[np.triu_indices(len(sub), k=1)]))


# ---------------------------------------------------------------------------
# matrix product states for the fidelity kernel
# ---------------------------------------------------------------------------
def build_mps(spec: CircuitSpec, phi: np.ndarray) -> list[np.ndarray]:
    """Exact MPS of the circuit's output state for every event.

    Returns one tensor per qubit of shape (n_events, left, 2, right).  The
    bond sizes (`left`, `right`) never exceed 16 for our 4-layer circuit."""
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    n = phi.shape[0]
    init = spec.initial_states()
    A = [np.broadcast_to(init[q][None, None, :, None], (n, 1, 2, 1)).astype(complex).copy()
         for q in range(spec.n_qubits)]
    for layer in spec.layers:
        for (q1, q2, slot) in layer:
            left, right = A[q1].shape[1], A[q2].shape[3]
            old_bond = A[q1].shape[3]
            theta = np.einsum("nasb,nbtc->nastc", A[q1], A[q2])
            c = np.cos(2.0 * phi[:, slot])[:, None, None, None, None]
            s = np.sin(2.0 * phi[:, slot])[:, None, None, None, None]
            theta = c * theta - 1j * s * np.swapaxes(theta, 2, 3)      # partial swap
            U, S, Vh = np.linalg.svd(theta.reshape(n, left * 2, 2 * right), full_matrices=False)
            # A two-qubit gate can multiply the entanglement across the cut by
            # at most 4, so singular values beyond that are exactly zero.
            keep = min(S.shape[1], 4 * old_bond)
            if S.shape[1] > keep and S[:, keep:].max() > 1e-9:
                raise RuntimeError("unexpected entanglement growth in the MPS")
            A[q1] = U[:, :, :keep].reshape(n, left, 2, keep)
            A[q2] = (S[:, :keep, None] * Vh[:, :keep, :]).reshape(n, keep, 2, right)
    return A


def mps_overlaps(rows: list[np.ndarray], cols: list[np.ndarray]) -> np.ndarray:
    """<psi_i | psi_j> for every event i in `rows` and j in `cols`
    (complex matrix).  Works site by site with large matrix products."""
    n_i, n_j = rows[0].shape[0], cols[0].shape[0]
    E = np.ones((n_i, n_j, 1, 1), dtype=complex)
    for A, B in zip(rows, cols):
        a, a2 = A.shape[1], A.shape[3]
        b, b2 = B.shape[1], B.shape[3]
        # step 1: attach the ket tensor B[j] on the right of E[:, j]
        T = np.matmul(E.transpose(1, 0, 2, 3).reshape(n_j, n_i * a, b),
                      B.reshape(n_j, b, 2 * b2))                       # (n_j, n_i*a, 2*b2)
        T = T.reshape(n_j, n_i, a, 2, b2).transpose(1, 2, 3, 0, 4).reshape(n_i, a * 2, n_j * b2)
        # step 2: attach the conjugated bra tensor A[i] on the left
        Ah = A.reshape(n_i, a * 2, a2).conj().transpose(0, 2, 1)       # (n_i, a2, a*2)
        E = np.matmul(Ah, T).reshape(n_i, a2, n_j, b2).transpose(0, 2, 1, 3)
    return E.reshape(n_i, n_j)


def take(mps: list[np.ndarray], index) -> list[np.ndarray]:
    return [A[index] for A in mps]


def _band_block(mps, start: int, stop: int, j0: int) -> np.ndarray:
    ov = mps_overlaps(take(mps, slice(start, stop)), take(mps, slice(j0, stop)))
    return (ov.real ** 2 + ov.imag ** 2).astype(np.float32)


def fidelity_kernel_band(mps: list[np.ndarray], width: int, block: int = 16,
                         n_jobs: int = 1) -> np.ndarray:
    """Fidelity kernel between every event i and the `width` events before it
    (events are in time order, and training sets only ever reach back a
    limited distance, so the far-apart pairs are never needed).

    Returns a dense (n, n) float32 matrix; entries further apart than `width`
    are left at 0 and must not be used."""
    n = mps[0].shape[0]
    tasks = [(start, min(start + block, n), max(0, start - width)) for start in range(0, n, block)]
    if n_jobs == 1:
        blocks = [_band_block(mps, *t) for t in tasks]
    else:
        from joblib import Parallel, delayed
        blocks = Parallel(n_jobs=n_jobs)(delayed(_band_block)(mps, *t) for t in tasks)
    K = np.zeros((n, n), dtype=np.float32)
    for (start, stop, j0), k in zip(tasks, blocks):
        K[start:stop, j0:stop] = k
        K[j0:stop, start:stop] = k.T
    return K


def _cross_block(rows, cols, start: int, stop: int) -> np.ndarray:
    ov = mps_overlaps(take(rows, slice(start, stop)), cols)
    return (ov.real ** 2 + ov.imag ** 2).astype(np.float32)


def fidelity_kernel_cross(rows: list[np.ndarray], cols: list[np.ndarray], block: int = 16,
                          n_jobs: int = 1) -> np.ndarray:
    """Fidelity kernel between two different groups of events (n_rows, n_cols)."""
    n = rows[0].shape[0]
    tasks = [(start, min(start + block, n)) for start in range(0, n, block)]
    if n_jobs == 1:
        parts = [_cross_block(rows, cols, *t) for t in tasks]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs)(delayed(_cross_block)(rows, cols, *t) for t in tasks)
    return np.vstack(parts)


# ---------------------------------------------------------------------------
# SVM helpers
# ---------------------------------------------------------------------------
C_GRID = (0.01, 0.1, 1.0, 10.0)
GAMMA_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0)


def _cv_auc(K: np.ndarray, y: np.ndarray, c_value: float, folds: list[np.ndarray]) -> float:
    from sklearn.metrics import roc_auc_score
    scores = []
    all_idx = np.arange(len(y))
    for val in folds:
        tr = np.setdiff1d(all_idx, val)
        svm = SVC(C=c_value, kernel="precomputed").fit(K[np.ix_(tr, tr)], y[tr])
        scores.append(roc_auc_score(y[val], svm.decision_function(K[np.ix_(val, tr)])))
    return float(np.mean(scores))


def make_folds(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(perm, n_folds)]


def tune_svm(kernels: dict, y: np.ndarray, folds: list[np.ndarray]) -> tuple:
    """Pick the best (kernel variant, C) by cross-validated AUC.
    `kernels` maps a label (e.g. a gamma value) to a training kernel matrix."""
    best = (-np.inf, None, None)
    for label, K in kernels.items():
        for c_value in C_GRID:
            score = _cv_auc(K, y, c_value, folds)
            if score > best[0]:
                best = (score, label, c_value)
    return best[1], best[2], best[0]


def svm_fit_predict(K_train: np.ndarray, y_train: np.ndarray, K_test: np.ndarray,
                    c_value: float) -> np.ndarray:
    """Train an SVM on a ready-made kernel and return p(fill) for the test
    events.  An SVM only outputs a score, so the scores are turned into
    probabilities with "Platt scaling": a logistic curve fitted on
    cross-validated scores of the training events."""
    from sklearn.calibration import CalibratedClassifierCV
    model = CalibratedClassifierCV(SVC(C=c_value, kernel="precomputed"),
                                   method="sigmoid", cv=5, ensemble=False)
    model.fit(K_train, y_train)
    return model.predict_proba(K_test)[:, 1]


# ---------------------------------------------------------------------------
# kernel ridge regression (the follow-up analysis of Part 2)
# ---------------------------------------------------------------------------
# An SVM tries to draw a boundary and ignores how confident it should be.
# When the signal is very weak and one outcome is more common (63% of quotes
# are not filled), the best "boundary" is to call everything "not filled",
# and the SVM's scores become meaningless.  Kernel ridge regression uses the
# SAME kernels but fits a smooth score to all training events at once:
#       score(x) = sum_i  c_i * k(x, x_i),     c = (K + lam * I)^-1 (y - mean)
# `lam` controls how strongly the fit is smoothed.
RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _psd_eigh(K: np.ndarray):
    """Eigen-decomposition of a kernel matrix.  Kernels estimated from a
    finite number of shots can have slightly negative eigenvalues; those are
    set to zero (the standard repair)."""
    w, V = np.linalg.eigh((K + K.T) / 2.0)
    return np.clip(w, 0.0, None), V


def ridge_scores(K_train: np.ndarray, y_train: np.ndarray, K_test: np.ndarray,
                 lam: float) -> np.ndarray:
    w, V = _psd_eigh(K_train)
    yc = y_train - y_train.mean()
    coef = V @ ((V.T @ yc) / (w + lam))
    return K_test @ coef + y_train.mean()


def tune_ridge(kernels: dict, y: np.ndarray, folds: list[np.ndarray]) -> tuple:
    """Pick the best (kernel variant, lam) by cross-validated AUC."""
    from sklearn.metrics import roc_auc_score
    best = (-np.inf, None, None)
    all_idx = np.arange(len(y))
    for label, K in kernels.items():
        scores = {lam: [] for lam in RIDGE_GRID}
        for val in folds:
            tr = np.setdiff1d(all_idx, val)
            w, V = _psd_eigh(K[np.ix_(tr, tr)])
            proj = V.T @ (y[tr] - y[tr].mean())
            K_val = K[np.ix_(val, tr)]
            for lam in RIDGE_GRID:
                scores[lam].append(roc_auc_score(y[val], K_val @ (V @ (proj / (w + lam)))))
        for lam in RIDGE_GRID:
            if np.mean(scores[lam]) > best[0]:
                best = (float(np.mean(scores[lam])), label, lam)
    return best[1], best[2], best[0]


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def concentration(K: np.ndarray) -> dict:
    """How much information is in a kernel matrix?  If every off-diagonal
    entry is ~0 (or all are ~equal), every request looks equally (dis)similar
    to every other and the kernel is useless ("exponential concentration")."""
    off = K[np.triu_indices(len(K), k=1)]
    return {"median": float(np.median(off)), "mean": float(off.mean()),
            "p99": float(np.quantile(off, 0.99)), "std": float(off.std())}


def kernel_target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """Between -1 and 1: how well 'similar according to the kernel' agrees
    with 'same outcome'.  Higher = the kernel suits this prediction task."""
    yc = np.where(np.asarray(y) > 0, 1.0, -1.0)
    yc = yc - yc.mean()
    Kc = K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()
    return float(yc @ Kc @ yc / (np.linalg.norm(Kc) * (yc @ yc)))


def geometric_difference(K_classical: np.ndarray, K_quantum: np.ndarray,
                         reg: float = 1e-3) -> float:
    """Huang et al.'s test for possible quantum advantage (Nature Comm. 12,
    2631, 2021).  It asks how differently the two kernels arrange the same
    data.  A value near 1 means the classical kernel can do everything the
    quantum one can on this data; only a large value (growing like sqrt(N))
    leaves room for an advantage - and even then it is not guaranteed.

        g = sqrt( || sqrt(KQ) sqrt(KC) (KC + reg*I)^-2 sqrt(KC) sqrt(KQ) || )

    Both kernels are first rescaled so that their diagonals average 1."""
    def normalise(K):
        K = (K + K.T) / 2.0
        return K * (len(K) / np.trace(K))

    def sqrtm_psd(K):
        w, V = np.linalg.eigh(K)
        return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T

    KC, KQ = normalise(np.asarray(K_classical, float)), normalise(np.asarray(K_quantum, float))
    sC, sQ = sqrtm_psd(KC), sqrtm_psd(KQ)
    inv = np.linalg.inv(KC + reg * np.eye(len(KC)))
    M = sQ @ sC @ inv @ inv @ sC @ sQ
    return float(np.sqrt(np.linalg.eigvalsh((M + M.T) / 2.0).max()))
