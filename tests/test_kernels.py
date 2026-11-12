"""
CHECK: the quantum kernels are computed correctly.

The fidelity kernel k = |<psi(x)|psi(x')>|^2 is compared with Qiskit's exact
statevectors on small chains, and the 109-qubit states behind it are checked
against the (already verified) light-cone simulator.
"""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qon.circuit import build_qiskit_circuit, make_spec
from qon.kernels import (build_mps, fidelity_kernel_band, geometric_difference,
                         kernel_target_alignment, median_gamma, mps_overlaps,
                         squared_distances, svm_fit_predict, take, tune_svm, make_folds)
from qon.simulator import simulate_noiseless


def _angles(rng, n, p):
    return 2.0 * np.pi * np.tanh(rng.normal(size=(n, p)) / 3.0)


@pytest.mark.parametrize("n_qubits,steps,n_features", [(4, 1, 3), (8, 2, 5), (11, 2, 20), (10, 3, 6)])
def test_fidelity_kernel_matches_qiskit_statevectors(n_qubits, steps, n_features):
    rng = np.random.default_rng(n_qubits)
    spec = make_spec(n_features, n_qubits=n_qubits, trotter_steps=steps, init_seed=2, order_seed=3)
    phi = spec.gate_angles(_angles(rng, 6, n_features))
    mps = build_mps(spec, phi)
    ours = np.abs(mps_overlaps(mps, mps)) ** 2
    qc, params = build_qiskit_circuit(spec)
    states = [Statevector(qc.assign_parameters(dict(zip(params, phi[e])))) for e in range(6)]
    theirs = np.array([[abs(a.inner(b)) ** 2 for b in states] for a in states])
    assert np.allclose(ours, theirs, atol=1e-10)
    assert np.allclose(np.diag(ours), 1.0, atol=1e-10)


def test_109_qubit_mps_reproduces_the_simulators_measurements():
    """Read <Z> of every qubit off the MPS and compare with the light-cone
    simulator: proves the 109-qubit MPS really is the circuit's state."""
    rng = np.random.default_rng(0)
    spec = make_spec(98)
    phi = spec.gate_angles(_angles(rng, 3, 98))
    mps = build_mps(spec, phi)
    assert max(max(A.shape[1], A.shape[3]) for A in mps) <= 16
    expect = simulate_noiseless(spec, phi)[:, 2 * 109:]               # the <Z_q> block
    Z = np.diag([1.0, -1.0])
    for q in (0, 1, 37, 54, 107, 108):
        changed = [A.copy() for A in mps]
        changed[q] = np.einsum("nasb,st->natb", mps[q], Z)             # apply Z to qubit q
        for e in range(3):
            val = mps_overlaps(take(mps, slice(e, e + 1)), take(changed, slice(e, e + 1)))[0, 0]
            assert np.isclose(val.real, expect[e, q], atol=1e-9) and abs(val.imag) < 1e-9


def test_band_kernel_is_a_valid_similarity_matrix():
    rng = np.random.default_rng(1)
    spec = make_spec(98, alpha=0.1)
    mps = build_mps(spec, spec.gate_angles(_angles(rng, 40, 98)))
    full = fidelity_kernel_band(mps, width=40, block=7)
    assert np.allclose(full, full.T) and np.allclose(np.diag(full), 1.0, atol=1e-5)
    assert np.linalg.eigvalsh(full.astype(float)).min() > -1e-5        # positive semi-definite
    assert 0.0 <= full.min() and full.max() <= 1.0 + 1e-5
    band = fidelity_kernel_band(mps, width=10, block=7)
    i, j = np.indices(full.shape)
    assert np.allclose(band[np.abs(i - j) <= 10], full[np.abs(i - j) <= 10], atol=1e-6)


def test_fidelity_kernel_concentrates_unless_alpha_is_small():
    """The textbook problem of quantum kernels, visible in our own circuit."""
    rng = np.random.default_rng(2)
    ang = _angles(rng, 30, 98)
    medians = {}
    for alpha in (1.0, 0.1, 0.03):
        spec = make_spec(98, alpha=alpha)
        K = np.abs(mps_overlaps(*(2 * [build_mps(spec, spec.gate_angles(ang))]))) ** 2
        medians[alpha] = np.median(K[np.triu_indices(30, k=1)])
    assert medians[1.0] < 1e-6 < medians[0.1] < medians[0.03] < 1.0


def test_svm_helpers_learn_a_simple_pattern():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 5))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 1.5).astype(int)              # a circle: not linear
    g = median_gamma(X)
    D2 = squared_distances(X[:300], X[:300])
    kernels = {m: np.exp(-m * g * D2) for m in (0.3, 1.0, 3.0)}
    kernels["linear"] = X[:300] @ X[:300].T
    label, c_value, score = tune_svm(kernels, y[:300], make_folds(300, 3, 0))
    assert label != "linear" and score > 0.9
    K_test = np.exp(-label * g * squared_distances(X[300:], X[:300]))
    p = svm_fit_predict(kernels[label], y[:300], K_test, c_value)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[300:], p) > 0.9 and 0.0 <= p.min() and p.max() <= 1.0


def test_diagnostics():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    K = np.exp(-0.5 * squared_distances(X, X))
    assert np.isclose(geometric_difference(K, K, reg=1e-8), 1.0, atol=1e-3)   # identical kernels
    good = np.outer(2 * y - 1, 2 * y - 1).astype(float)
    assert kernel_target_alignment(good, y) > 0.99
    assert abs(kernel_target_alignment(np.eye(120), y)) < 0.2
