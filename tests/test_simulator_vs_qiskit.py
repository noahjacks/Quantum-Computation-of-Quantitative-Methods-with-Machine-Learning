"""
CHECK: our fast simulator gives the same numbers as Qiskit.

Why this matters: every "quantum" number in the project comes from
src/qon/simulator.py.  If that file had a bug, everything downstream would be
wrong.  So we compare it with three independent Qiskit simulators:

1. Qiskit's exact statevector            (small chains, perfect gates)
2. Qiskit Aer's density-matrix simulator (small chains, noisy gates)
3. Qiskit Aer's matrix-product-state simulator on the full 109 qubits
   (this is the simulator HSBC used for their noiseless results)

Run with:   .venv/bin/python -m pytest tests -q
"""
import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Statevector

from qon.circuit import build_observables, build_qiskit_circuit, make_spec
from qon.simulator import (backward_cone, simulate_noiseless, simulate_noisy,
                           target_groups)


def _random_angles(rng, n_events, n_features):
    """Random scaled features in the range the real pipeline produces."""
    return 2.0 * np.pi * np.tanh(rng.normal(size=(n_events, n_features)) / 3.0)


@pytest.mark.parametrize("n_qubits,steps,n_features,alpha",
                         [(4, 1, 3, 1.0), (7, 2, 4, 1.0), (10, 2, 7, 1.0),
                          (11, 2, 30, 0.3), (9, 3, 5, 1.0), (12, 2, 11, 1.0)])
def test_noiseless_matches_qiskit_statevector(n_qubits, steps, n_features, alpha):
    rng = np.random.default_rng(100 + n_qubits)
    spec = make_spec(n_features, n_qubits=n_qubits, trotter_steps=steps, alpha=alpha,
                     init_seed=3, order_seed=4)
    phi = spec.gate_angles(_random_angles(rng, 4, n_features))
    ours = simulate_noiseless(spec, phi)
    qc, params = build_qiskit_circuit(spec)
    obs = build_observables(n_qubits)
    for e in range(phi.shape[0]):
        state = Statevector(qc.assign_parameters(dict(zip(params, phi[e]))))
        theirs = np.array([state.expectation_value(o).real for o in obs])
        assert np.allclose(ours[e], theirs, atol=1e-10)


def test_compact_heisenberg_gate_is_the_same_operation():
    """The 3-entangling-gate circuit we send to hardware equals
    exp(-i phi (XX+YY+ZZ)) = RXX(2phi) RYY(2phi) RZZ(2phi), and the whole
    feature-map circuit is the same in both spellings."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, Pauli
    from scipy.linalg import expm
    from qon.circuit import heisenberg_gate

    H = sum(Operator(Pauli(p)).data for p in ("XX", "YY", "ZZ"))
    for phi in (-0.78, -0.2, 0.0, 0.31, 0.78, 2.5):
        compact = QuantumCircuit(2)
        heisenberg_gate(compact, phi, 0, 1)
        assert Operator(compact).equiv(Operator(expm(-1j * phi * H)))
    spec = make_spec(5, n_qubits=6, trotter_steps=2)
    phi = spec.gate_angles(_random_angles(np.random.default_rng(0), 1, 5))[0]
    a, pa = build_qiskit_circuit(spec, compact=True)
    b, pb = build_qiskit_circuit(spec, compact=False)
    assert Operator(a.assign_parameters(dict(zip(pa, phi)))).equiv(
        Operator(b.assign_parameters(dict(zip(pb, phi)))))


def test_noisy_with_zero_noise_equals_noiseless():
    rng = np.random.default_rng(5)
    spec = make_spec(20, n_qubits=15, trotter_steps=2)
    phi = spec.gate_angles(_random_angles(rng, 6, 20))
    assert np.allclose(simulate_noisy(spec, phi, np.zeros(spec.n_slots)),
                       simulate_noiseless(spec, phi), atol=1e-12)


@pytest.mark.parametrize("n_qubits,steps,n_features", [(5, 1, 3), (8, 2, 5), (9, 2, 12)])
def test_noisy_matches_aer_density_matrix(n_qubits, steps, n_features):
    """Same circuit, same depolarizing error after every Heisenberg gate,
    simulated by Qiskit Aer's exact density-matrix method."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import depolarizing_error

    rng = np.random.default_rng(200 + n_qubits)
    spec = make_spec(n_features, n_qubits=n_qubits, trotter_steps=steps, init_seed=8, order_seed=9)
    phi = spec.gate_angles(_random_angles(rng, 3, n_features))
    lam = rng.uniform(0.0, 0.2, size=spec.n_slots)        # a different error per gate
    ours = simulate_noisy(spec, phi, lam)
    obs = build_observables(n_qubits)
    backend = AerSimulator(method="density_matrix")
    for e in range(phi.shape[0]):
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.u(float(spec.init_theta[q]), float(spec.init_phi[q]), 0.0, q)
        for layer in spec.layers:
            for (q1, q2, slot) in layer:
                qc.rxx(2 * phi[e, slot], q1, q2)
                qc.ryy(2 * phi[e, slot], q1, q2)
                qc.rzz(2 * phi[e, slot], q1, q2)
                qc.append(depolarizing_error(float(lam[slot]), 2), [q1, q2])
        qc.save_density_matrix()
        rho = DensityMatrix(backend.run(qc).result().data(0)["density_matrix"])
        theirs = np.array([rho.expectation_value(o).real for o in obs])
        assert np.allclose(ours[e], theirs, atol=1e-9)


def test_drifting_noise_is_applied_per_event():
    """A per-event error table must give the same answer as simulating each
    event separately with its own error rates."""
    rng = np.random.default_rng(7)
    spec = make_spec(9, n_qubits=10, trotter_steps=2)
    phi = spec.gate_angles(_random_angles(rng, 5, 9))
    lam = rng.uniform(0.0, 0.1, size=(5, spec.n_slots))
    together = simulate_noisy(spec, phi, lam)
    for e in range(5):
        assert np.allclose(together[e], simulate_noisy(spec, phi[e:e + 1], lam[e])[0], atol=1e-12)


def test_full_109_qubits_matches_aer_matrix_product_state():
    """The real 109-qubit circuit against the simulator HSBC used."""
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp

    rng = np.random.default_rng(42)
    spec = make_spec(98)                                   # the project's circuit
    assert spec.n_qubits == 109 and spec.n_slots == 216
    phi = spec.gate_angles(_random_angles(rng, 2, 98))
    ours = simulate_noiseless(spec, phi)
    qc, params = build_qiskit_circuit(spec)
    backend = AerSimulator(method="matrix_product_state",
                           matrix_product_state_truncation_threshold=1e-14)
    for e in range(phi.shape[0]):
        bound = qc.assign_parameters(dict(zip(params, phi[e])))
        bound.remove_final_measurements()
        for k, pauli in enumerate("XYZ"):
            for q in range(spec.n_qubits):
                bound.save_expectation_value(SparsePauliOp(pauli), [q], label=f"{pauli}{q}")
        data = backend.run(bound).result().data(0)
        theirs = np.array([data[f"{p}{q}"] for p in "XYZ" for q in range(spec.n_qubits)])
        assert np.allclose(ours[e], theirs, atol=1e-7)


def test_light_cone_is_small():
    """With 2 Trotter steps no qubit depends on more than 8 qubits / 10 gates."""
    spec = make_spec(98)
    groups = target_groups(spec)
    assert sorted(q for g in groups for q in g) == list(range(109))
    for g in groups:
        gates = backward_cone(spec, g)
        qubits = set(g).union(*[(a, b) for a, b, _ in gates])
        assert len(gates) <= 10 and len(qubits) <= 8


def test_total_spin_is_conserved():
    """Physics check: the Heisenberg gate conserves total spin, so the SUM of
    <X> over all qubits (and likewise Y, Z) must not depend on the data."""
    rng = np.random.default_rng(1)
    spec = make_spec(98)
    out = simulate_noiseless(spec, spec.gate_angles(_random_angles(rng, 5, 98)))
    totals = out.reshape(5, 3, 109).sum(axis=2)
    assert np.allclose(totals, totals[0], atol=1e-9)
    init = spec.initial_states()
    x0 = 2 * (init[:, 0].conj() * init[:, 1]).real.sum()
    assert np.isclose(totals[0, 0], x0, atol=1e-9)
