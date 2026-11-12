"""
circuit.py  --  THE QUANTUM CIRCUIT ("PROJECTED QUANTUM FEATURE MAP")
====================================================================

Plain-English summary
---------------------
HSBC did NOT use a quantum computer to make predictions.  They used it as a
strange kind of calculator that turns each row of market data into a NEW row
of numbers, and then gave those new numbers to ordinary machine-learning
models.  The recipe, for one quote request:

1. Start 109 qubits, each in its own fixed random state.
2. Apply a chain of two-qubit gates between neighbouring qubits.  The ANGLE
   of each gate is set by one market feature (after scaling).  This is how
   the data gets "loaded" into the quantum state.
3. Measure every qubit along three directions (X, Y, Z) many times and take
   the averages.  Those 3 x 109 = 327 averages are the new features.

The gate is the "Heisenberg" interaction between neighbours j and j+1,

    U(phi) = exp( -i * phi * (XX + YY + ZZ) ),

which in Qiskit is the product RXX(2 phi) * RYY(2 phi) * RZZ(2 phi).
Physically it is a *partial swap*: phi = 0 does nothing and phi = pi/4 fully
swaps the two qubits' states.

Layout (identical in shape to HSBC's "shorter" circuit)
-------------------------------------------------------
* 109 qubits in a line => 108 neighbouring pairs.
* One "Trotter step" = gates on the even pairs (0-1, 2-3, ...) followed by
  gates on the odd pairs (1-2, 3-4, ...).  We use M = 2 steps: 216 gates.
* HSBC had 216 features, one per gate.  We have 98 (97 market features plus
  the side of the request), so we walk through the shuffled feature list
  again and again until all 216 gates have a feature.  Every feature
  therefore sets 2 or 3 gates.  Which feature goes on which gate is
  arbitrary (it was in the paper too); ours is fixed by FEATURE_ORDER_SEED.
* Gate angle:  phi = ALPHA * angle / (4 M),  angle = 2 pi tanh(z / 3).

Everything here is a description of the circuit.  Two things can execute it:
`simulator.py` (fast, exact, checked against Qiskit) and real IBM hardware
via `hardware.py`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass(frozen=True)
class CircuitSpec:
    """A complete, hardware-independent description of the feature-map circuit."""

    n_qubits: int
    trotter_steps: int
    alpha: float
    n_features: int
    slot_feature: np.ndarray        # (n_slots,) which feature drives each gate
    layers: tuple                   # layers[l] = tuple of (qubit, qubit+1, slot)
    init_theta: np.ndarray          # (n_qubits,) polar angle of each start state
    init_phi: np.ndarray            # (n_qubits,) azimuth of each start state

    @property
    def n_slots(self) -> int:
        return len(self.slot_feature)

    @property
    def n_outputs(self) -> int:
        return 3 * self.n_qubits

    def gate_angles(self, angles: np.ndarray) -> np.ndarray:
        """Scaled features (n_events, n_features) -> the angle phi of every
        gate (n_events, n_slots)."""
        angles = np.atleast_2d(angles)
        return self.alpha * angles[:, self.slot_feature] / (4.0 * self.trotter_steps)

    def initial_states(self) -> np.ndarray:
        """(n_qubits, 2) complex amplitudes of each qubit's start state."""
        return np.stack([np.cos(self.init_theta / 2.0),
                         np.exp(1j * self.init_phi) * np.sin(self.init_theta / 2.0)], axis=1)


def make_spec(n_features: int, n_qubits: int = C.N_QUBITS,
              trotter_steps: int = C.TROTTER_STEPS, alpha: float = C.ALPHA,
              init_seed: int = C.INIT_STATE_SEED,
              order_seed: int = C.FEATURE_ORDER_SEED) -> CircuitSpec:
    n_pairs = n_qubits - 1
    n_slots = n_pairs * trotter_steps
    # Shuffle the features once, then deal them out to the gates in a cycle.
    order = np.random.default_rng(order_seed).permutation(n_features)
    slot_feature = order[np.arange(n_slots) % n_features]
    layers = []
    for step in range(trotter_steps):
        for parity in (0, 1):                       # even pairs first, then odd
            layers.append(tuple((j, j + 1, step * n_pairs + j)
                                for j in range(parity, n_pairs, 2)))
    # "Uniformly random" single-qubit states: uniform on the Bloch sphere.
    rng = np.random.default_rng(init_seed)
    init_theta = np.arccos(rng.uniform(-1.0, 1.0, n_qubits))
    init_phi = rng.uniform(0.0, 2.0 * np.pi, n_qubits)
    return CircuitSpec(n_qubits=n_qubits, trotter_steps=trotter_steps, alpha=alpha,
                       n_features=n_features, slot_feature=slot_feature,
                       layers=tuple(layers), init_theta=init_theta, init_phi=init_phi)


# ---------------------------------------------------------------------------
# Qiskit versions of the same circuit
# ---------------------------------------------------------------------------
def heisenberg_gate(qc, phi, q1: int, q2: int) -> None:
    """Append exp(-i phi (XX + YY + ZZ)) on qubits (q1, q2) using only THREE
    entangling gates (the textbook Vatan-Williams construction).

    Writing it naively as RXX * RYY * RZZ would cost six entangling gates.
    Entangling gates are by far the noisiest operation on real hardware, so
    halving their number halves the damage.  tests/test_simulator_vs_qiskit.py
    checks that both versions are the same operation."""
    half_pi = np.pi / 2.0
    qc.rz(-half_pi, q2)
    qc.cx(q2, q1)
    qc.rz(half_pi + 2.0 * phi, q1)
    qc.ry(-half_pi - 2.0 * phi, q2)
    qc.cx(q1, q2)
    qc.ry(half_pi + 2.0 * phi, q2)
    qc.cx(q2, q1)
    qc.rz(half_pi, q1)


def build_qiskit_circuit(spec: CircuitSpec, compact: bool = True):
    """The circuit as a Qiskit object with one free parameter per gate.

    Returns (circuit, parameters).  Bind `parameters` to one row of
    `spec.gate_angles(...)` to get the circuit for one quote request.
    compact=True uses the 3-entangling-gate form of the Heisenberg gate
    (what we send to hardware); compact=False spells it out as
    RXX(2 phi) RYY(2 phi) RZZ(2 phi), which is easier to read."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    phi = ParameterVector("phi", spec.n_slots)
    qc = QuantumCircuit(spec.n_qubits)
    for q in range(spec.n_qubits):
        qc.u(float(spec.init_theta[q]), float(spec.init_phi[q]), 0.0, q)
    for layer in spec.layers:
        qc.barrier()
        for (q1, q2, slot) in layer:
            if compact:
                heisenberg_gate(qc, phi[slot], q1, q2)
            else:
                qc.rxx(2.0 * phi[slot], q1, q2)
                qc.ryy(2.0 * phi[slot], q1, q2)
                qc.rzz(2.0 * phi[slot], q1, q2)
    return qc, phi


def build_observables(n_qubits: int):
    """The 3 x n_qubits measured quantities, in the order
    X_0..X_{n-1}, Y_0..Y_{n-1}, Z_0..Z_{n-1}."""
    from qiskit.quantum_info import SparsePauliOp

    return [SparsePauliOp.from_sparse_list([(p, [q], 1.0)], num_qubits=n_qubits)
            for p in "XYZ" for q in range(n_qubits)]


def output_names(n_qubits: int) -> list[str]:
    return [f"{p}{q}" for p in "XYZ" for q in range(n_qubits)]
