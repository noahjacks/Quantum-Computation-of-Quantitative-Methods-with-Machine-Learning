"""
simulator.py  --  A FAST, EXACT SIMULATOR FOR THE 109-QUBIT CIRCUIT
==================================================================

"Simulating 109 qubits is impossible" - usually true (the full state needs
2^109 numbers), but not for this circuit, because of a simple physical fact:

    THE LIGHT CONE.  Our circuit has only 4 layers of nearest-neighbour
    gates.  A gate can only pass information to the qubit next door, so
    after 4 layers a qubit can only have been influenced by qubits at most
    4 steps away.  To predict a measurement on qubits (a, a+1) we therefore
    only need the 8 qubits a-3 ... a+4 and the 10 gates that connect them.
    Everything else in the chain cannot matter and can be left out.

So instead of one impossible 109-qubit simulation we do 55 tiny ones.  The
result is EXACT (no approximation) - tests/test_simulator_vs_qiskit.py checks
it against Qiskit's own simulators, including Qiskit's 109-qubit
matrix-product-state simulator, which is what HSBC used.

Two simulators live here:

* `simulate_noiseless`  - a perfect quantum computer (feature set B).
* `simulate_noisy`      - gate errors included (the core of feature set C).
  It tracks density matrices (the description of a quantum state that is
  partly random) and discards each qubit as soon as it can no longer affect
  the answer, which keeps every matrix tiny (at most 16 x 16).

The Heisenberg gate exp(-i phi (XX+YY+ZZ)) equals, up to an irrelevant global
phase,  cos(2 phi) * I  -  i sin(2 phi) * SWAP,  so applying it is just "mix
the state with a copy of itself that has the two qubits swapped".

Noise model inside `simulate_noisy`: after every Heisenberg gate the two
qubits suffer a "depolarizing" error: with probability `lam` their state is
replaced by complete randomness.  This is the standard model of gate noise on
hardware that uses Pauli twirling (as HSBC did).  Measurement errors and the
randomness of a finite number of shots are added afterwards in noise.py.
"""
from __future__ import annotations

import string

import numpy as np

from .circuit import CircuitSpec

_LETTERS = string.ascii_letters


# ---------------------------------------------------------------------------
# light cones
# ---------------------------------------------------------------------------
def target_groups(spec: CircuitSpec) -> list[tuple[int, ...]]:
    """Group the qubits so that qubits sharing their final gate are simulated
    together (they have the same light cone)."""
    last_gate: dict[int, tuple] = {}
    for li, layer in enumerate(spec.layers):
        for (q1, q2, slot) in layer:
            last_gate[q1] = (li, q1, q2)
            last_gate[q2] = (li, q1, q2)
    groups: dict[tuple, list[int]] = {}
    for q in range(spec.n_qubits):
        groups.setdefault(last_gate.get(q, ("none", q)), []).append(q)
    return [tuple(v) for v in groups.values()]


def backward_cone(spec: CircuitSpec, targets: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """All gates that can influence `targets`, in the order they are applied.
    Within a layer, gates far from the targets come first so that their outer
    qubits can be discarded early."""
    support = set(targets)
    centre = float(np.mean(targets))
    cone_layers = []
    for layer in reversed(spec.layers):
        hit = [g for g in layer if g[0] in support or g[1] in support]
        for (q1, q2, _) in hit:
            support.update((q1, q2))
        hit.sort(key=lambda g: -abs((g[0] + g[1]) / 2.0 - centre))
        cone_layers.append(hit)
    return [g for layer in reversed(cone_layers) for g in layer]


# ---------------------------------------------------------------------------
# noiseless simulator (pure states)
# ---------------------------------------------------------------------------
def _expectations_from_state(psi: np.ndarray, axis: int):
    """<X>, <Y>, <Z> of the qubit stored on `axis` of the state tensor."""
    moved = np.moveaxis(psi, axis, 1).reshape(psi.shape[0], 2, -1)
    a, b = moved[:, 0, :], moved[:, 1, :]
    ab = np.einsum("ni,ni->n", a.conj(), b)
    z = np.einsum("ni,ni->n", a.conj(), a).real - np.einsum("ni,ni->n", b.conj(), b).real
    return 2.0 * ab.real, 2.0 * ab.imag, z


def _simulate_group_pure(spec: CircuitSpec, phi: np.ndarray, targets: tuple[int, ...]) -> np.ndarray:
    gates = backward_cone(spec, targets)
    qubits = sorted(set(targets).union(*[(g[0], g[1]) for g in gates]))
    pos = {q: i for i, q in enumerate(qubits)}
    init = spec.initial_states()
    psi0 = np.array([1.0 + 0.0j])
    for q in qubits:
        psi0 = np.multiply.outer(psi0, init[q])
    psi0 = psi0.reshape((2,) * len(qubits))
    n = phi.shape[0]
    psi = np.broadcast_to(psi0, (n,) + psi0.shape).copy()
    bshape = (n,) + (1,) * len(qubits)
    for (q1, q2, slot) in gates:
        c = np.cos(2.0 * phi[:, slot]).reshape(bshape)
        s = np.sin(2.0 * phi[:, slot]).reshape(bshape)
        psi = c * psi - 1j * s * np.swapaxes(psi, 1 + pos[q1], 1 + pos[q2])
    out = np.empty((n, len(targets), 3))
    for k, q in enumerate(targets):
        out[:, k, 0], out[:, k, 1], out[:, k, 2] = _expectations_from_state(psi, 1 + pos[q])
    return out


def simulate_noiseless(spec: CircuitSpec, phi: np.ndarray, chunk: int = 8000) -> np.ndarray:
    """Exact expectation values on a perfect quantum computer.

    phi : (n_events, n_slots) gate angles from `spec.gate_angles`.
    Returns (n_events, 3 * n_qubits) in the order X_0.., Y_0.., Z_0.."""
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    n, nq = phi.shape[0], spec.n_qubits
    out = np.empty((n, 3 * nq))
    groups = target_groups(spec)
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        for targets in groups:
            res = _simulate_group_pure(spec, phi[sl], targets)
            for k, q in enumerate(targets):
                out[sl, q], out[sl, nq + q], out[sl, 2 * nq + q] = res[:, k, 0], res[:, k, 1], res[:, k, 2]
    return out


# ---------------------------------------------------------------------------
# noisy simulator (density matrices with early discarding of qubits)
# ---------------------------------------------------------------------------
class _Cluster:
    """A group of qubits that are currently entangled with each other.
    `rho` has shape (batch, 2,..,2, 2,..,2): one "ket" axis and one "bra"
    axis per qubit."""

    def __init__(self, qubits: list[int], rho: np.ndarray):
        self.qubits = qubits
        self.rho = rho

    @property
    def k(self) -> int:
        return len(self.qubits)

    def _subs(self):
        k = self.k
        return _LETTERS[:k], _LETTERS[k: 2 * k]

    def apply_gate(self, q1: int, q2: int, c: np.ndarray, s: np.ndarray) -> None:
        """rho -> U rho U^dagger with U = c*I - i*s*SWAP(q1, q2)."""
        i, j, k = self.qubits.index(q1), self.qubits.index(q2), self.k
        rho = self.rho
        s_rho = np.swapaxes(rho, 1 + i, 1 + j)                 # SWAP acting on the left
        rho_s = np.swapaxes(rho, 1 + k + i, 1 + k + j)         # SWAP acting on the right
        s_rho_s = np.swapaxes(s_rho, 1 + k + i, 1 + k + j)
        shape = (-1,) + (1,) * (2 * k)
        c, s = c.reshape(shape), s.reshape(shape)
        self.rho = c * c * rho + s * s * s_rho_s + 1j * s * c * (rho_s - s_rho)

    def trace_out(self, q: int) -> None:
        """Discard qubit q (average over everything it could be doing)."""
        i = self.qubits.index(q)
        kets, bras = self._subs()
        bras_t = bras[:i] + kets[i] + bras[i + 1:]
        out = kets[:i] + kets[i + 1:] + bras[:i] + bras[i + 1:]
        self.rho = np.einsum(f"Z{kets}{bras_t}->Z{out}", self.rho)
        self.qubits = self.qubits[:i] + self.qubits[i + 1:]

    def depolarize(self, q1: int, q2: int, lam: np.ndarray) -> None:
        """With probability lam, replace the state of (q1, q2) by pure noise:
        rho -> (1 - lam) rho + lam * Tr_{q1 q2}(rho) (x) I/4."""
        i, j = self.qubits.index(q1), self.qubits.index(q2)
        kets, bras = self._subs()
        bras_t = list(bras)
        bras_t[i], bras_t[j] = kets[i], kets[j]
        keep_k = "".join(ch for n, ch in enumerate(kets) if n not in (i, j))
        keep_b = "".join(ch for n, ch in enumerate(bras) if n not in (i, j))
        reduced = np.einsum(f"Z{kets}{''.join(bras_t)}->Z{keep_k}{keep_b}", self.rho)
        eye = np.eye(2) / 2.0
        mixed = np.einsum(f"Z{keep_k}{keep_b},{kets[i]}{bras[i]},{kets[j]}{bras[j]}->Z{kets}{bras}",
                          reduced, eye, eye)
        lam = np.asarray(lam, dtype=float).reshape((-1,) + (1,) * (2 * self.k))
        self.rho = (1.0 - lam) * self.rho + lam * mixed


def _merge(c1: _Cluster, c2: _Cluster) -> _Cluster:
    k1, k2 = c1.k, c2.k
    a_k, a_b = _LETTERS[:k1], _LETTERS[k1: 2 * k1]
    b_k, b_b = _LETTERS[2 * k1: 2 * k1 + k2], _LETTERS[2 * k1 + k2: 2 * k1 + 2 * k2]
    rho = np.einsum(f"Z{a_k}{a_b},Z{b_k}{b_b}->Z{a_k}{b_k}{a_b}{b_b}", c1.rho, c2.rho)
    return _Cluster(c1.qubits + c2.qubits, rho)


def _simulate_group_noisy(spec: CircuitSpec, phi: np.ndarray, lam: np.ndarray,
                          targets: tuple[int, ...]) -> np.ndarray:
    gates = backward_cone(spec, targets)
    n = phi.shape[0]
    init = spec.initial_states()
    last_use: dict[int, int] = {}
    for gi, (q1, q2, _) in enumerate(gates):
        last_use[q1] = gi
        last_use[q2] = gi
    cluster_of: dict[int, _Cluster] = {}

    def get(q: int) -> _Cluster:
        if q not in cluster_of:
            rho0 = np.outer(init[q], init[q].conj())[None, :, :]
            cluster_of[q] = _Cluster([q], np.broadcast_to(rho0, (n, 2, 2)).copy())
        return cluster_of[q]

    for gi, (q1, q2, slot) in enumerate(gates):
        ca, cb = get(q1), get(q2)
        if ca is not cb:
            merged = _merge(ca, cb)
            for q in merged.qubits:
                cluster_of[q] = merged
            ca = merged
        ca.apply_gate(q1, q2, np.cos(2.0 * phi[:, slot]), np.sin(2.0 * phi[:, slot]))
        ca.depolarize(q1, q2, lam[:, slot])
        for q in (q1, q2):                    # discard qubits that are finished
            if last_use[q] == gi and q not in targets:
                ca.trace_out(q)
                del cluster_of[q]

    out = np.empty((n, len(targets), 3))
    for k, q in enumerate(targets):
        cl = get(q)
        kets, bras = cl._subs()
        i = cl.qubits.index(q)
        bras_t = "".join(kets[m] if m != i else bras[m] for m in range(cl.k))
        rho_q = np.einsum(f"Z{kets}{bras_t}->Z{kets[i]}{bras[i]}", cl.rho)   # (n, 2, 2)
        out[:, k, 0] = 2.0 * rho_q[:, 1, 0].real
        out[:, k, 1] = 2.0 * rho_q[:, 1, 0].imag
        out[:, k, 2] = (rho_q[:, 0, 0] - rho_q[:, 1, 1]).real
    return out


def simulate_noisy(spec: CircuitSpec, phi: np.ndarray, lam: np.ndarray,
                   chunk: int = 4000, n_jobs: int = 1) -> np.ndarray:
    """Exact expectation values when every gate is followed by a two-qubit
    depolarizing error.

    phi : (n_events, n_slots) gate angles.
    lam : depolarizing probability of each gate; shape (n_slots,) if it is
          the same for all events, or (n_events, n_slots) if it drifts.
    Returns (n_events, 3 * n_qubits), same order as `simulate_noiseless`."""
    phi = np.atleast_2d(np.asarray(phi, dtype=float))
    n, nq = phi.shape[0], spec.n_qubits
    lam = np.asarray(lam, dtype=float)
    if lam.ndim == 1:
        lam = np.broadcast_to(lam, (n, lam.shape[0]))
    groups = target_groups(spec)

    def run(targets):
        res = np.empty((n, len(targets), 3))
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            res[sl] = _simulate_group_noisy(spec, phi[sl], lam[sl], targets)
        return res

    if n_jobs == 1:
        results = [run(t) for t in groups]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(run)(t) for t in groups)
    out = np.empty((n, 3 * nq))
    for targets, res in zip(groups, results):
        for k, q in enumerate(targets):
            out[:, q], out[:, nq + q], out[:, 2 * nq + q] = res[:, k, 0], res[:, k, 1], res[:, k, 2]
    return out
