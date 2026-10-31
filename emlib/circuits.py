"""Resistor networks, modified nodal analysis for DC and AC circuits, and conducting sheets.

Three levels share one idea, Kirchhoff's current law at every node:

- solve_network: a graph of conductances with some node potentials fixed. Used directly for
  random resistor lattices and for a conducting sheet, which is a resistor network on a grid.
- mna: small circuits written as a list of components, including voltage sources, capacitors
  and inductors. With omega given, capacitors and inductors become complex admittances and the
  same solve gives the phasor response.
- solve_sheet: current flow in a 2D sheet of varying conductivity between two electrodes.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


def grid_edges(nx, ny):
    """Node pairs of a 2D grid of nx by ny nodes, node number i * ny + j.

    Returns (h_edges, v_edges): h_edges (nx-1, ny, 2) joins (i, j) to (i+1, j) and
    v_edges (nx, ny-1, 2) joins (i, j) to (i, j+1).
    """
    idx = np.arange(nx * ny).reshape(nx, ny)
    h_edges = np.stack([idx[:-1, :], idx[1:, :]], axis=-1)
    v_edges = np.stack([idx[:, :-1], idx[:, 1:]], axis=-1)
    return h_edges, v_edges


def solve_network(n_nodes, edges, G, fixed, V_fixed, I_inj=None):
    """Node potentials of a conductance network with Kirchhoff's current law at every free node.

    edges: (M, 2) node pairs. G: (M,) conductances (zero means no connection; complex allowed).
    fixed: (n_nodes,) bool, True where the potential is given in V_fixed. I_inj: (n_nodes,)
    current injected into each node from outside (default zero).

    Nodes that belong to a cluster with no fixed node carry no current and are set to zero
    potential, so isolated clusters of a random lattice do not make the system singular.
    Returns (V, I_edge) with I_edge = G * (V[a] - V[b]), the current from a to b.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    G = np.asarray(G).reshape(-1)
    fixed = np.asarray(fixed, dtype=bool).copy()
    dtype = np.result_type(G.dtype, np.asarray(V_fixed).dtype, float)
    V = np.zeros(n_nodes, dtype=dtype)
    V[fixed] = np.asarray(V_fixed, dtype=dtype)[fixed]
    inj = np.zeros(n_nodes, dtype=dtype) if I_inj is None else np.asarray(I_inj, dtype=dtype)

    live = G != 0
    a, b, g = edges[live, 0], edges[live, 1], G[live]
    adj = sps.coo_matrix((np.ones(len(a)), (a, b)), shape=(n_nodes, n_nodes))
    _, label = connected_components(adj, directed=False)
    anchored = np.zeros(label.max() + 1, dtype=bool)
    anchored[label[fixed]] = True
    floating = ~anchored[label]
    if np.any(inj[floating & ~fixed] != 0):
        raise ValueError("current is injected into a cluster with no fixed node")
    fixed = fixed | floating  # floating clusters held at V = 0

    free = ~fixed
    number = -np.ones(n_nodes, dtype=np.int64)
    number[free] = np.arange(free.sum())
    L = sps.coo_matrix(
        (np.concatenate([g, g, -g, -g]), (np.concatenate([a, b, a, b]), np.concatenate([a, b, b, a]))),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    A = L[free][:, free]
    rhs = inj[free] - L[free][:, fixed] @ V[fixed]
    if A.shape[0]:
        V[free] = spla.spsolve(A.tocsc(), rhs)
    I_edge = G * (V[edges[:, 0]] - V[edges[:, 1]])
    return V, I_edge


def mna(n_nodes, resistors=(), current_sources=(), voltage_sources=(), capacitors=(), inductors=(), omega=None, ground=0):
    """Modified nodal analysis of a lumped circuit.

    Components are tuples (a, b, value) between nodes a and b:
      resistors (ohm); current_sources (A: the source carries I from a to b through itself, so it
      draws I out of node a and delivers it into node b); voltage_sources (V, with
      V[a] - V[b] = value); capacitors (F); inductors (H).
    With omega=None the circuit is solved at DC: capacitors are open and inductors are shorts
    (an inductor is then a zero-volt voltage source). With omega given, every value may be a
    complex amplitude and capacitors and inductors become admittances j omega C and 1/(j omega L).

    Returns (V, I_vs): node potentials with V[ground] = 0, and the current through each voltage
    source from a to b inside the source (for DC, inductor currents are appended after the sources).
    """
    ac = omega is not None
    dtype = complex if ac else float
    Y = np.zeros((n_nodes, n_nodes), dtype=dtype)
    rhs = np.zeros(n_nodes, dtype=dtype)

    def stamp(a, b, y):
        Y[a, a] += y
        Y[b, b] += y
        Y[a, b] -= y
        Y[b, a] -= y

    for a, b, R in resistors:
        stamp(a, b, 1.0 / R)
    for a, b, I in current_sources:
        rhs[a] -= I
        rhs[b] += I
    vsrc = list(voltage_sources)
    if ac:
        for a, b, C in capacitors:
            stamp(a, b, 1j * omega * C)
        for a, b, Lh in inductors:
            stamp(a, b, 1.0 / (1j * omega * Lh))
    else:
        vsrc += [(a, b, 0.0) for a, b, _ in inductors]

    m = len(vsrc)
    A = np.zeros((n_nodes + m, n_nodes + m), dtype=dtype)
    A[:n_nodes, :n_nodes] = Y
    z = np.concatenate([rhs, np.zeros(m, dtype=dtype)])
    for k, (a, b, Vs) in enumerate(vsrc):
        A[a, n_nodes + k] += 1.0
        A[b, n_nodes + k] -= 1.0
        A[n_nodes + k, a] += 1.0
        A[n_nodes + k, b] -= 1.0
        z[n_nodes + k] = Vs
    keep = np.ones(n_nodes + m, dtype=bool)
    keep[ground] = False
    sol = np.zeros(n_nodes + m, dtype=dtype)
    sol[keep] = np.linalg.solve(A[np.ix_(keep, keep)], z[keep])
    return sol[:n_nodes], sol[n_nodes:]


def solve_sheet(sigma, V_left=1.0, V_right=0.0, h=1.0, thickness=1.0):
    """Steady current in a sheet of conductivity sigma (nx, ny) between two electrodes.

    Column i = 0 is held at V_left and column i = nx - 1 at V_right; the edges j = 0 and
    j = ny - 1 are insulating (no current crosses them). Each face between neighbouring nodes is
    a resistor with conductance sigma_face * thickness, where sigma_face is the harmonic mean of
    the two node values (zero if either is zero, so sigma = 0 marks a hole).

    Returns (V, Ix, Iy, I_total): Ix (nx-1, ny) is the current through each x face along +x,
    Iy (nx, ny-1) along +y, and I_total the current leaving the left electrode.
    """
    sigma = np.asarray(sigma, dtype=float)
    nx, ny = sigma.shape
    h_edges, v_edges = grid_edges(nx, ny)

    def face(s1, s2):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((s1 > 0) & (s2 > 0), 2.0 * s1 * s2 / (s1 + s2), 0.0)

    Gx = face(sigma[:-1, :], sigma[1:, :]) * thickness  # length h, cross-section h * thickness
    Gy = face(sigma[:, :-1], sigma[:, 1:]) * thickness
    edges = np.concatenate([h_edges.reshape(-1, 2), v_edges.reshape(-1, 2)])
    G = np.concatenate([Gx.ravel(), Gy.ravel()])
    fixed = np.zeros((nx, ny), dtype=bool)
    fixed[0, :] = fixed[-1, :] = True
    V_fixed = np.zeros((nx, ny))
    V_fixed[0, :], V_fixed[-1, :] = V_left, V_right
    V, I_edge = solve_network(nx * ny, edges, G, fixed.ravel(), V_fixed.ravel())
    nh = Gx.size
    Ix = I_edge[:nh].reshape(nx - 1, ny)
    Iy = I_edge[nh:].reshape(nx, ny - 1)
    return V.reshape(nx, ny), Ix, Iy, float(Ix[0, :].sum())


def rc_charging(t, R, C, V0):
    """Capacitor voltage for a capacitor charged through R from a step of V0 at t = 0."""
    return V0 * (1.0 - np.exp(-np.asarray(t) / (R * C)))


def resonance_omega(L, C):
    """Angular resonance frequency 1 / sqrt(L C) of an LC or series RLC circuit."""
    return 1.0 / np.sqrt(L * C)


def quality_factor(R, L, C):
    """Q of a series RLC circuit: omega0 L / R."""
    return np.sqrt(L / C) / R
