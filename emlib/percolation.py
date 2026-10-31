"""Random resistor lattices: bond percolation on a square grid and its effective conductivity.

The lattice has n by n nodes. Node (i, j) sits at x = i, y = j. Horizontal bonds join (i, j) to
(i+1, j) and vertical bonds join (i, j) to (i, j+1). Each bond is present (unit conductance)
with probability p. The column i = 0 is an electrode at V = 1 and the column i = n - 1 an
electrode at V = 0. On an infinite square lattice a spanning cluster first appears at p = 1/2.
"""

import numpy as np
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components

from .circuits import grid_edges, solve_network

P_C_BOND_SQUARE = 0.5


def random_bonds(n, p, rng=None):
    """Present bonds of an n by n node lattice. Returns (h_bonds (n-1, n), v_bonds (n, n-1)) bool."""
    rng = np.random.default_rng(rng)
    return rng.random((n - 1, n)) < p, rng.random((n, n - 1)) < p


def _edges_and_G(h_bonds, v_bonds):
    n = h_bonds.shape[1]
    h_edges, v_edges = grid_edges(n, n)
    edges = np.concatenate([h_edges.reshape(-1, 2), v_edges.reshape(-1, 2)])
    G = np.concatenate([h_bonds.ravel(), v_bonds.ravel()]).astype(float)
    return n, edges, G


def spans(h_bonds, v_bonds):
    """True if some cluster of present bonds connects the left column to the right column."""
    n, edges, G = _edges_and_G(h_bonds, v_bonds)
    live = G > 0
    a, b = edges[live, 0], edges[live, 1]
    adj = sps.coo_matrix((np.ones(len(a)), (a, b)), shape=(n * n, n * n))
    _, label = connected_components(adj, directed=False)
    label = label.reshape(n, n)
    return bool(np.intersect1d(label[0, :], label[-1, :]).size)


def solve_lattice(h_bonds, v_bonds):
    """Kirchhoff solution with the left column at V = 1 and the right column at V = 0.

    Returns (V (n, n), Ih (n-1, n) current along +x, Iv (n, n-1) current along +y, I_total).
    """
    n, edges, G = _edges_and_G(h_bonds, v_bonds)
    fixed = np.zeros((n, n), dtype=bool)
    fixed[0, :] = fixed[-1, :] = True
    V_fixed = np.zeros((n, n))
    V_fixed[0, :] = 1.0
    V, I_edge = solve_network(n * n, edges, G, fixed.ravel(), V_fixed.ravel())
    nh = (n - 1) * n
    Ih = I_edge[:nh].reshape(n - 1, n)
    Iv = I_edge[nh:].reshape(n, n - 1)
    return V.reshape(n, n), Ih, Iv, float(Ih[0, :].sum())


def effective_conductivity(h_bonds, v_bonds):
    """Sheet conductivity in units of the bond conductance, normalised so that the full lattice gives 1.

    The full lattice is n parallel rows of n - 1 unit bonds in series: conductance n / (n - 1).
    """
    n = h_bonds.shape[1]
    return solve_lattice(h_bonds, v_bonds)[3] * (n - 1) / n


def backbone(Ih, Iv, rel_tol=1e-9):
    """Bonds that carry current: the backbone of the spanning cluster. Dangling ends carry none."""
    scale = max(np.abs(Ih).max(initial=0.0), np.abs(Iv).max(initial=0.0), 1e-300)
    return np.abs(Ih) > rel_tol * scale, np.abs(Iv) > rel_tol * scale
