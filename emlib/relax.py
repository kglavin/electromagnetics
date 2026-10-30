"""Laplace and Poisson solvers on a uniform 2D grid.

The discrete Poisson equation at an interior node is

    V[i+1,j] + V[i-1,j] + V[i,j+1] + V[i,j-1] - 4 V[i,j] = -h^2 rho[i,j] / eps0

Nodes marked in the boolean mask ``fixed`` hold a Dirichlet value and are never updated.
The outer boundary is always treated as fixed. Three solvers share this convention:
Jacobi and Gauss-Seidel (optionally over-relaxed) sweep until the largest change falls below
a tolerance; the sparse solver assembles the linear system and solves it directly.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from numba import njit

from .fields import EPS0, gradient


def _source_term(rho, h, eps):
    return np.zeros_like(rho, dtype=float) if rho is None else h * h * np.asarray(rho, dtype=float) / eps


def _with_fixed_boundary(fixed):
    fixed = np.array(fixed, dtype=bool, copy=True)
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    return fixed


@njit(cache=True)
def jacobi_step(V, fixed, src):
    """One Jacobi update from V into a new array. Returns (V_new, max_change)."""
    nx, ny = V.shape
    Vn = V.copy()
    change = 0.0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if not fixed[i, j]:
                new = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1] + src[i, j])
                d = abs(new - V[i, j])
                if d > change:
                    change = d
                Vn[i, j] = new
    return Vn, change


@njit(cache=True)
def gauss_seidel_sweep(V, fixed, src, omega):
    """One in-place Gauss-Seidel sweep with over-relaxation factor omega. Returns max_change."""
    nx, ny = V.shape
    change = 0.0
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if not fixed[i, j]:
                gs = 0.25 * (V[i + 1, j] + V[i - 1, j] + V[i, j + 1] + V[i, j - 1] + src[i, j])
                new = V[i, j] + omega * (gs - V[i, j])
                d = abs(new - V[i, j])
                if d > change:
                    change = d
                V[i, j] = new
    return change


def solve_relax(V0, fixed, rho=None, h=1.0, method="gauss-seidel", omega=1.0, tol=1e-6, max_iter=200_000):
    """Iterate from V0 until the largest update is below tol.

    Returns (V, history) where history[k] is the max change at sweep k. The length of history
    is the number of sweeps taken. Use method="jacobi" or "gauss-seidel"; omega > 1 gives SOR.
    """
    V = np.array(V0, dtype=float, copy=True)
    fixed = _with_fixed_boundary(fixed)
    src = _source_term(rho if rho is not None else np.zeros_like(V), h, EPS0)
    history = []
    for _ in range(max_iter):
        if method == "jacobi":
            V, change = jacobi_step(V, fixed, src)
        elif method == "gauss-seidel":
            change = gauss_seidel_sweep(V, fixed, src, omega)
        else:
            raise ValueError("method must be 'jacobi' or 'gauss-seidel'")
        history.append(change)
        if change < tol:
            break
    return V, np.array(history)


def sor_omega(nx, ny):
    """Near-optimal over-relaxation factor for a rectangular grid."""
    rho_j = 0.5 * (np.cos(np.pi / (nx - 1)) + np.cos(np.pi / (ny - 1)))
    return 2.0 / (1.0 + np.sqrt(1.0 - rho_j**2))


def solve_sparse(V0, fixed, rho=None, h=1.0):
    """Direct solution of the discrete Poisson equation with Dirichlet values from V0 where fixed."""
    V0 = np.asarray(V0, dtype=float)
    fixed = _with_fixed_boundary(fixed)
    nx, ny = V0.shape
    n = nx * ny
    idx = np.arange(n).reshape(nx, ny)
    src = _source_term(rho if rho is not None else np.zeros_like(V0), h, EPS0)

    rows, cols, vals = [], [], []
    b = np.zeros(n)
    free = ~fixed
    fi, fj = np.nonzero(free)
    centre = idx[fi, fj]
    rows.append(centre); cols.append(centre); vals.append(np.full(centre.size, -4.0))
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = idx[fi + di, fj + dj]
        rows.append(centre); cols.append(nb); vals.append(np.ones(centre.size))
    b[centre] = -src[fi, fj]

    ki, kj = np.nonzero(fixed)
    known = idx[ki, kj]
    rows.append(known); cols.append(known); vals.append(np.ones(known.size))
    b[known] = V0[ki, kj]

    A = sps.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return spla.spsolve(A, b).reshape(nx, ny)


def field_energy(V, h, eps=EPS0):
    """Electrostatic energy (1/2) eps |E|^2 integrated over a 2D grid, per unit depth."""
    E = -gradient(V, h, h)
    u = 0.5 * eps * np.sum(E**2, axis=-1)
    return float(np.trapezoid(np.trapezoid(u, dx=h, axis=1), dx=h, axis=0))
