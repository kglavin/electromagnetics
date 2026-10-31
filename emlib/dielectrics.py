"""Polarization, bound charge, displacement, and Poisson's equation in a dielectric.

Conventions follow fields.py: vector grids have a trailing axis of length 3 (or ndim),
scalar grids match the leading shape, and spacings are passed one per axis.
"""

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from .fields import EPS0, divergence


def bound_volume_charge(P, *h):
    """rho_b = -div P on a grid."""
    return -divergence(P, *h)


def bound_surface_charge(P, normal):
    """sigma_b = P . n for polarization P (..., 3) at a surface with outward unit normal n."""
    return np.einsum("...k,k->...", P, np.asarray(normal, dtype=float))


def displacement(E, P):
    """D = eps0 E + P."""
    return EPS0 * E + P


def dipole_sources(positions, moments, delta):
    """Represent point dipoles as pairs of opposite charges.

    positions: (M, 3) dipole centres. moments: (M, 3) dipole moments p. delta: charge separation.
    Each dipole becomes +q at r + d/2 and -q at r - d/2 with q d = p, d parallel to p.
    Returns (sources, charges) ready for fields.point_charges_E / point_charges_V.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    moments = np.atleast_2d(np.asarray(moments, dtype=float))
    pmag = np.linalg.norm(moments, axis=1)
    direction = moments / np.where(pmag > 0, pmag, 1.0)[:, None]
    q = pmag / delta
    plus = positions + 0.5 * delta * direction
    minus = positions - 0.5 * delta * direction
    return np.vstack([plus, minus]), np.concatenate([q, -q])


def _with_fixed_boundary(fixed):
    fixed = np.array(fixed, dtype=bool, copy=True)
    for a in range(fixed.ndim):
        fixed[(slice(None),) * a + (0,)] = True
        fixed[(slice(None),) * a + (-1,)] = True
    return fixed


def solve_dielectric(V0, fixed, eps_r, rho_f=None, h=1.0, direct_limit=20_000, tol=1e-10):
    """Solution of div(eps_r eps0 grad V) = -rho_f on a uniform 2D or 3D grid.

    V0 holds the Dirichlet values where fixed is True (the outer boundary is always fixed).
    eps_r is the relative permittivity at each node. The coefficient on each face is the
    harmonic mean of the two adjacent nodes, which keeps the normal component of D continuous
    across a material interface. With eps_r = 1 everywhere this is the usual Poisson stencil.

    Only free nodes are unknowns, so the system is symmetric positive definite. Up to
    direct_limit unknowns it is solved directly; above that, by conjugate gradients with a
    Jacobi preconditioner, which is what makes 3D grids affordable.
    """
    V0 = np.asarray(V0, dtype=float)
    eps_r = np.broadcast_to(np.asarray(eps_r, dtype=float), V0.shape)
    fixed = _with_fixed_boundary(fixed)
    shape = V0.shape
    src = np.zeros(shape) if rho_f is None else h * h * np.asarray(rho_f, dtype=float) / EPS0

    free = ~fixed
    n_free = int(free.sum())
    number = -np.ones(shape, dtype=np.int64)
    number[free] = np.arange(n_free)
    free_idx = np.nonzero(free)
    row_of_free = number[free_idx]

    rows, cols, vals = [], [], []
    diag = np.zeros(n_free)
    b = src[free_idx].copy()
    for a in range(V0.ndim):
        for step in (+1, -1):
            nb_idx = list(free_idx)
            nb_idx[a] = nb_idx[a] + step
            nb_idx = tuple(nb_idx)
            c = 2.0 * eps_r[free_idx] * eps_r[nb_idx] / (eps_r[free_idx] + eps_r[nb_idx])
            diag += c
            nb_free = free[nb_idx]
            rows.append(row_of_free[nb_free]); cols.append(number[nb_idx][nb_free]); vals.append(-c[nb_free])
            b[~nb_free] += c[~nb_free] * V0[nb_idx][~nb_free]  # known neighbour moves to the right-hand side
    rows.append(row_of_free); cols.append(row_of_free); vals.append(diag)
    A = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_free, n_free))

    if n_free <= direct_limit:
        sol = spla.spsolve(A.tocsc(), b)
    else:
        M = sps.diags(1.0 / A.diagonal())
        sol, info = spla.cg(A, b, M=M, rtol=tol, maxiter=20_000)
        if info != 0:
            raise RuntimeError(f"conjugate gradient did not converge (info={info})")
    V = V0.copy()
    V[free_idx] = sol
    return V


def energy(V, eps_r, h):
    """Electrostatic energy (1/2) eps0 eps_r |E|^2 integrated with the solver's own discretisation.

    E is evaluated on the faces between nodes with the harmonic-mean face permittivity, so the
    result is consistent with solve_dielectric to rounding error. A node-based integral puts a
    single permittivity on each interface node and is only first-order accurate there.
    Returns energy per unit depth in 2D, total energy in 3D.
    """
    V = np.asarray(V, dtype=float)
    eps_r = np.broadcast_to(np.asarray(eps_r, dtype=float), V.shape)
    cell = h**V.ndim
    total = 0.0
    for a in range(V.ndim):
        lo = (slice(None),) * a + (slice(None, -1),)
        hi = (slice(None),) * a + (slice(1, None),)
        e_face = 2.0 * eps_r[lo] * eps_r[hi] / (eps_r[lo] + eps_r[hi])
        E_face = (V[hi] - V[lo]) / h
        density = e_face * E_face**2
        # Faces are centred along axis a but sit on nodes along the other axes, so integrate
        # those directions with the trapezoid rule: half weight on the two boundary rows.
        for b in range(V.ndim):
            if b != a:
                w = np.ones(V.shape[b]); w[0] = w[-1] = 0.5
                density = density * w.reshape([-1 if d == b else 1 for d in range(V.ndim)])
        total += 0.5 * EPS0 * np.sum(density) * cell
    return float(total)
