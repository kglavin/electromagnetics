"""Electrostatic fields of point charges and discrete vector calculus on grids."""

import numpy as np

EPS0 = 8.8541878128e-12  # F/m
K = 1.0 / (4.0 * np.pi * EPS0)  # Coulomb constant, N m^2 / C^2


def _as_sources(sources, charges):
    sources = np.atleast_2d(np.asarray(sources, dtype=float))
    charges = np.atleast_1d(np.asarray(charges, dtype=float))
    if sources.shape[0] != charges.shape[0]:
        raise ValueError("one charge per source position is required")
    return sources, charges


def point_charges_E(sources, charges, obs, chunk=512):
    """Electric field of point charges by superposition of Coulomb's law.

    sources: (M, 3) positions. charges: (M,) values in coulomb. obs: (..., 3) observation points.
    Returns E with the same leading shape as obs and a trailing axis of length 3.
    The contribution of a source evaluated exactly at its own position is set to zero.
    Sources are processed in chunks to bound memory for large grids.
    """
    sources, charges = _as_sources(sources, charges)
    obs = np.asarray(obs, dtype=float)
    flat = obs.reshape(-1, 3)
    E = np.zeros_like(flat)
    for start in range(0, sources.shape[0], chunk):
        s = sources[start : start + chunk]
        q = charges[start : start + chunk]
        r = flat[:, None, :] - s[None, :, :]  # (N, m, 3)
        d = np.linalg.norm(r, axis=-1)  # (N, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(d > 0, q[None, :] / d**3, 0.0)
        E += K * np.einsum("nm,nmk->nk", w, r)
    return E.reshape(obs.shape)


def point_charges_V(sources, charges, obs, chunk=512):
    """Electric potential of point charges with V = 0 at infinity.

    Same conventions as point_charges_E. Returns an array with the leading shape of obs.
    """
    sources, charges = _as_sources(sources, charges)
    obs = np.asarray(obs, dtype=float)
    flat = obs.reshape(-1, 3)
    V = np.zeros(flat.shape[0])
    for start in range(0, sources.shape[0], chunk):
        s = sources[start : start + chunk]
        q = charges[start : start + chunk]
        d = np.linalg.norm(flat[:, None, :] - s[None, :, :], axis=-1)
        with np.errstate(divide="ignore"):
            V += K * np.where(d > 0, q[None, :] / d, 0.0).sum(axis=1)
    return V.reshape(obs.shape[:-1])


def gradient(f, *h):
    """Gradient of a scalar grid f with spacings h (one per axis). Returns (..., ndim) array."""
    return np.stack(np.gradient(f, *h), axis=-1)


def divergence(F, *h):
    """Divergence of a vector grid F of shape (..., ndim) with spacings h."""
    ndim = F.shape[-1]
    return sum(np.gradient(F[..., i], h[i], axis=i) for i in range(ndim))


def curl(F, hx, hy, hz=None):
    """Curl of a vector grid.

    For a 3D grid F has shape (nx, ny, nz, 3). For a 2D grid F has shape (nx, ny, 3), the
    fields are taken independent of z, and only the z component of the curl is non-zero
    unless F has a z component that varies in the plane.
    """
    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]
    if hz is None:
        dFz_dy = np.gradient(Fz, hy, axis=1)
        dFz_dx = np.gradient(Fz, hx, axis=0)
        dFy_dx = np.gradient(Fy, hx, axis=0)
        dFx_dy = np.gradient(Fx, hy, axis=1)
        return np.stack([dFz_dy, -dFz_dx, dFy_dx - dFx_dy], axis=-1)
    dFz_dy, dFy_dz = np.gradient(Fz, hy, axis=1), np.gradient(Fy, hz, axis=2)
    dFx_dz, dFz_dx = np.gradient(Fx, hz, axis=2), np.gradient(Fz, hx, axis=0)
    dFy_dx, dFx_dy = np.gradient(Fy, hx, axis=0), np.gradient(Fx, hy, axis=1)
    return np.stack([dFz_dy - dFy_dz, dFx_dz - dFz_dx, dFy_dx - dFx_dy], axis=-1)


def line_integral(F, path):
    """Midpoint-rule line integral of a vector field along a polyline.

    F: callable mapping an (N, 3) array of points to an (N, 3) array of vectors.
    path: (N, 3) array of vertices, in order. Returns the scalar integral of F . dl.
    """
    path = np.asarray(path, dtype=float)
    dl = np.diff(path, axis=0)
    mid = 0.5 * (path[1:] + path[:-1])
    return float(np.einsum("nk,nk->", F(mid), dl))
