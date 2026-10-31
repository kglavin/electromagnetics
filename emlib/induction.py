"""Faraday induction and inductance: flux through loops, emf, and mutual inductance by the Neumann formula."""

import numpy as np
from scipy.special import ellipe, ellipk

from .biot_savart import MU0, _basis, segments


def disk_flux(B_func, centre, normal, R, nr=60, nphi=120):
    """Flux of B through a flat disk of radius R, by a midpoint rule in polar coordinates.

    B_func maps an (N, 3) array of points to an (N, 3) array of fields. The disk's orientation
    is given by normal; a loop around its rim traversed counter-clockwise about normal has
    the same orientation (right-hand rule).
    """
    u, v, n = _basis(normal)
    r = (np.arange(nr) + 0.5) * R / nr
    phi = (np.arange(nphi) + 0.5) * 2.0 * np.pi / nphi
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    pts = np.asarray(centre, dtype=float) + (rr * np.cos(pp))[..., None] * u + (rr * np.sin(pp))[..., None] * v
    dA = rr * (R / nr) * (2.0 * np.pi / nphi)
    Bn = B_func(pts.reshape(-1, 3)) @ n
    return float(np.sum(Bn * dA.ravel()))


def emf_from_flux(t, flux):
    """emf = -dPhi/dt by central differences on a time series."""
    return -np.gradient(np.asarray(flux, dtype=float), np.asarray(t, dtype=float))


def neumann_M(path1, path2, closed=True, chunk=512):
    """Mutual inductance of two wires by the Neumann formula, mu0/(4 pi) double-sum of dl1 . dl2 / |r1 - r2|.

    path1, path2: (N, 3) vertex arrays. The paths must not touch: segment midpoints of one wire
    should be farther from the other wire than a segment length.
    """
    m1, d1 = segments(path1, closed)
    m2, d2 = segments(path2, closed)
    total = 0.0
    for s in range(0, len(m1), chunk):
        dist = np.linalg.norm(m1[s : s + chunk, None, :] - m2[None, :, :], axis=-1)
        total += np.sum((d1[s : s + chunk] @ d2.T) / dist)
    return MU0 / (4.0 * np.pi) * total


def coaxial_loops_M(a, b, d):
    """Exact mutual inductance of two coaxial circular loops of radii a and b, a distance d apart."""
    m = 4.0 * a * b / ((a + b) ** 2 + d**2)  # parameter m = k^2
    k = np.sqrt(m)
    return MU0 * np.sqrt(a * b) * ((2.0 / k - k) * ellipk(m) - 2.0 / k * ellipe(m))


def solenoid_L(N, R, length):
    """Inductance mu0 N^2 pi R^2 / length of a long solenoid (end effects ignored)."""
    return MU0 * N**2 * np.pi * R**2 / length


def ring_self_L(R, a):
    """Low-frequency self-inductance of a circular ring of radius R made of round wire of radius a << R."""
    return MU0 * R * (np.log(8.0 * R / a) - 1.75)
