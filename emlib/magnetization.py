"""Magnetized matter: bound currents from a magnetization field and the auxiliary field H."""

import numpy as np

from . import fields
from .biot_savart import MU0


def bound_volume_current(M, hx, hy, hz=None):
    """J_b = curl M on a grid. M has shape (nx, ny, 3) or (nx, ny, nz, 3)."""
    return fields.curl(M, hx, hy, hz)


def bound_surface_current(M, normal):
    """K_b = M x n on a surface with outward unit normal n. Broadcasts over leading axes."""
    return np.cross(np.asarray(M, dtype=float), np.asarray(normal, dtype=float))


def H_field(B, M):
    """H = B / mu0 - M."""
    return np.asarray(B, dtype=float) / MU0 - np.asarray(M, dtype=float)


def B_from_H(H, chi_m=0.0):
    """B = mu0 (1 + chi_m) H for a linear magnetic material."""
    return MU0 * (1.0 + chi_m) * np.asarray(H, dtype=float)
