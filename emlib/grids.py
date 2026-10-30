"""Grid construction and reshaping.

All meshgrids use ``indexing="ij"`` so that array index i follows x and index j follows y.
NumPy's default ``"xy"`` mode swaps the first two axes, which silently rotates every field plot.
"""

import numpy as np


def grid2d(xlim, ylim, n, z=0.0):
    """Return (x, y, P) for a uniform 2D grid in the plane z = const.

    x, y are 1D coordinate vectors of length n (or n[0], n[1]).
    P has shape (nx, ny, 3) and holds the position vector of every grid point.
    """
    nx, ny = (n, n) if np.isscalar(n) else n
    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    P = np.stack([X, Y, np.full_like(X, z)], axis=-1)
    return x, y, P


def grid3d(xlim, ylim, zlim, n):
    """Return (x, y, z, P) for a uniform 3D grid. P has shape (nx, ny, nz, 3)."""
    nx, ny, nz = (n, n, n) if np.isscalar(n) else n
    x = np.linspace(*xlim, nx)
    y = np.linspace(*ylim, ny)
    z = np.linspace(*zlim, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    P = np.stack([X, Y, Z], axis=-1)
    return x, y, z, P


def spacing(*coords):
    """Grid step of each uniform coordinate vector."""
    return tuple(float(c[1] - c[0]) for c in coords)


def components(V):
    """Split a (..., 3) vector array into its three component arrays."""
    return V[..., 0], V[..., 1], V[..., 2]


def magnitude(V):
    """Length of each vector in a (..., 3) array."""
    return np.linalg.norm(V, axis=-1)
