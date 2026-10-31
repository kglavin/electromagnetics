"""Magnetostatics: wires as segments, Biot-Savart superposition, closed-form reference fields."""

import numpy as np

MU0 = 4e-7 * np.pi  # T m / A


def _basis(normal):
    """Two unit vectors u, v perpendicular to normal, forming a right-handed set (u, v, n)."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def segments(path, closed=True):
    """Midpoints and step vectors of a polyline.

    path: (N, 3) vertices in order. If closed, the last vertex is joined back to the first.
    Returns (mid, dl), both (M, 3). The current flows from each vertex to the next.
    """
    path = np.asarray(path, dtype=float)
    if closed:
        path = np.vstack([path, path[:1]])
    dl = np.diff(path, axis=0)
    mid = 0.5 * (path[1:] + path[:-1])
    return mid, dl


def loop(R, n=200, centre=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)):
    """Vertices of a circular loop of radius R, counter-clockwise about normal. Use with segments(closed=True)."""
    u, v, _ = _basis(normal)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.asarray(centre, dtype=float) + R * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)


def straight_wire(start, end, n=1000):
    """Vertices along a straight segment from start to end. Use with segments(closed=False)."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return start + t * (end - start)


def helix(R, L, turns, n_per_turn=40, centre=(0.0, 0.0, 0.0)):
    """Vertices of a helical winding of radius R and length L along z, centred on centre.

    turns full turns, n_per_turn vertices each. Use with segments(closed=False).
    """
    n = int(turns * n_per_turn)
    t = np.linspace(0.0, 2.0 * np.pi * turns, n + 1)
    z = -L / 2 + L * t / (2.0 * np.pi * turns)
    return np.asarray(centre, dtype=float) + np.stack([R * np.cos(t), R * np.sin(t), z], axis=-1)


def helmholtz(R, n=200, centre=(0.0, 0.0, 0.0)):
    """Segments of a Helmholtz pair: two coaxial loops of radius R along z, separated by R."""
    c = np.asarray(centre, dtype=float)
    m1, d1 = segments(loop(R, n, c + [0, 0, -R / 2]))
    m2, d2 = segments(loop(R, n, c + [0, 0, R / 2]))
    return np.vstack([m1, m2]), np.vstack([d1, d2])


def wire_B(mid, dl, I, obs, chunk=512):
    """Magnetic field of current I flowing along segments (mid, dl), by the Biot-Savart law.

    obs: (..., 3) observation points. Returns B with the same leading shape and a trailing 3.
    A segment evaluated at its own midpoint contributes zero.
    """
    mid, dl = np.asarray(mid, dtype=float), np.asarray(dl, dtype=float)
    obs = np.asarray(obs, dtype=float)
    flat = obs.reshape(-1, 3)
    B = np.zeros_like(flat)
    for start in range(0, mid.shape[0], chunk):
        m, d = mid[start : start + chunk], dl[start : start + chunk]
        r = flat[:, None, :] - m[None, :, :]  # (N, M, 3)
        dist = np.linalg.norm(r, axis=-1)  # (N, M)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(dist > 0, 1.0 / dist**3, 0.0)
        B += np.einsum("nm,nmk->nk", w, np.cross(d[None, :, :], r))
    return (MU0 * I / (4.0 * np.pi)) * B.reshape(obs.shape)


# Closed forms used as tests.


def loop_axis_B(R, I, z):
    """B_z on the axis of a circular loop of radius R, current I, at height z above its centre."""
    return MU0 * I * R**2 / (2.0 * (z**2 + R**2) ** 1.5)


def straight_wire_B(I, r):
    """|B| at distance r from an infinite straight wire."""
    return MU0 * I / (2.0 * np.pi * r)


def finite_wire_B(I, r, L):
    """|B| on the perpendicular bisector of a straight segment of half-length L, at distance r."""
    return MU0 * I / (4.0 * np.pi * r) * 2.0 * L / np.sqrt(r**2 + L**2)


def solenoid_axis_B(R, L, n_density, I, z):
    """B_z on the axis of a solenoid of radius R and length L with n_density turns per length, z from its centre."""
    a, b = L / 2 + z, L / 2 - z
    return 0.5 * MU0 * n_density * I * (a / np.sqrt(R**2 + a**2) + b / np.sqrt(R**2 + b**2))


def dipole_B(m, obs):
    """Field of a point magnetic dipole m at the origin, evaluated at obs (..., 3)."""
    m = np.asarray(m, dtype=float)
    obs = np.asarray(obs, dtype=float)
    r = np.linalg.norm(obs, axis=-1, keepdims=True)
    rhat = obs / r
    return MU0 / (4.0 * np.pi) * (3.0 * np.sum(m * rhat, axis=-1, keepdims=True) * rhat - m) / r**3
