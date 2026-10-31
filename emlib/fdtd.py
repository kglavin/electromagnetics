"""Finite-difference time-domain (Yee) solvers for Maxwell's equations in 1D and 2D, in SI units.

1D: a plane wave moving along x with E along y and H along z, so the Poynting vector E x H
points along x. Ey lives on integer nodes x_i = i dx, Hz on half nodes x_{i+1/2}, and Hz is
advanced half a time step out of phase with Ey. The update is

    Hz[i+1/2] -= dt / (mu0 dx) (Ey[i+1] - Ey[i])
    Ey[i]     -= dt / (eps dx) (Hz[i+1/2] - Hz[i-1/2])     (plus conduction loss sigma E)

which are Faraday's law dBz/dt = -dEy/dx and Ampere-Maxwell eps dEy/dt = -dHz/dx - sigma Ey.

2D TM (fields independent of z): Ez on nodes, Hx on (i, j+1/2), Hy on (i+1/2, j). An optional
absorbing frame of graded electric and matched magnetic conductivity (sigma_m = sigma mu0 / eps0)
soaks up outgoing waves; it is a simple matched lossy layer, not a perfectly matched layer, so
some reflection remains at oblique incidence.

The Courant number S = c dt / dx must satisfy S <= 1 in 1D and S <= 1/sqrt(2) in 2D.
"""

import numpy as np

from .biot_savart import MU0
from .fields import EPS0

C0 = 1.0 / np.sqrt(MU0 * EPS0)
ETA0 = np.sqrt(MU0 / EPS0)


def courant_limit(ndim):
    return 1.0 / np.sqrt(ndim)


def _check_courant(courant, ndim):
    if not 0 < courant <= courant_limit(ndim) + 1e-12:
        raise ValueError(f"Courant number {courant} violates the {ndim}D stability limit {courant_limit(ndim):.4f}")


def gaussian(t, t0, width):
    return np.exp(-(((t - t0) / width) ** 2))


def fdtd_1d(n_cells, dx, n_steps, courant=0.5, eps_r=1.0, sigma=0.0, source=None, src_index=None,
            boundary="mur", probes=(), snapshot_every=0, Ey0=None, Hz0=None):
    """Run a 1D Yee simulation.

    n_cells + 1 E nodes span x = 0 .. n_cells dx. eps_r and sigma are scalars or arrays on the E nodes.
    source(t) is added to Ey[src_index] every step (a soft source; it radiates both ways).
    boundary: "mur" (first-order absorbing) or "pec" (Ey = 0 at both ends).
    probes: E-node indices whose Ey is recorded every step, together with the co-located Hz
    (the average of the two neighbouring half nodes, at the same time level).
    Ey0, Hz0: optional initial fields (Hz0 on the n_cells half nodes, at time -dt/2).

    Returns a dict with x, x_h, dt, t (the times at which Ey is stored), Ey, Hz (final),
    probe_E and probe_H (n_steps + 1, n_probes), and snapshots (list of Ey copies).
    """
    _check_courant(courant, 1)
    dt = courant * dx / C0
    n = n_cells + 1
    x = np.arange(n) * dx
    x_h = (np.arange(n_cells) + 0.5) * dx
    eps = EPS0 * np.broadcast_to(np.asarray(eps_r, dtype=float), (n,)).copy()
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (n,)).copy()
    ca = (1 - sig * dt / (2 * eps)) / (1 + sig * dt / (2 * eps))
    cb = (dt / (eps * dx)) / (1 + sig * dt / (2 * eps))
    ch = dt / (MU0 * dx)

    Ey = np.zeros(n) if Ey0 is None else np.array(Ey0, dtype=float)
    Hz = np.zeros(n_cells) if Hz0 is None else np.array(Hz0, dtype=float)
    probes = list(probes)
    probe_E = np.zeros((n_steps + 1, len(probes)))
    probe_H = np.zeros((n_steps + 1, len(probes)))
    snapshots = []

    def record(k):
        # Ey is at time k dt and Hz at (k - 1/2) dt. Hz at k dt is the average of Hz now and
        # after the next update; it is co-located with Ey by averaging the two half nodes.
        h_next = Hz - ch * (Ey[1:] - Ey[:-1])
        h_now = 0.5 * (Hz + h_next)
        for p, i in enumerate(probes):
            probe_E[k, p] = Ey[i]
            probe_H[k, p] = 0.5 * (h_now[max(i - 1, 0)] + h_now[min(i, n_cells - 1)])

    c_left = 1.0 / np.sqrt(MU0 * eps[0])
    c_right = 1.0 / np.sqrt(MU0 * eps[-1])
    kl = (c_left * dt - dx) / (c_left * dt + dx)
    kr = (c_right * dt - dx) / (c_right * dt + dx)

    record(0)
    for k in range(n_steps):
        Hz -= ch * (Ey[1:] - Ey[:-1])
        e0, e1, em, em1 = Ey[0], Ey[1], Ey[-1], Ey[-2]
        Ey[1:-1] = ca[1:-1] * Ey[1:-1] - cb[1:-1] * (Hz[1:] - Hz[:-1])
        if source is not None:
            Ey[src_index] += source((k + 1) * dt)
        if boundary == "mur":
            Ey[0] = e1 + kl * (Ey[1] - e0)
            Ey[-1] = em1 + kr * (Ey[-2] - em)
        else:
            Ey[0] = Ey[-1] = 0.0
        if probes:
            record(k + 1)
        if snapshot_every and (k + 1) % snapshot_every == 0:
            snapshots.append(Ey.copy())
    t = np.arange(n_steps + 1) * dt
    return dict(x=x, x_h=x_h, dt=dt, t=t, Ey=Ey, Hz=Hz, probe_E=probe_E, probe_H=probe_H, snapshots=snapshots)


def energy_1d(Ey, Hz, dx, eps_r=1.0):
    """Electromagnetic energy per unit area, sum of (eps E^2 + mu0 H^2) / 2 over the grid."""
    eps = EPS0 * np.broadcast_to(np.asarray(eps_r, dtype=float), Ey.shape)
    return 0.5 * dx * (np.sum(eps * Ey**2) + MU0 * np.sum(Hz**2))


def absorbing_profile(n, n_layer, sigma_max, order=3):
    """Electric conductivity on n nodes rising from 0 to sigma_max over n_layer cells at each end."""
    s = np.zeros(n)
    if n_layer > 0:
        ramp = (np.arange(n_layer, 0, -1) / n_layer) ** order * sigma_max
        s[:n_layer] = ramp
        s[-n_layer:] = ramp[::-1]
    return s


def fdtd_2d_tm(nx, ny, dx, n_steps, courant=0.5, eps_r=1.0, source=None, src_index=None,
               n_layer=20, layer_strength=1.0, probes=(), snapshot_every=0, pec=None, callback=None):
    """Run a 2D TM Yee simulation (Ez, Hx, Hy) on nx by ny E nodes with spacing dx.

    eps_r: scalar or (nx, ny) array. source(t) is added to Ez[src_index] each step.
    n_layer cells of graded matched loss line every edge (n_layer = 0 gives PEC walls);
    layer_strength scales the conductivity (1 is a good default for a few-percent reflection).
    probes: list of (i, j) recorded every step.
    pec: optional (nx, ny) bool mask of perfect conductor; Ez is held at zero there, which makes
    walls, slits and metal obstacles.
    callback: optional callback(k, Ez, Hx, Hy) called after every step k = 1 .. n_steps, with Ez at
    time k dt and Hx, Hy at (k - 1/2) dt. Use it to record fluxes or energies; do not modify the arrays.

    Returns a dict with x, y, dt, t, Ez, Hx, Hy (final), probe_E (n_steps + 1, n_probes),
    snapshots (list of Ez copies) and sigma (the electric conductivity used).
    """
    _check_courant(courant, 2)
    dt = courant * dx / C0
    eps = EPS0 * np.broadcast_to(np.asarray(eps_r, dtype=float), (nx, ny)).copy()

    # graded conductivity: optimal-ish peak for a polynomial profile of order 3
    sig_max = layer_strength * 0.8 * 4.0 / (ETA0 * dx)
    sx = absorbing_profile(nx, n_layer, sig_max)
    sy = absorbing_profile(ny, n_layer, sig_max)
    sigma = np.maximum(sx[:, None], sy[None, :])
    sig_m = sigma * MU0 / EPS0

    ca = (1 - sigma * dt / (2 * eps)) / (1 + sigma * dt / (2 * eps))
    cb = (dt / (eps * dx)) / (1 + sigma * dt / (2 * eps))
    sm_x = 0.5 * (sig_m[:, :-1] + sig_m[:, 1:])  # at Hx positions (i, j+1/2)
    sm_y = 0.5 * (sig_m[:-1, :] + sig_m[1:, :])  # at Hy positions (i+1/2, j)
    da_x = (1 - sm_x * dt / (2 * MU0)) / (1 + sm_x * dt / (2 * MU0))
    db_x = (dt / (MU0 * dx)) / (1 + sm_x * dt / (2 * MU0))
    da_y = (1 - sm_y * dt / (2 * MU0)) / (1 + sm_y * dt / (2 * MU0))
    db_y = (dt / (MU0 * dx)) / (1 + sm_y * dt / (2 * MU0))

    Ez = np.zeros((nx, ny))
    Hx = np.zeros((nx, ny - 1))
    Hy = np.zeros((nx - 1, ny))
    probes = list(probes)
    probe_E = np.zeros((n_steps + 1, len(probes)))
    snapshots = []
    for k in range(n_steps):
        Hx = da_x * Hx - db_x * (Ez[:, 1:] - Ez[:, :-1])
        Hy = da_y * Hy + db_y * (Ez[1:, :] - Ez[:-1, :])
        curl = np.zeros((nx, ny))
        curl[1:-1, 1:-1] = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) - (Hx[1:-1, 1:] - Hx[1:-1, :-1])
        Ez = ca * Ez + cb * curl
        Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0.0
        if pec is not None:
            Ez[pec] = 0.0
        if source is not None:
            Ez[src_index] += source((k + 1) * dt)
        if callback is not None:
            callback(k + 1, Ez, Hx, Hy)
        for p, (i, j) in enumerate(probes):
            probe_E[k + 1, p] = Ez[i, j]
        if snapshot_every and (k + 1) % snapshot_every == 0:
            snapshots.append(Ez.copy())
    x = np.arange(nx) * dx
    y = np.arange(ny) * dx
    t = np.arange(n_steps + 1) * dt
    return dict(x=x, y=y, dt=dt, t=t, Ez=Ez, Hx=Hx, Hy=Hy, probe_E=probe_E, snapshots=snapshots, sigma=sigma)


def poynting_2d_tm(Ez, Hx, Hy):
    """Poynting vector (Sx, Sy) = (-Ez Hy, Ez Hx) on interior E nodes, with H averaged to the nodes.

    The H fields are half a time step from Ez; for a time-averaged flux this offset does not matter.
    """
    hx = 0.5 * (Hx[1:-1, 1:] + Hx[1:-1, :-1])
    hy = 0.5 * (Hy[1:, 1:-1] + Hy[:-1, 1:-1])
    ez = Ez[1:-1, 1:-1]
    return -ez * hy, ez * hx
