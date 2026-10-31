import numpy as np
import pytest

from emlib import fdtd as fd


def _pulse_run(eps_split=None, n=2000, S=0.5, steps=5000, width_steps=60, probes=(900, 1300, 1700), boundary="mur"):
    dx = 1e-3
    dt = S * dx / fd.C0
    w = width_steps * dt
    t0 = 4 * w
    eps = np.ones(n + 1)
    if eps_split is not None:
        eps[eps_split[0]:] = eps_split[1]
    r = fd.fdtd_1d(n, dx, steps, courant=S, eps_r=eps, source=lambda t: fd.gaussian(t, t0, w), src_index=500,
                   probes=probes, boundary=boundary)
    return r, t0, dx


def test_courant_limits():
    with pytest.raises(ValueError):
        fd.fdtd_1d(10, 1e-3, 1, courant=1.01)
    with pytest.raises(ValueError):
        fd.fdtd_2d_tm(10, 10, 1e-3, 1, courant=0.75)


def test_pulse_moves_at_c():
    r, t0, dx = _pulse_run()
    i = np.argmax(r["probe_E"][:, 0])
    assert abs(400 * dx / (r["t"][i] - t0) / fd.C0 - 1.0) < 2e-3


def test_reflection_and_transmission_at_dielectric_step():
    r, t0, dx = _pulse_run(eps_split=(1400, 4.0))
    t, a, b = r["t"], r["probe_E"][:, 1], r["probe_E"][:, 2]
    inc = a.max()
    refl = a[t > t[np.argmax(a)] + 100 * dx / fd.C0].min()  # the incident pulse (30 cells wide) has passed
    assert abs(refl / inc - (1 - 2) / (1 + 2)) < 0.01  # (n1 - n2) / (n1 + n2)
    assert abs(b.max() / inc - 2 / 3) < 0.01  # 2 n1 / (n1 + n2)
    # the transmitted pulse moves at c / 2: compare arrival at 1700 with the interface crossing
    t_cross = t0 + 900 * dx / fd.C0
    assert abs((t[np.argmax(b)] - t_cross) / (300 * dx / (fd.C0 / 2)) - 1.0) < 1e-2


def test_mur_boundary_absorbs_and_pec_reflects():
    r, _, _ = _pulse_run(steps=8000)
    assert np.abs(r["Ey"]).max() < 2e-3
    r, _, _ = _pulse_run(steps=8000, boundary="pec")
    assert np.abs(r["Ey"]).max() > 0.5


def test_energy_is_conserved_in_a_closed_cavity():
    n, dx, S = 400, 1e-3, 0.5
    x = np.arange(n + 1) * dx
    Ey0 = np.exp(-(((x - 0.2) / 0.02) ** 2))
    Ey0[0] = Ey0[-1] = 0.0
    energies = []
    for steps in (0, 500, 1500, 3000):
        r = fd.fdtd_1d(n, dx, steps, courant=S, boundary="pec", Ey0=Ey0)
        energies.append(fd.energy_1d(r["Ey"], r["Hz"], dx))
    energies = np.array(energies[1:])  # the staggered H makes step 0 a special case
    assert np.ptp(energies) / energies.mean() < 1e-2


def test_poynting_flux_carries_the_pulse_energy():
    # a pulse launched with matched E and H moves one way only; the time-integrated E H at a
    # probe equals the pulse energy
    n, dx, S = 1200, 1e-3, 0.5
    dt = S * dx / fd.C0
    x = np.arange(n + 1) * dx
    xh = (np.arange(n) + 0.5) * dx
    f = lambda s: np.exp(-(((s - 0.3) / 0.03) ** 2))
    Ey0 = f(x)
    Hz0 = f(xh + fd.C0 * dt / 2) / fd.ETA0  # H half a step earlier, so shifted forward by c dt / 2
    U0 = fd.energy_1d(Ey0, Hz0, dx)
    r = fd.fdtd_1d(n, dx, 4000, courant=S, Ey0=Ey0, Hz0=Hz0, probes=[800])
    flux = np.sum(r["probe_E"][:, 0] * r["probe_H"][:, 0]) * dt
    assert abs(flux / U0 - 1.0) < 1e-2
    assert np.abs(r["probe_E"][:, 0]).max() > 0.99  # it all went right


def test_2d_point_source_speed_isotropy_and_decay():
    n, dx, S = 301, 1e-3, 0.5
    dt = S * dx / fd.C0
    w = 20 * dt
    t0 = 4 * w
    r = fd.fdtd_2d_tm(n, n, dx, 900, courant=S, source=lambda t: fd.gaussian(t, t0, w), src_index=(150, 150),
                      probes=[(190, 150), (230, 150), (207, 207)])
    t, pe = r["t"], r["probe_E"]
    ta, tb, td = (t[np.argmax(pe[:, k])] for k in range(3))
    assert abs(40 * dx / (tb - ta) / fd.C0 - 1.0) < 2e-2  # speed from the delay between two radii
    assert abs(td - tb) < 3 * dt  # the diagonal probe at nearly the same radius arrives together
    assert abs(pe[:, 0].max() / pe[:, 1].max() - np.sqrt(2)) < 0.03  # cylindrical spreading: 1 / sqrt(r)


def test_2d_absorbing_frame():
    n, dx, S = 301, 1e-3, 0.5
    dt = S * dx / fd.C0
    w = 20 * dt
    src = lambda t: fd.gaussian(t, 4 * w, w)
    early = fd.fdtd_2d_tm(n, n, dx, 900, courant=S, source=src, src_index=(150, 150), probes=[(190, 150)])
    late = fd.fdtd_2d_tm(n, n, dx, 1400, courant=S, source=src, src_index=(150, 150))
    assert np.abs(late["Ez"]).max() < 5e-3 * early["probe_E"][:, 0].max()


def test_2d_pec_wall_blocks_and_callback_sees_every_step():
    n, dx, S = 201, 1e-3, 0.5
    dt = S * dx / fd.C0
    w = 20 * dt
    src = lambda t: fd.gaussian(t, 4 * w, w)
    wall = np.zeros((n, n), dtype=bool)
    wall[120:124, :] = True  # a solid conducting wall across the grid
    seen = []
    r = fd.fdtd_2d_tm(n, n, dx, 600, courant=S, source=src, src_index=(80, 100), probes=[(160, 100), (60, 100)],
                      pec=wall, callback=lambda k, Ez, Hx, Hy: seen.append((k, float(np.abs(Ez[wall]).max()))))
    open_run = fd.fdtd_2d_tm(n, n, dx, 600, courant=S, source=src, src_index=(80, 100), probes=[(160, 100)])
    assert [k for k, _ in seen] == list(range(1, 601))
    assert max(e for _, e in seen) == 0.0  # Ez stays zero inside the conductor
    assert np.abs(r["probe_E"][:, 0]).max() == 0.0  # nothing gets through a solid wall
    assert np.abs(open_run["probe_E"][:, 0]).max() > 1e-3
    # without pec or callback the result is unchanged
    base = fd.fdtd_2d_tm(n, n, dx, 200, courant=S, source=src, src_index=(80, 100))
    again = fd.fdtd_2d_tm(n, n, dx, 200, courant=S, source=src, src_index=(80, 100), pec=np.zeros((n, n), bool), callback=lambda *a: None)
    assert np.array_equal(base["Ez"], again["Ez"])
