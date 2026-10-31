import numpy as np

from emlib import integrators as it


Q, M, B0 = 1.6e-19, 1.67e-27, 0.01  # a proton in 10 mT


def test_boris_circle_radius_and_period():
    v0 = 1e5
    omega = it.cyclotron_frequency(Q, B0, M)
    T = 2 * np.pi / omega
    n = 400
    x, v = it.boris([0, 0, 0], [v0, 0, 0], Q, M, [0, 0, 0], [0, 0, B0], T / n, 3 * n)
    r = it.larmor_radius(M, v0, Q, B0)
    centre = x[:n].mean(axis=0)  # mean over one full period
    assert centre[1] < 0  # positive charge, v along +x, B along +z: it turns clockwise, centre below the start
    radii = np.linalg.norm(x - centre, axis=1)
    assert np.allclose(radii, r, rtol=1e-3)
    assert np.allclose(x[n], x[0], atol=1e-3 * r)  # back at the start after one period


def test_boris_conserves_speed_exactly_in_magnetic_field():
    x, v = it.boris([0, 0, 0], [2e5, 1e5, 3e4], Q, M, [0, 0, 0], [0, 0, B0], 1e-9, 20000)
    speed = np.linalg.norm(v, axis=1)
    assert np.abs(speed / speed[0] - 1.0).max() < 1e-12


def test_e_cross_b_drift():
    E0 = 100.0
    omega = it.cyclotron_frequency(Q, B0, M)
    T = 2 * np.pi / omega
    x, v = it.boris([0, 0, 0], [0, 0, 0], Q, M, [E0, 0, 0], [0, 0, B0], T / 200, 200 * 10)
    drift = (x[-1] - x[0]) / (10 * T)
    assert np.allclose(drift, [0.0, -E0 / B0, 0.0], rtol=1e-3, atol=1e-3 * E0 / B0)


def test_rk4_agrees_with_boris_for_small_steps():
    omega = it.cyclotron_frequency(Q, B0, M)
    T = 2 * np.pi / omega
    args = ([0, 0, 0], [1e5, 0, 2e4], Q, M, [10.0, 0, 0], [0, 0, B0], T / 500, 1000)
    xb, vb = it.boris(*args)
    xr, vr = it.rk4(*args)
    scale = np.linalg.norm(xb[-1] - xb[0])
    assert np.linalg.norm(xb[-1] - xr[-1]) < 1e-4 * scale


def test_rk4_energy_drifts_boris_does_not():
    omega = it.cyclotron_frequency(Q, B0, M)
    T = 2 * np.pi / omega
    dt, n = T / 12, 12 * 200  # a coarse step for 200 periods
    _, vb = it.boris([0, 0, 0], [1e5, 0, 0], Q, M, [0, 0, 0], [0, 0, B0], dt, n)
    _, vr = it.rk4([0, 0, 0], [1e5, 0, 0], Q, M, [0, 0, 0], [0, 0, B0], dt, n)
    eb = np.sum(vb**2, axis=1) / np.sum(vb[0] ** 2)
    er = np.sum(vr**2, axis=1) / np.sum(vr[0] ** 2)
    assert abs(eb[-1] - 1.0) < 1e-10
    assert abs(er[-1] - 1.0) > 1e-3
