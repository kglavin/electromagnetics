import numpy as np

from emlib import biot_savart as bs
from emlib import fields, grids
from emlib import induction as ind


def test_disk_flux_of_uniform_field():
    B = lambda p: np.tile([0.0, 0.3, 0.4], (len(p), 1))
    assert np.isclose(ind.disk_flux(B, (1, 2, 3), (0, 0, 1), 0.5), 0.4 * np.pi * 0.25)
    n = np.array([0.0, 0.6, 0.8])
    assert np.isclose(ind.disk_flux(B, (0, 0, 0), n, 0.5), 0.5 * np.pi * 0.25)


def test_neumann_matches_coaxial_closed_form_and_is_symmetric():
    a, b, d = 0.1, 0.15, 0.08
    l1, l2 = bs.loop(a, 400), bs.loop(b, 400, centre=(0, 0, d))
    M = ind.neumann_M(l1, l2)
    assert np.isclose(M, ind.coaxial_loops_M(a, b, d), rtol=1e-4)
    assert np.isclose(ind.neumann_M(l2, l1), M, rtol=1e-12)


def test_flux_linkage_equals_M_times_current():
    a, b, d, I = 0.1, 0.15, 0.08, 2.0
    mid, dl = bs.segments(bs.loop(a, 400))
    flux = ind.disk_flux(lambda p: bs.wire_B(mid, dl, I, p), (0, 0, d), (0, 0, 1), b, nr=80, nphi=160)
    assert np.isclose(flux / I, ind.coaxial_loops_M(a, b, d), rtol=2e-3)
    # the same flux from the line integral of A around the second loop
    m2, d2 = bs.segments(bs.loop(b, 400, centre=(0, 0, d)))
    circ = np.sum(bs.wire_A(mid, dl, I, m2) * d2)
    assert np.isclose(circ / I, ind.coaxial_loops_M(a, b, d), rtol=1e-4)


def test_curl_of_wire_A_is_wire_B():
    mid, dl = bs.segments(bs.loop(0.5, 200))
    x, y, z, P = grids.grid3d((0.8, 1.4), (0.8, 1.4), (-0.3, 0.3), 25)
    h = grids.spacing(x, y, z)
    curlA = fields.curl(bs.wire_A(mid, dl, 1.0, P), *h)[3:-3, 3:-3, 3:-3]
    B = bs.wire_B(mid, dl, 1.0, P)[3:-3, 3:-3, 3:-3]
    assert np.abs(curlA - B).max() < 2e-3 * np.abs(B).max()


def test_emf_of_a_sinusoidal_flux():
    t = np.linspace(0, 0.1, 20001)
    w, B0, R = 2 * np.pi * 50, 0.1, 0.05
    flux = B0 * np.sin(w * t) * np.pi * R**2
    emf = ind.emf_from_flux(t, flux)
    assert np.allclose(emf[1:-1], -B0 * w * np.cos(w * t[1:-1]) * np.pi * R**2, atol=1e-6 * B0 * w * np.pi * R**2 * 1e3)


def test_central_flux_of_a_long_solenoid():
    # flux through a disk well inside the windings, at the centre, against mu0 n I pi r^2 with the
    # finite-length factor; the disk stays clear of the wire, whose own field is singular
    N, R, L, I = 400, 0.02, 0.8, 1.0
    mid, dl = bs.segments(bs.helix(R, L, N, 40), closed=False)
    r = 0.6 * R
    flux = ind.disk_flux(lambda p: bs.wire_B(mid, dl, I, p), (0, 0, L / (2 * N)), (0, 0, 1), r, nr=20, nphi=40)
    expect = bs.solenoid_axis_B(R, L, N / L, I, 0.0) * np.pi * r**2
    assert np.isclose(flux, expect, rtol=5e-3)
    assert np.isclose(ind.solenoid_L(N, R, L), bs.MU0 * N**2 * np.pi * R**2 / L)
