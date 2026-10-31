import numpy as np

from emlib import circuits as ci
from emlib.integrators import rk4_ode


def test_voltage_divider_and_series_parallel():
    # 10 V source between node 1 and ground, 1 k and 3 k in series: node 2 sits at 7.5 V
    V, Ivs = ci.mna(3, resistors=[(1, 2, 1e3), (2, 0, 3e3)], voltage_sources=[(1, 0, 10.0)])
    assert np.isclose(V[2], 7.5)
    assert np.isclose(Ivs[0], -10.0 / 4e3)  # current inside the source from + to -: the source delivers 2.5 mA
    # two 2 k in parallel fed by a 1 mA current source into node 1: V = 1 mA x 1 k
    V, _ = ci.mna(2, resistors=[(1, 0, 2e3), (1, 0, 2e3)], current_sources=[(0, 1, 1e-3)])
    assert np.isclose(V[1], 1.0)


def test_wheatstone_bridge_balanced_and_unbalanced():
    def bridge(R1, R2, R3, R4, Rg=100.0, Vs=10.0):
        # nodes: 0 ground (bottom), 1 top, 2 left middle, 3 right middle
        res = [(1, 2, R1), (2, 0, R2), (1, 3, R3), (3, 0, R4), (2, 3, Rg)]
        V, _ = ci.mna(4, resistors=res, voltage_sources=[(1, 0, Vs)])
        return (V[2] - V[3]) / Rg

    assert abs(bridge(100, 200, 300, 600)) < 1e-15
    # unbalanced: compare with the Thevenin closed form
    R1, R2, R3, R4, Rg, Vs = 100.0, 200.0, 300.0, 400.0, 100.0, 10.0
    Vth = Vs * (R2 / (R1 + R2) - R4 / (R3 + R4))
    Rth = R1 * R2 / (R1 + R2) + R3 * R4 / (R3 + R4)
    assert np.isclose(bridge(R1, R2, R3, R4, Rg, Vs), Vth / (Rth + Rg))


def test_solve_network_matches_mna_and_ignores_floating_clusters():
    edges = np.array([[0, 1], [1, 2], [1, 3], [3, 2], [4, 5]])  # nodes 4 and 5 float
    G = np.array([1.0, 0.5, 2.0, 1.0, 3.0])
    fixed = np.array([True, False, True, False, False, False])
    V, I = ci.solve_network(6, edges, G, fixed, np.array([1.0, 0, 0, 0, 0, 0]))
    Vm, _ = ci.mna(4, resistors=[(0, 1, 1.0), (1, 2, 2.0), (1, 3, 0.5), (3, 2, 1.0)], voltage_sources=[(0, 2, 1.0)], ground=2)
    assert np.allclose(V[:4], Vm)
    assert np.allclose(V[4:], 0.0) and np.allclose(I[4], 0.0)
    assert np.isclose(I[0], I[1] + I[2])  # Kirchhoff at node 1


def test_rc_lowpass_phasor():
    R, C = 1e3, 1e-6
    for w in (10.0, 1.0 / (R * C), 1e5):
        V, _ = ci.mna(3, resistors=[(1, 2, R)], capacitors=[(2, 0, C)], voltage_sources=[(1, 0, 1.0)], omega=w)
        H = V[2]
        assert np.isclose(abs(H), 1.0 / np.sqrt(1.0 + (w * R * C) ** 2))
        assert np.isclose(np.angle(H), -np.arctan(w * R * C))


def test_series_rlc_resonance_current_peaks_at_omega0():
    R, L, C = 10.0, 1e-3, 1e-6
    w = np.linspace(0.5, 1.5, 2001) * ci.resonance_omega(L, C)
    I = [abs(ci.mna(4, resistors=[(1, 2, R)], inductors=[(2, 3, L)], capacitors=[(3, 0, C)], voltage_sources=[(1, 0, 1.0)], omega=wk)[1][0]) for wk in w]
    k = int(np.argmax(I))
    assert abs(w[k] / ci.resonance_omega(L, C) - 1.0) < 1e-3
    assert np.isclose(max(I), 1.0 / R, rtol=1e-6)


def test_dc_inductor_is_a_short_and_capacitor_is_open():
    V, I = ci.mna(3, resistors=[(1, 2, 5.0)], inductors=[(2, 0, 1e-3)], capacitors=[(1, 0, 1e-6)], voltage_sources=[(1, 0, 10.0)])
    assert np.isclose(V[2], 0.0) and np.isclose(abs(I[0]), 2.0) and np.isclose(I[1], 2.0)


def test_rc_transient_by_rk4_matches_closed_form():
    R, C, V0 = 2e3, 1e-6, 5.0
    t = np.linspace(0.0, 5 * R * C, 501)
    Vc = rk4_ode(lambda tk, v: (V0 - v) / (R * C), 0.0, t)
    assert np.abs(Vc - ci.rc_charging(t, R, C, V0)).max() < 1e-8 * V0


def test_uniform_sheet_and_series_layers():
    nx, ny = 41, 11
    V, Ix, Iy, I = ci.solve_sheet(np.ones((nx, ny)))
    assert np.allclose(V, np.linspace(1, 0, nx)[:, None])
    assert np.isclose(I, ny / (nx - 1)) and np.allclose(Iy, 0.0)
    # two layers in series along x: sigma = 1 then 4, interface at a half-node
    sigma = np.ones((nx, ny))
    sigma[21:, :] = 4.0
    _, Ix, _, I = ci.solve_sheet(sigma)
    R_row = 20.5 / 1.0 + 19.5 / 4.0  # face 20 is the harmonic-mean face: half a cell of each
    assert np.isclose(I, ny / R_row)
    assert np.allclose(Ix.sum(axis=1), I)  # the same current crosses every column of faces


def test_sheet_current_is_conserved_around_a_hole():
    nx, ny = 61, 31
    sigma = np.ones((nx, ny))
    sigma[25:36, 10:21] = 0.0
    V, Ix, Iy, I = ci.solve_sheet(sigma)
    net = np.zeros((nx, ny))
    net[:-1, :] += Ix
    net[1:, :] -= Ix
    net[:, :-1] += Iy
    net[:, 1:] -= Iy
    assert np.abs(net[1:-1, :]).max() < 1e-12
    assert np.allclose(Ix.sum(axis=1), I)
    assert I < ny / (nx - 1)
