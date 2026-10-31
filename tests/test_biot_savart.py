import numpy as np

from emlib import biot_savart as bs
from emlib import fields, grids


def test_loop_on_axis_matches_closed_form():
    R, I = 0.3, 2.0
    mid, dl = bs.segments(bs.loop(R, 400))
    z = np.array([0.0, 0.1, 0.3, 1.0, 3.0])
    obs = np.stack([0 * z, 0 * z, z], axis=-1)
    B = bs.wire_B(mid, dl, I, obs)
    assert np.allclose(B[:, 2], bs.loop_axis_B(R, I, z), rtol=1e-5)
    assert np.allclose(B[:, :2], 0.0, atol=1e-12)


def test_loop_direction_follows_right_hand_rule():
    mid, dl = bs.segments(bs.loop(1.0, 100))  # counter-clockwise about +z
    B = bs.wire_B(mid, dl, 1.0, np.array([0.0, 0.0, 0.0]))
    assert B[2] > 0


def test_long_straight_wire_matches_mu0_I_over_2_pi_r():
    I, L = 3.0, 500.0
    mid, dl = bs.segments(bs.straight_wire([0, 0, -L], [0, 0, L], 100000), closed=False)  # segments of 0.01
    r = np.array([0.2, 1.0, 3.0])
    obs = np.stack([r, 0 * r, 0 * r], axis=-1)
    B = bs.wire_B(mid, dl, I, obs)
    assert np.allclose(B[:, 1], bs.straight_wire_B(I, r), rtol=1e-3)  # along +y at points on +x
    assert np.allclose(B[:, [0, 2]], 0.0, atol=1e-12)


def test_finite_wire_on_bisector():
    I, L = 1.0, 0.5
    mid, dl = bs.segments(bs.straight_wire([-L, 0, 0], [L, 0, 0], 4000), closed=False)
    r = np.array([0.1, 0.5, 2.0])
    B = bs.wire_B(mid, dl, I, np.stack([0 * r, r, 0 * r], axis=-1))
    assert np.allclose(np.linalg.norm(B, axis=1), bs.finite_wire_B(I, r, L), rtol=1e-5)


def test_ampere_loop_integral_around_wire():
    I, L = 1.5, 400.0
    mid, dl = bs.segments(bs.straight_wire([0, 0, -L], [0, 0, L], 20000), closed=False)
    t = np.linspace(0, 2 * np.pi, 2001)
    circle = np.stack([0.7 * np.cos(t), 0.7 * np.sin(t), 0 * t], axis=-1)
    circ = fields.line_integral(lambda p: bs.wire_B(mid, dl, I, p), circle)
    assert abs(circ / (bs.MU0 * I) - 1.0) < 1e-4
    far = np.stack([3.0 + 0.5 * np.cos(t), 0.5 * np.sin(t), 0 * t], axis=-1)  # does not enclose the wire
    assert abs(fields.line_integral(lambda p: bs.wire_B(mid, dl, I, p), far)) < 1e-6 * bs.MU0 * I


def test_helmholtz_field_is_uniform_near_centre():
    R, I = 1.0, 1.0
    mid, dl = bs.helmholtz(R, 400)
    z = np.array([0.0, 0.05, 0.1, 0.2])
    B = bs.wire_B(mid, dl, I, np.stack([0 * z, 0 * z, z], axis=-1))[:, 2]
    assert np.isclose(B[0], (0.8) ** 1.5 * bs.MU0 * I / R, rtol=1e-5)
    assert np.all(np.abs(B[1:3] / B[0] - 1.0) < 2e-4)
    assert abs(B[3] / B[0] - 1.0) < 3e-3


def test_solenoid_axis_matches_closed_form_and_mu0_n_I():
    R, L, turns, I = 0.1, 2.0, 200, 1.0
    mid, dl = bs.segments(bs.helix(R, L, turns, 60), closed=False)
    z = np.array([0.0, 0.5, 0.9, 1.0, 1.5])
    B = bs.wire_B(mid, dl, I, np.stack([0 * z, 0 * z, z], axis=-1))[:, 2]
    assert np.allclose(B, bs.solenoid_axis_B(R, L, turns / L, I, z), rtol=2e-3)
    assert abs(B[0] / (bs.MU0 * turns / L * I) - 1.0) < 1e-2


def test_divergence_of_B_vanishes_on_grid():
    mid, dl = bs.segments(bs.loop(0.5, 200))
    x, y, z, P = grids.grid3d((0.8, 1.6), (0.8, 1.6), (-0.4, 0.4), 21)
    h = grids.spacing(x, y, z)
    B = bs.wire_B(mid, dl, 1.0, P)
    div = fields.divergence(B, *h)[2:-2, 2:-2, 2:-2]
    scale = np.linalg.norm(B, axis=-1).mean() / h[0]
    assert np.abs(div).max() < 1e-3 * scale


def test_loop_far_field_is_a_dipole():
    R, I = 0.1, 1.0
    mid, dl = bs.segments(bs.loop(R, 400))
    m = np.array([0.0, 0.0, I * np.pi * R**2])
    obs = np.array([[3.0, 0.0, 0.0], [2.0, 2.0, 1.0], [0.0, 0.0, 4.0]])
    assert np.allclose(bs.wire_B(mid, dl, I, obs), bs.dipole_B(m, obs), rtol=2e-3)
