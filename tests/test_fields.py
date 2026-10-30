import numpy as np
import pytest

from emlib import fields, grids


def test_point_charge_E_matches_coulomb():
    q = 2e-9
    E = fields.point_charges_E([[0, 0, 0]], [q], [1.0, 0.0, 0.0])
    assert E == pytest.approx([fields.K * q, 0.0, 0.0])


def test_point_charge_V_matches_coulomb():
    q = -3e-9
    V = fields.point_charges_V([[0, 0, 0]], [q], [[0.0, 2.0, 0.0]])
    assert V == pytest.approx([fields.K * q / 2.0])


def test_superposition_two_equal_charges_cancel_at_midpoint():
    E = fields.point_charges_E([[-1, 0, 0], [1, 0, 0]], [1e-9, 1e-9], [0.0, 0.0, 0.0])
    assert np.allclose(E, 0.0)


def test_contribution_at_source_position_is_zero_not_nan():
    E = fields.point_charges_E([[0, 0, 0]], [1e-9], [0.0, 0.0, 0.0])
    assert np.all(np.isfinite(E)) and np.allclose(E, 0.0)


def test_chunking_gives_same_result():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(100, 3))
    q = rng.normal(size=100) * 1e-9
    obs = rng.normal(size=(7, 3)) * 5
    assert np.allclose(
        fields.point_charges_E(src, q, obs, chunk=7), fields.point_charges_E(src, q, obs, chunk=1000)
    )


def test_E_equals_minus_grad_V_for_point_charge():
    x, y, z, P = grids.grid3d((1, 3), (1, 3), (1, 3), 41)
    h = grids.spacing(x, y, z)
    V = fields.point_charges_V([[0, 0, 0]], [1e-9], P)
    E_num = -fields.gradient(V, *h)
    E_exact = fields.point_charges_E([[0, 0, 0]], [1e-9], P)
    interior = (slice(2, -2),) * 3
    rel = np.linalg.norm(E_num[interior] - E_exact[interior], axis=-1) / np.linalg.norm(E_exact[interior], axis=-1)
    assert rel.max() < 2e-3


def test_divergence_of_radial_field_is_three():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 11)
    div = fields.divergence(P, *grids.spacing(x, y, z))
    assert np.allclose(div, 3.0)


def test_curl_of_rotation_field_is_two_z():
    x, y, P = grids.grid2d((-1, 1), (-1, 1), 11)
    F = np.stack([-P[..., 1], P[..., 0], np.zeros_like(P[..., 0])], axis=-1)
    c = fields.curl(F, *grids.spacing(x, y))
    assert np.allclose(c[..., 2], 2.0) and np.allclose(c[..., :2], 0.0)


def test_curl_of_gradient_is_zero():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 21)
    h = grids.spacing(x, y, z)
    f = P[..., 0] ** 2 * P[..., 1] + np.sin(P[..., 2])
    c = fields.curl(fields.gradient(f, *h), *h)
    interior = (slice(1, -1),) * 3
    assert np.abs(c[interior]).max() < 1e-10


def test_divergence_of_curl_is_zero():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 21)
    h = grids.spacing(x, y, z)
    F = np.stack([P[..., 1] * P[..., 2], P[..., 0] ** 2, np.cos(P[..., 0] * P[..., 1])], axis=-1)
    d = fields.divergence(fields.curl(F, *h), *h)
    interior = (slice(1, -1),) * 3
    assert np.abs(d[interior]).max() < 1e-10


def test_line_integral_of_gradient_field_is_path_independent():
    grad = lambda p: np.stack([2 * p[:, 0], 2 * p[:, 1], np.zeros(len(p))], axis=-1)  # grad of x^2 + y^2
    t = np.linspace(0, 1, 2001)
    straight = np.stack([t, t, 0 * t], axis=-1)
    arc = np.stack([np.sin(np.pi / 2 * t), 1 - np.cos(np.pi / 2 * t), 0 * t], axis=-1)
    assert fields.line_integral(grad, straight) == pytest.approx(2.0, rel=1e-6)
    assert fields.line_integral(grad, arc) == pytest.approx(2.0, rel=1e-6)


def test_line_integral_of_rotation_field_around_unit_circle_is_two_pi():
    rot = lambda p: np.stack([-p[:, 1], p[:, 0], np.zeros(len(p))], axis=-1)
    t = np.linspace(0, 2 * np.pi, 4001)
    circle = np.stack([np.cos(t), np.sin(t), 0 * t], axis=-1)
    assert fields.line_integral(rot, circle) == pytest.approx(2 * np.pi, rel=1e-6)
