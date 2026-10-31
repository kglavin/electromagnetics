import numpy as np

from emlib import grids
from emlib import magnetization as mg
from emlib.biot_savart import MU0


def test_uniform_magnetization_has_no_volume_current_and_K_equals_M():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 21)
    h = grids.spacing(x, y, z)
    M = np.zeros(P.shape)
    M[..., 2] = 2.0
    J = mg.bound_volume_current(M, *h)
    assert np.abs(J).max() == 0.0
    # cylinder magnetized along its axis: K = M x r_hat has magnitude M and runs along phi_hat
    phi = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    normal = np.stack([np.cos(phi), np.sin(phi), 0 * phi], axis=-1)
    K = mg.bound_surface_current(np.array([0.0, 0.0, 2.0]), normal)
    assert np.allclose(np.linalg.norm(K, axis=1), 2.0)
    assert np.allclose(K, 2.0 * np.stack([-np.sin(phi), np.cos(phi), 0 * phi], axis=-1))


def test_nonuniform_magnetization_curl_matches_formula():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 41)
    h = grids.spacing(x, y, z)
    M = np.zeros(P.shape)
    M[..., 2] = P[..., 0] ** 2  # M = x^2 z_hat, curl M = (0, -2x, 0)
    J = mg.bound_volume_current(M, *h)
    inner = (slice(2, -2),) * 3
    assert np.allclose(J[inner][..., 1], -2 * P[inner][..., 0], atol=1e-10)
    assert np.allclose(J[inner][..., [0, 2]], 0.0, atol=1e-12)


def test_H_and_B_relations():
    B = np.array([[0.0, 0.0, 1e-3]])
    M = np.array([[0.0, 0.0, 100.0]])
    H = mg.H_field(B, M)
    assert np.allclose(H, B / MU0 - M)
    assert np.allclose(mg.B_from_H(H, chi_m=0.0), MU0 * H)
