import numpy as np
import pytest

from emlib import dielectrics as di
from emlib import fields, grids, relax
from emlib.fields import EPS0


def test_uniform_polarization_has_no_volume_bound_charge():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 21)
    Pol = np.zeros_like(P); Pol[..., 2] = 2e-6
    assert np.allclose(di.bound_volume_charge(Pol, *grids.spacing(x, y, z)), 0.0)


def test_linear_polarization_gives_constant_bound_charge():
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 21)
    Pol = np.zeros_like(P); Pol[..., 0] = 3e-6 * P[..., 0]       # P = 3e-6 x  ->  rho_b = -3e-6
    assert np.allclose(di.bound_volume_charge(Pol, *grids.spacing(x, y, z)), -3e-6)


def test_slab_surface_charges_cancel():
    Pz = 1e-6
    Pol = np.array([0, 0, Pz])
    top = di.bound_surface_charge(Pol, [0, 0, 1])
    bottom = di.bound_surface_charge(Pol, [0, 0, -1])
    assert top == pytest.approx(Pz) and bottom == pytest.approx(-Pz)


def test_dipole_pair_matches_ideal_dipole_potential_far_away():
    p = np.array([0, 0, 1e-12])
    src, q = di.dipole_sources([[0, 0, 0]], [p], delta=1e-3)
    r = np.array([0.3, 0.0, 0.4])
    V = fields.point_charges_V(src, q, r[None])[0]
    V_ideal = (p @ r) / (4 * np.pi * EPS0 * np.linalg.norm(r) ** 3)
    assert V == pytest.approx(V_ideal, rel=1e-4)


def sphere_patches(R, N):
    """Fibonacci-spiral patches on a sphere: centres (N, 3), unit normals (N, 3), equal areas."""
    k = np.arange(N) + 0.5
    theta = np.arccos(1 - 2 * k / N)
    phi = np.pi * (1 + 5**0.5) * k
    n = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    return R * n, n, 4 * np.pi * R**2 / N


def test_uniformly_polarized_sphere_interior_field_from_bound_surface_charge():
    # sigma_b = P . n = P cos(theta). The field inside is uniform, E = -P / (3 eps0).
    R, Pz = 1.0, 1e-6
    centres, normals, dA = sphere_patches(R, 20000)
    sigma_b = di.bound_surface_charge(np.array([0.0, 0.0, Pz]), [0, 0, 1]) * normals[:, 2]  # P cos(theta)
    q = sigma_b * dA
    assert q.sum() == pytest.approx(0.0, abs=1e-12 * np.abs(q).sum())
    obs = np.array([[0, 0, 0], [0.3, 0.1, -0.2], [-0.4, 0.3, 0.3]])
    E = fields.point_charges_E(centres, q, obs)
    assert np.allclose(E[:, 2], -Pz / (3 * EPS0), rtol=1e-2)
    assert np.all(np.abs(E[:, :2]) < 1e-2 * Pz / (3 * EPS0))


def test_dipole_lattice_field_at_a_lattice_site_differs_from_macroscopic_field():
    # The microscopic field of a cubic dipole lattice vanishes at a lattice site (Lorentz), so it is
    # not the macroscopic -P/(3 eps0). This guards against using dipole sums as if they were macroscopic.
    R, Pz, n = 1.0, 1e-6, 25
    x, y, z, G = grids.grid3d((-R, R), (-R, R), (-R, R), n)
    h = grids.spacing(x, y, z)[0]
    inside = grids.magnitude(G) <= R
    cells = G[inside]
    moments = np.zeros_like(cells); moments[:, 2] = Pz * h**3
    src, q = di.dipole_sources(cells, moments, delta=0.1 * h)
    site = G[n // 2 + 2, n // 2 - 1, n // 2 + 1]            # a lattice site, but not the dipole there itself
    keep = ~np.all(np.isclose(src, site + 0.05 * h * np.array([0, 0, 1])), axis=1) & ~np.all(np.isclose(src, site - 0.05 * h * np.array([0, 0, 1])), axis=1)
    E = fields.point_charges_E(src[keep], q[keep], site)
    assert abs(E[2]) < 0.1 * Pz / (3 * EPS0)


def test_dielectric_solver_reduces_to_poisson_for_unit_permittivity():
    n = 41
    rho = np.full((n, n), 1e-9)
    V0 = np.zeros((n, n)); fixed = np.zeros((n, n), dtype=bool)
    V_a = relax.solve_sparse(V0, fixed, rho=rho, h=1 / (n - 1))
    V_b = di.solve_dielectric(V0, fixed, np.ones((n, n)), rho_f=rho, h=1 / (n - 1))
    assert np.allclose(V_a, V_b)


def _slab_capacitor(n, eps_r, fill):
    """Plates at x=0 (V=0) and x=1 (V=1); dielectric on nodes with x < fill; sides fixed to the exact 1D solution.

    With node permittivities and harmonic-mean face coefficients the discrete interface sits halfway
    between the last dielectric node and the first vacuum node, so the exact solution uses that position.
    """
    x = np.linspace(0, 1, n)
    slab = x < fill
    k = np.argmax(~slab)
    x_int = 0.5 * (x[k - 1] + x[k])
    D_over_eps0 = 1.0 / (x_int / eps_r + (1 - x_int))
    exact = np.where(x < x_int, D_over_eps0 * x / eps_r, D_over_eps0 * (x_int / eps_r + (x - x_int)))
    V0 = np.zeros((n, n)); V0[:, 0] = V0[:, -1] = exact; V0[-1, :] = 1.0
    fixed = np.zeros((n, n), dtype=bool)
    er = np.where(slab[:, None], eps_r, 1.0) * np.ones((1, n))
    return x, exact, V0, fixed, er, x_int


def test_slab_capacitor_matches_series_formula_and_D_is_continuous():
    n, eps_r, fill = 81, 4.0, 0.5
    x, exact, V0, fixed, er, x_int = _slab_capacitor(n, eps_r, fill)
    h = x[1] - x[0]
    V = di.solve_dielectric(V0, fixed, er, h=h)
    assert np.allclose(V, exact[:, None], atol=1e-10)
    E = -np.gradient(V[:, n // 2], h)
    D = EPS0 * er[:, n // 2] * E
    k = int(round(x_int / h))
    inner_slab, inner_gap = slice(3, k - 3), slice(k + 4, n - 3)
    assert np.std(D[inner_slab]) / np.mean(D) < 1e-9 and np.std(D[inner_gap]) / np.mean(D) < 1e-9
    assert np.mean(D[inner_slab]) == pytest.approx(np.mean(D[inner_gap]), rel=1e-9)
    assert np.mean(E[inner_gap]) / np.mean(E[inner_slab]) == pytest.approx(eps_r, rel=1e-9)


def test_dielectric_sphere_in_uniform_field_interior():
    # Exact: E_in = 3 / (eps_r + 2) E0, uniform. Outer boundary fixed to the exact exterior potential.
    eps_r, R, E0, n = 3.0, 1.0, 1.0, 41
    L = 2.5 * R
    x, y, z, G = grids.grid3d((-L, L), (-L, L), (-L, L), n)
    h = grids.spacing(x, y, z)[0]
    r = grids.magnitude(G)
    k = (eps_r - 1) / (eps_r + 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        V_ext = -E0 * G[..., 2] + k * R**3 * E0 * G[..., 2] / r**3
    V0 = np.where(r > R, V_ext, 0.0)
    fixed = np.zeros(G.shape[:3], dtype=bool)
    er = np.where(r <= R, eps_r, 1.0)
    V = di.solve_dielectric(V0, fixed, er, h=h)
    E = -fields.gradient(V, h, h, h)
    deep = r < 0.6 * R
    Ez = E[..., 2][deep]
    assert np.mean(Ez) == pytest.approx(3 / (eps_r + 2) * E0, rel=0.03)
    assert np.std(Ez) < 0.03 * E0


def test_energy_gives_series_capacitance_for_slab_capacitor():
    n, eps_r, fill = 81, 4.0, 0.5
    x, exact, V0, fixed, er, x_int = _slab_capacitor(n, eps_r, fill)
    h = x[1] - x[0]
    V = di.solve_dielectric(V0, fixed, er, h=h)
    C = 2 * di.energy(V, er, h)                     # per unit depth, plates of width 1 and separation 1
    C_exact = 1 / (x_int / (eps_r * EPS0) + (1 - x_int) / EPS0)
    assert C == pytest.approx(C_exact, rel=1e-9)


def test_energy_reduces_to_vacuum_field_energy_between_plates():
    n = 41
    x = np.linspace(0, 1, n)
    V = np.broadcast_to(x[:, None], (n, n))
    U = di.energy(V, np.ones((n, n)), x[1] - x[0])
    assert U == pytest.approx(0.5 * EPS0 * 1.0, rel=1e-12)   # (1/2) eps0 E^2 * area with E = 1
