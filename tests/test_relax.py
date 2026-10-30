import numpy as np
import pytest

from emlib import fields, grids, relax
from emlib.fields import EPS0, K


def test_box_flux_of_point_charge_is_q_over_eps0():
    q = 1e-9
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 81)
    h = grids.spacing(x, y, z)
    E = fields.point_charges_E([[0.05, -0.1, 0.02]], [q], P)
    flux = fields.box_flux(E, h, (0, 0, 0), (80, 80, 80))
    assert flux == pytest.approx(q / EPS0, rel=2e-3)


def test_box_flux_is_zero_when_charge_is_outside():
    q = 1e-9
    x, y, z, P = grids.grid3d((-1, 1), (-1, 1), (-1, 1), 61)
    h = grids.spacing(x, y, z)
    E = fields.point_charges_E([[3.0, 0.0, 0.0]], [q], P)
    flux = fields.box_flux(E, h, (0, 0, 0), (60, 60, 60))
    assert abs(flux) < 1e-3 * q / EPS0


def _plates(n=41):
    """Plates at x=0 (V=0) and x=1 (V=1), sides fixed to the linear solution."""
    x = np.linspace(0, 1, n)
    V0 = np.zeros((n, n))
    V0[:, 0] = V0[:, -1] = x
    V0[-1, :] = 1.0
    fixed = np.zeros((n, n), dtype=bool)
    fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
    return x, V0, fixed


@pytest.mark.parametrize("method", ["jacobi", "gauss-seidel"])
def test_relaxation_recovers_linear_potential_between_plates(method):
    x, V0, fixed = _plates()
    V, hist = relax.solve_relax(V0, fixed, method=method, tol=1e-9)
    assert np.allclose(V, x[:, None], atol=1e-6)
    assert hist[-1] < 1e-9


def test_sor_converges_in_fewer_sweeps_than_gauss_seidel():
    x, V0, fixed = _plates(61)
    _, gs = relax.solve_relax(V0, fixed, tol=1e-8)
    _, sor = relax.solve_relax(V0, fixed, omega=relax.sor_omega(61, 61), tol=1e-8)
    assert len(sor) < len(gs) / 3


def test_sparse_solver_matches_relaxation_for_uniform_charge():
    n = 41
    h = 1.0 / (n - 1)
    rho = np.full((n, n), 1e-9)
    V0 = np.zeros((n, n))
    fixed = np.zeros((n, n), dtype=bool)
    V_sp = relax.solve_sparse(V0, fixed, rho=rho, h=h)
    V_gs, _ = relax.solve_relax(V0, fixed, rho=rho, h=h, omega=relax.sor_omega(n, n), tol=1e-12)
    assert np.allclose(V_sp, V_gs, atol=1e-6 * np.abs(V_sp).max())
    assert V_sp.max() > 0  # positive charge gives a positive potential bump


def test_sparse_poisson_matches_analytic_one_dimensional_solution():
    # rho uniform, V=0 at x=0 and x=1, sides fixed to the exact parabola -> V = rho x (1 - x) / (2 eps0)
    n, rho0 = 81, 1e-9
    x = np.linspace(0, 1, n)
    exact = rho0 * x * (1 - x) / (2 * EPS0)
    V0 = np.zeros((n, n))
    V0[:, 0] = V0[:, -1] = exact
    fixed = np.zeros((n, n), dtype=bool)
    V = relax.solve_sparse(V0, fixed, rho=np.full((n, n), rho0), h=x[1] - x[0])
    assert np.allclose(V, exact[:, None], rtol=1e-6, atol=1e-9 * exact.max())


def test_image_line_charge_against_relaxed_potential():
    # 2D problem: a line charge lambda at (d, 0) beside a grounded plane x = 0.
    # Image solution: V = (lambda / 2 pi eps0) ln(r_minus / r_plus). Outer boundary fixed to it.
    lam, d = 1e-9, 0.5
    n = 121
    x = np.linspace(0, 2, n)
    y = np.linspace(-1, 1, n)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    r_plus = np.hypot(X - d, Y)
    r_minus = np.hypot(X + d, Y)
    with np.errstate(divide="ignore"):
        exact = lam / (2 * np.pi * EPS0) * np.log(r_minus / r_plus)
    V0 = np.where(np.isfinite(exact), exact, 0.0)
    V0[1:-1, 1:-1] = 0.0
    fixed = np.zeros((n, n), dtype=bool)
    rho = np.zeros((n, n))
    i0, j0 = np.argmin(abs(x - d)), np.argmin(abs(y))
    rho[i0, j0] = lam / h**2  # discrete delta: rho h^2 = lambda per unit depth
    V = relax.solve_sparse(V0, fixed, rho=rho, h=h)
    far = r_plus > 0.15
    rel = np.abs(V - exact)[far] / np.abs(exact)[far].max()
    assert rel.max() < 2e-2
    assert np.allclose(V[0, :], 0.0)


def test_capacitance_from_field_energy_matches_parallel_plates():
    n = 81
    x, V0, fixed = _plates(n)
    h = x[1] - x[0]
    V = relax.solve_sparse(V0, fixed)
    U = relax.field_energy(V, h)
    C = 2 * U / 1.0**2  # per unit depth, with plate separation 1 and width 1
    assert C == pytest.approx(EPS0 * 1.0 / 1.0, rel=1e-6)
