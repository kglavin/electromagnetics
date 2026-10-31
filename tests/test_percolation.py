import numpy as np

from emlib import percolation as pc


def test_full_lattice_has_unit_conductivity():
    n = 20
    h, v = np.ones((n - 1, n), bool), np.ones((n, n - 1), bool)
    assert pc.spans(h, v)
    assert np.isclose(pc.effective_conductivity(h, v), 1.0)


def test_empty_and_sparse_lattices_carry_no_current():
    n = 30
    h, v = pc.random_bonds(n, 0.2, rng=1)
    assert not pc.spans(h, v)
    assert pc.effective_conductivity(h, v) == 0.0


def test_dense_lattice_spans_and_conducts():
    h, v = pc.random_bonds(40, 0.8, rng=2)
    assert pc.spans(h, v)
    assert 0.3 < pc.effective_conductivity(h, v) < 1.0


def test_kirchhoff_and_backbone():
    n = 40
    h, v = pc.random_bonds(n, 0.6, rng=3)
    V, Ih, Iv, I = pc.solve_lattice(h, v)
    net = np.zeros((n, n))
    net[:-1, :] += Ih
    net[1:, :] -= Ih
    net[:, :-1] += Iv
    net[:, 1:] -= Iv
    assert np.abs(net[1:-1, :]).max() < 1e-10
    assert np.isclose(Ih[-1, :].sum(), I)
    bh, bv = pc.backbone(Ih, Iv)
    assert np.all(h[bh]) and np.all(v[bv])  # only present bonds carry current
    assert bh.sum() + bv.sum() < h.sum() + v.sum()  # dangling bonds exist at p = 0.6


def test_conductivity_rises_through_threshold():
    n, ps = 40, [0.4, 0.5, 0.6, 0.7]
    mean = [np.mean([pc.effective_conductivity(*pc.random_bonds(n, p, rng=s)) for s in range(20)]) for p in ps]
    assert mean[0] < 0.02 and mean[1] < mean[2] < mean[3]
    # effective-medium estimate 2 (p - 1/2) is close well above threshold
    assert abs(mean[3] - 2 * (0.7 - 0.5)) < 0.08
