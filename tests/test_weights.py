import numpy as np
import pytest

import groupunit as gu


def test_variance_components_recover_simulation_values():
    rng = np.random.default_rng(4)
    n = gu.lognormal_sizes(200, 20, rng=rng)
    X, groups, Y = gu.simulate(n, p=1, q=2, beta=1.0, tau2=0.5, sigma2_w=3.0, rng=rng)
    vc = gu.variance_components(X[:, 0], groups, Y)
    assert vc.sigma2_w == pytest.approx(3.0, rel=0.05)
    assert vc.tau2 == pytest.approx(0.5, rel=0.3)


def test_variance_components_refuse_saturated_annotations():
    groups = np.repeat(np.arange(4), 5)
    with pytest.raises(ValueError):
        gu.variance_components(np.arange(20.0), groups, np.eye(4))


def test_effective_groups_bounds():
    n = np.array([700, 200, 50, 20, 10, 10, 5], float)
    tau2, sigma2_w = 0.2, 5.0
    kish = gu.effective_groups(n)
    model = gu.effective_groups(n, n, tau2, sigma2_w)
    best = gu.max_effective_groups(n, tau2, sigma2_w)
    optimal = gu.effective_groups(gu.optimal_weights(n, tau2, sigma2_w), n, tau2, sigma2_w)
    assert model <= kish
    assert model <= best <= len(n)
    assert optimal == pytest.approx(best)


def test_optimal_weights_limits():
    n = np.array([10, 100, 1000], float)
    np.testing.assert_allclose(gu.optimal_weights(n, 1e6, 1.0), 1 / 3, atol=1e-4)
    np.testing.assert_allclose(gu.optimal_weights(n, 1e-9, 1.0), n / n.sum(), atol=1e-4)
