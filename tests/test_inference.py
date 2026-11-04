import numpy as np
import pytest

import groupunit as gu
from groupunit.inference import draw_permutations, p_value


def test_permutations_stay_within_strata():
    strata = np.array([0, 0, 0, 1, 1, 2])
    for perm in draw_permutations(6, 50, rng=0, strata=strata):
        np.testing.assert_array_equal(strata[perm], strata)


def test_p_value_counts_ties():
    assert p_value(1.0, np.ones(99)) == 1.0
    assert p_value(2.0, np.ones(99)) == pytest.approx(0.01)


def test_correlation_test_finds_real_relation(design):
    X, groups, Y, _ = design
    result = gu.correlation_test(X[:, :2], groups, Y, n_perm=999, rng=0)
    assert result.r.shape == (2, 4)
    assert (result.p_fwer >= result.p_value - 1e-12).all()
    assert result.p_value[:, 0].max() < 0.05


def test_pls_and_cca_tests_detect_relation(design):
    X, groups, Y, _ = design
    pls = gu.pls_test(X, groups, Y, n_perm=499, rng=0)
    assert pls.p_value[0] < 0.05
    assert (np.diff(pls.p_value) >= 0).all()
    cca = gu.cca_test(X, groups, Y[:, :2], n_perm=499, rng=0)
    assert cca.p_value[0] < 0.05


@pytest.mark.parametrize("q", [3, 12])
def test_cca_test_holds_level_whatever_the_number_of_annotations(q):
    """Fewer and more annotations than features (p = 6); a reduced SVD in the deflation broke q > p."""
    rng = np.random.default_rng(13)
    p_values = []
    for _ in range(60):
        X, groups, Y = gu.simulate([20] * 20, p=6, q=q, beta=0.0, tau2=0.5, rng=rng)
        p_values.append(gu.cca_test(X, groups, Y, n_components=2, n_perm=99, rng=rng).p_value[0])
    assert np.mean(np.array(p_values) < 0.05) < 0.15


def test_exchangeable_weights_restore_level_when_large_group_is_atypical():
    """A large group with an extreme profile makes size-weighted tests liberal; 1/sd weights do not."""
    rng = np.random.default_rng(14)
    n = np.array([300] + [10] * 11)
    tau2, sigma2_w = 0.05, 1.0
    rejections = {"size": 0, "exchangeable": 0}
    for _ in range(80):
        Y = rng.standard_normal((12, 4))
        Y[0] *= 4
        X, groups, Y = gu.simulate(n, p=4, Y=Y, tau2=tau2, sigma2_w=sigma2_w, rng=rng)
        vc = gu.pooled_variance_components(X, groups)
        w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
        rejections["size"] += gu.pls_test(X, groups, Y, 1, n_perm=99, rng=rng).p_value[0] < 0.05
        rejections["exchangeable"] += gu.pls_test(X, groups, Y, 1, n_perm=99, rng=rng, weights=w).p_value[0] < 0.05
    assert rejections["size"] / 80 > 0.3
    assert rejections["exchangeable"] / 80 < 0.12


def test_exchangeable_correlation_test_holds_level_for_fixed_scores():
    rng = np.random.default_rng(17)
    n = np.array([300] + [10] * 11)
    rejections = 0
    for _ in range(80):
        Y = rng.standard_normal((12, 4))
        Y[0] *= 4
        X, groups, Y = gu.simulate(n, p=4, Y=Y, tau2=0.5, sigma2_w=1.0, rng=rng)
        vc = gu.pooled_variance_components(X, groups)
        scores = X @ np.eye(4)[:, :2]                 # chosen in advance, not from the data
        w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
        rejections += gu.correlation_test(scores, groups, Y, n_perm=99, rng=rng, weights=w).p_fwer.min() < 0.05
    assert rejections / 80 < 0.12


def test_pairing_ratio():
    rng = np.random.default_rng(15)
    n = np.array([300] + [10] * 11)
    Y = rng.standard_normal((12, 4))
    Y[0] *= 4
    assert gu.pairing_ratio(Y, n) > 2
    assert gu.pairing_ratio(Y, n, weights=gu.exchangeable_weights(n, 0.05, 1.0), tau2=0.05, sigma2_w=1.0) \
        == pytest.approx(1, abs=0.05)
    Y[0] /= 4
    Y[-1] *= 4
    assert gu.pairing_ratio(Y, n) < 1


def test_cca_test_refuses_saturated_design():
    rng = np.random.default_rng(7)
    X, groups, Y = gu.simulate([10] * 6, p=3, q=5, rng=rng)
    with pytest.raises(ValueError, match="kappa = 1"):
        gu.cca_test(X, groups, Y, n_perm=10)


def test_group_level_test_holds_nominal_rate_where_individual_level_does_not():
    """Under the null with group heterogeneity, permuting groups is calibrated (Section 7.1)."""
    rng = np.random.default_rng(8)
    n = [40] * 12
    p_groups, p_individuals = [], []
    for _ in range(100):
        X, groups, Y = gu.simulate(n, p=1, q=1, beta=0.0, tau2=1.0, sigma2_w=1.0, rng=rng)
        p_groups.append(gu.correlation_test(X[:, 0], groups, Y, n_perm=99, rng=rng).p_value[0, 0])
        # the naive test: reassign annotation values among individuals
        y = Y[groups, 0]
        r = abs(np.corrcoef(X[:, 0], y)[0, 1])
        null = [abs(np.corrcoef(X[:, 0], rng.permutation(y))[0, 1]) for _ in range(99)]
        p_individuals.append(p_value(r, null))
    assert np.mean(np.array(p_groups) < 0.05) < 0.15
    assert np.mean(np.array(p_individuals) < 0.05) > 0.3


def test_generic_permutation_test_repeats_selection(design):
    """Select the best annotation inside the statistic, so the selection is part of the null."""
    X, groups, Y, n = design
    tbar = gu.group_scores(X[:, 0], groups)
    result = gu.permutation_test(lambda Yp: np.max(np.abs(gu.weighted_corr(tbar, Yp, n))), Y, n_perm=499, rng=0)
    assert result.null.shape == (499,)
    assert 0 < result.p_value <= 1
