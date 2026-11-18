"""Edge cases from the package review: undefined statistics, one annotation column, masks, singletons."""
import numpy as np
import pytest

import groupunit as gu
from groupunit.inference import max_t_p_value, min_p_value, p_value, permutation_weights


def test_undefined_statistic_is_not_significant():
    assert np.isnan(p_value(np.nan, np.ones(99)))
    # an undefined permuted statistic counts as extreme: 1 + 99 of 99
    assert p_value(2.0, np.full(99, np.nan)) == 1.0
    observed, null = np.array([3.0, np.nan]), np.zeros((99, 2))
    np.testing.assert_allclose(p_value(observed, null), [0.01, np.nan])
    assert np.isnan(max_t_p_value(observed, null)[1])
    # a NaN cell leaves min-p of the other cells as if it were not there
    np.testing.assert_allclose(min_p_value(observed, null), [min_p_value(observed[:1], null[:, :1])[0], np.nan])


def test_similarity_comparison_needs_four_groups():
    rng = np.random.default_rng(0)
    S = np.corrcoef(rng.standard_normal((3, 10)))
    with pytest.raises(ValueError, match="four groups"):
        gu.compare_similarity(S, S, n_perm=9, rng=0)


def test_a_single_annotation_vector_is_one_column():
    rng = np.random.default_rng(1)
    n = np.array([3, 4, 5, 6, 7, 8], float)
    t, y = rng.standard_normal(6), rng.standard_normal(6)
    Y = y[:, None]
    np.testing.assert_allclose(gu.contributions(t, y, n), gu.contributions(t, Y, n))
    assert gu.contributions(t, y, n).shape == (6, 1)
    np.testing.assert_allclose(gu.group_shares(t, y, n).share, gu.group_shares(t, Y, n).share)
    np.testing.assert_allclose(gu.leave_out(t, y, n, 2).loading, gu.leave_out(t, Y, n, 2).loading)
    np.testing.assert_allclose(gu.lens(y, n).matrix, gu.lens(Y, n).matrix)
    np.testing.assert_allclose(gu.lens_null(y, n, 50, rng=0), gu.lens_null(Y, n, 50, rng=0))
    X, groups, _ = gu.simulate(n.astype(int), 4, Y=Y, rng=2)
    np.testing.assert_allclose(gu.pls(X, groups, y).strength, gu.pls(X, groups, Y).strength)
    np.testing.assert_allclose(gu.cca(X, groups, y).strength, gu.cca(X, groups, Y).strength)
    np.testing.assert_allclose(gu.pls_test(X, groups, y, n_perm=49, rng=0).p_value,
                               gu.pls_test(X, groups, Y, n_perm=49, rng=0).p_value)
    assert gu.describe(groups, y).q == 1
    gu.size_check(lambda X, g, Y, rng: 1.0, n.astype(int), y, 1.0, 1.0, n_sim=2, rng=0)


def test_leave_out_accepts_masks_and_matches_direct_computation():
    rng = np.random.default_rng(3)
    n, Y, t = rng.integers(5, 50, 6).astype(float), rng.standard_normal((6, 2)), rng.standard_normal(6)
    mask = np.zeros(6, bool)
    mask[[1, 4]] = True
    by_mask = gu.leave_out(t, Y, n, mask)
    np.testing.assert_array_equal(by_mask.dropped, [1, 4])
    np.testing.assert_allclose(by_mask.loading, gu.leave_out(t, Y, n, [4, 1, 1]).loading)
    keep = ~mask
    direct = gu.contributions(t[keep], Y[keep], n[keep]).sum(0)
    np.testing.assert_allclose(by_mask.loading, direct, atol=1e-12)
    with pytest.raises(ValueError):
        gu.leave_out(t, Y, n, np.ones(6, bool))


def test_variance_components_refuse_singleton_groups():
    with pytest.raises(ValueError, match="single individual"):
        gu.variance_components(np.arange(4.0), np.arange(4))


def test_pooled_variance_components_average_the_columns():
    rng = np.random.default_rng(5)
    n = gu.lognormal_sizes(12, 8, rng=rng)
    X, groups, _ = gu.simulate(n, p=5, q=1, rng=rng)
    parts = [gu.variance_components(col, groups) for col in X.T]
    pooled = gu.pooled_variance_components(X, groups)
    assert pooled.sigma2_w == pytest.approx(np.mean([c.sigma2_w for c in parts]))
    assert pooled.tau2 == pytest.approx(np.mean([c.tau2 for c in parts]))


def test_exchangeable_weights_sum_to_one_with_equal_sizes():
    rng = np.random.default_rng(6)
    X, groups, Y = gu.simulate(np.full(8, 10), 3, q=2, rng=rng)
    w = permutation_weights("exchangeable", X, groups, np.bincount(groups))
    assert w.sum() == pytest.approx(1)
