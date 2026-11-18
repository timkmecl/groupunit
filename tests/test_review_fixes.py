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


def test_test_options_after_rng_are_keyword_only():
    rng = np.random.default_rng(7)
    X, groups, Y = gu.simulate(np.full(6, 5), 3, q=2, rng=rng)
    with pytest.raises(TypeError):
        gu.pls_test(X, groups, Y, 1, 9, 0, "size")
    with pytest.raises(TypeError):
        gu.cca_test(X, groups, Y, 1, 9, 0, "size")
    with pytest.raises(TypeError):
        gu.correlation_test(X[:, 0], groups, Y, 9, 0, "size")
    # the options themselves still work by name, in any order
    a = gu.cca_test(X, groups, Y, 1, 19, 0, strata=None, weights="size").p_value
    b = gu.cca_test(X, groups, Y, 1, 19, 0, weights="size", strata=None).p_value
    np.testing.assert_array_equal(a, b)


def test_weights_share_one_vocabulary():
    rng = np.random.default_rng(8)
    n = np.array([3, 5, 8, 4, 6, 7])
    X, groups, Y = gu.simulate(n, 4, q=2, rng=rng)
    for fit in (gu.pls, gu.cca):
        np.testing.assert_allclose(fit(X, groups, Y, weights="size").strength, fit(X, groups, Y).strength)
        np.testing.assert_allclose(fit(X, groups, Y, weights=n).strength, fit(X, groups, Y).strength)
    np.testing.assert_allclose(gu.lens(Y, n, weights="size").matrix, gu.lens(Y, n).matrix)
    assert gu.pairing_ratio(Y, n, weights="size") == pytest.approx(gu.pairing_ratio(Y, n))
    # None means size weights in the tests as well
    np.testing.assert_allclose(gu.correlation_test(X[:, 0], groups, Y, 49, 0, weights=None).p_value,
                               gu.correlation_test(X[:, 0], groups, Y, 49, 0, weights="size").p_value)
    for bad in ("exchangeable", np.ones(5), -np.ones(6), np.zeros(6)):
        with pytest.raises(ValueError):
            gu.pls(X, groups, Y, weights=bad)


def test_public_names_are_listed():
    assert all(hasattr(gu, name) for name in gu.__all__)
    assert "group_weights" in gu.__all__
    from groupunit.methods import Components
    assert isinstance(gu.pls_test(np.random.default_rng(9).standard_normal((30, 3)), np.repeat(np.arange(6), 5),
                                  np.random.default_rng(10).standard_normal((6, 2)), n_perm=9, rng=0).fit, Components)


def test_diagnostic_options_are_keyword_only():
    n, Y = np.array([4, 6, 9, 5]), np.random.default_rng(11).standard_normal((4, 2))
    for call in (lambda: gu.pairing_ratio(Y, n, None, 0.1, 1.0), lambda: gu.pairing_terms(Y, n, None),
                 lambda: gu.lens(Y, n, None), lambda: gu.lens_null(Y, n, 10, None)):
        with pytest.raises(TypeError):
            call()
    assert gu.pairing_ratio(Y, n, tau2=0.1, sigma2_w=1.0) == pytest.approx(
        gu.pairing_ratio(Y, n, sigma2_w=1.0, tau2=0.1))


def test_variance_components_fields_follow_the_function_order():
    from dataclasses import fields
    from groupunit.weights import VarianceComponents
    assert [f.name for f in fields(VarianceComponents)] == ["tau2", "sigma2_w"]
