"""Design summaries and the diagnostics of Section 6, including the numbers quoted for the paper's toy data."""
from itertools import permutations

import numpy as np
import pytest

import groupunit as gu
from conftest import TOY_N, TOY_T, TOY_Y


def test_toy_numbers_of_figures_2_and_3():
    shares = gu.group_shares(TOY_T, TOY_Y, TOY_N)
    assert shares.abs_share[2] + shares.abs_share[3] == pytest.approx(0.94, abs=0.01)
    assert shares.m_eff == pytest.approx(2.2, abs=0.05)
    assert gu.effective_number(TOY_N) == pytest.approx(4.6, abs=0.05)
    dropped = gu.leave_out(TOY_T, TOY_Y, TOY_N, drop=2)
    assert dropped.congruence == pytest.approx(0.58, abs=0.005)
    assert dropped.norm_ratio == pytest.approx(0.29, abs=0.005)
    L = gu.lens(TOY_Y, TOY_N)
    assert L.nu == pytest.approx(2.4, abs=0.05)
    assert L.cumulative_share[1] == pytest.approx(0.90, abs=0.01)
    null = gu.lens_congruence_null(shares.loading, TOY_Y, TOY_N, rng=0)
    assert np.median(null) == pytest.approx(0.66, abs=0.02)


def test_shares_sum_to_one_and_are_shapley_values():
    rng = np.random.default_rng(3)
    n, Y, t = rng.integers(5, 50, 5).astype(float), rng.standard_normal((5, 3)), rng.standard_normal(5)
    shares = gu.group_shares(t, Y, n)
    assert shares.share.sum() == pytest.approx(1)
    np.testing.assert_allclose(gu.contributions(t, Y, n).sum(0), shares.loading)

    # brute-force Shapley values of f(S) = ‖Σ_{m∈S} u_m‖²
    u = gu.contributions(t, Y, n)
    phi = np.zeros(5)
    orders = list(permutations(range(5)))
    for order in orders:
        running = np.zeros(3)
        for m in order:
            phi[m] += np.sum((running + u[m]) ** 2) - np.sum(running**2)
            running += u[m]
    phi /= len(orders)
    np.testing.assert_allclose(phi / np.sum(shares.loading**2), shares.share, atol=1e-12)


@pytest.mark.parametrize("drop", [2, [2, 3], [0, 5, 7]])
def test_leave_out_equals_recomputation_without_the_groups(drop):
    keep = np.setdiff1d(np.arange(len(TOY_N)), drop)
    direct = gu.contributions(TOY_T[keep], TOY_Y[keep], TOY_N[keep]).sum(0)
    np.testing.assert_allclose(gu.leave_out(TOY_T, TOY_Y, TOY_N, drop).loading, direct, atol=1e-12)


def test_leave_out_reverses_loading_when_share_exceeds_bound():
    """The projection of ℓ₋m on ℓ is ‖ℓ‖(1 − share_m · W/(W − w_m))."""
    shares = gu.group_shares(TOY_T, TOY_Y, TOY_N)
    W = TOY_N.sum()
    for m in range(len(TOY_N)):
        rest = gu.leave_out(TOY_T, TOY_Y, TOY_N, m)
        projection = rest.congruence * rest.norm_ratio
        assert projection == pytest.approx(1 - shares.share[m] * W / (W - TOY_N[m]))


def test_shares_for_several_components_match_one_at_a_time(design):
    X, groups, Y, n = design
    fit = gu.pls(X, groups, Y)
    together = gu.group_shares(fit.group_scores, Y, n)
    for c in range(fit.group_scores.shape[1]):
        alone = gu.group_shares(fit.group_scores[:, c], Y, n)
        np.testing.assert_allclose(together.share[:, c], alone.share)
        assert together.m_eff[c] == pytest.approx(alone.m_eff)
    # PLS annotation loadings are proportional to the group-level loading of the identity
    np.testing.assert_allclose(gu.congruence(together.loading, fit.y_loadings).diagonal(), 1)


def test_participation_ratio_of_product_can_exceed_factors():
    eps = 1e-3
    A, B = np.diag([1, eps]), np.diag([eps, 1])
    assert gu.participation_ratio(A) < 1.01 and gu.participation_ratio(B) < 1.01
    assert gu.participation_ratio(A @ B) == pytest.approx(2)


def test_describe_flags_saturation_and_single_group_annotations():
    groups = np.repeat(np.arange(5), [10, 20, 5, 8, 7])
    Y = np.column_stack([np.eye(5)[:, 0], np.arange(5.0), [1, 1, 2, 3, 5.0], [0, 2, 1, 4, 3.0]])
    summary = gu.describe(groups, Y, annotation_names=["only A", "b", "c", "d"])
    assert summary.r == 4 and summary.kappa == 1
    assert summary.single_group == [0]
    assert "SATURATED" in str(summary) and "only A" in str(summary)


def test_describe_flags_atypical_large_group():
    rng = np.random.default_rng(16)
    sizes = [300] + [10] * 11
    groups = np.repeat(np.arange(12), sizes)
    Y = rng.standard_normal((12, 4))
    Y[0] *= 4
    summary = gu.describe(groups, Y)
    assert summary.pairing_ratio > 1.1 and "liberal" in str(summary)
    Y[0] /= 4
    Y[-1] *= 4
    assert "conservative" in str(gu.describe(groups, Y))


def test_r2_null_mean():
    assert gu.r2_null(4, 21).mean() == pytest.approx(4 / 20)
    with pytest.raises(ValueError):
        gu.r2_null(20, 21)


def test_reduce_annotations_gives_requested_rank(design):
    _, groups, Y, n = design
    scores, explained = gu.reduce_annotations(Y, n, 2)
    assert scores.shape == (len(Y), 2) and gu.annotation_rank(scores, n) == 2
    assert explained[0] >= explained[1]
