"""Per-group pairing terms (Result 5d), the design-based size check and the refitted
PLS lens null (Result 2b), each against direct computation."""
import itertools

import numpy as np
import pytest

import groupunit as gu
from groupunit.diagnostics import _first_pls_loading


def _atypical_design(M=12, big=300, small=10, factor=4.0, q=4, seed=15):
    rng = np.random.default_rng(seed)
    n = np.array([big] + [small] * (M - 1))
    Y = rng.standard_normal((M, q))
    Y[0] = Y[1:].mean(0) + factor * (Y[0] - Y[1:].mean(0))
    return n, Y


# ---------- pairing terms ----------

@pytest.mark.parametrize("weights, components", [
    (None, (None, None)),
    ("equal", (0.3, 1.0)),
    ("exchangeable", (0.3, 1.0)),
    (None, (0.05, 2.0)),
])
def test_pairing_terms_sum_to_the_excess_of_the_ratio(weights, components):
    n, Y = _atypical_design()
    tau2, sigma2_w = components
    w = {None: None, "equal": np.ones(len(n)),
         "exchangeable": None if tau2 is None else gu.exchangeable_weights(n, tau2, sigma2_w)}[weights]
    pt = gu.pairing_terms(Y, n, weights=w, tau2=tau2, sigma2_w=sigma2_w)
    assert pt.ratio == pytest.approx(gu.pairing_ratio(Y, n, weights=w, tau2=tau2, sigma2_w=sigma2_w))
    assert pt.terms.sum() == pytest.approx(pt.ratio - 1)
    assert pt.profile_share.sum() == pytest.approx(1)
    assert pt.variance_ratio.mean() == pytest.approx(1)


def test_pairing_ratio_matches_brute_force_over_draws_and_all_permutations():
    """ϱ = E‖ℓ‖² / mean over all M! reassignments, from simulated group means and explicit loadings."""
    rng = np.random.default_rng(3)
    n = np.array([120, 60, 30, 15, 8, 4])
    Y = rng.standard_normal((6, 3))
    Y[0] *= 3
    tau2, sigma2_w = 0.2, 1.5
    w = n.astype(float)
    v = tau2 + sigma2_w / n
    tbar = rng.standard_normal((20000, 6)) * np.sqrt(v)

    def mean_sq_loading(Yp):
        Yc = gu.wcentre(Yp, w)
        t = tbar - (tbar @ w)[:, None] / w.sum()
        return ((t * w) @ Yc).__pow__(2).sum(1).mean()

    observed = mean_sq_loading(Y)
    permuted = np.mean([mean_sq_loading(Y[list(perm)]) for perm in itertools.permutations(range(6))])
    pt = gu.pairing_terms(Y, n, tau2=tau2, sigma2_w=sigma2_w)
    assert observed / permuted == pytest.approx(pt.ratio, rel=0.03)
    assert pt.ratio > 1.5


def test_pairing_terms_point_at_the_atypical_large_group():
    n, Y = _atypical_design()
    pt = gu.pairing_terms(Y, n, tau2=0.3, sigma2_w=1.0)
    assert pt.largest == 0
    assert pt.terms[0] > 0.5 * (pt.ratio - 1)
    # the same group on the smallest size makes size weights conservative
    pt_small = gu.pairing_terms(Y[::-1], n, tau2=0.3, sigma2_w=1.0)
    assert pt_small.ratio < 1


def test_pairing_terms_approximation_when_no_group_dominates_the_weight():
    rng = np.random.default_rng(4)
    M = 60
    n = rng.integers(20, 80, M)
    Y = rng.standard_normal((M, 3))
    Y[5] *= 4
    pt = gu.pairing_terms(Y, n, tau2=0.1, sigma2_w=1.0)
    approx = pt.profile_share * (pt.variance_ratio - 1)
    assert np.abs(pt.terms - approx).max() < 0.1 * np.abs(pt.terms).max()
    assert approx.sum() == pytest.approx(pt.ratio - 1, rel=0.1, abs=0.01)


def test_pairing_terms_refuse_identical_profiles():
    with pytest.raises(ValueError, match="undefined"):
        gu.pairing_terms(np.ones((5, 2)), np.arange(1, 6))


# ---------- size check ----------

def test_size_check_detects_a_liberal_design_and_its_remedy():
    """Size weights with the atypical profile on the largest group reject far too often;
    exchangeable weights do not (compare Figure 7)."""
    n, Y = _atypical_design()
    tau2, sigma2_w = 0.3, 1.0

    def pls_size(X, g, Yt, rng):
        return gu.pls_test(X, g, Yt, n_components=1, n_perm=99, rng=rng, weights="size").p_value[0]

    def pls_exch(X, g, Yt, rng):
        return gu.pls_test(X, g, Yt, n_components=1, n_perm=99, rng=rng).p_value[0]

    liberal = gu.size_check(pls_size, n, Y, tau2, sigma2_w, p=3, n_sim=100, rng=1)
    fixed = gu.size_check(pls_exch, n, Y, tau2, sigma2_w, p=3, n_sim=100, rng=1)
    assert liberal.rate > 0.2
    assert fixed.rate < 0.13
    assert liberal.p_values.shape == (100,)
    assert liberal.se == pytest.approx(np.sqrt(liberal.rate * (1 - liberal.rate) / 100))


def test_size_check_is_near_nominal_for_equal_groups():
    rng = np.random.default_rng(5)
    n = np.full(12, 20)
    Y = rng.standard_normal((12, 3))

    def corr(X, g, Yt, rng):
        return gu.correlation_test(X[:, 0], g, Yt, n_perm=99, rng=rng).p_fwer

    check = gu.size_check(corr, n, Y, tau2=0.5, sigma2_w=1.0, n_sim=100, rng=2)
    assert check.rate < 0.12


def test_size_check_validates_inputs():
    with pytest.raises(ValueError, match="rows"):
        gu.size_check(lambda *a: 1.0, [5, 5, 5], np.zeros((4, 2)), 0.1, 1.0, n_sim=1)
    with pytest.raises(ValueError, match="non-negative"):
        gu.size_check(lambda *a: 1.0, [5, 5, 5], np.zeros((3, 2)), 0.0, 0.0, n_sim=1)


# ---------- refitted PLS lens null ----------

@pytest.mark.parametrize("custom_weights", [False, True])
def test_first_pls_loading_matches_pls_on_individuals(custom_weights):
    rng = np.random.default_rng(6)
    M, p, q = 8, 5, 4
    n = rng.integers(5, 40, M)
    Y = rng.standard_normal((M, q))
    Tbar = rng.standard_normal((M, p))
    # individuals with exactly these group means
    groups = np.repeat(np.arange(M), n)
    noise = rng.standard_normal((len(groups), p))
    noise -= gu.group_means(noise, groups, M)[groups]
    X = Tbar[groups] + noise
    w = rng.uniform(0.5, 2, M) if custom_weights else n.astype(float)
    fit = gu.pls(X, groups, Y, n_components=1, weights=w if custom_weights else None)
    WY = w[:, None] * gu.wcentre(Y, w)
    direct = _first_pls_loading(Tbar[None], WY, w)[0]
    assert abs(gu.congruence(direct, fit.y_loadings[:, 0])) == pytest.approx(1, abs=1e-10)


def test_refitted_lens_null_approaches_the_leading_lens_direction_as_p_grows():
    """Result 2b: the first PLS loading of p unrelated features tends to the lens's leading eigenvector."""
    rng = np.random.default_rng(7)
    n = gu.lognormal_sizes(15, 40, rng=rng)
    Y = rng.standard_normal((15, 6))
    lead = gu.lens(Y, n).eigenvectors[:, 0]
    med = {p: np.median(np.abs(gu.lens_null(Y, n, n_draws=400, rng=8, p=p) @ lead)) for p in (1, 10, 400)}
    single = np.median(np.abs(gu.lens_null(Y, n, n_draws=4000, rng=9) @ lead))
    assert med[1] < med[10] < med[400]
    assert med[400] > 0.95
    assert med[1] == pytest.approx(single, abs=0.06)       # one feature: the single-score null


def test_lens_congruence_null_passes_p():
    rng = np.random.default_rng(10)
    n = gu.lognormal_sizes(10, 30, rng=rng)
    Y = rng.standard_normal((10, 4))
    loading = gu.lens(Y, n).eigenvectors[:, 0]
    refit = gu.lens_congruence_null(loading, Y, n, n_draws=300, rng=1, p=50)
    single = gu.lens_congruence_null(loading, Y, n, n_draws=300, rng=1)
    assert refit.shape == single.shape == (300,)
    assert np.median(refit) > np.median(single)
    with pytest.raises(ValueError, match="positive integer"):
        gu.lens_null(Y, n, n_draws=5, p=0)
