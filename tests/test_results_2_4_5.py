"""Results 2b, 4a, 4b, 5b and 5c, and row-weighted CCA, against direct computation."""
import numpy as np
import pytest
from scipy.linalg import orth, subspace_angles

import groupunit as gu


# ---------- Result 2b: the lens is the covariance of the loading ----------

def test_result2b_lens_is_the_covariance_of_random_loadings():
    rng = np.random.default_rng(1)
    n = gu.lognormal_sizes(9, 30, rng=rng).astype(float)
    Y = rng.standard_normal((9, 4))
    tau2, sigma2_w = 0.4, 2.0
    v = tau2 + sigma2_w / n
    tbar = rng.standard_normal((200000, 9)) * np.sqrt(v)
    loadings = tbar @ (n[:, None] * gu.wcentre(Y, n))        # ℓ = Ỹᵀ D t̄, one row per draw
    empirical = loadings.T @ loadings / len(loadings)
    L = gu.lens(Y, n, tau2=tau2, sigma2_w=sigma2_w).matrix
    np.testing.assert_allclose(empirical, L, rtol=0.03, atol=0.03 * np.abs(L).max())


# ---------- Result 4a: the Mantel correlation splits into typicality and pattern ----------

@pytest.mark.parametrize("kind", ["random", "distance"])
def test_result4a_mantel_is_typicality_part_plus_pattern_part(kind):
    rng = np.random.default_rng(2)
    M = 10
    if kind == "random":
        S1, S2 = (A + A.T for A in rng.standard_normal((2, M, M)))
    else:
        F = rng.standard_normal((M, 3)) * rng.uniform(0.5, 3, (M, 1))
        S1 = gu.distance_similarity(F)
        S2 = gu.distance_similarity(F + rng.standard_normal((M, 3)))
    dec = gu.similarity_decomposition(S1, S2)
    iu = np.triu_indices(M, 1)
    assert dec.mantel == pytest.approx(np.corrcoef(S1[iu], S2[iu])[0, 1])
    assert dec.typicality_part + dec.pattern_part == pytest.approx(dec.mantel, abs=1e-12)
    # the parts against direct correlations of the fitted effects and residuals
    d1, d2 = gu.typicality_pattern(S1), gu.typicality_pattern(S2)
    assert dec.r_typicality == pytest.approx(np.corrcoef(d1.typicality, d2.typicality)[0, 1])
    assert dec.r_pattern == pytest.approx(np.corrcoef(d1.pattern[iu], d2.pattern[iu])[0, 1])
    # λ²: the share of variation over pairs due to the fitted additive effects
    fitted = (d1.typicality[:, None] + d1.typicality[None, :])[iu]
    total = ((S1[iu] - S1[iu].mean()) ** 2).sum()
    assert dec.typicality_share[0] == pytest.approx((fitted**2).sum() / total)


def test_result4a_shared_typicality_alone_gives_a_large_mantel_correlation():
    rng = np.random.default_rng(3)
    M = 12
    a = rng.standard_normal(M) * 3
    noise = [rng.standard_normal((M, M)) for _ in range(2)]
    S1, S2 = (a[:, None] + a[None, :] + (E + E.T) / 2 for E in noise)
    dec = gu.similarity_decomposition(S1, S2)
    assert dec.mantel > 0.7
    assert dec.typicality_part > 0.95 * dec.mantel
    assert abs(dec.r_pattern) < 0.3


def test_similarity_decomposition_validates_inputs():
    with pytest.raises(ValueError, match="shape"):
        gu.similarity_decomposition(np.zeros((4, 4)), np.zeros((5, 5)))
    with pytest.raises(ValueError, match="constant"):
        gu.similarity_decomposition(np.ones((4, 4)), np.eye(4))


# ---------- Result 4b: closed form of the pattern for squared distances ----------

def test_result4b_pattern_of_squared_distances_in_closed_form():
    rng = np.random.default_rng(4)
    M = 9
    F = rng.standard_normal((M, 4))
    Fc = F - F.mean(0)
    g = Fc @ Fc.T
    d = np.diag(g)
    expected = 2 * (g + (d[:, None] + d[None, :]) / (M - 2) - M * d.mean() / ((M - 1) * (M - 2)))
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(gu.typicality_pattern(gu.distance_similarity(F)).pattern, expected, atol=1e-12)


# ---------- Results 5b and 5c: when the pairing ratio is 1 ----------

def test_result5b_equal_weights_and_variances_give_ratio_one():
    rng = np.random.default_rng(5)
    for _ in range(5):
        Y = rng.standard_normal((8, 3)) * rng.uniform(0.2, 5, (8, 1))
        assert gu.pairing_ratio(Y, np.full(8, 25)) == pytest.approx(1)


def test_result5c_exchangeable_weights_in_closed_form():
    rng = np.random.default_rng(6)
    M = 14
    n = gu.lognormal_sizes(M, 40, sigma=1.0, rng=rng).astype(float)
    Y = rng.standard_normal((M, 3))
    Y[0] *= 4
    tau2, sigma2_w = 0.2, 1.5
    w = gu.exchangeable_weights(n, tau2, sigma2_w)
    Yc = Y - Y.mean(0)
    shift = ((w @ Y / w.sum() - Y.mean(0)) ** 2).sum()
    kish = w.sum() ** 2 / (w**2).sum()
    expected = (M - 1) * (1 + M * shift / (Yc**2).sum()) / (M - 2 + M / kish)
    assert gu.pairing_ratio(Y, n, weights=w, tau2=tau2, sigma2_w=sigma2_w) == pytest.approx(expected)
    # and every group's term is equally variable
    pt = gu.pairing_terms(Y, n, weights=w, tau2=tau2, sigma2_w=sigma2_w)
    assert pt.variance_ratio.std() < 0.1


# ---------- row-weighted CCA ----------

def _weighted_cca_direct(X, groups, Y, w):
    """Canonical correlations of R^{1/2}-scaled, weighted-centred individual data."""
    n = np.bincount(groups).astype(float)
    r = (w / n)[groups]
    Yi = Y[groups]
    Xc = X - r @ X / r.sum()
    Yc = Yi - r @ Yi / r.sum()
    return np.cos(subspace_angles(orth(np.sqrt(r)[:, None] * Xc), orth(np.sqrt(r)[:, None] * Yc)))


def test_weighted_cca_matches_direct_computation(design):
    X, groups, Y, n = design
    w = np.random.default_rng(7).uniform(0.5, 3, len(n))
    fit = gu.cca(X, groups, Y, weights=w)
    np.testing.assert_allclose(np.sort(fit.strength), np.sort(_weighted_cca_direct(X, groups, Y, w)), atol=1e-10)
    # scores have unit weighted variance, and loadings are weighted covariances
    r = (w / n)[groups]
    t = fit.scores(X)
    np.testing.assert_allclose(r @ t / r.sum(), 0, atol=1e-10)
    np.testing.assert_allclose(r @ t**2 / r.sum(), 1, atol=1e-10)
    tbar = gu.group_means(t, groups, len(n))
    np.testing.assert_allclose(fit.group_scores, tbar, atol=1e-10)
    np.testing.assert_allclose(fit.y_loadings, gu.wcentre(Y, w).T @ (w[:, None] * tbar) / w.sum(), atol=1e-10)


def test_size_weights_reproduce_ordinary_cca(design):
    X, groups, Y, n = design
    a, b = gu.cca(X, groups, Y), gu.cca(X, groups, Y, weights=n.astype(float))
    for field in ("strength", "x_weights", "y_weights", "group_scores", "y_loadings"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-10)


def test_weighted_cca_still_saturates():
    """Result 3c does not depend on the weights: at r = M − 1 every table gives the same fit."""
    rng = np.random.default_rng(8)
    n = np.array([40, 5, 12, 30, 8, 20])
    X, groups, Y = gu.simulate(n, p=4, q=5, rng=rng)
    w = rng.uniform(0.5, 2, 6)
    ref = gu.cca(X, groups, Y, weights=w).strength
    np.testing.assert_allclose(gu.cca(X, groups, rng.standard_normal((6, 5)), weights=w).strength, ref, atol=1e-10)
    with pytest.raises(ValueError, match="kappa = 1"):
        gu.cca_test(X, groups, Y, n_perm=10, weights="exchangeable")


def test_weighted_cca_test_is_calibrated_with_equal_groups():
    rng = np.random.default_rng(9)
    rejections = 0
    for _ in range(80):
        X, groups, Y = gu.simulate([15] * 12, p=3, q=2, tau2=0.5, rng=rng)
        rejections += gu.cca_test(X, groups, Y, n_components=1, n_perm=99, rng=rng,
                                  weights="exchangeable").p_value[0] < 0.05
    assert rejections / 80 < 0.12


def test_cca_rejects_bad_weights(design):
    X, groups, Y, n = design
    with pytest.raises(ValueError, match="positive"):
        gu.cca(X, groups, Y, weights=np.ones(len(n) + 1))
