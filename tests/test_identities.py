"""The central identity and Results 1-3 of the paper, checked against direct computation."""
import numpy as np
import pytest
from scipy.linalg import subspace_angles

import groupunit as gu


def inherited(X, groups, Y):
    """Centred X and the annotations as inherited by individuals, Y[groups], centred."""
    Xc = X - X.mean(0)
    Yi = Y[groups]
    return Xc, Yi - Yi.mean(0)


def test_encode_groups_follows_order():
    codes, names = gu.encode_groups(["b", "a", "b"], order=["b", "a"])
    assert codes.tolist() == [0, 1, 0] and names.tolist() == ["b", "a"]
    with pytest.raises(ValueError):
        gu.encode_groups(["c"], order=["a"])


def test_central_identity(design):
    X, groups, Y, n = design
    Xc, Yi = inherited(X, groups, Y)
    Xbar = gu.group_means(Xc, groups)
    np.testing.assert_allclose(Xc.T @ Yi, Xbar.T @ (n[:, None] * gu.wcentre(Y, n)), atol=1e-9)


def test_scatter_matrices_add_up(design):
    X, groups, _, _ = design
    S_T, S_B, S_W = gu.scatter_matrices(X, groups)
    np.testing.assert_allclose(S_T, S_B + S_W, atol=1e-9)


def test_result1_correlation_is_attenuated_group_correlation(design):
    X, groups, Y, _ = design
    t = X @ np.arange(1, 7)
    d = gu.decompose_score(t, groups, Y)
    direct = [np.corrcoef(t, Y[groups, j])[0, 1] for j in range(Y.shape[1])]
    np.testing.assert_allclose(d.corr, direct, atol=1e-10)
    np.testing.assert_allclose(d.corr, np.sqrt(d.eta2) * d.group_corr)


def test_result3a_factorisation(design):
    X, groups, Y, _ = design
    t = X[:, 0] - X[:, 0].mean()
    _, Yi = inherited(X, groups, Y)
    fitted = Yi @ np.linalg.lstsq(Yi, t, rcond=None)[0]
    r2_individual = fitted @ fitted / (t @ t)
    assert gu.decompose_score(t, groups, Y).max_corr2 == pytest.approx(r2_individual)


def test_pls_weights_are_svd_of_individual_cross_product(design):
    X, groups, Y, _ = design
    Xc, Yi = inherited(X, groups, Y)
    U, s, _ = np.linalg.svd(Xc.T @ Yi, full_matrices=False)
    fit = gu.pls(X, groups, Y)
    k = len(fit.strength)
    assert k <= min(X.shape[1], Y.shape[1], len(Y) - 1)
    np.testing.assert_allclose(fit.strength, s[:k])
    np.testing.assert_allclose(np.abs(fit.x_weights), np.abs(U[:, :k]), atol=1e-8)
    np.testing.assert_allclose(fit.group_scores, gu.group_means(fit.scores(X), groups), atol=1e-9)


def test_cca_matches_principal_angles(design):
    X, groups, Y, _ = design
    Xc, Yi = inherited(X, groups, Y)
    expected = np.sort(np.cos(subspace_angles(Xc, Yi)))[::-1]
    fit = gu.cca(X, groups, Y)
    np.testing.assert_allclose(fit.strength, expected[: len(fit.strength)], atol=1e-10)
    # scores have unit variance and the stated canonical correlations
    T = fit.scores(X)
    U = (Y[groups] - Y[groups].mean(0)) @ fit.y_weights
    np.testing.assert_allclose(T.var(0), 1, atol=1e-10)
    np.testing.assert_allclose([np.corrcoef(T[:, c], U[:, c])[0, 1] for c in range(T.shape[1])],
                               fit.strength, atol=1e-8)


def test_result3c_saturated_cca_ignores_the_annotations():
    rng = np.random.default_rng(1)
    n = gu.lognormal_sizes(10, 25, rng=rng)
    X, groups, Y = gu.simulate(n, p=5, q=9, beta=1.0, rng=rng)   # r = M - 1 = 9
    reference = gu.cca(X, groups, Y).strength
    for other in (Y[rng.permutation(10)], rng.standard_normal((10, 12)), np.eye(10)):
        np.testing.assert_allclose(gu.cca(X, groups, other).strength, reference, atol=1e-10)
    np.testing.assert_allclose(gu.lda(X, groups).strength, reference, atol=1e-10)
    # squared canonical correlations of LDA are the between-group shares of its scores
    fit = gu.lda(X, groups)
    np.testing.assert_allclose(gu.between_share(fit.scores(X), groups), fit.strength**2, atol=1e-10)


def test_result3d_cca_invariant_pls_not(design):
    X, groups, Y, _ = design
    B = np.random.default_rng(2).standard_normal((4, 4))
    np.testing.assert_allclose(gu.cca(X, groups, Y @ B).strength, gu.cca(X, groups, Y).strength, atol=1e-10)
    assert not np.allclose(gu.pls(X, groups, Y @ B).strength, gu.pls(X, groups, Y).strength)
