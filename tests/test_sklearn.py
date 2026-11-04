"""Cross-checks against scikit-learn, and scikit-learn models passed to groupunit. Skipped without scikit-learn."""
import numpy as np
import pytest

import groupunit as gu

pytest.importorskip("sklearn")
from sklearn.cross_decomposition import CCA, PLSSVD, PLSRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402


def test_pls_matches_plssvd_without_scaling(design):
    X, groups, Y, _ = design
    ours = gu.pls(X, groups, Y, n_components=3)
    theirs = PLSSVD(n_components=3, scale=False).fit(X, Y[groups])
    np.testing.assert_allclose(np.abs(ours.x_weights), np.abs(theirs.x_weights_), atol=1e-8)


def test_cca_matches_sklearn(design):
    X, groups, Y, _ = design
    ours = gu.cca(X, groups, Y, n_components=3)
    x_scores, y_scores = CCA(n_components=3, max_iter=5000).fit(X, Y[groups]).transform(X, Y[groups])
    theirs = [np.corrcoef(x_scores[:, c], y_scores[:, c])[0, 1] for c in range(3)]
    np.testing.assert_allclose(ours.strength, theirs, atol=1e-4)   # scikit-learn's CCA is iterative


def test_estimator_test_detects_relation_and_respects_the_null(design):
    X, groups, Y, _ = design
    result = gu.estimator_test(PLSRegression(n_components=2, scale=False), X, groups, Y, n_perm=199, rng=0)
    assert result.statistic.shape == (2,)
    assert result.p_value[0] < 0.05

    rng = np.random.default_rng(1)
    X0, groups0, Y0 = gu.simulate(gu.lognormal_sizes(12, 30, rng=rng), p=6, q=3, beta=0.0, rng=rng)
    null = gu.estimator_test(PLSRegression(n_components=1, scale=False), X0, groups0, Y0, n_perm=199, rng=0)
    assert null.p_value[0] > 0.05


def test_estimator_test_with_custom_statistic(design):
    X, groups, Y, _ = design

    def r2(fitted, X, Y_individuals):
        return fitted.score(X, Y_individuals)

    result = gu.estimator_test(PLSRegression(n_components=2, scale=False), X, groups, Y, statistic=r2,
                               n_perm=99, rng=0)
    assert result.null.shape == (99,)


def test_leave_one_group_out_accepts_estimators():
    rng = np.random.default_rng(12)
    Y = rng.standard_normal((20, 4))
    target = Y @ np.array([1.0, -1.0, 0.5, 0.0]) + 0.1 * rng.standard_normal(20)
    w = rng.integers(5, 50, 20).astype(float)

    for weights in (None, w):
        ours = gu.leave_one_group_out(target, Y, weights=weights)
        theirs = gu.leave_one_group_out(target, Y, weights=weights, model=LinearRegression())
        np.testing.assert_allclose(theirs.predictions, ours.predictions, atol=1e-10)

    # a pipeline refits its PCA on the training groups in every fold
    piped = gu.leave_one_group_out(target, Y, model=make_pipeline(PCA(3), LinearRegression()))
    nested = gu.leave_one_group_out(target, Y, model=gu.pca_regression(3))
    np.testing.assert_allclose(piped.predictions, nested.predictions, atol=1e-10)
