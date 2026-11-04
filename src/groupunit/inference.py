"""Permutation tests over groups (Section 7).

The annotation profile belongs to the group, so the exchangeable units are
groups: every test here reassigns rows of the annotation table among groups,
keeping each group's individuals together. What must be refitted depends on
the design (Section 7.2):

* scores built without the annotations are held fixed - `correlation_test`;
* components that depend on the annotations are refitted in every
  permutation - `pls_test`, `cca_test`;
* any other data-dependent step (annotation subsets, numbers of components,
  selected cells) must go inside the statistic of `permutation_test`;
* models from other libraries (scikit-learn's PLSRegression, CCA, pipelines)
  are refitted on individuals in every permutation - `estimator_test`.

Reassigning rows among groups is exact only if the groups' terms in the
statistic are exchangeable. With unequal group sizes they may not be: check
`pairing_ratio`, and use `groupunit.weights.exchangeable_weights` when it is
far from 1.
"""
import copy
from dataclasses import dataclass

import numpy as np

from .core import check_groups, group_means, wcentre, weighted_basis, weighted_corr
from .methods import _rank_cutoff, cca, pls
from .scores import between_share


def pairing_ratio(Y, n, weights=None, tau2=None, sigma2_w=None):
    """How unusual the observed pairing of group weights with annotation profiles is.

    Under the null, a weighted loading ℓ = Ỹᵀ W (x̄ − x̄_w) of independent group
    means with variances v_m = τ² + σ²_w / n_m has E‖ℓ‖² = tr(Yᵀ A Y), with
    A = W C V Cᵀ W and C the weighted centring. The ratio of this to its
    average over reassignments of annotation rows among groups, tr(A) ·
    Σ‖y_m − ȳ‖² / (M − 1), says how much larger than under permutation the
    statistic is expected to be by design alone. Near 1, group permutation
    tests with these weights hold their level. Above 1, atypical profiles sit
    on high-variance terms (for size weights: large groups) and the tests are
    liberal. Below 1, they are conservative. In the paper's simulations, tests
    held their level at ratios of 0.9-1.1. Without variance components, group
    means are taken as equally variable. Weights from
    `groupunit.weights.exchangeable_weights` give a ratio close to 1.
    """
    n = np.asarray(n, float)
    Y = np.asarray(Y, float).reshape(len(n), -1)
    M = len(n)
    w = n if weights is None else np.asarray(weights, float)
    if tau2 is None and sigma2_w is None:
        v = np.ones_like(n)
    elif tau2 is None or sigma2_w is None:
        raise ValueError("give both tau2 and sigma2_w, or neither")
    else:
        v = tau2 + sigma2_w / n
    WC = w[:, None] * (np.eye(M) - np.outer(np.ones(M), w) / w.sum())   # W C
    A = WC @ (v[:, None] * WC.T)
    observed = np.trace(Y.T @ A @ Y)
    permuted = np.trace(A) * ((Y - Y.mean(0)) ** 2).sum() / (M - 1)
    return float(observed / permuted)


def draw_permutations(M, n_perm, rng=None, strata=None):
    """Random permutations of M groups, (n_perm, M).

    With ``strata`` (one label per group, for example a drug class), groups are
    permuted only within their stratum, which tests the annotations beyond that
    group-level variable (Section 7.5).
    """
    rng = np.random.default_rng(rng)
    if strata is None:
        return np.array([rng.permutation(M) for _ in range(n_perm)])
    strata = np.asarray(strata)
    if strata.shape != (M,):
        raise ValueError(f"strata needs one label per group ({M}), got shape {strata.shape}")
    perms = np.tile(np.arange(M), (n_perm, 1))
    for label in np.unique(strata):
        idx = np.flatnonzero(strata == label)
        for perm in perms:
            perm[idx] = rng.permutation(idx)
    return perms


def p_value(observed, null):
    """Permutation p-values, (1 + #{null >= observed}) / (1 + n_perm), elementwise.

    A small tolerance counts exact ties as ties: under saturation every
    permutation reproduces the observed statistic up to rounding.
    """
    observed = np.asarray(observed, float)
    null = np.asarray(null, float)
    tol = 1e-10 * np.maximum(1.0, np.abs(observed))
    return (1 + (null >= observed - tol).sum(0)) / (1 + len(null))


def max_t_p_value(observed, null):
    """Family-wise p-values by the maximum statistic over all cells (Westfall-Young max-T)."""
    observed = np.asarray(observed, float)
    null = np.asarray(null, float)
    null_max = null.reshape(len(null), -1).max(1)
    return p_value(observed, null_max.reshape((-1,) + (1,) * observed.ndim))


@dataclass(frozen=True)
class PermutationResult:
    """Observed statistic, its permutation null, and uncorrected and family-wise p-values."""

    statistic: np.ndarray
    null: np.ndarray
    p_value: np.ndarray
    p_fwer: np.ndarray


def permutation_test(statistic, Y, n_perm=5000, rng=None, strata=None):
    """Generic group-level permutation test.

    ``statistic(Y_perm)`` receives the annotation table with its rows
    reassigned among groups and returns a number or an array, larger meaning
    more extreme (pass absolute values for two-sided tests). It must repeat
    every fit and selection step. For arrays, ``p_fwer`` controls family-wise
    error over all cells by max-T, so the cells should share a scale.
    """
    Y = np.asarray(Y, float)
    observed = np.asarray(statistic(Y), float)
    null = np.array([statistic(Y[perm]) for perm in draw_permutations(len(Y), n_perm, rng, strata)], float)
    return PermutationResult(observed, null, p_value(observed, null), max_t_p_value(observed, null))


def paired_score_correlation(fitted, X, Y_individuals):
    """|corr| between the paired feature and annotation scores of each component, (k,).

    The default statistic of `estimator_test`, for any fitted estimator whose
    ``transform(X, Y)`` returns ``(x_scores, y_scores)``, as scikit-learn's
    PLSRegression, PLSCanonical, PLSSVD and CCA do.
    """
    x_scores, y_scores = fitted.transform(X, Y_individuals)
    x_scores = x_scores - x_scores.mean(0)
    y_scores = y_scores - y_scores.mean(0)
    return np.abs((x_scores * y_scores).sum(0) / np.sqrt((x_scores**2).sum(0) * (y_scores**2).sum(0)))


def estimator_test(estimator, X, groups, Y, statistic=paired_score_correlation, n_perm=1000, rng=None,
                   strata=None):
    """Group-level permutation test of a model fitted on individuals, refitted in every permutation.

    ``estimator`` is any object with a scikit-learn style ``fit(X, Y)``. It is
    copied and fitted on ``X`` and the inherited annotations ``Y[groups]``,
    once for the real table and once per reassignment of its rows among groups.
    ``statistic(fitted, X, Y_individuals)`` returns a number or an array,
    larger meaning more extreme. If the model's components can come out in a
    different order under permutation, as with PLS, later components are tested
    conservatively. For sequential tests use `pls_test` or `cca_test`.

    Iterative fits are not exact. At saturation, where every permutation
    should reproduce the observed statistic, their rounding noise makes the
    p-value arbitrary rather than 1. Fits on individuals weight groups by
    size, so the test is liberal when large groups carry atypical profiles
    (`pairing_ratio`). Check `groupunit.describe` first.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    groups, _ = check_groups(groups, len(Y))

    def refit(Y_perm):
        Y_individuals = Y_perm[groups]
        # a fresh copy each time, so no fitted state carries over between permutations
        fitted = copy.deepcopy(estimator).fit(X, Y_individuals)
        return statistic(fitted, X, Y_individuals)

    return permutation_test(refit, Y, n_perm, rng, strata)


@dataclass(frozen=True)
class CorrelationTest:
    """Correlations of fixed scores with annotations, tested over groups.

    Attributes
    ----------
    r : (k, q) individual-level correlations, η · r_n; with custom weights,
        the weighted group-level correlations r_w.
    p_value : (k, q) uncorrected two-sided p-values.
    p_fwer : (k, q) family-wise p-values over all k × q cells (max-T).
    """

    r: np.ndarray
    p_value: np.ndarray
    p_fwer: np.ndarray


def correlation_test(scores, groups, Y, n_perm=5000, rng=None, strata=None, weights=None):
    """Test correlations of scores built without the annotations (Sections 4.1, 7.2).

    ``scores`` (N,) or (N, k) are held fixed: single features chosen in
    advance, PCA components, LDA scores, or CCA factors at saturation. Each
    score's η is unchanged by permutation, so the test is a test of the
    group-level correlations r_n, which rest on M groups. If atypical profiles
    sit on large groups (`pairing_ratio` well above 1), the size-weighted test
    is liberal. Pass ``weights`` from `groupunit.weights.exchangeable_weights`
    to test the weighted group-level correlations instead. Their p-values then
    come from the weighted cross-product Σ w_m t̄_m y_m, scaled by quantities
    that do not change under permutation. A correlation's denominator would
    change with the pairing and make the test slightly liberal.
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    scores = np.asarray(scores, float).reshape(len(groups), -1)
    tbar = group_means(scores - scores.mean(0), groups, len(Y))
    Y2 = Y.reshape(len(Y), -1)
    if weights is None:
        eta = np.sqrt(np.atleast_1d(between_share(scores, groups)))
        r = eta[:, None] * weighted_corr(tbar, Y2, n).reshape(len(eta), -1)

        def statistic(Yp):
            return np.abs(eta[:, None] * weighted_corr(tbar, Yp, n).reshape(len(eta), -1))
    else:
        w = np.asarray(weights, float)
        r = weighted_corr(tbar, Y2, w).reshape(tbar.shape[1], -1)
        wt = w[:, None] * wcentre(tbar, w)          # sums to zero, so Y needs no re-centring
        scale = np.outer(np.sqrt((wt * wcentre(tbar, w)).sum(0)), Y2.std(0))

        def statistic(Yp):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.abs(wt.T @ Yp / scale)

    result = permutation_test(lambda Yp: np.nan_to_num(statistic(Yp)), Y2, n_perm, rng, strata)
    return CorrelationTest(r=r, p_value=result.p_value, p_fwer=result.p_fwer)


@dataclass(frozen=True)
class SequentialTest:
    """Sequential permutation tests of components.

    Attributes
    ----------
    strength : (k,) observed singular values or canonical correlations.
    p_value : (k,) p-value of component c after removing components 1..c-1,
        made non-decreasing so that testing stops at the first non-significant one.
    fit : the fitted `Components` on the real annotations.
    """

    strength: np.ndarray
    p_value: np.ndarray
    fit: object


def pls_test(X, groups, Y, n_components=None, n_perm=5000, rng=None, weights=None, strata=None):
    """Sequential group-level permutation test of PLS components (Sections 7.2-7.3).

    PLS weights depend on the annotations, so the model is refitted in every
    permutation. For component c, the first c - 1 components are removed from
    both sides before permuting (Winkler et al. 2020). Only the group means
    enter, so the test is fast. Selection of annotations or of the number of
    components is not repeated here; wrap the whole pipeline in
    `permutation_test` if it was data-driven.

    With size weights the test is liberal when large groups carry atypical
    profiles; check `pairing_ratio`, and pass ``weights`` from
    `groupunit.weights.exchangeable_weights` to make every group's term
    exchangeable.
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    w = n.astype(float) if weights is None else np.asarray(weights, float)
    fit = pls(X, groups, Y, n_components, weights)
    Xbar = group_means(X, groups, len(Y))
    Xb = Xbar - w @ Xbar / w.sum()
    Yc = wcentre(Y, w)
    perms = draw_permutations(len(Y), n_perm, rng, strata)

    p = np.empty(len(fit.strength))
    for c, observed in enumerate(fit.strength):
        Wd, Vd = fit.x_weights[:, :c], fit.y_weights[:, :c]
        XtW = (Xb - Xb @ Wd @ Wd.T).T * w
        Yd = Yc - Yc @ Vd @ Vd.T
        null = [np.linalg.norm(XtW @ wcentre(Yd[perm], w), 2) for perm in perms]
        p[c] = p_value(observed, null)
    return SequentialTest(fit.strength, np.maximum.accumulate(p), fit)


def cca_test(X, groups, Y, n_components=None, n_perm=5000, rng=None, strata=None):
    """Sequential group-level permutation test of canonical correlations.

    Refits CCA in every permutation, removing the first c - 1 canonical
    variates before testing the c-th. If the annotations have rank M - 1
    (κ = 1), every permutation gives the same result and no test of the
    annotations is possible (Result 3c); this raises an error. CCA weights
    groups by size, so like size-weighted PLS it is liberal when large groups
    carry atypical profiles (`pairing_ratio` well above 1).
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    Q = weighted_basis(Y, n)
    if Q.shape[1] >= len(Y) - 1:
        raise ValueError("the annotations span all M - 1 between-group directions (kappa = 1): CCA equals "
                         "LDA for any annotation table and cannot be tested against the annotations. "
                         "Reduce the annotations first, e.g. with reduce_annotations().")
    fit = cca(X, groups, Y, n_components)

    X = np.asarray(X, float)
    Ux, sx, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    Ux = Ux[:, _rank_cutoff(sx)]
    P = np.sqrt(n)[:, None] * group_means(Ux, groups, len(Y))   # canonical correlations = svd(Pᵀ Q)
    # full SVD: deflation must keep every remaining direction of both spaces, not only the paired ones
    A, rho, Bt = np.linalg.svd(P.T @ Q, full_matrices=True)
    perms = draw_permutations(len(Y), n_perm, rng, strata)

    p = np.empty(len(fit.strength))
    for c, observed in enumerate(fit.strength):
        Pd = P @ A[:, c:]
        Yd = (Q @ Bt[c:].T) / np.sqrt(n)[:, None]   # annotation table spanning the deflated space
        null = [np.linalg.norm(Pd.T @ weighted_basis(Yd[perm], n), 2) for perm in perms]
        p[c] = p_value(observed, null)
    return SequentialTest(fit.strength, np.maximum.accumulate(p), fit)
