"""Permutation tests over groups (Section 4).

The annotation profile belongs to the group, so the exchangeable units are
groups: every test here reassigns rows of the annotation table among groups,
keeping each group's individuals together. What must be refitted depends on
the design (Section 4.4):

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
from .methods import _cca_spaces, cca, pls
from .scores import between_share
from .simulate import simulate
from .weights import exchangeable_weights, pooled_variance_components


def permutation_weights(weights, values, groups, n):
    """Resolve the ``weights`` argument of a group permutation test to one weight per group.

    ``"exchangeable"`` (the default of the tests) estimates τ² and σ²_w from
    ``values`` (features or scores, pooled over columns, under the null) and
    returns weights ∝ 1/sqrt(τ² + σ²_w/n_m), so every group's term is equally
    variable. With equal group sizes these are equal weights. ``"size"`` gives
    group sizes, the weighting of an analysis over individuals. An array is
    used as given.
    """
    if not isinstance(weights, str):
        return np.asarray(weights, float)
    if weights == "size":
        return n.astype(float)
    if weights != "exchangeable":
        raise ValueError(f"weights must be 'exchangeable', 'size' or an array, not {weights!r}")
    if np.all(n == n[0]):
        return np.ones(len(n))
    vc = pooled_variance_components(values, groups)
    if not np.all(vc.tau2 + vc.sigma2_w / n > 0):
        return np.ones(len(n))      # no variation to standardise: every term is already alike
    return exchangeable_weights(n, vc.tau2, vc.sigma2_w)


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
    liberal. Below 1, they are conservative. No range of the ratio is safe for
    every design: one atypical group's share of the profile spread shrinks
    like 1/M, so the same ratio implies a more variable term, and a more
    liberal test, when there are more groups (Result 5d). Use the ratio as a
    warning and `size_check` to check a test's level. Without variance components, group
    means are taken as equally variable. Weights from
    `groupunit.weights.exchangeable_weights` give a ratio close to 1.
    """
    Y, A = _pairing_matrix(Y, n, weights, tau2, sigma2_w)
    observed = np.trace(Y.T @ A @ Y)
    permuted = np.trace(A) * (Y**2).sum() / (len(Y) - 1)
    return float(observed / permuted)


def _pairing_matrix(Y, n, weights, tau2, sigma2_w):
    """Y centred without weights, and A = W C V Cᵀ W, the null covariance of the weighted terms."""
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
    # A 1 = 0, so the statistic does not depend on how Y is centred; unweighted centring
    # makes the permutation average a function of Σ‖y_m‖² alone
    return Y - Y.mean(0), WC @ (v[:, None] * WC.T)


@dataclass(frozen=True)
class PairingTerms:
    """The pairing ratio split into one term per group (Result 5d).

    Attributes
    ----------
    ratio : the pairing ratio ϱ, as returned by `pairing_ratio`.
    terms : (M,) the excess ϱ − 1 split exactly over groups; they sum to ϱ − 1.
        A large positive term marks a group whose atypical profile sits on a
        highly variable term.
    profile_share : (M,) f_m = ‖y_m − ȳ‖² / Σ_m ‖y_m − ȳ‖², each group's share
        of the spread of the annotation profiles (unweighted mean ȳ).
    variance_ratio : (M,) g_m = A_mm / (tr A / M), the null variance of group
        m's weighted term relative to the average group; about w_m² v_m up to
        scale. Exchangeable weights make every g_m close to 1.
    largest : index of the group with the largest term.
    """

    ratio: float
    terms: np.ndarray
    profile_share: np.ndarray
    variance_ratio: np.ndarray
    largest: int


def pairing_terms(Y, n, weights=None, tau2=None, sigma2_w=None):
    """Which groups make a group permutation test liberal or conservative (Result 5d).

    Splits the excess of the `pairing_ratio` over 1 into one term per group,

        ϱ_m = (M − 1) c_m / (tr(A) S_Y) − f_m,   c_m = Σ_m' A_mm' y_mᵀ y_m',

    with the profiles y_m centred without weights, S_Y = Σ‖y_m‖² and A as in
    `pairing_ratio`. The terms sum exactly to ϱ − 1. When no group holds a
    large share of the total weight, ϱ_m ≈ f_m (g_m − 1): a group pushes the
    test towards liberal when its profile is unusual (large f_m) and its term
    is more variable than average (g_m > 1). Under size weights these are the
    large groups; under equal weights, and in similarity designs, the small
    ones. Report the ratio together with its largest term.

    Arguments are those of `pairing_ratio`.
    """
    Y, A = _pairing_matrix(Y, n, weights, tau2, sigma2_w)
    M = len(Y)
    S_Y = (Y**2).sum()
    if S_Y == 0:
        raise ValueError("all annotation profiles are equal; the pairing ratio is undefined")
    trA = np.trace(A)
    c = ((A @ Y) * Y).sum(1)                 # c_m = Σ_m' A_mm' y_mᵀ y_m'
    f = (Y**2).sum(1) / S_Y
    terms = (M - 1) * c / (trA * S_Y) - f
    return PairingTerms(
        ratio=float(1 + terms.sum()),
        terms=terms,
        profile_share=f,
        variance_ratio=np.diag(A) / (trA / M),
        largest=int(np.argmax(terms)),
    )


def draw_permutations(M, n_perm, rng=None, strata=None):
    """Random permutations of M groups, (n_perm, M).

    With ``strata`` (one label per group, for example a drug class), groups are
    permuted only within their stratum, which tests the annotations beyond that
    group-level variable (Section 4.5).
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
    """Family-wise p-values by the maximum statistic over all cells (Westfall-Young max-T).

    Suited to cells that share a scale and a null spread. If one cell's null
    is much wider than the others', it sets the bar for all of them; use
    `min_p_value` then.
    """
    observed = np.asarray(observed, float)
    null = np.asarray(null, float)
    null_max = null.reshape(len(null), -1).max(1)
    return p_value(observed, null_max.reshape((-1,) + (1,) * observed.ndim))


def min_p_value(observed, null):
    """Family-wise p-values by the minimum p-value over all cells (Westfall-Young min-p).

    Each cell's statistic is first turned into a p-value against that cell's
    own permutation null, and the family-wise p-value of a cell is the share of
    permutations whose smallest p-value over the family is at most the cell's
    own. Cells with different null spreads are thereby put on one scale, so a
    cell with a wide null does not take the power of the others. The observed
    data count as one of the n_perm + 1 equally likely assignments, as in
    `p_value`. P-values are discrete with floor 1 / (n_perm + 1), so large
    families need many permutations.
    """
    observed = np.asarray(observed, float)
    null = np.asarray(null, float)
    values = np.concatenate([observed.reshape(1, -1), null.reshape(len(null), -1)])   # (n_perm + 1, cells)
    B1 = len(values)
    tol = 1e-10 * np.maximum(1.0, np.abs(values))
    # p-value of every assignment (observed and permuted) against its own cell's distribution
    ranks = np.empty_like(values)
    for j in range(values.shape[1]):
        ordered = np.sort(values[:, j])
        ranks[:, j] = B1 - np.searchsorted(ordered, values[:, j] - tol[:, j], side="left")
    p_all = ranks / B1
    min_p = p_all.min(1)
    family = (min_p[None, :] <= p_all[0][:, None] + 1e-12).sum(1) / B1
    return family.reshape(observed.shape)


CORRECTIONS = {"max-t": max_t_p_value, "min-p": min_p_value}


def _family_wise(correction):
    try:
        return CORRECTIONS[correction]
    except KeyError:
        raise ValueError(f"correction must be one of {sorted(CORRECTIONS)}, not {correction!r}") from None


@dataclass(frozen=True)
class PermutationResult:
    """Observed statistic, its permutation null, and uncorrected and family-wise p-values.

    Attributes
    ----------
    statistic : the observed statistic, a number or an array of cells.
    null : (n_perm, ...) the statistic under each reassignment of annotation rows.
    p_value : uncorrected p-values, each cell against its own null.
    p_fwer : family-wise p-values over all cells, by ``correction``.
    correction : "max-t" or "min-p".
    """

    statistic: np.ndarray
    null: np.ndarray
    p_value: np.ndarray
    p_fwer: np.ndarray
    correction: str = "max-t"


def permutation_test(statistic, Y, n_perm=5000, rng=None, strata=None, correction="max-t"):
    """Generic group-level permutation test.

    ``statistic(Y_perm)`` receives the annotation table with its rows
    reassigned among groups and returns a number or an array, larger meaning
    more extreme (pass absolute values for two-sided tests). It must repeat
    every fit and selection step.

    For arrays, ``p_fwer`` controls family-wise error over all cells.
    ``correction="max-t"`` compares each cell with the maximum over cells and
    suits cells on a common scale with similar null spread. ``"min-p"`` first
    turns each cell into a p-value against its own null, which suits cells
    whose null spreads differ: selection statistics such as the best of many
    annotations per component, or components concentrated on different
    numbers of groups. Choose the correction before looking at the results.
    """
    family_wise = _family_wise(correction)
    Y = np.asarray(Y, float)
    observed = np.asarray(statistic(Y), float)
    null = np.array([statistic(Y[perm]) for perm in draw_permutations(len(Y), n_perm, rng, strata)], float)
    return PermutationResult(observed, null, p_value(observed, null), family_wise(observed, null), correction)


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
                   strata=None, correction="max-t"):
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
    (`pairing_ratio`). Check `groupunit.describe` first. ``correction`` is
    passed to `permutation_test`.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    groups, _ = check_groups(groups, len(Y))

    def refit(Y_perm):
        Y_individuals = Y_perm[groups]
        # a fresh copy each time, so no fitted state carries over between permutations
        fitted = copy.deepcopy(estimator).fit(X, Y_individuals)
        return statistic(fitted, X, Y_individuals)

    return permutation_test(refit, Y, n_perm, rng, strata, correction)


@dataclass(frozen=True)
class CorrelationTest:
    """Correlations of fixed scores with annotations, tested over groups.

    Attributes
    ----------
    r : (k, q) individual-level correlations, η · r_n; with custom weights,
        the weighted group-level correlations r_w.
    p_value : (k, q) uncorrected two-sided p-values.
    p_fwer : (k, q) family-wise p-values over all k × q cells, by ``correction``.
    correction : "max-t" or "min-p".
    """

    r: np.ndarray
    p_value: np.ndarray
    p_fwer: np.ndarray
    correction: str = "max-t"


def correlation_test(scores, groups, Y, n_perm=5000, rng=None, strata=None, weights="exchangeable",
                     correction="max-t"):
    """Test correlations of scores built without the annotations (Result 1, Section 4.4).

    ``scores`` (N,) or (N, k) are held fixed: single features chosen in
    advance, PCA components, LDA scores, or CCA factors at saturation. The
    test concerns the group-level correlations, which rest on M groups.

    By default groups are weighted so that their terms are exchangeable
    (`permutation_weights`), which keeps the test valid when atypical profiles sit on
    large or small groups. ``r`` is then the weighted group-level correlation,
    and the p-values come from the weighted cross-product Σ w_m t̄_m y_m,
    scaled by quantities that do not change under permutation.
    ``weights="size"`` tests the size-weighted correlations instead: ``r`` is
    then the individual-level correlation η · r_n. That test is liberal when
    large groups carry atypical profiles (`pairing_ratio` well above 1).

    ``correction`` sets the family-wise correction over the k × q cells (see
    `permutation_test`). Max-T suits scores with similar null spreads. If some
    scores are concentrated on a few groups (a small effective number of
    groups behind them), their cells have wider nulls and min-p is the fairer
    choice. Decide before looking at the results.
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    scores = np.asarray(scores, float).reshape(len(groups), -1)
    tbar = group_means(scores - scores.mean(0), groups, len(Y))
    Y2 = Y.reshape(len(Y), -1)
    if isinstance(weights, str) and weights == "size":
        eta = np.sqrt(np.atleast_1d(between_share(scores, groups)))
        r = eta[:, None] * weighted_corr(tbar, Y2, n).reshape(len(eta), -1)

        def statistic(Yp):
            return np.abs(eta[:, None] * weighted_corr(tbar, Yp, n).reshape(len(eta), -1))
    else:
        w = permutation_weights(weights, scores, groups, n)
        r = weighted_corr(tbar, Y2, w).reshape(tbar.shape[1], -1)
        wt = w[:, None] * wcentre(tbar, w)          # sums to zero, so Y needs no re-centring
        scale = np.outer(np.sqrt((wt * wcentre(tbar, w)).sum(0)), Y2.std(0))

        def statistic(Yp):
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.abs(wt.T @ Yp / scale)

    result = permutation_test(lambda Yp: np.nan_to_num(statistic(Yp)), Y2, n_perm, rng, strata, correction)
    return CorrelationTest(r=r, p_value=result.p_value, p_fwer=result.p_fwer, correction=correction)


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


def pls_test(X, groups, Y, n_components=None, n_perm=5000, rng=None, weights="exchangeable", strata=None):
    """Sequential group-level permutation test of PLS components (Section 4.4).

    PLS weights depend on the annotations, so the model is refitted in every
    permutation. For component c, the first c - 1 components are removed from
    both sides before permuting (Winkler et al. 2020). Only the group means
    enter, so the test is fast. Selection of annotations or of the number of
    components is not repeated here; wrap the whole pipeline in
    `permutation_test` if it was data-driven.

    By default groups are weighted so that their terms are exchangeable
    (`permutation_weights`), so ``fit`` is the PLS fit with those weights. For a
    size-weighted description of the data, fit `groupunit.pls` separately.
    ``weights="size"`` tests the size-weighted fit, which is liberal when
    large groups carry atypical profiles (`pairing_ratio` well above 1).
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    w = permutation_weights(weights, X, groups, n)
    fit = pls(X, groups, Y, n_components, w)
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


@dataclass(frozen=True)
class SizeCheck:
    """Estimated size of a test for one design (Result 5, Section 4.3).

    Attributes
    ----------
    rate : share of simulated null data sets in which the test rejected.
    se : its binomial standard error, sqrt(rate (1 − rate) / n_sim).
    alpha : the nominal level.
    p_values : (n_sim,) the smallest p-value the test returned for each data set.
    """

    rate: float
    se: float
    alpha: float
    p_values: np.ndarray


def size_check(test, n, Y, tau2, sigma2_w, p=1, n_sim=200, alpha=0.05, rng=None):
    """How often a test rejects for this design when features and annotations are unrelated.

    The pairing ratio compares averages, while a test's level depends on the
    tails of its null distribution, so a ratio near 1 does not guarantee the
    level (Result 5a). This checks the level directly for the design at hand.
    It keeps the group sizes ``n`` and the annotation table ``Y`` as observed,
    draws features with no relation to the annotations,

        x_i = u_m(i) + ε_i,   u_m ~ N(0, τ² I),   ε_i ~ N(0, σ²_w I),

    so that each group mean has variance v_m = τ² + σ²_w / n_m, runs ``test``
    on each data set and records how often it rejects at ``alpha``.

    ``test(X, groups, Y, rng)`` runs the test and returns a p-value or an
    array of p-values; the data set counts as a rejection if the smallest is
    at most ``alpha``, so return family-wise p-values for a family of cells.
    For example, for the first PLS component::

        size_check(lambda X, g, Y, rng: pls_test(X, g, Y, n_components=1, n_perm=199,
                                                 rng=rng).p_value[0],
                   n, Y, vc.tau2, vc.sigma2_w)

    with ``vc`` from `groupunit.weights.pooled_variance_components`. Estimate
    τ² around the grand mean, not around the annotations, so that the check
    is possible at any annotation ratio. ``p`` is the number of simulated
    features: use the number the test sees, or a smaller one when only a
    rough check is needed. The features are Gaussian and independent, so the
    check covers the design (sizes, profiles, variance components), not
    heavy tails or correlated features. With 200 data sets the standard error
    at a true rate of 0.05 is about 0.015.
    """
    rng = np.random.default_rng(rng)
    n = np.asarray(n, int)
    Y = np.asarray(Y, float)
    if Y.shape[0] != len(n):
        raise ValueError(f"Y has {Y.shape[0]} rows but there are {len(n)} groups")
    if tau2 < 0 or sigma2_w < 0 or tau2 + sigma2_w == 0:
        raise ValueError("tau2 and sigma2_w must be non-negative and not both zero")
    p_values = np.empty(n_sim)
    for s in range(n_sim):
        X, groups, _ = simulate(n, p, Y=Y, beta=0.0, tau2=tau2, sigma2_w=sigma2_w, rng=rng)
        p_values[s] = np.min(test(X, groups, Y, rng))
    rate = float((p_values <= alpha).mean())
    return SizeCheck(rate, float(np.sqrt(rate * (1 - rate) / n_sim)), alpha, p_values)


def cca_test(X, groups, Y, n_components=None, n_perm=5000, rng=None, strata=None, weights="size"):
    """Sequential group-level permutation test of canonical correlations.

    Refits CCA in every permutation, removing the first c - 1 canonical
    variates before testing the c-th. If the annotations have rank M - 1
    (κ = 1), every permutation gives the same result and no test of the
    annotations is possible (Result 3c); this raises an error.

    Ordinary CCA weights groups by size (``weights="size"``, the default), so
    like size-weighted PLS it is liberal when large groups carry atypical
    profiles (`pairing_ratio` well above 1). ``weights="exchangeable"`` or an
    array fits and tests row-weighted CCA instead (see `groupunit.cca`), in
    which group m carries total weight w_m. Canonical correlations are
    ratios, so whether exchangeable weights restore the level as they do for
    PLS is an empirical question; check the design with `size_check`.
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    X = np.asarray(X, float)
    w = permutation_weights(weights, X, groups, n)
    Ux, sx, Vxt, x_mean, P, Q = _cca_spaces(X, groups, Y, w)
    if Q.shape[1] >= len(Y) - 1:
        raise ValueError("the annotations span all M - 1 between-group directions (kappa = 1): CCA equals "
                         "LDA for any annotation table and cannot be tested against the annotations. "
                         "Reduce the annotations first, e.g. with reduce_annotations().")
    fit = cca(X, groups, Y, n_components, weights=w)

    # canonical correlations = svd(Pᵀ Q); full SVD: deflation must keep every remaining
    # direction of both spaces, not only the paired ones
    A, rho, Bt = np.linalg.svd(P.T @ Q, full_matrices=True)
    perms = draw_permutations(len(Y), n_perm, rng, strata)

    p = np.empty(len(fit.strength))
    for c, observed in enumerate(fit.strength):
        Pd = P @ A[:, c:]
        Yd = (Q @ Bt[c:].T) / np.sqrt(w)[:, None]   # annotation table spanning the deflated space
        null = [np.linalg.norm(Pd.T @ weighted_basis(Yd[perm], w), 2) for perm in perms]
        p[c] = p_value(observed, null)
    return SequentialTest(fit.strength, np.maximum.accumulate(p), fit)
