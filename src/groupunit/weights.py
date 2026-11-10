"""Weighting groups: variance components, optimal weights and effective numbers of groups (Section 4.1, Appendix B.7).

A group mean deviates from what the annotations predict through sampling
error, with variance σ²_w / n_m, and through group-level heterogeneity, with
variance τ². Their balance decides whether size weights are efficient.
"""
from dataclasses import dataclass

import numpy as np

from .core import check_groups, group_means
from .diagnostics import effective_number


@dataclass(frozen=True)
class VarianceComponents:
    """Within-group variance σ²_w and between-group heterogeneity τ² of one score."""

    sigma2_w: float
    tau2: float


def variance_components(t, groups, Y=None):
    """Estimate σ²_w and τ² for a score ``t`` (N,), such as a component or a single feature.

    σ²_w is the pooled within-group variance. τ² is a moment estimate from the
    residuals of an equally weighted regression of the group means on the
    annotations ``Y`` (on an intercept alone if ``Y`` is None), with the expected
    sampling-error part removed; it is truncated at zero.
    """
    t = np.asarray(t, float)
    M = None if Y is None else len(Y)
    groups, n = check_groups(groups, M)
    M = len(n)
    tbar = group_means(t, groups, M)
    sigma2_w = float(((t - tbar[groups]) ** 2).sum() / (len(t) - M))

    Z = np.ones((M, 1)) if Y is None else np.column_stack([np.ones(M), np.asarray(Y, float)])
    H = Z @ np.linalg.pinv(Z)
    df = M - np.linalg.matrix_rank(Z)
    if df <= 0:
        raise ValueError("the annotations fit the group means exactly (saturated); τ² cannot be "
                         "estimated from their residuals. Reduce the annotations or pass Y=None.")
    resid = tbar - H @ tbar
    # E[RSS] = Σ (1 - h_mm)(τ² + σ²_w / n_m), with Σ (1 - h_mm) = df
    leverage_free = 1 - np.diag(H)
    tau2 = (resid @ resid - sigma2_w * (leverage_free / n).sum()) / df
    return VarianceComponents(sigma2_w=sigma2_w, tau2=float(max(tau2, 0.0)))


def optimal_weights(n, tau2, sigma2_w):
    """Most precise group weights, ∝ 1 / (τ² + σ²_w / n_m), normalised to sum to one."""
    w = 1 / (tau2 + sigma2_w / np.asarray(n, float))
    return w / w.sum()


def exchangeable_weights(n, tau2, sigma2_w):
    """Group weights 1 / sqrt(τ² + σ²_w / n_m) for permutation tests, normalised to sum to one.

    With these weights every group's term in a weighted cross-product has the
    same variance under the null, so reassigning annotation rows among groups
    is a valid test whichever groups carry atypical profiles (see
    `groupunit.inference.pairing_ratio`). They differ from `optimal_weights`,
    which are the square of these and are the most precise for estimation.
    """
    w = 1 / np.sqrt(tau2 + sigma2_w / np.asarray(n, float))
    return w / w.sum()


def pooled_variance_components(X, groups):
    """σ²_w and τ² averaged over the columns of ``X``, for weights shared by all features.

    Each column's τ² is estimated around its grand mean (no annotations), which
    is what the null hypothesis of no association implies.
    """
    X = np.asarray(X, float).reshape(len(groups), -1)
    parts = [variance_components(col, groups) for col in X.T]
    return VarianceComponents(sigma2_w=float(np.mean([c.sigma2_w for c in parts])),
                              tau2=float(np.mean([c.tau2 for c in parts])))


def effective_groups(w, n=None, tau2=None, sigma2_w=None):
    """Effective number of groups for weights ``w``.

    Without variance components this is Kish's formula, (Σw)² / Σw², which
    assumes every group mean is equally noisy. With ``n``, ``tau2`` and
    ``sigma2_w`` it is the model-based value
    τ²(Σw)² / Σ w²(τ² + σ²_w / n), never larger than Kish's.
    """
    w = np.asarray(w, float)
    if tau2 is None and sigma2_w is None:
        return float(effective_number(w))
    if n is None or tau2 is None or sigma2_w is None:
        raise ValueError("the model-based value needs n, tau2 and sigma2_w")
    v = tau2 + sigma2_w / np.asarray(n, float)
    return float(tau2 * w.sum() ** 2 / (w**2 * v).sum())


def max_effective_groups(n, tau2, sigma2_w):
    """Largest achievable effective number of groups, Σ τ² / (τ² + σ²_w / n_m) <= M."""
    n = np.asarray(n, float)
    return float((tau2 / (tau2 + sigma2_w / n)).sum())
