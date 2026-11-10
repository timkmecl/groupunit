"""Between-group share of scores and the decomposition of their correlations (Results 1 and 3a)."""
from dataclasses import dataclass

import numpy as np

from .core import check_groups, group_means, weighted_corr, weighted_r2


def between_share(t, groups):
    """Between-group share η² of the variance of a score (Results 1 and 3a).

    ``t`` is a score vector ``(N,)`` or a matrix of scores ``(N, k)``; returns
    a float or one value per column.
    """
    groups, n = check_groups(groups)
    t = np.asarray(t, float)
    t = t - t.mean(0)
    tbar = group_means(t, groups)
    n_ = n.reshape((-1,) + (1,) * (t.ndim - 1))
    eta2 = (n_ * tbar**2).sum(0) / (t**2).sum(0)
    return float(eta2) if t.ndim == 1 else eta2


def feature_between_share(X, groups):
    """Between-group share of the feature space as a whole, tr S_B / tr S_T (Section 5)."""
    groups, n = check_groups(groups)
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    Xbar = group_means(Xc, groups)
    return float((n[:, None] * Xbar**2).sum() / (Xc**2).sum())


@dataclass(frozen=True)
class ScoreDecomposition:
    """How a score relates to the annotations, split into group separation and group-level fit.

    Attributes
    ----------
    eta2 : between-group share of the score's variance.
    group_corr : (q,) size-weighted correlation of the score's group means with each annotation, r_n.
    corr : (q,) individual-level correlation with each annotation, ``sqrt(eta2) * group_corr`` (Result 1).
    group_r2 : size-weighted R² of the group means on all annotations, R²_n.
    max_corr2 : largest squared correlation of the score with any combination of
        annotations, ``eta2 * group_r2`` (Result 3a); what CCA maximises.
    """

    eta2: float
    group_corr: np.ndarray
    corr: np.ndarray
    group_r2: float
    max_corr2: float


def decompose_score(t, groups, Y):
    """Decompose the relation of one score ``t`` (N,) with the annotation table ``Y`` (M, q).

    Correlations computed over individuals equal the group-level correlations
    attenuated by η, and rest on M group means, not on N individuals.
    """
    groups, n = check_groups(groups, len(Y))
    t = np.asarray(t, float)
    if t.ndim != 1:
        raise ValueError("decompose_score takes a single score vector")
    eta2 = between_share(t, groups)
    tbar = group_means(t - t.mean(), groups)
    group_corr = np.atleast_1d(weighted_corr(tbar, np.asarray(Y, float), n))
    group_r2 = weighted_r2(tbar, Y, n)
    return ScoreDecomposition(
        eta2=eta2,
        group_corr=group_corr,
        corr=np.sqrt(eta2) * group_corr,
        group_r2=group_r2,
        max_corr2=eta2 * group_r2,
    )


def group_scores(t, groups, M=None):
    """Group means of a score centred over individuals: the only part of it the annotations see."""
    t = np.asarray(t, float)
    return group_means(t - t.mean(0), groups, M)
