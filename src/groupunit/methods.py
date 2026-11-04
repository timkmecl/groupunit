"""PLS correlation, CCA and LDA, written through the group means (Sections 3-4).

These are plain reference implementations. They exist so that the tests in
`groupunit.inference` can refit them under permutation, and so that each
result can be checked against the identities of the paper. Every fit returns
a `Components` object with the same fields.
"""
from dataclasses import dataclass

import numpy as np

from .core import check_groups, group_means, wcentre, weighted_basis


@dataclass(frozen=True)
class Components:
    """Paired feature-side and annotation-side components.

    Attributes
    ----------
    strength : (k,) singular values (PLS) or canonical correlations (CCA).
    x_weights : (p, k) feature weights; scores are ``(X - x_mean) @ x_weights``.
    y_weights : (q, k) annotation weights. Not unique when q > r (Result 3e);
        the minimum-norm solution is returned.
    group_scores : (M, k) group means of the feature-side scores, t̄.
    y_loadings : (q, k) covariances of the annotations with the feature-side
        scores, ``Ỹᵀ D t̄ / N``. Unique, and a fixed function of ``group_scores``.
    x_mean : (p,) feature means used for centring.
    """

    strength: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray
    group_scores: np.ndarray
    y_loadings: np.ndarray
    x_mean: np.ndarray

    def scores(self, X):
        """Feature-side scores of individuals, ``(N, k)``."""
        return (np.asarray(X, float) - self.x_mean) @ self.x_weights


def _fix_signs(*arrays, ref):
    """Flip each component so that its largest annotation loading is positive."""
    signs = np.sign(ref[np.argmax(np.abs(ref), axis=0), np.arange(ref.shape[1])])
    signs[signs == 0] = 1
    return [a * signs for a in arrays]


def _rank_cutoff(s):
    return s > s.max(initial=0) * max(s.shape[0], 1) * np.finfo(float).eps * 10


def pls(X, groups, Y, n_components=None, weights=None):
    """PLS correlation (PLS-SVD) of individual features with group-level annotations.

    The weights are the singular vectors of ``XᵀY = X̄ᵀ D Ỹ`` (Result 2), so they
    depend on the group means, the annotations and the group sizes only. Pass
    ``weights`` (one per group, e.g. ``np.ones(M)``) to target the population of
    groups instead of individuals (Section 5). The first component equals the
    first component of PLS regression.
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    w = n.astype(float) if weights is None else np.asarray(weights, float)
    Xbar = group_means(X, groups, len(Y))
    x_mean = w @ Xbar / w.sum()
    Xb, Yc = Xbar - x_mean, wcentre(Y, w)

    U, s, Vt = np.linalg.svd(Xb.T @ (w[:, None] * Yc), full_matrices=False)
    k = int(_rank_cutoff(s).sum())
    if n_components is not None:
        k = min(k, n_components)
    W, V, s = U[:, :k], Vt[:k].T, s[:k]
    T = Xb @ W
    loadings = Yc.T @ (w[:, None] * T) / w.sum()
    W, V, T, loadings = _fix_signs(W, V, T, loadings, ref=loadings)
    return Components(s, W, V, T, loadings, x_mean)


def cca(X, groups, Y, n_components=None):
    """Canonical correlation analysis of individual features with group-level annotations.

    Canonical correlations are the cosines of the principal angles between the
    column spaces of X and of the inherited annotations ``Y[groups]``. The
    annotation space is built from the group-level table, so the fit never
    forms the N × q matrix. If the annotations have rank M - 1, the result is
    the same for every annotation table of that rank and equals `lda` (Result 3c).
    Feature scores are scaled to unit variance over individuals.
    """
    Y = np.asarray(Y, float)
    groups, n = check_groups(groups, len(Y))
    X = np.asarray(X, float)
    N = len(groups)
    x_mean = X.mean(0)

    # orthonormal basis of the feature space
    Ux, sx, Vxt = np.linalg.svd(X - x_mean, full_matrices=False)
    keep = _rank_cutoff(sx)
    Ux, sx, Vxt = Ux[:, keep], sx[keep], Vxt[keep]
    # orthonormal basis of the annotation space is G D^{-1/2} Q, so Uxᵀ(G D^{-1/2} Q) = Ūxᵀ D^{1/2} Q
    Q = weighted_basis(Y, n)
    C = group_means(Ux, groups, len(Y)).T @ (np.sqrt(n)[:, None] * Q)
    A, rho, Bt = np.linalg.svd(C, full_matrices=False)

    k = len(rho) if n_components is None else min(len(rho), n_components)
    A, B, rho = A[:, :k], Bt[:k].T, np.clip(rho[:k], 0, 1)
    x_weights = Vxt.T @ (A / sx[:, None]) * np.sqrt(N)
    T = group_means(Ux @ A, groups, len(Y)) * np.sqrt(N)
    Yc = wcentre(Y, n)
    # minimum-norm annotation weights b with D^{1/2} Ỹ b = Q B (unit-variance annotation scores)
    y_weights = np.linalg.pinv(np.sqrt(n)[:, None] * Yc) @ Q @ B * np.sqrt(N)
    loadings = Yc.T @ (n[:, None] * T) / N
    x_weights, y_weights, T, loadings = _fix_signs(x_weights, y_weights, T, loadings, ref=loadings)
    return Components(rho, x_weights, y_weights, T, loadings, x_mean)


def lda(X, groups, n_components=None):
    """Linear discriminant analysis of individuals by group, as CCA with group indicators.

    The squared canonical correlations are the between-group shares η² of the
    discriminant scores. With p large relative to M or N, groups separate in
    sample even when they do not differ; estimate η² on held-out individuals.
    """
    groups, n = check_groups(groups)
    return cca(X, groups, np.eye(len(n)), n_components)
