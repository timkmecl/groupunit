"""Diagnostics computable from group means, annotations and group sizes (Section 6).

All functions here work on group-level inputs, so they also apply to
published results: digitised group means of a component, the annotation
table and the group sizes are enough.
"""
from dataclasses import dataclass

import numpy as np

from .core import wcentre


def effective_number(w, axis=0):
    """(Σ|w|)² / Σw²: the number of equal weights with the same concentration (Section 6.2).

    With group sizes it is Kish's effective number of groups, with eigenvalues
    the participation ratio, and with group shares the effective number of
    groups behind a component.
    """
    w = np.abs(np.asarray(w, float))
    return w.sum(axis) ** 2 / (w**2).sum(axis)


def participation_ratio(A):
    """Participation ratio ν of a symmetric matrix, or of a vector of its eigenvalues."""
    A = np.asarray(A, float)
    eigenvalues = np.linalg.eigvalsh(A) if A.ndim == 2 else A
    return float(effective_number(np.clip(eigenvalues, 0, None)))


def congruence(a, b):
    """Tucker's congruence coefficient, the cosine between loading vectors (Section 6.3).

    For two vectors returns a float; for matrices, the congruences between
    their columns.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a2 = a.reshape(len(a), -1)
    b2 = b.reshape(len(b), -1)
    c = (a2 / np.linalg.norm(a2, axis=0)).T @ (b2 / np.linalg.norm(b2, axis=0))
    if a.ndim == 1 and b.ndim == 1:
        return float(c[0, 0])
    return c[0] if a.ndim == 1 else c[:, 0] if b.ndim == 1 else c


def contributions(tbar, Y, w):
    """Group × annotation contribution matrix with entries w_m t̄_m ỹ_mj (Section 6.1).

    Column sums are the (unnormalised) annotation loadings of the component.
    """
    tbar = wcentre(np.asarray(tbar, float), w)
    if tbar.ndim != 1:
        raise ValueError("contributions takes the group means of a single component")
    return (np.asarray(w, float) * tbar)[:, None] * wcentre(Y, w)


@dataclass(frozen=True)
class GroupShares:
    """Additive group shares of a component's annotation loading.

    Shapes are (M,) for one component, (M, k) for k components.

    Attributes
    ----------
    loading : ℓ = Σ_m w_m t̄_m ỹ_m, shape (q,) or (q, k).
    terms : s_m, each group's term projected onto the unit loading; sums to ‖ℓ‖.
    share : s_m / Σ s; sums to one, and equals the groups' Shapley values for ‖ℓ‖².
        Negative for groups working against the component.
    abs_share : |s_m| / Σ|s|, never negative.
    m_eff : effective number of groups behind the component (descriptive).
    """

    loading: np.ndarray
    terms: np.ndarray
    share: np.ndarray
    abs_share: np.ndarray
    m_eff: float | np.ndarray

    def top(self, k=3):
        """Indices of the k groups with the largest absolute shares (first component if several)."""
        a = self.abs_share if self.abs_share.ndim == 1 else self.abs_share[:, 0]
        return np.argsort(-a)[:k]


def group_shares(tbar, Y, w):
    """Group shares of the annotation loading of one or more components (Sections 6.1-6.2).

    ``tbar`` holds each group's mean score, (M,) or (M, k); ``w`` are the group
    weights used by the analysis (group sizes for analyses of individuals).
    Both ``tbar`` and ``Y`` are centred with ``w``, which shares require.
    """
    w = np.asarray(w, float)
    tbar = np.asarray(tbar, float)
    T = wcentre(tbar.reshape(len(tbar), -1), w)
    Yc = wcentre(Y, w)
    wT = w[:, None] * T
    loading = Yc.T @ wT
    unit = loading / np.linalg.norm(loading, axis=0)
    terms = wT * (Yc @ unit)
    share = terms / terms.sum(0)
    abs_share = np.abs(terms) / np.abs(terms).sum(0)
    m_eff = effective_number(terms)
    if tbar.ndim == 1:
        return GroupShares(loading[:, 0], terms[:, 0], share[:, 0], abs_share[:, 0], float(m_eff[0]))
    return GroupShares(loading, terms, share, abs_share, m_eff)


@dataclass(frozen=True)
class LeaveOut:
    """A component's loading with some groups' terms removed, the component held fixed.

    Attributes
    ----------
    dropped : indices of the removed groups.
    loading : (q,) the loading recomputed over the remaining groups.
    congruence : Tucker's congruence of the remaining loading with the full one.
    norm_ratio : ‖remaining loading‖ / ‖full loading‖.
    """

    dropped: np.ndarray
    loading: np.ndarray
    congruence: float
    norm_ratio: float


def leave_out(tbar, Y, w, drop):
    """A component's loading without groups ``drop``, the component held fixed (Section 6.4).

    The other groups' means are kept unchanged, and the loading is the one the
    data without the dropped groups would give, re-centred over the remaining
    groups. With group terms u_m (see `contributions`), dropping one group gives

        ℓ₋m = ℓ − u_m · W / (W − w_m),

    so a group removes more than its own term, because the centroid moves away
    from it, and the loading reverses if the group's share exceeds (W − w_m) / W.
    For several groups S, ℓ₋S = ℓ − Σ_S u_m − (Σ_S w_m t̄_m)(Σ_S w_m ỹ_m) / (W − W_S).
    To refit instead, rerun the method without those groups. Choose ``drop``
    with `GroupShares.top`.
    """
    w = np.asarray(w, float)
    t = wcentre(np.asarray(tbar, float), w)
    Yc = wcentre(Y, w)
    terms = (w * t)[:, None] * Yc
    full = terms.sum(0)
    S = np.atleast_1d(np.asarray(drop, int))
    W_rest = w.sum() - w[S].sum()
    rest = full - terms[S].sum(0) - (w[S] @ t[S]) * (w[S] @ Yc[S]) / W_rest
    return LeaveOut(
        dropped=S,
        loading=rest,
        congruence=congruence(rest, full),
        norm_ratio=float(np.linalg.norm(rest) / np.linalg.norm(full)),
    )


def _group_mean_cov(n, tau2, sigma2_w):
    """Diagonal of K, the covariance of random group means (identity if no components given)."""
    n = np.asarray(n, float)
    if tau2 is None and sigma2_w is None:
        return np.ones_like(n)
    if tau2 is None or sigma2_w is None:
        raise ValueError("give both tau2 and sigma2_w, or neither")
    return tau2 + sigma2_w / n


@dataclass(frozen=True)
class Lens:
    """The lens Λ = Ỹᵀ W K W Ỹ through which loadings see random group means (Section 4.2).

    Attributes
    ----------
    matrix : (q, q) the lens matrix.
    eigenvalues : (q,) in decreasing order.
    eigenvectors : (q, q) matching columns; the leading ones are where loadings concentrate.
    nu : participation ratio, the effective number of annotation directions.
    """

    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    nu: float

    @property
    def cumulative_share(self):
        """Cumulative share of the lens held by its leading directions."""
        return np.cumsum(self.eigenvalues) / self.eigenvalues.sum()


def lens(Y, n, weights=None, tau2=None, sigma2_w=None):
    """The lens of a design, from the annotation table and group sizes (Sections 4.2, 6.3).

    By default group means are taken as independent with equal variance (K = I),
    which gives Λ = Ỹᵀ D² Ỹ for size weights. Pass ``tau2`` and ``sigma2_w``
    (see `groupunit.weights.variance_components`) to use K = τ²I + σ²_w D⁻¹.
    ``weights`` are the group weights of the analysis (default: group sizes).
    """
    n = np.asarray(n, float)
    w = n if weights is None else np.asarray(weights, float)
    Yc = wcentre(Y, w)
    k = _group_mean_cov(n, tau2, sigma2_w)
    WY = w[:, None] * Yc
    L = WY.T @ (k[:, None] * WY)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    return Lens(L, eigenvalues, eigenvectors[:, order], participation_ratio(eigenvalues))


def lens_null(Y, n, n_draws=5000, weights=None, tau2=None, sigma2_w=None, rng=None):
    """Unit-length loadings produced by random group means through the lens, (n_draws, q).

    These are the loadings a design yields with no information from the
    features; compare published loadings against them (Section 6.3).
    """
    rng = np.random.default_rng(rng)
    n = np.asarray(n, float)
    w = n if weights is None else np.asarray(weights, float)
    Yc = wcentre(Y, w)
    sd = np.sqrt(_group_mean_cov(n, tau2, sigma2_w))
    T = wcentre(rng.standard_normal((len(n), n_draws)) * sd[:, None], w)
    L = Yc.T @ (w[:, None] * T)
    return (L / np.linalg.norm(L, axis=0)).T


def lens_congruence_null(loading, Y, n, n_draws=5000, weights=None, tau2=None, sigma2_w=None, rng=None):
    """Absolute congruence of a loading with random-group-mean loadings, (n_draws,).

    A loading, or the agreement between two analyses of the same groups, is
    informative only to the extent that it exceeds this baseline.
    """
    null = lens_null(Y, n, n_draws, weights, tau2, sigma2_w, rng)
    loading = np.asarray(loading, float)
    return np.abs(null @ (loading / np.linalg.norm(loading)))
