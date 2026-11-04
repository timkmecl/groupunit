"""Basic quantities: group codes, group means, weighted centring and weighted fits.

Conventions used throughout the package (Section 2 of the paper):

    X       (N, p)  features of individuals
    groups  (N,)    integer group codes 0..M-1 (see `encode_groups`)
    Y       (M, q)  annotation table, one row per group (the paper's Ỹ)
    n       (M,)    group sizes
    w       (M,)    group weights; size weights w = n unless stated otherwise
    tbar    (M,) or (M, k)  group means of one or more scores

Functions centre their inputs themselves (X over individuals, group-level
quantities with the group weights), so raw values can be passed.
"""
import numpy as np


def encode_groups(labels, order=None):
    """Turn arbitrary group labels into integer codes 0..M-1.

    Returns ``(codes, names)``, where ``names[m]`` is the label of group m.
    Pass ``order`` (the labels in the row order of your annotation table) so
    that code m refers to row m of that table.
    """
    labels = np.asarray(labels)
    if order is None:
        names, codes = np.unique(labels, return_inverse=True)
        return codes, names
    names = np.asarray(order)
    index = {label: m for m, label in enumerate(names.tolist())}
    try:
        codes = np.array([index[label] for label in labels.tolist()], dtype=int)
    except KeyError as err:
        raise ValueError(f"label {err.args[0]!r} is not in `order`") from None
    return codes, names


def check_groups(groups, M=None):
    """Validate integer group codes and return ``(groups, n)``.

    ``M`` is the number of groups (rows of the annotation table); every group
    must contain at least one individual.
    """
    groups = np.asarray(groups)
    if groups.ndim != 1 or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("groups must be a 1-D array of integer codes; use encode_groups() for labels")
    if groups.size == 0 or groups.min() < 0:
        raise ValueError("group codes must be non-negative integers")
    M = groups.max() + 1 if M is None else M
    if groups.max() >= M:
        raise ValueError(f"group code {groups.max()} is out of range for {M} groups")
    n = np.bincount(groups, minlength=M)
    if (n == 0).any():
        raise ValueError(f"groups {np.flatnonzero(n == 0).tolist()} have no individuals")
    return groups, n


def group_sizes(groups, M=None):
    """Number of individuals in each group, ``n``."""
    return check_groups(groups, M)[1]


def group_means(X, groups, M=None):
    """Group means of a vector or of the columns of a matrix: ``(M,)`` or ``(M, p)``."""
    groups, n = check_groups(groups, M)
    X = np.asarray(X, float)
    sums = np.zeros((len(n),) + X.shape[1:])
    np.add.at(sums, groups, X)
    return sums / n.reshape((-1,) + (1,) * (X.ndim - 1))


def wcentre(A, w):
    """Centre the rows of ``A`` (vector or matrix) on their ``w``-weighted mean."""
    A = np.asarray(A, float)
    w = np.asarray(w, float)
    return A - np.tensordot(w, A, axes=1) / w.sum()


def weighted_corr(a, b, w):
    """Weighted correlation across groups, r_w(a, b) (Section 2.4).

    ``a`` and ``b`` are vectors or matrices with one row per group. Returns a
    float for two vectors, otherwise the matrix of correlations between the
    columns of ``a`` (rows of the result) and those of ``b``.
    """
    w = np.asarray(w, float)
    a, b = wcentre(a, w), wcentre(b, w)
    a2, b2 = a.reshape(len(a), -1), b.reshape(len(b), -1)
    cov = a2.T @ (w[:, None] * b2)
    sd_a = np.sqrt(w @ a2**2)
    sd_b = np.sqrt(w @ b2**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.outer(sd_a, sd_b)
    if a.ndim == 1 and b.ndim == 1:
        return float(r[0, 0])
    if a.ndim == 1:
        return r[0]
    if b.ndim == 1:
        return r[:, 0]
    return r


def weighted_basis(A, w):
    """Orthonormal basis of the column space of ``diag(sqrt(w)) @ wcentre(A, w)``.

    Its number of columns is the rank of the weighted-centred table, which for
    an annotation table is r (Section 2.2).
    """
    A = np.asarray(A, float)
    A = A.reshape(len(A), -1)
    w = np.asarray(w, float)
    Z = np.sqrt(w)[:, None] * wcentre(A, w)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    tol = s.max() * max(Z.shape) * np.finfo(float).eps
    return U[:, s > tol]


def annotation_rank(Y, w):
    """Rank r of the annotation table after weighted centring; r <= min(q, M - 1)."""
    return weighted_basis(Y, w).shape[1]


def weighted_r2(tbar, Y, w):
    """R²_w of a weighted regression of group means on the annotations (Result 3a).

    Returns a float for a vector ``tbar``, or one value per column of a matrix.
    """
    w = np.asarray(w, float)
    tbar = np.asarray(tbar, float)
    z = np.sqrt(w).reshape((-1,) + (1,) * (tbar.ndim - 1)) * wcentre(tbar, w)
    Q = weighted_basis(Y, w)
    explained = ((Q.T @ z) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = explained / (z**2).sum(0)
    return float(r2) if tbar.ndim == 1 else r2


def scatter_matrices(X, groups):
    """Total, between-group and within-group sums of squares and cross-products.

    Returns ``(S_T, S_B, S_W)`` with ``S_T = S_B + S_W`` (Section 2.3).
    """
    groups, n = check_groups(groups)
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    Xbar = group_means(Xc, groups)
    E = Xc - Xbar[groups]
    return Xc.T @ Xc, Xbar.T @ (n[:, None] * Xbar), E.T @ E
