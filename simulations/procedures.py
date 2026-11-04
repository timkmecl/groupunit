"""Test procedures compared in the simulations, including the invalid ones.

The invalid procedures (resampling individuals, or permuting groups without
refitting) are deliberately kept out of the package. Each function returns a
p-value for the largest statistic of its family.
"""
import numpy as np
from scipy import stats

import groupunit as gu
from groupunit.inference import draw_permutations, p_value


# ---------------------------------------------------------------- correlation of fixed scores

def corr_individuals(scores, G, Y, n, n_perm, rng):
    """Max |corr| of fixed scores with annotations, annotations reassigned among individuals."""
    T = scores - scores.mean(0)
    Yc = gu.wcentre(Y, n)
    scale = np.outer(np.linalg.norm(T, axis=0), np.sqrt(n @ Yc**2))

    def stat(Tp):
        return np.abs((G.T @ Tp).T @ Yc / scale).max()

    null = [stat(T[rng.permutation(len(T))]) for _ in range(n_perm)]
    return p_value(stat(T), null)


def corr_groups(scores, groups, Y, n_perm, rng):
    """The valid test with size weights: max-T over cells, annotation rows permuted among groups."""
    return gu.correlation_test(scores, groups, Y, n_perm, rng, weights="size").p_fwer.min()


def corr_parametric(scores, groups, Y):
    """Per-cell Pearson test with N - 2 degrees of freedom, Bonferroni over cells."""
    Yi = Y[groups]
    r = np.array([[np.corrcoef(t, y)[0, 1] for y in Yi.T] for t in np.atleast_2d(scores.T)])
    N = len(groups)
    t = np.abs(r) * np.sqrt((N - 2) / (1 - r**2))
    return min(1.0, 2 * stats.t.sf(t.max(), N - 2) * r.size)


# ---------------------------------------------------------------- PLS, first singular value

def pls_individuals(X, G, Y, n, n_perm, rng):
    Xc = X - X.mean(0)
    Yc = gu.wcentre(Y, n)

    def stat(Xp):
        return np.linalg.norm((G.T @ Xp).T @ Yc, 2)   # Xᵀ Y[groups] via group sums

    null = [stat(Xc[rng.permutation(len(Xc))]) for _ in range(n_perm)]
    return p_value(stat(Xc), null)


def pls_groups_fixed(X, groups, Y, n, n_perm, rng):
    """Group permutation, the real first component held fixed (not refitted)."""
    fit = gu.pls(X, groups, Y, n_components=1)
    nt = n * fit.group_scores[:, 0]
    null = [np.linalg.norm(gu.wcentre(Y[perm], n).T @ nt) for perm in draw_permutations(len(Y), n_perm, rng)]
    return p_value(fit.strength[0], null)


def pls_groups_refit(X, groups, Y, n_perm, rng):
    return gu.pls_test(X, groups, Y, n_components=1, n_perm=n_perm, rng=rng, weights="size").p_value[0]


# ---------------------------------------------------------------- CCA, first canonical correlation

def _x_basis(X):
    U, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return U[:, s > s.max() * 1e-10]


def cca_individuals(X, G, Y, n, n_perm, rng):
    Ux = _x_basis(X)
    Qy = gu.weighted_basis(Y, n) / np.sqrt(n)[:, None]

    def stat(U):
        return np.linalg.norm((G.T @ U).T @ Qy, 2)

    null = [stat(Ux[rng.permutation(len(Ux))]) for _ in range(n_perm)]
    return p_value(stat(Ux), null)


def cca_groups_fixed(X, groups, Y, n, n_perm, rng):
    """Group permutation, the real first canonical score held fixed: ρ = η · R_n(t̄ ~ Ỹ_perm)."""
    fit = gu.cca(X, groups, Y, n_components=1)
    t = fit.scores(X)[:, 0]
    eta2 = gu.between_share(t, groups)
    tbar = gu.group_scores(t, groups)
    null = [np.sqrt(eta2 * gu.weighted_r2(tbar, Y[perm], n)) for perm in draw_permutations(len(Y), n_perm, rng)]
    return p_value(fit.strength[0], null)


def cca_groups_refit(X, groups, Y, n_perm, rng):
    try:
        return gu.cca_test(X, groups, Y, n_components=1, n_perm=n_perm, rng=rng).p_value[0]
    except ValueError:      # saturated: every permutation equals the observed result
        return 1.0


# ---------------------------------------------------------------- similarity

def mantel_pairs(S1, S2):
    """Pearson test over the M(M-1)/2 pairs as if they were independent (one-sided)."""
    iu = np.triu_indices(len(S1), 1)
    res = stats.pearsonr(S1[iu], S2[iu], alternative="greater")
    return res.pvalue


def mantel_groups(S1, S2, n_perm, rng):
    iu = np.triu_indices(len(S1), 1)
    stat = lambda perm: np.corrcoef(S1[iu], S2[np.ix_(perm, perm)][iu])[0, 1]
    return p_value(stat(np.arange(len(S1))), [stat(p) for p in draw_permutations(len(S1), n_perm, rng)])
