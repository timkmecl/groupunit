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


# ---------------------------------------------------------------- comparators for one score fixed in advance

def _group_summaries(t, groups, M):
    n = np.bincount(groups, minlength=M).astype(float)
    tbar = np.bincount(groups, weights=t, minlength=M) / n
    within_ss = ((t - tbar[groups]) ** 2).sum()
    return n, tbar, within_ss


def mixed_model(t, groups, Y):
    """Random-intercept model t_i = z_mᵀβ + u_m + e_i fitted by REML, and two tests of the annotations.

    Because the annotations are constant within groups, the REML likelihood of the
    individuals factorises exactly into that of the group means, with variances
    v_m = τ² + σ²/n_m, and that of the within-group sum of squares, σ² χ²(N − M).
    Returns the Wald test of the q annotation coefficients against χ²(q), which is
    what mixed-model software reports by default, the same statistic divided by q
    against F(q, M − q − 1), the between-groups ("containment") degrees of freedom,
    and whether τ² was estimated at zero.
    """
    from scipy.optimize import minimize

    M = len(Y)
    n, tbar, wss = _group_summaries(t, groups, M)
    N = n.sum()
    Z = np.column_stack([np.ones(M), Y])
    k = Z.shape[1]

    def gls(v):
        Zw = Z / v[:, None]
        cov = np.linalg.pinv(Z.T @ Zw)
        beta = cov @ (Zw.T @ tbar)
        return beta, cov

    def neg_reml(theta):
        log_s2, tau2 = theta
        s2 = np.exp(log_s2)
        v = tau2 + s2 / n
        beta, cov = gls(v)
        r = tbar - Z @ beta
        _, logdet = np.linalg.slogdet(np.linalg.pinv(cov))
        return 0.5 * ((N - M) * log_s2 + wss / s2 + np.log(v).sum() + logdet + (r**2 / v).sum())

    s2_0 = wss / (N - M)
    tau2_0 = max(tbar.var() - s2_0 * (1 / n).mean(), 1e-3 * s2_0)
    fit = minimize(neg_reml, [np.log(s2_0), tau2_0], method="L-BFGS-B",
                   bounds=[(None, None), (0.0, None)])
    log_s2, tau2 = fit.x
    beta, cov = gls(tau2 + np.exp(log_s2) / n)
    b, C = beta[1:], cov[1:, 1:]
    wald = float(b @ np.linalg.pinv(C) @ b)
    q = k - 1
    df2 = M - k
    return {"wald": stats.chi2.sf(wald, q),
            "F": stats.f.sf(wald / q, q, df2) if df2 > 0 else 1.0,
            "tau2 zero": tau2 <= 1e-8 * np.exp(log_s2),
            "tau2": tau2, "sigma2": np.exp(log_s2)}


def ols_individuals(t, groups, Y):
    """OLS F test of the inherited annotations with N − q − 1 residual degrees of freedom."""
    M = len(Y)
    n, tbar, wss = _group_summaries(t, groups, M)
    Z = np.column_stack([np.ones(M), Y])
    beta = np.linalg.lstsq(np.sqrt(n)[:, None] * Z, np.sqrt(n) * tbar, rcond=None)[0]
    rss = wss + (n * (tbar - Z @ beta) ** 2).sum()
    tss = ((t - t.mean()) ** 2).sum()
    q, df2 = Z.shape[1] - 1, n.sum() - Z.shape[1]
    return stats.f.sf((tss - rss) / q / (rss / df2), q, df2)


def wild_cluster_bootstrap(t, groups, Y, n_boot, rng):
    """Wild cluster restricted bootstrap of the cluster-robust Wald test of the annotations.

    OLS of t on the inherited annotations, CR1 cluster-robust covariance by group,
    Rademacher weights per group applied to residuals from the null model
    (intercept only), as recommended for few clusters (Cameron, Gelbach and
    Miller 2008). With annotations constant within groups only the group sums of
    residuals enter, so everything is computed at group level.
    """
    M = len(Y)
    n, tbar, _ = _group_summaries(t, groups, M)
    N = n.sum()
    Z = np.column_stack([np.ones(M), Y])
    k = Z.shape[1]
    bread = np.linalg.pinv(Z.T @ (n[:, None] * Z))
    c = M / (M - 1) * (N - 1) / (N - k)

    def wald(tb):
        beta = bread @ (Z.T @ (n * tb))
        s = Z * (n * (tb - Z @ beta))[:, None]      # cluster scores z_m Σ_i e_i
        V = c * bread @ (s.T @ s) @ bread
        b = beta[1:]
        return b @ np.linalg.pinv(V[1:, 1:]) @ b

    observed = wald(tbar)
    alpha = (n @ tbar) / N                           # restricted (null) fit: the grand mean
    signs = rng.choice([-1.0, 1.0], size=(n_boot, M))
    null = [wald(alpha + s * (tbar - alpha)) for s in signs]
    return p_value(observed, null)
