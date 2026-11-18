"""Outcome 14: the refitted lens null of a PLS loading (Result 2b, Section 5).

Under β = 0 we fit PLS and take the congruence of its first annotation
loading with the leading direction of the model lens. If a null is right,
the observed congruence is a uniform draw from it, so its percentile in that
null has mean 0.5 and a uniform distribution. We compare three nulls, each
computed for the table and group sizes at hand with the true variance
components: a single random score (`lens_null` without p), PLS refitted on p
random features (the nominal number), and PLS refitted on as many features as
there are latent dimensions (the effective number). Result 2b predicts that the
single-score null is too diffuse for PLS and that p should be the effective
number of feature dimensions.

Two estimates of the effective number, usable on real data, are checked
against the known one: the participation ratio of the between-group
covariance of the group means (size-weighted, as the `lens_null` docstring
suggests) and that of the pooled within-group covariance, each rounded to
the nearest integer. Their nulls are drawn after the others, so the columns
above reproduce earlier runs.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

import groupunit as gu
from common import Design, run, save

N_DRAWS = 500


def cells():
    grid = [Design(M=20, p=p, feature_rank=r, h=h)
            for h in (1, 10) for p, r in ((10, None), (100, None), (100, 5), (100, 20), (10, 3))]
    return [(d, {}) for d in grid]


def p_eff_estimates(X, groups, n):
    """Participation ratios of the between-group and the pooled within-group feature covariance."""
    Xbar = gu.group_means(X, groups)
    B = np.sqrt(n / n.sum())[:, None] * gu.wcentre(Xbar, n)      # B^T B: size-weighted between covariance
    between = gu.participation_ratio(np.linalg.svd(B, compute_uv=False) ** 2)
    E = X - Xbar[groups]
    within = gu.participation_ratio(np.linalg.svd(E, compute_uv=False) ** 2)
    return between, within


def replicate(d, rng):
    X, groups, Y, n = d.draw(rng)
    loading = gu.pls(X, groups, Y, n_components=1).y_loadings[:, 0]
    lead = gu.lens(Y, n, tau2=d.tau2, sigma2_w=1.0).eigenvectors[:, 0]
    observed = abs(gu.congruence(loading, lead))
    eff = d.p if d.feature_rank is None else d.feature_rank
    row = {"observed congruence": observed}
    for name, p in (("single score", None), ("refit, nominal p", d.p), ("refit, effective p", eff)):
        null = np.abs(gu.lens_null(Y, n, N_DRAWS, tau2=d.tau2, sigma2_w=1.0, rng=rng, p=p) @ lead)
        row[f"{name}: median"] = np.median(null)
        row[f"{name}: percentile"] = (null < observed).mean()
    between, within = p_eff_estimates(X, groups, n.astype(float))
    row["p_eff estimate, between"], row["p_eff estimate, within"] = between, within
    for name, est in (("refit, between PR", between), ("refit, within PR", within)):
        null = np.abs(gu.lens_null(Y, n, N_DRAWS, tau2=d.tau2, sigma2_w=1.0, rng=rng, p=max(1, round(est))) @ lead)
        row[f"{name}: median"] = np.median(null)
        row[f"{name}: percentile"] = (null < observed).mean()
    return row


NULLS = ("single score", "refit, nominal p", "refit, effective p", "refit, between PR", "refit, within PR")


def summarise(df):
    df = df.fillna({"feature_rank": 0})
    keys = ["h", "p", "feature_rank"]
    g = df.groupby(keys)
    out = g[["observed congruence"]].median()
    out[["p_eff estimate, between", "p_eff estimate, within"]] = g[["p_eff estimate, between", "p_eff estimate, within"]].mean()
    for name in NULLS:
        out[f"{name}: median"] = g[f"{name}: median"].median()
        out[f"{name}: mean percentile"] = g[f"{name}: percentile"].mean()
        out[f"{name}: KS p"] = g[f"{name}: percentile"].apply(lambda u: stats.kstest(u, "uniform").pvalue)
    return out


# ---------------------------------------------------------------- group effects in fewer dimensions

# Group effects in k_between latent dimensions, nested in the k_within dimensions of the
# within-group noise, so no single p_eff is known. The null is computed for a grid of p
# to find the p that calibrates it, and for the two estimates; variance components are
# estimated from the data, as in an analysis.
P_GRID = (1, 2, 3, 5, 8, 12, 20, 40)
MIXED = ((5, 20, 100), (5, 100, 100), (20, 20, 100))      # (k_between, k_within, p); the last is a control


def mixed_cells():
    return [(Design(M=20, p=p, h=h), {"k_between": kb, "k_within": kw})
            for h in (1, 10) for kb, kw, p in MIXED]


def draw_mixed(d, kb, kw, rng, truth=False):
    """Like Design.draw under β = 0, with group effects in the first kb of kw latent dimensions.

    With ``truth=True`` also returns the true between- and within-group covariances (Σ_b, Σ_w).
    """
    n = d.sizes()
    Y = d.annotations(rng)
    groups = np.repeat(np.arange(d.M), n)
    A = np.eye(d.p) if kw == d.p else np.linalg.qr(rng.standard_normal((d.p, kw)))[0].T * np.sqrt(d.p / kw)
    U = np.zeros((d.M, kw))
    U[:, :kb] = np.sqrt(d.tau2) * rng.standard_normal((d.M, kb))
    X = (U[groups] + rng.standard_normal((len(groups), kw))) @ A
    if kw < d.p:
        X += 0.1 * rng.standard_normal(X.shape)
    if not truth:
        return X, groups, Y, n
    Sb = d.tau2 * A[:kb].T @ A[:kb]
    Sw = A.T @ A + (0.01 * np.eye(d.p) if kw < d.p else 0)
    return X, groups, Y, n, Sb, Sw


def mixed_replicate(d, rng, k_between, k_within):
    X, groups, Y, n = draw_mixed(d, k_between, k_within, rng)
    vc = gu.pooled_variance_components(X, groups)
    loading = gu.pls(X, groups, Y, n_components=1).y_loadings[:, 0]
    lead = gu.lens(Y, n, tau2=vc.tau2, sigma2_w=vc.sigma2_w).eigenvectors[:, 0]
    observed = abs(gu.congruence(loading, lead))
    between, within = p_eff_estimates(X, groups, n.astype(float))
    row = {"p_eff estimate, between": between, "p_eff estimate, within": within}
    for name, p in [(f"p = {q}", q) for q in P_GRID] + [("between PR", max(1, round(between))),
                                                         ("within PR", max(1, round(within)))]:
        null = np.abs(gu.lens_null(Y, n, N_DRAWS, tau2=vc.tau2, sigma2_w=vc.sigma2_w, rng=rng, p=p) @ lead)
        row[f"{name}: percentile"] = (null < observed).mean()
    return row


def mixed_summarise(df):
    g = df.groupby(["h", "k_between", "k_within", "p"])
    out = g[["p_eff estimate, between", "p_eff estimate, within"]].mean()
    for name in [f"p = {q}" for q in P_GRID] + ["between PR", "within PR"]:
        out[f"{name}: mean percentile"] = g[f"{name}: percentile"].mean()
        out[f"{name}: KS p"] = g[f"{name}: percentile"].apply(lambda u: stats.kstest(u, "uniform").pvalue)
    return out


# ---------------------------------------------------------------- a null without p_eff

def covariance_null(X, groups, Y, n, lead, n_draws, rng, Sb=None, Sw=None):
    """|congruence| with ``lead`` of first PLS loadings refitted on random group means, (n_draws,).

    Each draw gives group m the mean N(0, Σ_b + Σ_w / n_m), with Σ_w the pooled within-group
    covariance and Σ_b the ANOVA moment estimate, (MSB − Σ_w) / n₀, with negative eigenvalues
    set to zero. The draws keep the observed feature covariance, so no number of feature
    dimensions has to be chosen. Pass ``Sb`` and ``Sw`` to use known covariances instead.
    """
    n = n.astype(float)
    M, N = len(n), len(groups)
    if Sb is None:
        Xbar = gu.group_means(X, groups)
        E = X - Xbar[groups]
        Sw = E.T @ E / (N - M)
        D = Xbar - (n @ Xbar) / N
        msb = (n[:, None] * D).T @ D / (M - 1)
        n0 = (N - (n**2).sum() / N) / (M - 1)
        Sb = (msb - Sw) / n0
    lam, V = np.linalg.eigh(Sb)
    Lb = V * np.sqrt(np.clip(lam, 0, None))                   # Σ_b = Lb Lbᵀ
    lam, V = np.linalg.eigh(Sw)
    Lw = V * np.sqrt(np.clip(lam, 0, None))                   # Σ_w = Lw Lwᵀ
    WY = n[:, None] * gu.wcentre(Y, n)
    out = np.empty(n_draws)
    for start in range(0, n_draws, 100):
        k = min(100, n_draws - start)
        T = rng.standard_normal((k, M, Lb.shape[1])) @ Lb.T
        T += (rng.standard_normal((k, M, Lw.shape[1])) @ Lw.T) / np.sqrt(n)[None, :, None]
        C = np.einsum("mq,kmp->kqp", WY, T)                   # Ỹᵀ D X̄ for every draw
        vecs = np.linalg.eigh(C @ C.transpose(0, 2, 1))[1][:, :, -1]
        out[start:start + k] = np.abs(vecs @ lead)
    return out


def permutation_null(X, groups, Y, n, lead, n_draws, rng, sd=None):
    """|congruence| with ``lead`` of first PLS loadings refitted on group means shuffled among groups.

    The annotation table and group sizes stay in place. With ``sd`` (each group mean's standard
    deviation), the centred means are divided by their own sd before the shuffle and multiplied by
    the receiving group's sd after it, so that each group keeps its own noise level.
    """
    n = n.astype(float)
    Xbar = gu.group_means(X, groups)
    Z = gu.wcentre(Xbar, n)
    sd = np.ones(len(n)) if sd is None else sd
    Z = Z / sd[:, None]
    WY = n[:, None] * gu.wcentre(Y, n)
    perms = np.array([rng.permutation(len(n)) for _ in range(n_draws)])
    out = np.empty(n_draws)
    for start in range(0, n_draws, 100):
        P = perms[start:start + 100]
        T = Z[P] * sd[None, :, None]
        C = np.einsum("mq,kmp->kqp", WY, T)
        vecs = np.linalg.eigh(C @ C.transpose(0, 2, 1))[1][:, :, -1]
        out[start:start + len(P)] = np.abs(vecs @ lead)
    return out


def cov_cells():
    """All sixteen designs as (k_between, k_within): the ten of the first part, then the six mixed ones."""
    out = []
    for h in (1, 10):
        for p, r in ((10, None), (100, None), (100, 5), (100, 20), (10, 3)):
            k = p if r is None else r
            out.append((Design(M=20, p=p, h=h), {"k_between": k, "k_within": k}))
    return out + mixed_cells()


def equal_cells():
    """The sixteen designs with equal group sizes, where shuffling group means is exact."""
    return [(replace(d, kish=None), extra) for d, extra in cov_cells()]


def cov_replicate(d, rng, k_between, k_within):
    X, groups, Y, n, Sb, Sw = draw_mixed(d, k_between, k_within, rng, truth=True)
    vc = gu.pooled_variance_components(X, groups)
    loading = gu.pls(X, groups, Y, n_components=1).y_loadings[:, 0]
    lead = gu.lens(Y, n, tau2=vc.tau2, sigma2_w=vc.sigma2_w).eigenvectors[:, 0]
    observed = abs(gu.congruence(loading, lead))
    # Σ_b estimates: the ANOVA moment estimate, the within-group shape scaled to the pooled τ²
    # (exact if group effects vary like individuals), and the average of the two
    N, M = len(groups), d.M
    Xbar = gu.group_means(X, groups)
    E = X - Xbar[groups]
    Sw_hat = E.T @ E / (N - M)
    Dm = Xbar - (n @ Xbar) / N
    n0 = (N - (n**2).sum() / N) / (M - 1)
    Sb_mom = ((n[:, None] * Dm).T @ Dm / (M - 1) - Sw_hat) / n0
    Sb_shape = vc.tau2 * d.p / np.trace(Sw_hat) * Sw_hat
    lam, V = np.linalg.eigh(Sb_mom)
    Sb_avg = 0.5 * (V * np.clip(lam, 0, None)) @ V.T + 0.5 * Sb_shape
    row = {}
    for name, cov in (("estimated", {}), ("true", {"Sb": Sb, "Sw": Sw}),
                      ("within-shaped", {"Sb": Sb_shape, "Sw": Sw_hat}), ("averaged", {"Sb": Sb_avg, "Sw": Sw_hat})):
        null = covariance_null(X, groups, Y, n, lead, N_DRAWS, rng, **cov)
        row[f"{name} covariances: percentile"] = (null < observed).mean()
    sd = np.sqrt(vc.tau2 + vc.sigma2_w / n)
    for name, s_ in (("permuted means, raw", None), ("permuted means, standardised", sd)):
        null = permutation_null(X, groups, Y, n, lead, N_DRAWS, rng, s_)
        row[f"{name}: percentile"] = (null < observed).mean()
    return row


def cov_summarise(df):
    g = df.groupby(["h", "p", "k_between", "k_within"])
    out = pd.DataFrame(index=g.size().index)
    names = [f"{v} covariances" for v in ("estimated", "true", "within-shaped", "averaged")]
    for name in names + ["permuted means, raw", "permuted means, standardised"]:
        out[f"{name}: mean percentile"] = g[f"{name}: percentile"].mean()
        out[f"{name}: KS p"] = g[f"{name}: percentile"].apply(lambda u: stats.kstest(u, "uniform").pvalue)
    return out


def main(reps=300, n_perm=None, seed=15):
    df = run(replicate, cells(), reps, seed)
    save(df, summarise(df), "s14_lens_refit")
    df = run(mixed_replicate, mixed_cells(), reps, seed + 100)
    save(df, mixed_summarise(df), "s14_lens_mixed")
    df = run(cov_replicate, cov_cells(), reps, seed + 200)
    save(df, cov_summarise(df), "s14_lens_covariance")
    df = run(cov_replicate, equal_cells(), reps, seed + 300)
    save(df, cov_summarise(df), "s14_lens_equal")


if __name__ == "__main__":
    main()
