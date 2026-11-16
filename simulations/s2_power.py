"""Outcome 2: power of the valid tests under β ≠ 0 at realistic numbers of groups.

Annotation 1 drives features 1-2 (group-level R² = 0.3 or 0.6). Every cell is
run twice: under the standard design of Section 6.1, and with a strong
within-group direction unrelated to groups (a "length" direction, variance 3
along the mean of the 10 features) that dominates the first principal
component. Each test is run with size weights and with exchangeable weights
(estimated from the data); the Mantel test uses equal weights.

A second part fixes the population (τ² and the relation β) and varies the
mean number of individuals per group with the number of groups: more
individuals sharpen each group mean, but add no observations of the
annotations, so power should level off in n̄ and keep rising with M.
"""
import numpy as np

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, run, save

TESTS = ["corr: PCA scores, size", "corr: LDA scores, size", "PLS: refit, size", "CCA: refit, size",
         "corr: PCA scores, exchangeable", "corr: LDA scores, exchangeable", "PLS: refit, exchangeable",
         "CCA: refit, exchangeable", "Mantel: groups"]


def cells(n_perm):
    grid = [Design(M=M, kappa=k, h=1, signal=s, length_var=lv)
            for lv in (3.0, 0.0) for s in (0.3, 0.6) for M in (20, 50) for k in (0.1, 0.5, 0.9, 1.0)]
    return [(d, {"n_perm": n_perm}) for d in grid]


def cca_exchangeable(X, groups, Y, n_perm, rng):
    """Row-weighted CCA with exchangeable weights (as in outcome 15); p = 1 when saturated."""
    try:
        return gu.cca_test(X, groups, Y, 1, n_perm, rng, weights="exchangeable").p_value[0]
    except ValueError:
        return 1.0


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    Xc = X - X.mean(0)
    pca = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
    lda = gu.lda(X, groups, n_components=3).scores(X)
    S_features = gu.distance_similarity(gu.group_means(X, groups))
    p = [
        pr.corr_groups(pca, groups, Y, n_perm, rng),
        pr.corr_groups(lda, groups, Y, n_perm, rng),
        pr.pls_groups_refit(X, groups, Y, n_perm, rng),
        pr.cca_groups_refit(X, groups, Y, n_perm, rng),
        gu.correlation_test(pca, groups, Y, n_perm, rng).p_fwer.min(),
        gu.correlation_test(lda, groups, Y, n_perm, rng).p_fwer.min(),
        gu.pls_test(X, groups, Y, n_components=1, n_perm=n_perm, rng=rng).p_value[0],
        cca_exchangeable(X, groups, Y, n_perm, rng),
        pr.mantel_groups(S_features, gu.distance_similarity(Y), n_perm, rng),
    ]
    eta2 = {"eta2 PC1": gu.between_share(pca[:, 0], groups), "eta2 LD1": gu.between_share(lda[:, 0], groups)}
    return {**eta2, **dict(zip(TESTS, p))}


def summarise(df):
    keys = [df.length_var, df.signal, df.M, df.kappa]
    rates = (df[TESTS] < ALPHA).groupby(keys).mean()
    return df[["eta2 PC1", "eta2 LD1"]].groupby(keys).mean().join(rates)


# ---------------------------------------------------------------- power against group size

TAU2 = 0.02        # the reference scenario's τ² (h = 1 at n̄ = 50)
R2_POP = 0.5       # group-level R² of the population means, fixed across n̄
SIZE_TESTS = ["PLS: exchangeable", "PLS: size weights", "CCA: refit"]


def size_cells(n_perm):
    grid = []
    for M in (10, 20, 50, 100):
        for nbar in (10, 25, 50, 100, 250, 1000):
            # Design sets β from the R² of the observed means (τ² + 1/n̄ around βᵀỹ);
            # convert so that β, and the R² of the population means, stay fixed.
            odds = R2_POP / (1 - R2_POP) * TAU2 / (TAU2 + 1 / nbar)
            grid.append((Design(M=M, nbar=nbar, h=TAU2 * nbar, signal=odds / (1 + odds)), {"n_perm": n_perm}))
    return grid


def size_replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    return dict(zip(SIZE_TESTS, [
        gu.pls_test(X, groups, Y, n_components=1, n_perm=n_perm, rng=rng).p_value[0],
        pr.pls_groups_refit(X, groups, Y, n_perm, rng),
        pr.cca_groups_refit(X, groups, Y, n_perm, rng),
    ]))


def size_summarise(df):
    return (df[SIZE_TESTS] < ALPHA).groupby([df.M, df.nbar]).mean()


def main(reps=300, n_perm=199, seed=2):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s2_power")
    df = run(size_replicate, size_cells(n_perm), reps, seed + 100)
    save(df, size_summarise(df), "s2_group_size")


if __name__ == "__main__":
    main()
