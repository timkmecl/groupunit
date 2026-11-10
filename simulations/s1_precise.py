"""Outcome 1 at high precision: false-positive rates of the valid group-level tests.

The cells of `s1_false_positives` with more replicates and permutations, for
the procedures the paper recommends only: the invalid ones reject so often
that their rates need no precision. Each rate is compared with the binomial
95% band around α = 0.05 for the number of replicates, and the summary counts
the cells outside it. CCA at κ = 1 is left out: it cannot be tested
(Result 3c) and its p-value is 1 by construction.
"""
import numpy as np
from scipy import stats

import groupunit as gu
import procedures as pr
from common import ALPHA, run, save
from s1_false_positives import cells

TESTS = ["corr: groups, size", "corr: groups, exchangeable", "PLS: groups, refit, size",
         "PLS: groups, refit, exchangeable", "CCA: groups, refit", "Mantel: groups"]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    Xc = X - X.mean(0)
    scores = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T     # first three PCs, built without Y
    p = [
        gu.correlation_test(scores, groups, Y, n_perm, rng, weights="size").p_fwer.min(),
        gu.correlation_test(scores, groups, Y, n_perm, rng).p_fwer.min(),
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights="size").p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng).p_value[0],
        pr.cca_groups_refit(X, groups, Y, n_perm, rng) if d.kappa < 1 else np.nan,
        pr.mantel_groups(gu.distance_similarity(gu.group_means(X, groups)), gu.distance_similarity(Y), n_perm, rng),
    ]
    return dict(zip(TESTS, p))


def band(reps, alpha=ALPHA):
    """Central 95% binomial interval for a rejection rate at the nominal level."""
    lo, hi = stats.binom.interval(0.95, reps, alpha)
    return lo / reps, hi / reps


def summarise(df):
    keys = [df.M, df.kappa, df.h]
    rates = (df[TESTS] < ALPHA).groupby(keys).mean()
    rates[df[TESTS].isna().groupby(keys).all()] = np.nan
    reps = int(df.groupby(["M", "kappa", "h"]).size().min())
    lo, hi = band(reps)
    outside = ((rates < lo) | (rates > hi)) & rates.notna()
    rates.loc[("cells outside band", "", ""), :] = outside.sum().astype(float)
    rates.loc[(f"band low ({reps} reps)", "", ""), :] = lo
    rates.loc[(f"band high ({reps} reps)", "", ""), :] = hi
    return rates


def main(reps=2000, n_perm=999, seed=11):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s1_precise")


if __name__ == "__main__":
    main()
