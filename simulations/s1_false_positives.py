"""Outcome 1: false-positive rates under β = 0.

Each design is tested by resampling individuals, by permuting groups with the
components held fixed, and by permuting groups with every component refitted.
"""
import numpy as np

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, indicator, run, save

TESTS = ["corr: parametric, N-2 df", "corr: individuals", "corr: groups",
         "PLS: individuals", "PLS: groups, fixed", "PLS: groups, refit",
         "CCA: individuals", "CCA: groups, fixed", "CCA: groups, refit",
         "Mantel: pairs", "Mantel: groups"]


def cells(n_perm):
    grid = [Design(M=M, h=h) for M in (10, 20, 50, 100) for h in (0.1, 1, 10)]
    grid += [Design(M=20, h=1, kappa=k) for k in (0.1, 0.9, 1.0)]
    return [(d, {"n_perm": n_perm}) for d in grid]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    G = indicator(groups, d.M)
    Xc = X - X.mean(0)
    scores = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T     # first three PCs, built without Y
    S_features = gu.distance_similarity(gu.group_means(X, groups))
    S_annotations = gu.distance_similarity(Y)
    p = [
        pr.corr_parametric(scores, groups, Y),
        pr.corr_individuals(scores, G, Y, n, n_perm, rng),
        pr.corr_groups(scores, groups, Y, n_perm, rng),
        pr.pls_individuals(X, G, Y, n, n_perm, rng),
        pr.pls_groups_fixed(X, groups, Y, n, n_perm, rng),
        pr.pls_groups_refit(X, groups, Y, n_perm, rng),
        pr.cca_individuals(X, G, Y, n, n_perm, rng),
        pr.cca_groups_fixed(X, groups, Y, n, n_perm, rng),
        pr.cca_groups_refit(X, groups, Y, n_perm, rng),
        pr.mantel_pairs(S_features, S_annotations),
        pr.mantel_groups(S_features, S_annotations, n_perm, rng),
    ]
    return {"eta2": gu.feature_between_share(X, groups), **dict(zip(TESTS, p))}


def summarise(df):
    rates = (df[TESTS] < ALPHA).groupby([df.M, df.kappa, df.h]).mean()
    rates.insert(0, "eta2", df.groupby(["M", "kappa", "h"]).eta2.mean())
    return rates


def main(reps=500, n_perm=199, seed=1):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s1_false_positives")


if __name__ == "__main__":
    main()
