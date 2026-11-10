"""Outcome 11: more features than groups, correlated features, heavy tails.

The other outcomes use p = 10 features, fewer than M − 1, whereas the studies
that motivate the paper use 100 or more. Here p is 10, 100 or 300 at M = 20
and 50, with isotropic features or features generated from 10 latent
dimensions, and with Gaussian or Student t (3 df) group effects and
within-group noise. Under β = 0 the rates are false-positive rates; with
signal (annotation 1 drives features 1-2, group-level R² 0.3), power. One
pairing design (an atypical profile on the largest group) checks that
exchangeable weights still work when they are estimated from many features.
"""
import numpy as np

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, indicator, run, save

TESTS = ["corr: parametric, N-2 df", "PLS: individuals", "corr: groups, exchangeable",
         "PLS: groups, refit, size", "PLS: groups, refit, exchangeable", "CCA: groups, refit",
         "Mantel: groups"]


def cells(n_perm):
    grid = []
    for M in (20, 50):
        for p in (10, 100, 300):
            for rank in (None, 10):
                if rank is not None and rank >= p:
                    continue
                for s in (0.0, 0.3):
                    grid.append(Design(M=M, p=p, feature_rank=rank, signal=s))
        for p in (10, 100):
            grid.append(Design(M=M, p=p, tails="t3"))
        grid.append(Design(M=M, p=100, feature_rank=10, structure="extreme", extreme_at="largest", extremity=3.0))
    return [(d, {"n_perm": n_perm}) for d in grid]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    Xc = X - X.mean(0)
    scores = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
    p = [
        pr.corr_parametric(scores, groups, Y),
        pr.pls_individuals(X, indicator(groups, d.M), Y, n, n_perm, rng) if d.p <= 100 else np.nan,
        gu.correlation_test(scores, groups, Y, n_perm, rng).p_fwer.min(),
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights="size").p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng).p_value[0],
        pr.cca_groups_refit(X, groups, Y, n_perm, rng),
        pr.mantel_groups(gu.distance_similarity(gu.group_means(X, groups)), gu.distance_similarity(Y), n_perm, rng),
    ]
    return {"eta2": gu.feature_between_share(X, groups), **dict(zip(TESTS, p))}


def summarise(df):
    df = df.fillna({"feature_rank": 0})          # 0: isotropic
    keys = [df.structure, df.tails, df.signal, df.M, df.p, df.feature_rank]
    rates = (df[TESTS] < ALPHA).groupby(keys).mean()
    rates[df[TESTS].isna().groupby(keys).all()] = np.nan
    return df[["eta2"]].groupby(keys).mean().join(rates)


def main(reps=500, n_perm=199, seed=12):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s11_highdim")


if __name__ == "__main__":
    main()
