"""Outcome 3: indifference of saturated CCA, and departure from it as κ falls (Results 3b, 3c).

A real relation with annotation 1 is present. For each κ we compare the first
canonical correlation for the real table with shuffles of its rows among
groups and with LDA, and check the Beta null of the group-level R².
"""
import numpy as np
from scipy import stats

import groupunit as gu
from common import ALPHA, Design, run, save
from groupunit.inference import p_value

KAPPAS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def cells(n_perm):
    return [(Design(M=20, kappa=k, h=1, signal=0.3), {"n_perm": n_perm}) for k in KAPPAS]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    rho = gu.cca(X, groups, Y, n_components=1).strength[0]
    null = np.array([gu.cca(X, groups, Y[rng.permutation(d.M)], n_components=1).strength[0]
                     for _ in range(n_perm)])
    random_table = gu.cca(X, groups, rng.standard_normal(Y.shape), n_components=1).strength[0]
    lda = gu.lda(X, groups, n_components=1).strength[0]
    # R² of a score whose size-weighted group means point in a random direction (Result 3b)
    r2_random = gu.weighted_r2(rng.standard_normal(d.M) / np.sqrt(n), Y, n)
    return {
        "rho real": rho, "rho shuffled (mean)": null.mean(), "rho LDA": lda,
        "max |real - other|": max(np.abs(null - rho).max(), abs(random_table - rho), abs(lda - rho)),
        "p": p_value(rho, null), "r2 random score": r2_random,
    }


def summarise(df):
    g = df.groupby("kappa")
    out = g[["rho real", "rho shuffled (mean)", "rho LDA", "r2 random score"]].mean()
    out["max |real - other|"] = g["max |real - other|"].max()
    out["power"] = g.p.apply(lambda p: (p < ALPHA).mean())
    out["r2 null mean (theory)"] = [d.q / (d.M - 1) for d, _ in cells(0)]
    out["KS p vs Beta"] = [
        stats.kstest(grp["r2 random score"], stats.beta(d.q / 2, (d.M - 1 - d.q) / 2).cdf).pvalue
        if d.q < d.M - 1 else np.nan
        for (d, _), (_, grp) in zip(cells(0), g)
    ]
    return out


def main(reps=300, n_perm=199, seed=3):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s3_saturation")


if __name__ == "__main__":
    main()
