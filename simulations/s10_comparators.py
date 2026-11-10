"""Outcome 10: established remedies for one score chosen in advance (Section 4.6).

The score is feature 1, fixed before the annotations are seen (under the
alternative, annotation 1 drives it). We compare the group permutation tests
with the remedies an applied researcher would reach for first:

* OLS with N − q − 1 residual degrees of freedom (pseudoreplication);
* a random-intercept mixed model fitted by REML, tested by the Wald χ² that
  software reports by default, and by F with the between-groups degrees of
  freedom M − q − 1;
* the wild cluster restricted bootstrap with cluster-robust (CR1) errors and
  Rademacher weights, the econometric remedy for few clusters;
* group permutation of the correlations (max-T over the q annotations), with
  size and with exchangeable weights.

The tests differ in their statistic but share the null hypothesis that the
score is unrelated to the annotations, so their rejection rates under it are
comparable. We also record how often REML put τ² at zero.
"""
import numpy as np

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, run, save

TESTS = ["OLS, N-q-1 df", "mixed model, Wald chi2", "mixed model, F(q, M-q-1)",
         "wild cluster bootstrap", "groups permuted, size", "groups permuted, exchangeable"]


def cells(n_perm):
    grid = [Design(M=M, h=h) for M in (10, 20, 50, 100) for h in (0.1, 1, 10)]
    grid += [Design(M=20, h=1, kappa=k) for k in (0.1, 0.9)]
    grid += [Design(M=M, h=1, structure="extreme", extreme_at="largest", extremity=4.0) for M in (20, 50)]
    grid += [Design(M=M, h=1, kappa=0.1, signal=0.3) for M in (10, 20, 50)]
    grid += [Design(M=M, h=1, signal=0.3) for M in (20, 50)]
    return [(d, {"n_perm": n_perm}) for d in grid]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    t = X[:, 0]
    mm = pr.mixed_model(t, groups, Y)
    p = [
        pr.ols_individuals(t, groups, Y),
        mm["wald"],
        mm["F"],
        pr.wild_cluster_bootstrap(t, groups, Y, n_perm, rng),
        gu.correlation_test(t, groups, Y, n_perm, rng, weights="size").p_fwer.min(),
        gu.correlation_test(t, groups, Y, n_perm, rng).p_fwer.min(),
    ]
    return {"tau2 at zero": float(mm["tau2 zero"]), **dict(zip(TESTS, p))}


def summarise(df):
    keys = [df.signal, df.structure, df.kappa, df.M, df.h]
    rates = (df[TESTS] < ALPHA).groupby(keys).mean()
    return df[["tau2 at zero"]].groupby(keys).mean().join(rates)


def main(reps=1000, n_perm=499, seed=10):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s10_comparators")


if __name__ == "__main__":
    main()
