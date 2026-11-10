"""Outcome 15: power of row-weighted CCA against ordinary CCA and reweighted PLS.

Outcome 12 showed that row-weighted CCA with exchangeable weights holds its
level where ordinary (size-weighted) CCA does not, but is often conservative.
The paper recommends reweighted PLS until the power of row-weighted CCA is
known. Here annotation 1 drives features 1-2 with group-level R² = 0.3, in the
pairing designs of outcome 12 and in designs whose profiles are independent of
group sizes (κ = 0.1, 0.5, 0.9), at M = 20 and 50, h = 1.

Every design is also run under the null. Raw power of a liberal test is
inflated, and that of a conservative test deflated, so we also report
size-adjusted power: the share of data sets under the alternative whose p-value
is at most the 5th percentile of the same test's p-values under the null in the
same design. Raw power is what an analyst gets; adjusted power compares what
each statistic can detect at equal size.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

import groupunit as gu
from common import ALPHA, Design, run, save

TESTS = ["CCA: size", "CCA: exchangeable", "PLS: size", "PLS: exchangeable"]


def designs(M):
    out = [Design(M=M, kappa=k) for k in (0.1, 0.5, 0.9)]
    out += [Design(M=M, structure="several", n_atypical=k, extremity=2.0) for k in (3, 5)]
    out += [Design(M=M, structure="continuous", gamma=g) for g in (0.25, 0.5)]
    out += [Design(M=M, structure="extreme", extreme_at="largest", extremity=3.0)]
    return out


def cells(n_perm):
    return [(replace(d, signal=s), {"n_perm": n_perm})
            for M in (20, 50) for d in designs(M) for s in (0.0, 0.3)]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    vc = gu.pooled_variance_components(X, groups)
    w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
    p = [
        gu.cca_test(X, groups, Y, 1, n_perm, rng, weights="size").p_value[0],
        gu.cca_test(X, groups, Y, 1, n_perm, rng, weights=w).p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights="size").p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights=w).p_value[0],
    ]
    return dict(zip(TESTS, p))


KEYS = ["structure", "n_atypical", "gamma", "kappa", "M"]


def summarise(df):
    null, alt = df[df.signal == 0], df[df.signal > 0]
    out = (null[TESTS] < ALPHA).groupby([null[k] for k in KEYS]).mean().add_suffix(" | size")
    out = out.join((alt[TESTS] < ALPHA).groupby([alt[k] for k in KEYS]).mean().add_suffix(" | power"))
    crit = null.groupby(KEYS)[TESTS].quantile(ALPHA)                 # each test's own 5% point under the null
    adjusted = {}
    for key, grp in alt.groupby(KEYS):
        adjusted[key] = {t: float((grp[t] <= crit.loc[key, t]).mean()) for t in TESTS}
    adjusted = (
        pd.DataFrame.from_dict(adjusted, orient="index")
        .rename_axis(KEYS).add_suffix(" | adjusted power")
    )
    out = out.join(adjusted)
    order = [f"{t} | {kind}" for kind in ("size", "power", "adjusted power") for t in TESTS]
    return out[order]


def main(reps=300, n_perm=199, seed=16):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s15_cca_power")


if __name__ == "__main__":
    main()
