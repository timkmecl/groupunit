"""Outcome 12: harder pairing designs, and row-weighted CCA (Result 5).

Outcome 5 places one atypical profile on one group. Real tables are less
tidy, so here, under β = 0:

* several: the 3 or 5 largest groups each have a profile twice as far out;
* continuous: every profile's distance from the centre grows with its group's
  size, as (n_m / n̄)^γ, so no single group stands out;
* heavy tails: one profile three times as far out on the largest group, with
  Student t (3 df) group effects and within-group noise, against the same
  design with Gaussian noise.

For each design we record the pairing ratio under size and under estimated
exchangeable weights, the largest per-group term (Result 5d), and the
rejection rates of correlation, PLS, CCA and Mantel tests. CCA is tested both
as ordinary (size-weighted) CCA and as row-weighted CCA with exchangeable
weights (`gu.cca_test(weights="exchangeable")`), which the paper has not yet
evaluated.
"""
import numpy as np

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, run, save

TESTS = ["corr: size", "corr: exchangeable", "PLS: size", "PLS: equal", "PLS: exchangeable",
         "CCA: size", "CCA: exchangeable", "Mantel: equal"]


def cells(n_perm):
    grid = []
    for M in (20, 50):
        hs = (1, 10) if M == 20 else (1,)
        for h in hs:
            for k in (3, 5):
                grid.append(Design(M=M, h=h, structure="several", n_atypical=k, extremity=2.0))
            for g in (0.25, 0.5):
                grid.append(Design(M=M, h=h, structure="continuous", gamma=g))
            for tails in ("normal", "t3"):
                grid.append(Design(M=M, h=h, structure="extreme", extreme_at="largest", extremity=3.0, tails=tails))
    return [(d, {"n_perm": n_perm}) for d in grid]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    vc = gu.pooled_variance_components(X, groups)
    w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
    Xc = X - X.mean(0)
    pcs = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
    size_terms = gu.pairing_terms(Y, n, tau2=d.tau2, sigma2_w=1.0)
    p = [
        gu.correlation_test(pcs, groups, Y, n_perm, rng, weights="size").p_fwer.min(),
        gu.correlation_test(pcs, groups, Y, n_perm, rng, weights=w).p_fwer.min(),
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights="size").p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights=np.ones(d.M)).p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights=w).p_value[0],
        gu.cca_test(X, groups, Y, 1, n_perm, rng, weights="size").p_value[0],
        gu.cca_test(X, groups, Y, 1, n_perm, rng, weights=w).p_value[0],
        pr.mantel_groups(gu.distance_similarity(gu.group_means(X, groups)), gu.distance_similarity(Y), n_perm, rng),
    ]
    return {
        "pairing ratio (size)": size_terms.ratio,
        "largest term (size)": size_terms.terms.max(),
        "pairing ratio (exchangeable)": gu.pairing_ratio(Y, n, w, d.tau2, 1.0),
        "pairing ratio (equal)": gu.pairing_ratio(Y, n, np.ones(d.M), d.tau2, 1.0),
        **dict(zip(TESTS, p)),
    }


DESCRIPTIVE = ["pairing ratio (size)", "largest term (size)", "pairing ratio (exchangeable)", "pairing ratio (equal)"]


def summarise(df):
    keys = [df.structure, df.n_atypical, df.gamma, df.tails, df.M, df.h]
    return df[DESCRIPTIVE].groupby(keys).median().join((df[TESTS] < ALPHA).groupby(keys).mean())


def main(reps=500, n_perm=199, seed=13):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s12_pairing")


if __name__ == "__main__":
    main()
