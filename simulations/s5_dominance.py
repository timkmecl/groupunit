"""Outcome 5: dominant groups. One group has an extreme annotation profile.

The extreme group is the largest, the smallest, or a random one, and its
profile is 1.5 to 4 times as far out as the others'. We record its
share of the first PLS component and the effective number of groups behind
the component, the pairing ratio of the design, and the rejection rates of the
group-level tests with size weights, equal weights and exchangeable weights
1/sqrt(τ² + σ²_w/n), the last with pooled estimated variance components.
Under β = 0 the rejection rates are false-positive rates; with signal, power.
"""
import numpy as np

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, run, save

TESTS = ["corr: size", "corr: exchangeable", "PLS: size", "PLS: equal", "PLS: exchangeable",
         "CCA: size", "Mantel: equal"]


def cells(n_perm):
    return [(Design(M=20, kappa=0.5, h=h, structure="extreme", extreme_at=at, extremity=x, signal=s),
             {"n_perm": n_perm})
            for x in (1.5, 2.0, 3.0, 4.0) for at in ("random", "largest", "smallest")
            for h in (1, 10) for s in (0.0, 0.3)]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    extreme = int(np.argmax((gu.wcentre(Y, np.ones(d.M)) ** 2).sum(1)))
    vc = gu.pooled_variance_components(X, groups)
    w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
    Xc = X - X.mean(0)
    pcs = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:3].T
    fit = gu.pls(X, groups, Y, n_components=1)
    shares = gu.group_shares(fit.group_scores[:, 0], Y, n)
    p = [
        gu.correlation_test(pcs, groups, Y, n_perm, rng).p_fwer.min(),
        gu.correlation_test(pcs, groups, Y, n_perm, rng, weights=w).p_fwer.min(),
        pr.pls_groups_refit(X, groups, Y, n_perm, rng),
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights=np.ones(d.M)).p_value[0],
        gu.pls_test(X, groups, Y, 1, n_perm, rng, weights=w).p_value[0],
        pr.cca_groups_refit(X, groups, Y, n_perm, rng),
        pr.mantel_groups(gu.distance_similarity(gu.group_means(X, groups)), gu.distance_similarity(Y), n_perm, rng),
    ]
    return {"share of extreme group": shares.abs_share[extreme], "M_eff,c": shares.m_eff,
            "pairing ratio (size)": gu.pairing_ratio(Y, n, tau2=d.tau2, sigma2_w=1.0),
            "pairing ratio (equal)": gu.pairing_ratio(Y, n, np.ones(d.M), d.tau2, 1.0),
            **dict(zip(TESTS, p))}


def summarise(df):
    keys = [df.extremity, df.extreme_at, df.h, df.signal]
    descriptive = df[["share of extreme group", "M_eff,c", "pairing ratio (size)", "pairing ratio (equal)"]]
    return descriptive.groupby(keys).median().join((df[TESTS] < ALPHA).groupby(keys).mean())


def main(reps=300, n_perm=199, seed=5):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s5_dominance")


if __name__ == "__main__":
    main()
