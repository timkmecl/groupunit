"""Outcome 9: PLS regression (PLSR), as used by Suresh et al. (2026), against PLS correlation.

PLSR extracts components one at a time and deflates X after each, so later
components depend on within-group variation (Result 2); its first component
equals that of PLS correlation. We test the statistic such studies report,
the R² of predicting the inherited annotations from the features with k
components, and validate it out of sample.

(a) Tests of R² under β = 0: resampling individuals, permuting groups with
    PLSR refitted on individuals (size weights), and permuting groups with
    PLSR fitted to group means under exchangeable weights. The PLS-SVD tests
    of the first component on the same data are included for comparison.
(b) Cross-validated R², splitting individuals into folds versus leaving
    groups out, under β = 0 and β ≠ 0.

Needs scikit-learn.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression

import groupunit as gu
import procedures as pr
from common import ALPHA, Design, indicator, run, save
from groupunit.inference import draw_permutations, p_value

M, P = 21, 20


def cells(n_perm):
    grid = []
    for q in (19, 10):                                    # κ = 0.95 (as Suresh et al.) and 0.5
        for structure, at in (("independent", "largest"), ("extreme", "largest")):
            for k in ("1", "3", "all"):
                grid.append((Design(M=M, kappa=q / (M - 1), p=P, h=1, structure=structure, extreme_at=at),
                             {"n_perm": n_perm, "k": k, "part": "test"}))
    for q in (19, 10):
        for h in (1, 10):
            for s in (0.0, 0.3):
                for k in ("3", "all"):
                    grid.append((Design(M=M, kappa=q / (M - 1), p=P, h=h, signal=s),
                                 {"n_perm": 0, "k": k, "part": "cv"}))
    return grid


def n_components(k, d):
    return min(P, d.q) if k == "all" else int(k)


def r2_individuals(X, Yi, k):
    return PLSRegression(k, scale=False).fit(X, Yi).score(X, Yi)


def r2_groups_weighted(Xbar, Y, w, k):
    """PLSR on group means with group weights: rows scaled by sqrt(w) after weighted centring."""
    sw = np.sqrt(w)[:, None]
    Xw, Yw = sw * gu.wcentre(Xbar, w), sw * gu.wcentre(Y, w)
    fit = PLSRegression(k, scale=False).fit(Xw, Yw)
    resid = Yw - fit.predict(Xw)
    return 1 - (resid**2).sum() / (Yw**2).sum()


def test_part(d, rng, n_perm, k):
    X, groups, Y, n = d.draw(rng)
    G = indicator(groups, d.M)
    kk = n_components(k, d)
    Yi = Y[groups]
    perms = draw_permutations(d.M, n_perm, rng)

    observed = r2_individuals(X, Yi, kk)
    null_ind = [r2_individuals(X, Yi[rng.permutation(len(Yi))], kk) for _ in range(n_perm)]
    null_grp = [r2_individuals(X, Y[perm][groups], kk) for perm in perms]

    vc = gu.pooled_variance_components(X, groups)
    w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
    Xbar = gu.group_means(X, groups, d.M)
    observed_w = r2_groups_weighted(Xbar, Y, w, kk)
    null_w = [r2_groups_weighted(Xbar, Y[perm], w, kk) for perm in perms]
    return {
        "R2 in sample": observed,
        "PLSR R2: individuals": p_value(observed, null_ind),
        "PLSR R2: groups, size": p_value(observed, null_grp),
        "PLSR R2: groups, exchangeable": p_value(observed_w, null_w),
        "PLS-SVD: individuals": pr.pls_individuals(X, G, Y, n, n_perm, rng),
        "PLS-SVD: groups, size": pr.pls_groups_refit(X, groups, Y, n_perm, rng),
        "pairing ratio (size)": gu.pairing_ratio(Y, n, tau2=d.tau2, sigma2_w=1.0),
    }


def cv_r2(X, Yi, folds, k):
    sse = sst = 0.0
    for test in folds:
        train = np.ones(len(Yi), bool)
        train[test] = False
        fit = PLSRegression(k, scale=False).fit(X[train], Yi[train])
        sse += ((Yi[test] - fit.predict(X[test])) ** 2).sum()
        sst += ((Yi[test] - Yi[train].mean(0)) ** 2).sum()
    return 1 - sse / sst


def cv_part(d, rng, k):
    X, groups, Y, n = d.draw(rng)
    kk = n_components(k, d)
    Yi = Y[groups]
    individual_folds = np.array_split(rng.permutation(len(Yi)), 5)
    group_folds = [np.flatnonzero(groups == m) for m in range(d.M)]
    return {"CV R2: individual split": cv_r2(X, Yi, individual_folds, kk),
            "CV R2: leave one group out": cv_r2(X, Yi, group_folds, kk)}


def replicate(d, rng, n_perm, k, part):
    return test_part(d, rng, n_perm, k) if part == "test" else cv_part(d, rng, k)


TESTS = ["PLSR R2: individuals", "PLSR R2: groups, size", "PLSR R2: groups, exchangeable",
         "PLS-SVD: individuals", "PLS-SVD: groups, size"]


def summarise(df):
    tests = df[df.part == "test"]
    keys = ["kappa", "structure", "k"]
    a = (tests[TESTS] < ALPHA).groupby([tests[c] for c in keys]).mean()
    a.insert(0, "R2 in sample", tests.groupby(keys)["R2 in sample"].mean())
    a.insert(1, "pairing ratio", tests.groupby(keys)["pairing ratio (size)"].median())
    cv = df[df.part == "cv"]
    b = cv.groupby(["kappa", "h", "signal", "k"])[["CV R2: individual split", "CV R2: leave one group out"]].mean()
    return a, b


def main(reps=200, n_perm=99, seed=9):
    df = run(replicate, cells(n_perm), reps, seed, chunk=5)
    a, b = summarise(df)
    save(df, a, "s9_plsr_tests")
    save(df, b, "s9_plsr_cv")


if __name__ == "__main__":
    main()
