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
    # h = 10 is appended after h = 1, so the seeds of the earlier cells are unchanged
    return [(Design(M=20, kappa=k, h=h, signal=0.3), {"n_perm": n_perm}) for h in (1, 10) for k in KAPPAS]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    rho = gu.cca(X, groups, Y, n_components=1).strength[0]
    null = np.array([gu.cca(X, groups, Y[rng.permutation(d.M)], n_components=1).strength[0]
                     for _ in range(n_perm)])
    random_table = gu.cca(X, groups, rng.standard_normal(Y.shape), n_components=1).strength[0]
    lda = gu.lda(X, groups, n_components=1).strength[0]
    # R² of a score whose size-weighted group means point in a random direction (Result 3b)
    r2_random = gu.weighted_r2(rng.standard_normal(d.M) / np.sqrt(n), Y, n)
    # R² of null group means under the model (variances v_m = τ² + σ²_w/n_m), with size and with
    # optimal weights; drawn last so that the columns above reproduce earlier runs (Appendix B.3)
    v = d.tau2 + 1.0 / n
    tbar = rng.standard_normal(d.M) * np.sqrt(v)
    r2_size = gu.weighted_r2(tbar, Y, n) if d.q < d.M - 1 else np.nan
    r2_opt = gu.weighted_r2(tbar, Y, 1 / v) if d.q < d.M - 1 else np.nan
    # ratio-of-expectations approximation to the size-weighted null mean: with z = D^{1/2} C t̄,
    # E‖P z‖² / E‖z‖² = tr(Qᵀ Σ Q) / tr Σ, Σ = Cov z, Q an orthonormal basis of the annotation space
    C = np.eye(d.M) - np.outer(np.ones(d.M), n) / n.sum()
    Sigma = np.sqrt(n)[:, None] * (C * v) @ C.T * np.sqrt(n)[None, :]
    Q = gu.weighted_basis(Y, n)
    approx = np.trace(Q.T @ Sigma @ Q) / np.trace(Sigma)
    return {
        "r2 null, size weights": r2_size, "r2 null, size weights (approx)": approx, "r2 null, optimal weights": r2_opt,
        "rho real": rho, "rho shuffled (mean)": null.mean(), "rho LDA": lda,
        "max |real - other|": max(np.abs(null - rho).max(), abs(random_table - rho), abs(lda - rho)),
        "p": p_value(rho, null), "r2 random score": r2_random,
    }


def summarise(df):
    g = df.groupby(["h", "kappa"])
    out = g[["rho real", "rho shuffled (mean)", "rho LDA", "r2 random score"]].mean()
    out["max |real - other|"] = g["max |real - other|"].max()
    out["power"] = g.p.apply(lambda p: (p < ALPHA).mean())
    out["r2 null mean (theory)"] = [d.q / (d.M - 1) for d, _ in cells(0)]
    cols = ["r2 null, size weights", "r2 null, size weights (approx)", "r2 null, optimal weights"]
    out[cols] = g[cols].mean()
    out["r / M_eff (Kish)"] = [d.q / gu.effective_number(d.sizes()) for d, _ in cells(0)]
    out["KS p, optimal weights vs Beta"] = [
        stats.kstest(grp["r2 null, optimal weights"], stats.beta(d.q / 2, (d.M - 1 - d.q) / 2).cdf).pvalue
        if d.q < d.M - 1 else np.nan
        for (d, _), (_, grp) in zip(cells(0), g)
    ]
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
