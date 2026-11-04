"""Outcome 6: weighting groups (Section 5).

One feature, one annotation. (a) Mean squared error of the group-level slope
under size, equal and optimal weights (optimal with true and with estimated
variance components). (b) Accuracy of the effective-number formulas: the
variance of a weighted mean of group-level residuals against τ² / M_eff, with
M_eff from Kish's formula and from the model with estimated components.
"""
import numpy as np

import groupunit as gu
from common import Design, run, save

WEIGHTS = ["size", "equal", "optimal (true)", "optimal (estimated)"]


def cells():
    return [(Design(M=M, kappa=1 / (M - 1), p=1, h=h, signal=0.3), {}) for M in (20, 50) for h in (0.1, 1, 10)]


def slope(t, y, w):
    yc, tc = gu.wcentre(y, w), gu.wcentre(t, w)
    return (w * yc) @ tc / ((w * yc) @ yc)


def replicate(d, rng):
    X, groups, Y, n = d.draw(rng)
    t, y, c = gu.group_means(X[:, 0], groups), Y[:, 0], d.beta()[0, 0]
    vc = gu.variance_components(X[:, 0], groups, Y)
    weights = dict(zip(WEIGHTS, [n.astype(float), np.ones(d.M), gu.optimal_weights(n, d.tau2, 1.0),
                                 gu.optimal_weights(n, max(vc.tau2, 1e-12), vc.sigma2_w)]))
    row = {f"sq err: {k}": (slope(t, y, w) - c) ** 2 for k, w in weights.items()}
    # a weighted mean (size weights) of the group-level residuals t̄ - c·ỹ, whose variance is τ² / M_eff
    resid = t - c * y
    row["weighted mean"] = n @ resid / n.sum()
    row["var predicted: Kish"] = d.tau2 / gu.effective_groups(n)
    row["var predicted: model (estimated)"] = (
        vc.tau2 / gu.effective_groups(n, n, vc.tau2, vc.sigma2_w) if vc.tau2 > 0 else np.nan)
    row["var true"] = d.tau2 / gu.effective_groups(n, n, d.tau2, 1.0)
    row["tau2 estimate"] = vc.tau2
    row["tau2 true"] = d.tau2
    return row


def summarise(df):
    g = df.groupby(["M", "h"])
    mse = g[[f"sq err: {k}" for k in WEIGHTS]].mean()
    out = mse.div(mse["sq err: optimal (true)"], axis=0).rename(columns=lambda c: "rel MSE " + c[8:])
    empirical = g["weighted mean"].var()
    out["var ratio: Kish / empirical"] = g["var predicted: Kish"].mean() / empirical
    out["var ratio: model (est.) / empirical"] = g["var predicted: model (estimated)"].mean() / empirical
    out["var ratio: model (true) / empirical"] = g["var true"].mean() / empirical
    out["tau2 estimate / true"] = g["tau2 estimate"].mean() / g["tau2 true"].first()
    return out


def main(reps=2000, n_perm=None, seed=6):
    df = run(replicate, cells(), reps, seed, chunk=100)
    save(df, summarise(df), "s6_weighting")


if __name__ == "__main__":
    main()
