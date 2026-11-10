"""Outcome 14: the refitted lens null of a PLS loading (Result 2b, Section 5).

Under β = 0 we fit PLS and take the congruence of its first annotation
loading with the leading direction of the model lens. If a null is right,
the observed congruence is a uniform draw from it, so its percentile in that
null has mean 0.5 and a uniform distribution. We compare three nulls, each
computed for the table and group sizes at hand with the true variance
components: a single random score (`lens_null` without p), PLS refitted on p
random features (the nominal number), and PLS refitted on as many features as
there are latent dimensions (the effective number). Result 2b predicts that the
single-score null is too diffuse for PLS and that p should be the effective
number of feature dimensions.
"""
import numpy as np
from scipy import stats

import groupunit as gu
from common import Design, run, save

N_DRAWS = 500


def cells():
    grid = [Design(M=20, p=p, feature_rank=r, h=h)
            for h in (1, 10) for p, r in ((10, None), (100, None), (100, 5), (100, 20), (10, 3))]
    return [(d, {}) for d in grid]


def replicate(d, rng):
    X, groups, Y, n = d.draw(rng)
    loading = gu.pls(X, groups, Y, n_components=1).y_loadings[:, 0]
    lead = gu.lens(Y, n, tau2=d.tau2, sigma2_w=1.0).eigenvectors[:, 0]
    observed = abs(gu.congruence(loading, lead))
    eff = d.p if d.feature_rank is None else d.feature_rank
    row = {"observed congruence": observed}
    for name, p in (("single score", None), ("refit, nominal p", d.p), ("refit, effective p", eff)):
        null = np.abs(gu.lens_null(Y, n, N_DRAWS, tau2=d.tau2, sigma2_w=1.0, rng=rng, p=p) @ lead)
        row[f"{name}: median"] = np.median(null)
        row[f"{name}: percentile"] = (null < observed).mean()
    return row


NULLS = ("single score", "refit, nominal p", "refit, effective p")


def summarise(df):
    df = df.fillna({"feature_rank": 0})
    keys = ["h", "p", "feature_rank"]
    g = df.groupby(keys)
    out = g[["observed congruence"]].median()
    for name in NULLS:
        out[f"{name}: median"] = g[f"{name}: median"].median()
        out[f"{name}: mean percentile"] = g[f"{name}: percentile"].mean()
        out[f"{name}: KS p"] = g[f"{name}: percentile"].apply(lambda u: stats.kstest(u, "uniform").pvalue)
    return out


def main(reps=300, n_perm=None, seed=15):
    df = run(replicate, cells(), reps, seed)
    save(df, summarise(df), "s14_lens_refit")


if __name__ == "__main__":
    main()
