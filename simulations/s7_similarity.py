"""Outcome 7: typicality and pattern in similarity comparisons (Result 4).

Group profiles are generated directly. In the "typicality" scenario the two
tables share only how atypical each group is (deviations in independent
directions, as in Figure 6): any pattern rejection is a false positive. In the
"pattern" scenario they share which groups resemble which, through common
latent coordinates, and no typicality: rejections are power.
"""
import numpy as np

import groupunit as gu
from common import ALPHA, run, save

DIM = 30
SIMILARITIES = {"correlation": np.corrcoef, "distance": gu.distance_similarity}
TESTS = ["mantel", "typicality", "pattern raw", "pattern standardised"]


def cells(n_perm):
    return [(None, {"M": M, "scenario": s, "n_perm": n_perm}) for M in (10, 20, 50) for s in ("typicality", "pattern")]


def profiles(rng, M, scenario):
    if scenario == "typicality":
        u = np.exp(rng.normal(0, 0.55, M))
        spreads = [u, u * np.exp(rng.normal(0, 0.15, M))]
        return [rng.standard_normal(DIM) + s[:, None] * rng.standard_normal((M, DIM)) for s in spreads]
    Z = rng.standard_normal((M, 3))
    return [rng.standard_normal(DIM) + Z @ (0.5 * rng.standard_normal((3, DIM))) + rng.standard_normal((M, DIM))
            for _ in range(2)]


def replicate(_, rng, M, scenario, n_perm):
    F, A = profiles(rng, M, scenario)
    rows = []
    for name, similarity in SIMILARITIES.items():
        S1, S2 = similarity(F), similarity(A)
        raw = gu.compare_similarity(S1, S2, n_perm, rng, standardise=False)
        std = gu.compare_similarity(S1, S2, n_perm, rng, standardise=True)
        rows.append({"similarity": name, "mantel": raw.mantel.p, "typicality": raw.typicality.p,
                     "pattern raw": raw.pattern.p, "pattern standardised": std.pattern.p})
    return rows


def summarise(df):
    return (df[TESTS] < ALPHA).groupby([df.scenario, df.similarity, df.M]).mean()


def main(reps=500, n_perm=199, seed=7):
    df = run(replicate, cells(n_perm), reps, seed)
    save(df, summarise(df), "s7_similarity")


if __name__ == "__main__":
    main()
