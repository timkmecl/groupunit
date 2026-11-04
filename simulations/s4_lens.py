"""Outcome 4: the lens. Under β = 0, how closely do PLS loadings follow the lens?

For each design we record the congruence of the first PLS annotation loading
with the leading direction of three lenses: Ỹᵀ D² Ỹ (heterogeneity dominates),
Ỹᵀ D Ỹ (within-group noise dominates) and the model lens with the true τ² and
σ²_w. We also record the agreement between two independent null analyses
that share groups and annotations.
"""
import numpy as np
from scipy.special import gammaln

import groupunit as gu
from common import Design, run, save


def cells():
    return [(Design(M=20, kappa=0.5, structure=s, kish=k, h=h), {})
            for s in ("independent", "correlated", "extreme") for k in (None, 0.75, 0.4) for h in (0.1, 1, 10)]


def replicate(d, rng):
    Y = d.annotations(rng)
    loadings = []
    for _ in range(2):   # two independent analyses of the same groups and annotations
        X, groups, _, n = d.draw(rng, Y=Y)
        loadings.append(gu.pls(X, groups, Y, n_components=1).y_loadings[:, 0])
    lenses = {"D2": gu.lens(Y, n), "D": gu.lens(Y, n, tau2=0.0, sigma2_w=1.0),
              "model": gu.lens(Y, n, tau2=d.tau2, sigma2_w=1.0)}
    row = {f"cong lens {k}": abs(gu.congruence(loadings[0], L.eigenvectors[:, 0])) for k, L in lenses.items()}
    row.update({"nu lens D2": lenses["D2"].nu, "nu lens D": lenses["D"].nu,
                "agreement": abs(gu.congruence(*loadings))})
    return row


def random_congruence(q):
    """E|cos| between a uniformly random direction in q dimensions and a fixed one."""
    return float(np.exp(gammaln(q / 2) - gammaln((q + 1) / 2)) / np.sqrt(np.pi))


def summarise(df):
    keys = ["structure", "kish", "h"]
    out = df.fillna({"kish": 1.0}).groupby(keys).mean(numeric_only=True)
    out = out[[c for c in out if c.startswith(("cong", "nu", "agreement"))]]
    out["agreement > 0.85"] = df.fillna({"kish": 1.0}).groupby(keys).agreement.apply(lambda a: (a > 0.85).mean())
    out["random baseline"] = random_congruence(Design(M=20, kappa=0.5).q)
    return out


def main(reps=300, n_perm=None, seed=4):
    df = run(replicate, cells(), reps, seed)
    save(df, summarise(df), "s4_lens")


if __name__ == "__main__":
    main()
