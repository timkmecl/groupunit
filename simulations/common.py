"""Simulation design of Section 8.1, and a small parallel runner shared by the outcome scripts."""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

import groupunit as gu

RESULTS = os.path.join(os.path.dirname(__file__), "results")
ALPHA = 0.05


@dataclass(frozen=True)
class Design:
    """One cell of the simulation design.

    Within-group variance is fixed at σ²_w = 1, so τ² = h / nbar, where h is the
    heterogeneity ratio τ² / (σ²_w / nbar). The between-group share η² of the
    features then follows from h and nbar; it is not a free factor.
    """

    M: int = 20
    kappa: float = 0.5                 # annotation ratio r / (M - 1)
    kish: float | None = 0.4           # Kish effective number as a fraction of M; None for equal sizes
    h: float = 1.0                     # heterogeneity ratio
    nbar: int = 50                     # mean group size
    structure: str = "independent"     # "independent", "correlated" or "extreme"
    extreme_at: str = "largest"        # for "extreme": the "largest", "smallest" or a "random" group
    p: int = 10                        # features
    signal: float = 0.0                # group-level R² of features 1-2 on annotation 1; 0 is the null
    length_var: float = 0.0            # variance of a within-group direction unrelated to groups

    @property
    def q(self):
        return max(1, round(self.kappa * (self.M - 1)))

    @property
    def tau2(self):
        return self.h / self.nbar

    def sizes(self):
        """Group sizes, largest first, with the requested Kish effective number."""
        if self.kish is None:
            return np.full(self.M, self.nbar)
        z = norm.ppf((np.arange(self.M) + 0.5) / self.M)[::-1]   # deterministic, largest first

        def make(sigma):
            return np.maximum(2, np.round(self.nbar * np.exp(sigma * z - sigma**2 / 2))).astype(int)

        lo, hi = 0.0, 4.0
        for _ in range(60):   # bisection on the log-normal spread
            mid = (lo + hi) / 2
            if gu.effective_number(make(mid)) / self.M > self.kish:
                lo = mid
            else:
                hi = mid
        return make(lo)

    def annotations(self, rng):
        Y = rng.standard_normal((self.M, self.q))
        if self.structure == "correlated":
            lag = np.abs(np.subtract.outer(np.arange(self.q), np.arange(self.q)))
            Y = Y @ np.linalg.cholesky(0.8**lag).T
        elif self.structure == "extreme":
            # one group's profile is four times as far out; groups are ordered largest first
            m = {"largest": 0, "smallest": self.M - 1}.get(self.extreme_at)
            Y[rng.integers(self.M) if m is None else m] *= 4
        elif self.structure != "independent":
            raise ValueError(self.structure)
        return Y

    def beta(self):
        """The true relation: annotation 1 drives features 1 and 2, with group-level R² = signal."""
        beta = np.zeros((self.q, self.p))
        if self.signal > 0:
            noise = self.tau2 + 1 / self.nbar      # variance of an observed group mean around βᵀỹ
            beta[0, :2] = np.sqrt(self.signal / (1 - self.signal) * noise)
        return beta

    def draw(self, rng, Y=None):
        """Draw ``(X, groups, Y, n)``: x_i = μ_m(i) + ε_i, μ_m = βᵀỹ_m + u_m."""
        n = self.sizes()
        Y = self.annotations(rng) if Y is None else Y
        X, groups, Y = gu.simulate(n, self.p, Y=Y, beta=self.beta(), tau2=self.tau2, sigma2_w=1.0, rng=rng)
        if self.length_var > 0:
            d = np.ones(self.p) / np.sqrt(self.p)
            X += np.sqrt(self.length_var) * rng.standard_normal(len(X))[:, None] * d
        return X, groups, Y, n


def indicator(groups, M):
    """Dense N × M group indicator matrix G."""
    G = np.zeros((len(groups), M))
    G[np.arange(len(groups)), groups] = 1
    return G


# ---------------------------------------------------------------- runner

def _run_chunk(args):
    replicate, design, extra, seeds = args
    rows = []
    for seed in seeds:
        out = replicate(design, np.random.default_rng(seed), **extra)
        for row in out if isinstance(out, list) else [out]:
            rows.append({**(asdict(design) if design else {}), **extra, **row})
    return rows


def run(replicate, cells, reps, seed, n_jobs=None, chunk=10):
    """Run ``replicate(design, rng, **extra)`` ``reps`` times for every cell ``(design, extra)``.

    Returns a data frame with one row per replicate (or per row a replicate returns).
    Seeds are spawned from one ``seed``, so results do not depend on scheduling.
    """
    root = np.random.SeedSequence(seed)
    tasks = []
    for (design, extra), cell_seed in zip(cells, root.spawn(len(cells))):
        seeds = cell_seed.spawn(reps)
        tasks += [(replicate, design, extra, seeds[i:i + chunk]) for i in range(0, reps, chunk)]
    with ProcessPoolExecutor(n_jobs) as pool:
        rows = [row for chunk_rows in pool.map(_run_chunk, tasks) for row in chunk_rows]
    return pd.DataFrame(rows)


def save(df, summary, name):
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, f"{name}_replicates.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS, f"{name}.csv"))
    with open(os.path.join(RESULTS, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write(summary.to_markdown(floatfmt=".3f"))
    print(f"\n== {name} ==\n{summary.to_string(float_format=lambda v: f'{v:.3f}')}")
