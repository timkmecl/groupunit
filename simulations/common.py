"""Simulation design of Section 6.1, and a small parallel runner shared by the outcome scripts."""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

import groupunit as gu

RESULTS = os.environ.get("GROUPUNIT_SIM_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
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
    extremity: float = 4.0             # for "extreme": how many times as far out that profile is
    p: int = 10                        # features
    signal: float = 0.0                # group-level R² of features 1-2 on annotation 1; 0 is the null
    length_var: float = 0.0            # variance of a within-group direction unrelated to groups
    n_atypical: int = 1                # for "several": how many of the largest groups are atypical
    gamma: float = 0.5                 # for "continuous": profile scale grows as (n_m / mean n)^gamma
    feature_rank: int | None = None    # features from this many latent dimensions (None: isotropic)
    tails: str = "normal"              # "normal" or "t3" (Student t, 3 df, scaled to unit variance)

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
            # one group's profile is `extremity` times as far out; groups are ordered largest first
            m = {"largest": 0, "smallest": self.M - 1}.get(self.extreme_at)
            Y[rng.integers(self.M) if m is None else m] *= self.extremity
        elif self.structure == "several":
            # the `n_atypical` largest groups are each `extremity` times as far out
            Y[:self.n_atypical] *= self.extremity
        elif self.structure == "continuous":
            # how far a profile lies out grows smoothly with its group's size
            n = self.sizes()
            Y *= ((n / n.mean()) ** self.gamma)[:, None]
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
        if self.feature_rank is None and self.tails == "normal":
            X, groups, Y = gu.simulate(n, self.p, Y=Y, beta=self.beta(), tau2=self.tau2, sigma2_w=1.0, rng=rng)
        else:
            X, groups = self._draw_features(n, Y, rng)
        if self.length_var > 0:
            d = np.ones(self.p) / np.sqrt(self.p)
            X += np.sqrt(self.length_var) * rng.standard_normal(len(X))[:, None] * d
        return X, groups, Y, n


    def _noise(self, rng, shape):
        if self.tails == "normal":
            return rng.standard_normal(shape)
        if self.tails == "t3":
            return rng.standard_t(3, shape) / np.sqrt(3)       # unit variance
        raise ValueError(self.tails)

    def _draw_features(self, n, Y, rng):
        """Group effects and within-group noise in ``feature_rank`` latent dimensions, mapped to p features.

        The map A (k × p) has orthogonal rows with A Aᵀ = (p / k) I, so the total
        variance is that of p isotropic features; a little isotropic noise
        (variance 0.01) keeps the features of full rank. With isotropic features
        the relation β acts on features 1-2; with latent dimensions it acts on
        latent dimensions 1-2, so that it lies in the same subspace as the noise.
        """
        k = self.p if self.feature_rank is None else self.feature_rank
        groups = np.repeat(np.arange(self.M), n)
        if self.feature_rank is None:
            A = np.eye(self.p)
        else:
            A = np.linalg.qr(rng.standard_normal((self.p, k)))[0].T * np.sqrt(self.p / k)
        U = np.sqrt(self.tau2) * self._noise(rng, (self.M, k))
        E = self._noise(rng, (len(groups), k))
        if self.feature_rank is None:
            X = (U[groups] + E) @ A + (Y @ self.beta())[groups]
        else:
            # the relation acts in the latent space, so the signal shares the subspace (and the
            # scale) of the noise; on the observed features it would lie partly outside it, where
            # scale-invariant CCA would find a direction with almost no noise
            B = self.beta()[:, :k]
            X = ((U + Y @ B)[groups] + E) @ A
        if self.feature_rank is not None:
            X += 0.1 * rng.standard_normal(X.shape)
        return X, groups


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
