"""Simulated designs with group-level annotations (Section 6.1)."""
import numpy as np


def lognormal_sizes(M, mean_size, sigma=0.6, minimum=2, rng=None):
    """Unequal group sizes drawn from a log-normal distribution; larger ``sigma`` means more imbalance."""
    rng = np.random.default_rng(rng)
    sizes = np.round(np.exp(rng.normal(np.log(mean_size), sigma, M))).astype(int)
    return np.maximum(sizes, minimum)


def simulate(n, p, Y=None, q=None, beta=0.0, tau2=1.0, sigma2_w=1.0, rng=None):
    """Simulate individuals' features: x_i = μ_m(i) + ε_i, with μ_m = βᵀỹ_m + u_m.

    Parameters
    ----------
    n : group sizes, (M,).
    p : number of features.
    Y : annotation table (M, q); drawn as standard normal with ``q`` columns if None.
    beta : the true relation. A (q, p) matrix, or a number used as the scale of a
        random standard-normal β; 0 gives the null.
    tau2 : variance of the group-level heterogeneity u_m.
    sigma2_w : variance of the within-group noise ε_i.

    Returns
    -------
    X, groups, Y
        Features (N, p), group codes (N,) and the annotation table (M, q).
    """
    rng = np.random.default_rng(rng)
    n = np.asarray(n, int)
    M = len(n)
    if Y is None:
        if q is None:
            raise ValueError("give either Y or q")
        Y = rng.standard_normal((M, q))
    Y = np.asarray(Y, float)
    B = np.asarray(beta, float)
    if B.ndim == 0:
        B = B * rng.standard_normal((Y.shape[1], p))
    mu = Y @ B + np.sqrt(tau2) * rng.standard_normal((M, p))
    groups = np.repeat(np.arange(M), n)
    X = mu[groups] + np.sqrt(sigma2_w) * rng.standard_normal((len(groups), p))
    return X, groups, Y
