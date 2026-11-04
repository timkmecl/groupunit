import numpy as np
import pytest

import groupunit as gu


@pytest.fixture
def design():
    """A small simulated design with a real relation to the first annotation."""
    rng = np.random.default_rng(0)
    n = gu.lognormal_sizes(12, 30, rng=rng)
    beta = np.zeros((4, 6))
    beta[0, :2] = 1.5
    X, groups, Y = gu.simulate(n, p=6, q=4, beta=beta, tau2=0.3, sigma2_w=2.0, rng=rng)
    return X, groups, Y, n


# Toy data of Figures 2 and 3 of the paper (invented values, groups A-H, annotations a1-a8)
TOY_N = np.array([90, 60, 120, 80, 14, 10, 8, 6], float)
TOY_Y = np.array([
    [.9, .8, .7, 1.0, .5, .3, .2, .1],
    [.4, .2, .8, .7, .9, .8, .1, .2],
    [.4, .2, .3, .6, .3, .3, .9, .1],
    [.5, .1, .3, .8, .4, .9, .0, 1.0],
    [.5, .3, .2, .9, .6, .3, .4, .1],
    [.3, .1, .3, .7, .5, .2, .2, .0],
    [.4, .5, .1, .6, .4, .4, .0, .3],
    [.2, .0, .2, .5, .3, .1, .5, .0],
])
TOY_T = np.array([.5, .3, -2.0, 1.6, .2, -.3, .4, -.1])
