import numpy as np
import pytest

import groupunit as gu


def test_logo_recovers_linear_relation():
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((20, 3))
    target = Y @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.standard_normal(20)
    assert gu.leave_one_group_out(target, Y).r2 > 0.99


def test_logo_is_not_better_than_baseline_on_noise():
    rng = np.random.default_rng(10)
    Y, target = rng.standard_normal((15, 10)), rng.standard_normal(15)
    assert gu.leave_one_group_out(target, Y).r2 < 0.1


def test_pca_regression_and_multiple_targets():
    rng = np.random.default_rng(11)
    Y = rng.standard_normal((25, 8))
    target = np.column_stack([Y[:, 0] + Y[:, 1], rng.standard_normal(25)])
    w = rng.integers(5, 50, 25).astype(float)
    result = gu.leave_one_group_out(target, Y, weights=w, model=gu.pca_regression(3))
    assert result.predictions.shape == (25, 2) and result.r2.shape == (2,)
    assert result.baseline[0, 0] == pytest.approx(np.average(target[1:, 0], weights=w[1:]))
