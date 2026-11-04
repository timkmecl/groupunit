import numpy as np
import pytest

import groupunit as gu


def random_similarity(rng, M=10, dim=20):
    return np.corrcoef(rng.standard_normal((M, dim)))


def test_decomposition_reconstructs_off_diagonal():
    S = random_similarity(np.random.default_rng(5))
    d = gu.typicality_pattern(S)
    fitted = d.mean + d.typicality[:, None] + d.typicality[None, :] + d.pattern
    off = ~np.eye(len(S), dtype=bool)
    np.testing.assert_allclose(fitted[off], S[off])
    assert d.typicality.sum() == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(d.pattern.sum(1), 0, atol=1e-12)


def test_shared_typicality_without_pattern():
    """Two matrices that share group effects and nothing else (Result 4)."""
    rng = np.random.default_rng(3)
    M = 15
    a = rng.standard_normal(M)

    def matrix():
        E = rng.standard_normal((M, M)) * 0.5
        return a[:, None] + a[None, :] + E + E.T

    result = gu.compare_similarity(matrix(), matrix(), n_perm=999, rng=0)
    assert result.mantel.p < 0.05
    assert result.typicality.p < 0.05
    assert result.pattern.p > 0.05


def test_distance_similarity_decomposes_exactly():
    """Typicality is linear in the squared distance from the centroid, the pattern twice the U-centred inner products."""
    M = 8
    F = np.random.default_rng(7).standard_normal((M, 5))
    d = gu.typicality_pattern(gu.distance_similarity(F))
    Fc = F - F.mean(0)
    sq = (Fc**2).sum(1)
    np.testing.assert_allclose(d.typicality, -M / (M - 2) * (sq - sq.mean()), atol=1e-12)
    np.testing.assert_allclose(d.pattern, 2 * gu.typicality_pattern(Fc @ Fc.T).pattern, atol=1e-12)


def test_standardised_pattern_test_is_calibrated_under_shared_typicality():
    """Groups share only how atypical they are; their profiles deviate in independent directions."""
    rng = np.random.default_rng(2)
    M, dim = 12, 30

    def profiles(u):
        return rng.standard_normal(dim)[None, :] + u[:, None] * rng.standard_normal((M, dim))

    rejections = {True: 0, False: 0}
    for _ in range(150):
        u = np.exp(rng.normal(0, 0.55, M))
        S1, S2 = gu.distance_similarity(profiles(u)), gu.distance_similarity(profiles(u))
        for standardise in rejections:
            p = gu.compare_similarity(S1, S2, n_perm=99, rng=rng, standardise=standardise).pattern.p
            rejections[standardise] += p < 0.05
    assert rejections[True] / 150 < 0.10
    assert rejections[False] > rejections[True]


def test_identical_matrices_share_everything():
    S = random_similarity(np.random.default_rng(6))
    result = gu.compare_similarity(S, S, n_perm=199, rng=0)
    assert result.mantel.r == pytest.approx(1) and result.pattern.r == pytest.approx(1)
    assert result.pattern.p == pytest.approx(1 / 200)
