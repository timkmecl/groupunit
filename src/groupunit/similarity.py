"""Comparison of group similarity matrices: typicality versus pattern (Results 4a-4c)."""
from dataclasses import dataclass

import numpy as np

from .inference import p_value, draw_permutations


@dataclass(frozen=True)
class TypicalityPattern:
    """Additive split of a similarity matrix, s_ij = μ + a_i + a_j + e_ij, over off-diagonal entries.

    Attributes
    ----------
    mean : μ.
    typicality : (M,) group effects a_i, how similar each group is to the set as a whole.
    pattern : (M, M) residuals e_ij, which groups resemble each other beyond
        their typicality; zero on the diagonal.
    """

    mean: float
    typicality: np.ndarray
    pattern: np.ndarray


def typicality_pattern(S):
    """Least-squares fit of the additive model to the off-diagonal entries of ``S``."""
    S = np.array(S, float)
    M = len(S)
    if S.shape != (M, M) or not np.allclose(S, S.T):
        raise ValueError("S must be a symmetric square matrix")
    if M < 3:
        raise ValueError("need at least three groups")
    np.fill_diagonal(S, 0)
    rho = S.sum(1)
    T = rho.sum()
    a = rho / (M - 2) - T / (M * (M - 2))
    E = S - (rho[:, None] + rho[None, :]) / (M - 2) + T / ((M - 1) * (M - 2))
    np.fill_diagonal(E, 0)
    return TypicalityPattern(mean=T / (M * (M - 1)), typicality=a, pattern=E)


@dataclass(frozen=True)
class SimilarityDecomposition:
    """The Mantel correlation of two similarity matrices split into typicality and pattern (Result 4a).

    Attributes
    ----------
    mantel : r_M, the correlation of the two matrices over the M(M − 1)/2 pairs.
    typicality_share : (2,) λ² and λ'², the share of each matrix's variation
        over pairs that is due to typicality, (M − 2) Σ a_i² / Σ_{i<j} (s_ij − μ)².
    r_typicality : r_a, the correlation of the two sets of group effects across groups.
    r_pattern : r_e, the correlation of the raw (unstandardised) residuals across pairs.
    typicality_part : sqrt(λ² λ'²) r_a, the part of r_M carried by typicality.
    pattern_part : sqrt((1 − λ²)(1 − λ'²)) r_e, the part carried by pattern.
        The two parts sum exactly to ``mantel``.
    """

    mantel: float
    typicality_share: np.ndarray
    r_typicality: float
    r_pattern: float
    typicality_part: float
    pattern_part: float


def similarity_decomposition(S1, S2):
    """What a Mantel correlation is made of: shared typicality and shared pattern (Result 4a).

    With s_ij = μ + a_i + a_j + e_ij fitted to each matrix (`typicality_pattern`),

        r_M = sqrt(λ² λ'²) r_a + sqrt((1 − λ²)(1 − λ'²)) r_e,

    exactly. A positive Mantel correlation can therefore come entirely from
    groups that are unusual in both tables (the first term) without any pair
    of groups being specifically alike (the second). The claim that "groups
    with similar annotations have similar features" concerns r_e. This
    function only describes; `compare_similarity` tests the parts, with the
    pattern on standardised residuals.
    """
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    if S1.shape != S2.shape:
        raise ValueError(f"the matrices differ in shape: {S1.shape} and {S2.shape}")
    M = len(S1)
    iu = np.triu_indices(M, 1)
    parts = [typicality_pattern(S) for S in (S1, S2)]
    # over pairs, effects and residuals are orthogonal (residual row sums are zero), so
    # Σ (s_ij − μ)(s'_ij − μ') = (M − 2) Σ a_i a'_i + Σ e_ij e'_ij
    typ_ss = np.array([(M - 2) * (d.typicality**2).sum() for d in parts])
    pat_ss = np.array([(d.pattern[iu] ** 2).sum() for d in parts])
    if np.any(typ_ss + pat_ss == 0):
        raise ValueError("a matrix is constant over its off-diagonal entries; its correlation is undefined")
    with np.errstate(invalid="ignore", divide="ignore"):
        r_a = (parts[0].typicality @ parts[1].typicality) * (M - 2) / np.sqrt(typ_ss.prod())
        r_e = (parts[0].pattern[iu] @ parts[1].pattern[iu]) / np.sqrt(pat_ss.prod())
    share = typ_ss / (typ_ss + pat_ss)
    typ_part = np.sqrt(share.prod()) * np.nan_to_num(r_a)       # a part with no variation contributes 0
    pat_part = np.sqrt((1 - share).prod()) * np.nan_to_num(r_e)
    return SimilarityDecomposition(
        mantel=float(np.corrcoef(S1[iu], S2[iu])[0, 1]),
        typicality_share=share,
        r_typicality=float(r_a),
        r_pattern=float(r_e),
        typicality_part=float(typ_part),
        pattern_part=float(pat_part),
    )


def distance_similarity(profiles):
    """Negative squared Euclidean distances between group profiles, (M, M).

    For this similarity the additive model is exact: typicality is
    −M/(M−2) times each group's centred squared distance from the centroid, and
    the pattern is twice the U-centred matrix of inner products of the centred
    profiles. Correlation and
    cosine similarities divide by each group's norm, which turns shared
    typicality into apparent resemblance, so their pattern test is liberal
    when typicality is shared.
    """
    F = np.asarray(profiles, float)
    F = F - F.mean(0)
    G = F @ F.T
    d = np.diag(G)
    return 2 * G - d[:, None] - d[None, :]


def standardised_pattern(E):
    """Pattern residuals divided by the residual scale of both groups, e_ij / (σ_i σ_j).

    Atypical groups have larger residuals; when two tables share typicality
    they share this scale, and permuting groups breaks it, which makes the raw
    pattern test liberal. Standardising removes the shared scale.
    """
    E = np.asarray(E, float)
    sd = np.sqrt((E**2).sum(1) / (len(E) - 1))
    return E / np.outer(sd, sd)


@dataclass(frozen=True)
class Correlation:
    r: float
    p: float


@dataclass(frozen=True)
class SimilarityComparison:
    """Mantel correlation of two similarity matrices and its two parts, each with a group-permutation p-value."""

    mantel: Correlation
    typicality: Correlation
    pattern: Correlation

    def __str__(self):
        return "\n".join(f"{name:<11} r = {c.r:+.3f}  p = {c.p:.4f}"
                         for name, c in vars(self).items())


def compare_similarity(S1, S2, n_perm=5000, rng=None, alternative="greater", strata=None,
                       standardise=True):
    """Mantel test, and separate tests of shared typicality and shared pattern.

    Groups are permuted jointly over the rows and columns of ``S2``. A positive
    Mantel correlation alone does not show that groups with similar annotations
    have similar features; that claim concerns the pattern. Run the same
    comparison on a pair expected to share a pattern as a positive control.
    ``alternative`` is "greater" or "two-sided".

    With ``standardise`` (the default) the pattern test correlates
    `standardised_pattern` residuals, which keeps it calibrated when the tables
    share typicality. That holds for distance-based similarities (see
    `distance_similarity`); for correlation or cosine similarities the pattern
    test stays somewhat liberal under shared typicality, so confirm a
    significant pattern with distances.
    """
    S1, S2 = np.asarray(S1, float), np.asarray(S2, float)
    M = len(S1)
    iu = np.triu_indices(M, 1)
    d1, d2 = typicality_pattern(S1), typicality_pattern(S2)
    E1, E2 = d1.pattern, d2.pattern
    if standardise:
        E1, E2 = standardised_pattern(E1), standardised_pattern(E2)

    def stats(perm):
        # the decomposition (and the standardisation) is permutation-equivariant,
        # so permuting its parts equals decomposing the permuted matrix
        mantel = np.corrcoef(S1[iu], S2[np.ix_(perm, perm)][iu])[0, 1]
        typ = np.corrcoef(d1.typicality, d2.typicality[perm])[0, 1]
        pat = np.corrcoef(E1[iu], E2[np.ix_(perm, perm)][iu])[0, 1]
        return np.array([mantel, typ, pat])

    observed = stats(np.arange(M))
    null = np.array([stats(perm) for perm in draw_permutations(M, n_perm, rng, strata)])
    if alternative == "two-sided":
        p = p_value(np.abs(observed), np.abs(null))
    elif alternative == "greater":
        p = p_value(observed, null)
    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    return SimilarityComparison(*(Correlation(float(r), float(q)) for r, q in zip(observed, p)))
