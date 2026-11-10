"""groupunit: analysis of individual-level features with group-level annotations.

Companion software to Kmecl, "The group is the unit: limits of correlation,
PLS and CCA with group-level variables". See `groupunit.core` for the
notation and array conventions shared by all modules.
"""
from .core import (
    annotation_rank,
    encode_groups,
    group_means,
    group_sizes,
    scatter_matrices,
    wcentre,
    weighted_basis,
    weighted_corr,
    weighted_r2,
)
from .design import describe, r2_null, reduce_annotations, single_group_annotations
from .diagnostics import (
    congruence,
    contributions,
    effective_number,
    group_shares,
    leave_out,
    lens,
    lens_congruence_null,
    lens_null,
    participation_ratio,
)
from .inference import (
    cca_test,
    correlation_test,
    estimator_test,
    pairing_ratio,
    pairing_terms,
    permutation_test,
    pls_test,
    size_check,
)
from .methods import cca, lda, pls
from .scores import between_share, decompose_score, feature_between_share, group_scores
from .similarity import (
    compare_similarity,
    distance_similarity,
    similarity_decomposition,
    standardised_pattern,
    typicality_pattern,
)
from .simulate import lognormal_sizes, simulate
from .validation import estimator_model, leave_one_group_out, linear_model, pca_regression
from .weights import (
    effective_groups,
    exchangeable_weights,
    max_effective_groups,
    optimal_weights,
    pooled_variance_components,
    variance_components,
)

__version__ = "0.1.0"
