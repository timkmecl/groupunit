"""groupunit: analysis of individual-level features with group-level annotations.

Companion software to Kmecl, "The group is the unit: limits of correlation,
PLS and CCA with group-level variables". See `groupunit.core` for the
notation and array conventions shared by all modules.
"""
from .core import (
    annotation_rank,
    annotation_table,
    encode_groups,
    group_means,
    group_sizes,
    group_weights,
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

# the public interface; result classes and permutation helpers are imported from their modules
__all__ = [
    "annotation_rank",
    "annotation_table",
    "between_share",
    "cca",
    "cca_test",
    "compare_similarity",
    "congruence",
    "contributions",
    "correlation_test",
    "decompose_score",
    "describe",
    "distance_similarity",
    "effective_groups",
    "effective_number",
    "encode_groups",
    "estimator_model",
    "estimator_test",
    "exchangeable_weights",
    "feature_between_share",
    "group_means",
    "group_scores",
    "group_shares",
    "group_sizes",
    "group_weights",
    "lda",
    "leave_one_group_out",
    "leave_out",
    "lens",
    "lens_congruence_null",
    "lens_null",
    "linear_model",
    "lognormal_sizes",
    "max_effective_groups",
    "optimal_weights",
    "pairing_ratio",
    "pairing_terms",
    "participation_ratio",
    "pca_regression",
    "permutation_test",
    "pls",
    "pls_test",
    "pooled_variance_components",
    "r2_null",
    "reduce_annotations",
    "scatter_matrices",
    "similarity_decomposition",
    "simulate",
    "single_group_annotations",
    "size_check",
    "standardised_pattern",
    "typicality_pattern",
    "variance_components",
    "wcentre",
    "weighted_basis",
    "weighted_corr",
    "weighted_r2",
]

__version__ = "0.2.0"
