# API reference

Functions are importable from the top-level package, for example
`groupunit.pls`. The result classes (such as `Components` or `GroupShares`)
and the permutation helpers `p_value`, `max_t_p_value`, `draw_permutations`
and `paired_score_correlation` are imported from their modules, for example
`from groupunit.inference import p_value`. Modules are listed in the order of
the paper.

This page is generated from the docstrings when the site is built
(`mkdocs serve`). Read as plain Markdown it shows only the directives.

## Basic quantities

::: groupunit.core
    options:
      members: [encode_groups, group_sizes, group_means, wcentre, weighted_corr, weighted_basis, annotation_rank, weighted_r2, scatter_matrices]

## Design

::: groupunit.design
    options:
      members: [describe, DesignSummary, single_group_annotations, r2_null, reduce_annotations]

## Scores

::: groupunit.scores
    options:
      members: [between_share, feature_between_share, group_scores, decompose_score, ScoreDecomposition]

## Methods

::: groupunit.methods
    options:
      members: [pls, cca, lda, Components]

## Diagnostics

::: groupunit.diagnostics
    options:
      members: [group_shares, GroupShares, contributions, effective_number, participation_ratio, lens, Lens, lens_null, lens_congruence_null, congruence, leave_out, LeaveOut]

## Weights

::: groupunit.weights
    options:
      members: [variance_components, pooled_variance_components, VarianceComponents, optimal_weights, exchangeable_weights, effective_groups, max_effective_groups]

## Similarity

::: groupunit.similarity
    options:
      members: [typicality_pattern, TypicalityPattern, compare_similarity, SimilarityComparison, distance_similarity, standardised_pattern]

## Inference

::: groupunit.inference
    options:
      members: [correlation_test, CorrelationTest, pls_test, cca_test, SequentialTest, estimator_test, paired_score_correlation, permutation_test, PermutationResult, pairing_ratio, draw_permutations, p_value, max_t_p_value]

## Validation

::: groupunit.validation
    options:
      members: [leave_one_group_out, LOGOResult, linear_model, pca_regression, estimator_model]

## Simulation

::: groupunit.simulate
    options:
      members: [simulate, lognormal_sizes]
