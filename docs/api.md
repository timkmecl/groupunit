# API reference

Functions are importable from the top-level package, for example
`groupunit.pls`. The result classes (such as `Components` or `GroupShares`)
and the permutation helpers `p_value`, `max_t_p_value`, `min_p_value`,
`draw_permutations`, `permutation_weights` and `paired_score_correlation` are
imported from their modules, for example
`from groupunit.inference import p_value`. Modules are listed in the order of
the paper.

Two conventions hold throughout. Data are passed by position and options by
name: in the tests (`correlation_test`, `pls_test`, `cca_test`,
`estimator_test`, `permutation_test`, `compare_similarity`) every argument
after `rng` is keyword-only, and in `pairing_ratio`, `pairing_terms`, `lens`,
`lens_null` and `lens_congruence_null` so are `weights`, `tau2`, `sigma2_w`,
`rng` and `p`. Where the variance components are required inputs
(`exchangeable_weights`, `optimal_weights`, `max_effective_groups`,
`size_check`) they come in the order `tau2, sigma2_w`, as in
`VarianceComponents`. Every `weights` argument accepts
`"size"` (the group sizes; `None` means the same) or one weight per group, and
the tests also accept `"exchangeable"` (`group_weights`, `permutation_weights`).

This page is generated from the docstrings when the site is built
(`mkdocs serve`). Read as plain Markdown it shows only the directives.

## Basic quantities

::: groupunit.core
    options:
      members: [encode_groups, group_sizes, group_means, group_weights, annotation_table, wcentre, weighted_corr, weighted_basis, annotation_rank, weighted_r2, scatter_matrices]

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
      members: [typicality_pattern, TypicalityPattern, similarity_decomposition, SimilarityDecomposition, compare_similarity, SimilarityComparison, distance_similarity, standardised_pattern]

## Inference

::: groupunit.inference
    options:
      members: [correlation_test, CorrelationTest, pls_test, cca_test, SequentialTest, estimator_test, paired_score_correlation, permutation_test, PermutationResult, pairing_ratio, pairing_terms, PairingTerms, size_check, SizeCheck, permutation_weights, draw_permutations, p_value, max_t_p_value, min_p_value]

## Validation

::: groupunit.validation
    options:
      members: [leave_one_group_out, LOGOResult, linear_model, pca_regression, estimator_model]

## Simulation

::: groupunit.simulate
    options:
      members: [simulate, lognormal_sizes]
