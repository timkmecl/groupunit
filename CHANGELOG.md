# Changelog

## Unreleased

### Fixed

- An undefined (NaN) observed statistic got the smallest possible p-value,
  1/(n_perm + 1), and was reported as significant. `p_value` now returns NaN
  for it and counts an undefined permuted statistic as at least as extreme
  (conservative); `max_t_p_value` and `min_p_value` follow, and a NaN cell no
  longer affects the family-wise p-values of the other cells.
- `compare_similarity` with three groups returned a pattern correlation of
  NaN with p = 1/(n_perm + 1). With three groups the typicality model fits
  every pair exactly, so it now raises a `ValueError`; it also checks that
  the two matrices have the same shape.
- A single annotation passed as a vector (M,) was broadcast into an M × M
  outer product by `contributions`, `leave_out`, `lens`, `lens_null`, `pls`,
  `cca` and `pls_test`, which returned wrong results without an error, and
  made `group_shares`, `size_check` and `describe` fail. A vector is now one
  annotation column everywhere (`annotation_table`).
- `leave_out` read a boolean mask as the indices 0 and 1. It now accepts a
  mask, removes a group listed twice only once, rejects indices out of range,
  and raises if no group is left.
- `variance_components` returned NaN when every group has a single
  individual; it now raises a `ValueError`. `describe` skips the variance
  components in that case.
- With equal group sizes, `permutation_weights("exchangeable", ...)` returned
  weights summing to M instead of one, so `pls_test(...).strength` was M times
  larger than with unequal sizes. P-values were not affected.

### Changed

- In the tests (`correlation_test`, `pls_test`, `cca_test`, `estimator_test`,
  `permutation_test`, `compare_similarity`) every argument after `rng` is
  keyword-only. `pls_test` took `weights, strata` in that order and `cca_test`
  `strata, weights`; a call that passed them by position now raises a
  `TypeError` instead of silently swapping them. Calls that pass `n_perm` and
  `rng` by position are unaffected.
- In `pairing_ratio`, `pairing_terms`, `lens`, `lens_null` and
  `lens_congruence_null`, `weights`, `tau2`, `sigma2_w`, `rng` and `p` are
  keyword-only, so the two variance components cannot be swapped by position.
  Three calls in `simulations/` were updated; their results are unchanged.
- `VarianceComponents` lists its fields as `tau2, sigma2_w`, the order of every
  function that takes them. Code that reads the fields by name is unaffected.
- One vocabulary for `weights`: `pls`, `cca`, `lens`, `lens_null`,
  `lens_congruence_null`, `pairing_ratio` and `pairing_terms` accept `"size"`
  as well as `None` and an array, and the tests accept `None` as size weights.
  Arrays are checked (one finite, non-negative weight per group with a positive
  sum). New `group_weights` resolves the argument.
- `groupunit.__all__` lists the public functions. `SequentialTest.fit` is
  typed as `Components`.
- `pooled_variance_components` estimates all features at once and forms the
  regression's hat matrix once instead of once per feature, much faster
  with many features. Results are unchanged.
- New `annotation_table(Y)` in `groupunit.core`: the annotation table as an
  (M, q) array, with a vector as one column.
- `describe` computes the pairing terms once.

### Documentation

- Docstrings, README and guide pages cite the current numbering of the paper:
  the pairing ratio and its terms are Result 4 (4a–4d), the similarity
  results are Results C1–C3 in Appendix C, the proofs of the shares and of
  the variance components are Appendices B.7 and B.6, and the software is
  described in Appendix F.
- The testing and guide pages describe row-weighted CCA (`cca_test(...,
  weights="exchangeable")`), which they had said did not exist, and quote the
  current simulation results for size-weighted, exchangeable-weighted and CCA
  tests.
- `r2_null` states when the Beta null holds, as in Result 3b.
- `leave_one_group_out` warns that the target must not depend on the
  annotations: the group means of a PLS or CCA component fitted on all groups
  leak the held-out group.
- `lens_null` explains that the effective number of feature dimensions cannot
  be estimated reliably and recommends running the null at `p=1` and at the
  number of features.

## 0.2.0 (2026-11-10)

- `pairing_terms`: the pairing ratio split exactly into one term per group,
  with each group's profile share f_m and variance ratio g_m (Result 5d), so
  a report can name the group that makes a test liberal.
- `size_check`: the rejection rate of any test on simulated null data with
  the observed group sizes, annotation table and variance components, a
  direct check of a test's level for one design.
- `lens_null` and `lens_congruence_null` take `p`: with it, the null refits
  PLS on p random features and keeps the first loading, the right baseline
  for a PLS component (Result 2b).
- `similarity_decomposition`: the Mantel correlation split exactly into a
  typicality part and a pattern part, with λ², r_a and the raw r_e (Result 4a).
- `cca` and `cca_test` take `weights`: row-weighted CCA, in which group m
  carries total weight w_m. `cca_test` accepts `"exchangeable"`; its default
  stays `"size"`, ordinary CCA.
- `describe` no longer treats a pairing ratio within 0.8–1.1 as safe: where
  tests fail depends on the number of groups (Result 5d). It flags departures
  from 1 of more than 0.05, names the group with the largest term
  (`DesignSummary.pairing_group`), and always points to `size_check`.
- Docstrings and guide pages cite the restructured paper: results by number,
  sections by their new numbers.

## 0.1.0 (2026-11-04)

First release, accompanying the methods paper *The group is the unit: limits
of correlation, PLS and CCA with group-level variables*.

- Design summaries: annotation rank r, annotation ratio κ, Kish effective
  number of groups, the pairing ratio, flags for saturated designs and for
  single-group annotations (`describe`).
- Between-group share η² and the decomposition of correlations into η and
  group-level correlation (Results 1 and 3a).
- PLS correlation, CCA and LDA computed through the group means.
- Diagnostics: group shares (Shapley values of the squared loading), effective
  numbers of groups, the lens and its null, leaving out dominant groups.
- Weighting: variance components, optimal and exchangeable weights,
  model-based effective numbers of groups.
- Similarity designs: typicality–pattern decomposition, Mantel, typicality and
  standardised pattern tests, distance similarities.
- Inference: group permutation of fixed scores with max-T correction,
  sequential refitted tests of PLS and CCA, permutation within strata, tests of
  any scikit-learn style estimator or user-defined pipeline. `pls_test` and
  `correlation_test` weight groups so that their terms are exchangeable by
  default (`weights="exchangeable"`), which keeps them valid with unequal group
  sizes; `weights="size"` gives the size-weighted tests. Family-wise
  p-values use max-T or, with `correction="min-p"`, Westfall–Young min-p, for
  families whose cells differ in null spread.
- Validation: leave-one-group-out prediction against a mean baseline, with
  nested selection.
- Simulation helpers, and the paper's simulation study in `simulations/`.
