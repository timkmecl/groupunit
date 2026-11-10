# Changelog

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
