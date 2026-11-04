# Changelog

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
