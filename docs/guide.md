# Workflow

This page goes from data to reporting along the lines of the paper's
checklist (Box 1). The code runs on any data in the
[conventions](index.md#conventions). To try it on simulated data, use

```python
import numpy as np
import groupunit as gu

rng = np.random.default_rng(0)
n = gu.lognormal_sizes(20, 60, sigma=0.8, rng=rng)
X, groups, Y = gu.simulate(n, p=8, q=15, beta=0.5, tau2=0.5, sigma2_w=4.0, rng=rng)
```

`examples/walkthrough.py` runs the same steps end to end.

## 1. Prepare the data

Group codes must match the rows of the annotation table. Build them from
labels with the table's row order:

```python
groups, names = gu.encode_groups(report_drug, order=drug_table.index)
Y = drug_table.to_numpy(float)
n = gu.group_sizes(groups, len(Y))
```

Missing or censored annotation values filled in with a fixed number can
encode unrelated group properties (§8, Limitations). Code them deliberately before
using the package.

## 2. Describe the design

```python
summary = gu.describe(groups, Y, X, annotation_names=drug_table.columns)
print(summary)
```

`describe` reports N, M, q, the annotation rank r and κ = r/(M − 1), the Kish
effective number of groups, the largest number of non-trivial components, and
the between-group share of the features. It flags saturated and
near-saturated designs, and annotations in which only one group differs from
the rest (`single_group_annotations`). Such an annotation correlates with any
score exactly as that group's mean does.

## 3. Choose the target and the weights

Size weights (the default everywhere) describe the population of
individuals; equal weights (`weights=np.ones(M)`) describe the population of
groups. Choose by the research question, then check efficiency:

```python
vc = gu.variance_components(score, groups, Y)       # σ²_w and τ² for a score
gu.effective_groups(n)                               # Kish
gu.effective_groups(n, n, vc.tau2, vc.sigma2_w)      # model-based, never larger
gu.max_effective_groups(n, vc.tau2, vc.sigma2_w)     # the ceiling, at most M
```

## 4. Reduce the annotations if κ is close to 1

```python
Y3, explained = gu.reduce_annotations(Y, n, k=3, standardise=True)
```

Choose k in advance. If k is chosen from the data, the choice must be
repeated inside every test and validation fold (see
[Testing and validation](testing.md)).

With many features, prefer PLS when they share a few dimensions, and reduce
them without using the annotations (for example to a few principal
components) before CCA. If the relation is confined to a few of many
independent features, expect low power even with a valid test (§6.2).

## 5. Fit

The design decides the method:

| Design                                      | Function                                    |
| ------------------------------------------- | ------------------------------------------- |
| scores from the features alone, then correlate | any scores, e.g. PCA, or `lda(X, groups)` |
| PLS correlation                             | `pls(X, groups, Y)`                         |
| CCA                                         | `cca(X, groups, Y)`                         |
| similarity matrices                         | see [Similarity designs](similarity.md)     |

All three methods return `Components`. It holds feature weights,
annotation weights, the group means of the feature scores
(`group_scores`) and the annotation loadings (`y_loadings`, covariances of the
annotations with the scores). Use `.scores(X)` for individual scores.

```python
fit = gu.pls(X, groups, Y3)
fit.strength          # singular values
fit.group_scores      # (M, k): what the annotations see
```

For any single score, `decompose_score` separates group separation from the
relation to the annotations:

```python
d = gu.decompose_score(fit.scores(X)[:, 0], groups, Y3)
d.eta2, d.group_corr, d.corr      # corr = sqrt(eta2) * group_corr
```

## 6. Diagnose

```python
shares = gu.group_shares(fit.group_scores, Y3, n)       # all components at once
shares.share[:, 0], shares.m_eff                         # shares of component 1; M_eff per component
top = shares.top(2)

gu.leave_out(fit.group_scores[:, 0], Y3, n, top[0])      # without the dominant group
lens = gu.lens(Y, n)                                      # the lens of the full table
lens.nu, lens.cumulative_share[:3]
fit_full = gu.pls(X, groups, Y)
loading = fit_full.y_loadings[:, 0]
for p in (1, X.shape[1]):                                 # PLS refitted on p random features
    null = gu.lens_congruence_null(loading, Y, n, rng=0, p=p)
    print(p, np.quantile(null, [0.5, 0.95]))              # congruence from random group means

gu.between_share(fit.scores(X), groups)                   # η² of each component
gu.feature_between_share(X, groups)                       # η² of the feature space
```

The effective number of feature dimensions behind a PLS loading lies
between 1 and the number of features and cannot be estimated reliably, so
the null is run at both ends. Rely only on conclusions that hold at both.
With equal group sizes, pass `means=gu.group_means(X, groups)` instead of
`p`. The observed group means are then shuffled among groups, and no bracket
is needed (§5).

`contributions(tbar, Y, n)` gives the group × annotation matrix of terms
behind a loading, before projection.

## 7. Test

See [Testing and validation](testing.md). In short:

```python
gu.correlation_test(scores, groups, Y)       # scores built without Y: held fixed
gu.pls_test(X, groups, Y3)                   # PLS: refitted, sequential
gu.cca_test(X, groups, Y3)                   # CCA: refitted, sequential; refuses κ = 1
gu.pairing_terms(Y3, n)                      # the pairing ratio and its largest term; report both
```

`correlation_test` and `pls_test` weight groups so that their terms are
exchangeable. This corrects the pairing problem for PLS, and for scores that
were not computed over individuals. `cca_test` weights groups by size unless
you pass `weights="exchangeable"` (row-weighted CCA). With a pairing ratio
well above 1, prefer reweighted PLS. In every case, check the level of the
test you run with `size_check` (see
[Testing and validation](testing.md#unequal-group-sizes-check-the-pairing)).

For one score fixed in advance, a random-intercept mixed model tested with
F(r, M − r − 1) is an alternative to group permutation (§4.6). See
[Testing and validation](testing.md#one-score-a-mixed-model).

## 8. Validate on new groups

```python
target = gu.group_scores(X[:, 0], groups)
gu.leave_one_group_out(target, Y, weights=n, model=gu.pca_regression(3)).r2
```

The out-of-sample R² is relative to predicting each held-out group by the
mean of the others, and it can be negative.

## 9. Other group-level variables

Any other variable that is constant within groups (class, route, dose,
region) is confounded with the annotations (§4.5). Permuting within its
levels tests the annotations beyond it:

```python
gu.correlation_test(scores, groups, Y, strata=drug_class)
```
