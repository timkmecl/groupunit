# Testing and validation

## Permute groups, never individuals

Every test reassigns rows of the annotation table among groups, keeping each
group's individuals and size together (§4.2). Reassigning annotation values
among individuals breaks their constancy within groups. Such a test asks only
whether individuals differ between groups, and it rejects almost always once
groups differ at all (see [Simulations](simulations.md)).

With M groups there are M! reassignments, more than enough above about 7
groups.

## What to refit

| Scores or components                                              | Test                                   |
| ----------------------------------------------------------------- | -------------------------------------- |
| built without the annotations: features chosen in advance, PCA, LDA on group labels, CCA factors at κ = 1 | `correlation_test`, scores held fixed |
| PLS                                                               | `pls_test`, refitted in every permutation |
| CCA with κ < 1                                                    | `cca_test`, refitted in every permutation |
| a model from another library                                      | `estimator_test`                       |
| any pipeline with data-driven choices                             | `permutation_test`                     |

Holding components fixed when they were chosen using the annotations is
invalid. The test then reuses components already selected to match the
real annotations.

### Fixed scores

```python
res = gu.correlation_test(scores, groups, Y, n_perm=5000, rng=0)
res.r          # (k, q) correlations over individuals, η · r_n
res.p_value    # per cell, uncorrected
res.p_fwer     # family-wise over all k × q cells (max-T)
```

The family-wise p-values already account for reporting the most extreme
cell. Each score's η is unchanged by permutation, so the test concerns the
group-level correlations.

### Max-T or min-p

`correlation_test`, `permutation_test` and `estimator_test` take a
`correction`. Both corrections control family-wise error. They differ in power
when the cells' null distributions differ:

- **`correction="max-t"`** (the default) compares each cell with the largest
  statistic over all cells in each permutation. It suits a family of the same
  statistic with similar null spread: correlations of balanced scores, one
  statistic across many annotations.
- **`correction="min-p"`** first turns each cell into a p-value against its
  own null, then compares with the smallest p-value over the family. It suits
  cells whose nulls differ. One example is selection statistics, such as each
  component's best |r| over many annotations. Another is components
  concentrated on different numbers of groups, which you can see from
  `group_shares(...).m_eff`. Under max-T, the cell with the widest null sets
  the bar for all of them.

In the package's tests, with one cell being a best-of-30 statistic, max-T
detected a real relation in the other cell 22% of the time and min-p 55%.
Choose the correction before looking at the results, from the kind of family
you have, not from which gives the smaller p-value. Min-p p-values are
discrete with floor 1/(n_perm + 1), so large families need more permutations.

`pls_test` and `cca_test` are not affected. They test each component against
its own null and step down sequentially.

### Refitted components, tested sequentially

```python
res = gu.pls_test(X, groups, Y, n_perm=5000, rng=0)
res.strength, res.p_value       # p-values are non-decreasing: stop at the first non-significant one
res.fit                          # the fitted Components on the real annotations
```

Component c is tested after removing components 1..c−1 from both sides
(Winkler et al. 2020). Comparing the c-th observed value with the c-th
permuted value would be conservative after a strong first component.

`cca_test` does the same for CCA. It raises an error when κ = 1, because
every reassignment then gives the same result (Result 3c).

### Pipelines and selection steps

`pls_test` and `cca_test` refit the model but cannot know about choices made
before it: the annotation subset, the number of components, which cell to
report. If those choices depended on the data, the whole pipeline goes into
the statistic of `permutation_test`, which receives the reassigned table:

```python
tbar = gu.group_scores(score, groups)

def statistic(Y_perm):
    # selection repeated in every permutation
    best = np.argmax(np.abs(gu.weighted_corr(tbar, Y_perm, n)))
    return abs(gu.weighted_corr(tbar, Y_perm[:, best], n))

gu.permutation_test(statistic, Y, n_perm=5000, rng=0)
```

## Unequal group sizes: check the pairing

Group permutation is exact when groups are exchangeable. With unequal sizes
they need not be. Under the null, group m enters a weighted cross-product
through a term of variance \(w_m^2(\tau^2 + \sigma_w^2/n_m)\). Suppose the
groups with high-variance terms also have atypical annotation profiles, for
example the largest group under size weights, or the smallest and noisiest
group under equal weights. Then the observed statistic is larger than under
most reassignments, by design alone, and the test is liberal.

`pairing_ratio` measures this from the annotation table and group sizes:

```python
gu.pairing_ratio(Y, n)                                  # size weights, equally variable group means
gu.pairing_ratio(Y, n, tau2=vc.tau2, sigma2_w=vc.sigma2_w)
gu.pairing_ratio(Y, n, weights=np.ones(len(n)), tau2=vc.tau2, sigma2_w=vc.sigma2_w)
```

The ratio is the expected squared loading under the null for the observed
assignment, divided by its average over reassignments. Above 1 a test with
those weights leans liberal, below 1 conservative, and no range of the ratio
is safe for every design: one atypical group's share of the profile spread
shrinks like 1/M, so the same ratio does more harm with more groups (Result
4c). In the paper's simulations, size-weighted PLS rejected a true null in
3%, 5% and 22% of data sets at ratios of 1.1–1.2, 1.2–1.3 and 1.3–1.45 with 20
groups, but in 12%, 38% and 68% with 50 groups (§4.3, Figure 6).
`describe` reports the ratio and the group with the largest term
(`pairing_terms`), and flags departures from 1 of more than 0.05; a ratio near
1 is not a guarantee. To check the level of the test you actually run, use
`size_check`, which simulates null data with your group sizes, annotation table
and variance components:

```python
vc = gu.pooled_variance_components(X, groups)   # around the grand mean, so possible at any κ
check = gu.size_check(
    lambda X, g, Y, rng: gu.pls_test(X, g, Y, n_components=1, n_perm=199, rng=rng).p_value[0],
    n, Y, vc.tau2, vc.sigma2_w, p=X.shape[1], n_sim=400)
check.rate, check.se                             # estimated size of the test for this design
```

The remedy is to weight groups so that every term is equally variable, and
`pls_test` and `correlation_test` do this by default (`weights="exchangeable"`).
They estimate τ² and σ²_w under the null, pooled over the features (or
scores), and weight groups ∝ 1/√(τ² + σ²_w / n_m). With equal group sizes these
are equal weights, so nothing changes. If the estimate of τ² is zero they are
∝ √n_m, between equal and size weights: right if groups differ only by
sampling error, and only a partial correction if they also differ beyond it,
which is most likely to go unnoticed with few groups. Report the pairing ratio and its largest term as a
description of the design.

```python
gu.pls_test(X, groups, Y)                        # exchangeable weights (default)
gu.pls_test(X, groups, Y, weights="size")        # size weights: liberal when the ratio is above 1
gu.pls_test(X, groups, Y, weights=my_weights)    # any weights, one per group

vc = gu.pooled_variance_components(X, groups)    # the same weights by hand
w = gu.exchangeable_weights(n, vc.tau2, vc.sigma2_w)
```

With the default, `pls_test(...).fit` is the PLS fit with exchangeable
weights, and `correlation_test(...).r` holds weighted group-level
correlations. For a size-weighted description of the data, such as loadings
and group shares that describe the individuals, fit `gu.pls(X, groups, Y)`
separately, or pass `weights="size"` to `correlation_test` to get
individual-level correlations.

These weights change the statistic, not the hypothesis: under the null of no
association, any weighting is a valid statistic, and these make the
permutation distribution correct. In the paper's simulations they kept PLS
correlation at about 8% or below at a nominal 5% in every design, including
several atypical groups, profiles that grow with group size and heavy-tailed
noise, and were more powerful than size weights; PLS regression exceeded
that in one cell (11%). They are not the optimal weights for estimation
(`optimal_weights`, the square of these).

The scores must not bring the size weighting back in. Principal components
computed over individuals line up with the largest groups' means, so those
groups' terms come out extreme. In the simulations, exchangeable-weight
correlation tests of principal components stayed liberal in some designs (up
to 15%). Prefer scores chosen in advance, or computed from group means with
the same weights.

Ordinary CCA weights groups by size internally, and `cca_test` does so by
default; in the simulations it was less affected than PLS but still liberal
(up to 24%). `cca_test(..., weights="exchangeable")` fits row-weighted CCA, in
which each individual of group m gets weight w_m / n_m. It held its level in
every design but was often conservative and, in pairing designs, less powerful
than reweighted PLS. When the ratio is far from 1, test with reweighted PLS;
row-weighted CCA is a valid alternative when the annotations are few (κ around
0.1), where it was as powerful.

## Other group-level variables

Pass `strata` (one label per group) to permute only within levels of another
group-level variable, such as drug class. The test then asks whether the
annotations explain anything beyond that variable (§4.5):

```python
gu.pls_test(X, groups, Y, strata=drug_class)
```

## Leave-one-group-out validation

A claim that a relation holds for groups not yet studied is a prediction
(§4.5). `leave_one_group_out` predicts each group's group-level target from
its annotations, fitting on the other groups. It compares the result with
predicting the weighted mean of the other groups:

```python
target = gu.group_scores(score, groups)
res = gu.leave_one_group_out(target, Y, weights=n)     # default: weighted least squares
res.r2, res.predictions, res.baseline
```

The model must contain every data-dependent step, so that each step is
refitted in every fold. `pca_regression(k)` reduces the annotations on the
training groups only. Any scikit-learn regressor or pipeline also works (see
[Other models and scores](other-models.md)):

```python
gu.leave_one_group_out(target, Y, model=gu.pca_regression(3))
```

Splitting individuals instead of groups puts every held-out individual's group
in the training set. Such a split measures recognition of known groups, not
generalisation to new ones. With few groups, even correct estimates carry
wide error bars.
