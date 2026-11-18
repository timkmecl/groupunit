# Reanalysing published results

The diagnostics need only group-level quantities: each group's mean score on
a component, the annotation table and the group sizes. These are often
available from published figures and supplements, so a published component
can be examined without the individual-level data.

```python
tbar = np.array(digitised_drug_means)       # (M,) or (M, k), one row per drug
Y = binding_table                            # (M, q), the table as analysed
n = reports_per_drug                         # (M,)

shares = gu.group_shares(tbar, Y, n)         # which drugs each component depends on
gu.leave_out(tbar, Y, n, shares.top(1))      # the loading without the dominant drug
gu.lens(Y, n).nu                             # how concentrated null loadings are (r bounds their directions)
gu.lens_congruence_null(published_loading, Y, n, rng=0)   # baseline for a score fixed in advance
```

For a published PLS or CCA loading, the baseline must refit the method on
random group means. Run it with one feature and with the number of features
the study used, and rely only on conclusions that hold at both (§5):

```python
for p in (1, n_features):
    null = gu.lens_congruence_null(published_loading, Y, n, rng=0, p=p)
```

## Checking digitised values

Two properties help to check digitised values:

- Scores centred over individuals have size-weighted group means that sum to
  zero: `n @ tbar` should be close to 0.
- Loadings rebuilt from the identity, `gu.contributions(tbar, Y, n).sum(0)`,
  should be proportional to the published loadings. Check with
  `gu.congruence(rebuilt, published)`.

## Tests from group means

`correlation_test` takes individual scores, but its statistic depends on them
only through η and the group means. With fixed scores η does not change
under permutation, so testing the group-level correlations is equivalent.
That test can run on group means directly:

```python
def statistic(Y_perm):
    return np.abs(gu.weighted_corr(tbar, Y_perm, n)).max()

gu.permutation_test(statistic, Y, n_perm=10000, rng=0)
```

This is valid only for components that were fixed before the annotations
entered (PCA, text components, CCA factors at κ = 1). Published PLS components,
or CCA components with κ < 1, were already selected using the real
annotations and cannot be tested from published figures.

## Reconstructing group means from loadings

When only loadings are published, the identity can be inverted. Solve
\(\tilde Y^\top D \bar t \propto \ell\) together with \(\sum_m n_m \bar t_m = 0\).
The solution is unique only up to the null space of \(\tilde Y^\top D\). The
constraint removes the constant vector from that space, but whenever
r < M − 1 other directions remain. Check how much of the solution falls on
lightly weighted groups before relying on the shares it implies.
