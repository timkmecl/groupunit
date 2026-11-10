# Similarity designs

In a similarity design, each pair of groups gets a similarity from their
features and another from their annotations. The two M × M matrices are then
compared (§3.4, Results 4a-4c), as in the Mantel test or representational similarity
analysis.

## Typicality and pattern

`typicality_pattern` fits \(s_{ij} = \mu + a_i + a_j + e_{ij}\) to the
off-diagonal entries:

```python
d = gu.typicality_pattern(S)
d.mean, d.typicality, d.pattern
```

`compare_similarity` tests three things, each by permuting groups jointly over
the rows and columns of the second matrix:

```python
res = gu.compare_similarity(S_features, S_annotations, n_perm=5000, rng=0)
print(res)
```

prints one line per test, for example (illustrative values):

```text
mantel      r = +0.583  p = 0.0004
typicality  r = +0.871  p = 0.0002
pattern     r = +0.081  p = 0.1700
```

- **Mantel**: the usual comparison of all pairs. Shared typicality alone
  makes it positive.
- **Typicality**: groups unusual in one table are unusual in the other.
  This is a real association, and it predicts how distinctive a new group
  will be. It does not say which groups a new group will resemble.
- **Pattern**: groups that resemble each other in one table resemble each
  other in the other, beyond their typicality. This is usually the claim
  that motivates a similarity analysis.

Run the same comparison on a pair of matrices expected to share a pattern,
as a positive control that the pattern test can detect one.

## Which similarity to use

Typicality affects the pattern residuals in two ways.

1. **Scale.** Atypical groups have larger residuals. When two tables share
   typicality, they share this scale, and permuting groups breaks it. The raw
   pattern test is then liberal. `compare_similarity` therefore divides each
   residual by both groups' residual scales by default (`standardise=True`;
   `standardised_pattern`).
2. **Normalisation.** Correlation and cosine similarities divide by each
   group's norm. That turns shared typicality into apparent resemblance:
   atypical groups really do resemble each other once profiles are
   normalised. No additive model removes this.

For negative squared Euclidean distances the additive model is exact.
Typicality is then a linear function of each group's squared distance from
the centroid, and the pattern is twice the U-centred inner products. Use
these where you can:

```python
S_features = gu.distance_similarity(gu.group_means(X, groups))
S_annotations = gu.distance_similarity(Y)
```

With standardised residuals and distance similarities, the pattern test held
its level in the simulations. With correlation similarities it remained
liberal under shared typicality, so confirm a significant pattern with
distances. A non-significant pattern remains informative, because the bias
favours rejection.

## Unequal group sizes

Group means of small groups are noisier, and so are their similarities. If
the noisiest groups also have the most atypical annotation profiles, Mantel
and pattern tests over groups are liberal, for the reason described in
[Testing and validation](testing.md#unequal-group-sizes-check-the-pairing).
Check `pairing_ratio(Y, n, weights=np.ones(M), tau2=..., sigma2_w=...)`, and
consider excluding or pooling the smallest groups.
