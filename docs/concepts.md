# Concepts

This page summarises the ideas behind the package. The paper gives the
derivations; section numbers below refer to it.

## One observation of the annotation per group

Each of N individuals belongs to one of M groups, and inherits its group's
row of the annotation table Ỹ. For any score t computed from the features,

$$ t^\top Y = \sum_{m=1}^{M} n_m \, \bar t_m \, \tilde y_m, \qquad X^\top Y = \bar X^\top D \tilde Y, $$

where \(\bar t_m\) is the mean score in group m and D holds the group sizes
(§2.2). Everything that relates features to annotations through this
cross-product sees the M group means, weighted by size. Correlation, PLS, CCA
and regression all work this way. Variation within groups cancels.
Large N sharpens each group mean, but the annotation is still observed M
times.

The methods in `groupunit.methods` compute their fits from the group means
for this reason, and never form the N × q matrix of inherited annotations.

## Between-group share and attenuation

The **between-group share** η² of a score is the fraction of its variance that
lies between groups (`between_share`). The correlation of a score with an
annotation, computed over individuals, is the group-level correlation
attenuated by η (Result 1):

$$ \operatorname{corr}(t, y) = \eta_t \cdot r_n(\bar t, \tilde y). $$

The group-level correlation is computed from M points. A score that mostly describes
within-group variation, such as a principal component tracking report length,
correlates weakly with every annotation even when its group means correlate
strongly. `decompose_score` returns both factors.

## The annotation ratio and saturation

The rank r of the centred annotation table is the number of independent
between-group directions the annotations can express. The **annotation ratio**
κ = r / (M − 1) sets the regime (§3.3), and `describe` reports it.

- **κ = 1, saturated.** The annotations can express every difference between
  groups. CCA then gives identical results for *any* annotation table of full
  rank, including shuffled or random ones, and equals discriminant analysis
  of the group labels (Result 3c). It cannot respond to the annotation values.
- **κ near 1.** The group-level R² of a score on the annotations has a null
  mean of about κ (Result 3b, `r2_null`), and more with size weights and
  unequal group sizes, so CCA mostly maximises group separation.
- **κ ≪ 1, informative.** The annotations constrain which between-group
  patterns can be expressed. Reducing the annotations in advance
  (`reduce_annotations`) moves a design here.

## The lens

A component's annotation loading is \(\ell \propto \tilde Y^\top D \bar t\). The
annotation table and group sizes are fixed by the design, and only \(\bar t\)
comes from the features. Random group means therefore still produce
structured loadings, concentrated on the leading eigenvectors of the **lens**
\(\Lambda = \tilde Y^\top D^2 \tilde Y\) (Result 2b). The largest groups with the most
unusual profiles set those eigenvectors. `lens` computes Λ and its participation
ratio ν, the number of annotation directions on which null loadings tend to
concentrate. `lens_null` and `lens_congruence_null` give the loadings and
congruences that random group means produce. Published loadings, and
agreement between two analyses of the same groups, should be judged against
this baseline.

PLS chooses the best of many feature directions, so its loadings concentrate
on the leading direction of the lens more than a fixed score's do. Its null
must refit PLS on random group means (`p=`). The right number of feature
dimensions lies between 1 and the number of features and cannot be estimated
reliably, so the null is run at both ends (§5). With equal group sizes,
shuffling the observed group means among groups (`means=`) avoids the
bracket.

## Which groups a component depends on

Each group contributes one term \(n_m \bar t_m \tilde y_m\) to a loading.
Projected onto the loading, the terms give **group shares** that sum to one and
equal the groups' Shapley values for the squared loading (§5, Appendix B.7,
`group_shares`). The **effective number of groups** behind a component is the
inverse Simpson index of the absolute shares. A value of 2 or 3 means the
component is a contrast between a few groups. `leave_out` recomputes the
loading without the dominant groups. Removing a group also moves the centroid,
so the loading loses more than that group's own term.

## Weights and effective sample size

Size weights describe the population of individuals; equal weights describe
the population of groups (§4.1). A group mean deviates from what the
annotations predict through sampling error (σ²_w / n_m) and group-level
heterogeneity (τ²). The most precise weights are ∝ 1 / (τ² + σ²_w / n_m), and
no weighting gives more than M effective groups (`variance_components`,
`optimal_weights`, `effective_groups`, `max_effective_groups`).

## Groups are the exchangeable units

A permutation test is valid when the units it reassigns are exchangeable
under the null. The annotation profile belongs to the group, so tests
reassign annotation rows among groups, never among individuals (§4.2).
Reassigning among individuals tests only whether groups differ.

With unequal group sizes, exchangeability also needs the annotation profiles
to be unrelated to how variable each group's term is. Under a weighted
statistic, group m enters through a term of variance
\(w_m^2 (\tau^2 + \sigma_w^2 / n_m)\). If atypical profiles belong to groups
with high-variance terms, as when a large group has an unusual profile and
size weights are used, the observed assignment is atypical among
reassignments, and the test is liberal (Result 4a). `pairing_ratio` measures
this from the annotation table and group sizes alone. `exchangeable_weights`,
the default weighting of `pls_test` and `correlation_test`, gives every term
the same variance (Result 4d). This corrects PLS, but not correlation tests of
scores computed over individuals, and a ratio near 1 does not guarantee the
level, so check it with `size_check`. See [Testing and validation](testing.md).

## Typicality and pattern

Any similarity matrix between groups splits into each group's **typicality**
(how similar it is to the set as a whole) and the **pattern** (which groups
resemble each other beyond that), \(s_{ij} = \mu + a_i + a_j + e_{ij}\) (Appendix C, Result C1).
Two similarity matrices can correlate through shared typicality alone. The
claim that similar annotations go with similar features concerns the
pattern. See [Similarity designs](similarity.md).
