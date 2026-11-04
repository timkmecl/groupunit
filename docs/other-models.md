# Other models and scores

## Scores from any method

All score-level tools apply to scores from any method: correlations, η²,
group shares, the lens null and leaving out groups. The method can be linear
or not. The identity \(t^\top Y = \sum_m n_m \bar t_m \tilde y_m\) holds for any
vector t, so a UMAP coordinate, an autoencoder unit, a topic proportion or a
language-model rating is handled exactly like a principal component.

Which test applies depends on one question: did the method see the
annotations when it built the scores?

**No** (PCA, NMF, topic models, between-group PCA, LDA on group labels, UMAP,
autoencoders, embeddings). Compute the scores once and hold them fixed:

```python
from sklearn.decomposition import PCA

scores = PCA(5).fit_transform(X)
gu.correlation_test(scores, groups, Y)
gu.group_shares(gu.group_scores(scores, groups), Y, n)
gu.between_share(scores, groups)
```

Discriminant scores and between-group PCA separate groups in sample even
when groups do not differ. The test stays valid, but estimate η² on held-out
individuals before reporting it.

**Yes** (PLS regression, CCA, any supervised model). Refit in every
permutation with `estimator_test`, below. For diagnostics, take the fitted
model's scores:

```python
from sklearn.cross_decomposition import PLSRegression

fitted = PLSRegression(3, scale=False).fit(X, Y[groups])
tbar = gu.group_scores(fitted.transform(X), groups)
gu.group_shares(tbar, Y, n)
```

**Partly** (unsupervised scores, but the number of components or the
features kept was chosen by how well they relate to the annotations). Put
the choice inside a `permutation_test` statistic.

## Models from scikit-learn

The package implements PLS correlation, CCA and LDA itself, as SVDs computed
through the group means. Its results agree with scikit-learn's:
`pls` matches `PLSSVD(scale=False)` exactly, and `cca` matches `CCA` to the
tolerance of scikit-learn's iterative fit. The package keeps its own versions
because they are exact, which matters at saturation, where permutation tests
must count exact ties. They are also faster under permutation, and they never
rescale silently: scikit-learn's default `scale=True` standardises columns
over individuals and changes the PLS weights.

Any scikit-learn style model can still be passed in.

### Permutation tests

`estimator_test` copies the estimator and fits it on the individuals with
the inherited annotations, once for the real table and once per reassignment
of its rows among groups:

```python
res = gu.estimator_test(PLSRegression(n_components=3, scale=False), X, groups, Y, n_perm=1000, rng=0)
res.p_value, res.p_fwer

r2 = lambda fitted, X, Y_individuals: fitted.score(X, Y_individuals)
gu.estimator_test(PLSRegression(3, scale=False), X, groups, Y, statistic=r2)
```

By default it reports, for each component, the correlation between the paired
scores from `transform(X, Y)`. Components are compared by position, so later
components are tested conservatively. Use `pls_test` or `cca_test` for
sequential tests. Iterative fits are not exact, so at saturation the p-value
is arbitrary rather than 1. Check `describe` first.

### Validation

`leave_one_group_out` accepts any regressor with `fit` and `predict`, and
refits a fresh copy in every fold. A pipeline therefore standardises or
reduces the annotations on the training groups only:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), PCA(3), RidgeCV())
gu.leave_one_group_out(target, Y, model=model).r2
```

Unequal `weights` are passed to `fit` as `sample_weight`. Pipelines don't
accept that argument, so use equal weights with them.

### Regularised discriminant analysis

When features outnumber groups or individuals, use regularised LDA scores as
fixed scores:

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, groups)
gu.correlation_test(lda.transform(X)[:, :3], groups, Y)
```
