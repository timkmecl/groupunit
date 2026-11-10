# groupunit

Software for designs in which detailed measurements of individuals are related
to variables that exist only for the groups they belong to: text features of
drug experience reports and each drug's receptor profile, brain regions and
parcel-level atlas maps, patients and characteristics of their hospital.

It accompanies Kmecl, *The group is the unit: limits of correlation, PLS and
CCA with group-level variables* (methods paper, Appendix D). The package
computes the quantities the paper derives, and the diagnostics, tests and
validation it recommends. Every test resamples groups, never individuals.

## Installation

```bash
pip install -e .              # NumPy and SciPy only
pip install -e ".[test]"      # + pytest and scikit-learn; run `pytest`
pip install -e ".[docs]"      # + MkDocs; run `mkdocs serve`
```

Requires Python 3.10+.

## Example

```python
import numpy as np
import groupunit as gu

groups, names = gu.encode_groups(report_drug, order=drug_table.index)   # labels -> 0..M-1
Y = drug_table.to_numpy(float)                                         # (M, q), one row per group
n = gu.group_sizes(groups, len(Y))

print(gu.describe(groups, Y, X))           # N, M, r, kappa = r/(M-1), effective groups, warnings

Y3, _ = gu.reduce_annotations(Y, n, 3)     # reduce in advance if kappa is near 1
test = gu.pls_test(X, groups, Y3, n_perm=5000, rng=0)   # refitted under group permutation
shares = gu.group_shares(test.fit.group_scores, Y3, n)  # which groups carry each component
print(test.p_value, shares.m_eff, gu.pairing_ratio(Y3, n))
```

`examples/walkthrough.py` goes through the paper's checklist on simulated data.

## Documentation

The documentation is published at https://timkmecl.github.io/groupunit/.
Its source is in `docs/`; build it locally with `mkdocs serve`, or read the
Markdown directly:

- [Concepts](docs/concepts.md): the ideas behind the package, in a few pages.
- [Workflow](docs/guide.md): from data to reporting, following the paper's checklist.
- [Testing and validation](docs/testing.md): what to permute, what to refit, unequal group sizes.
- [Similarity designs](docs/similarity.md): typicality versus pattern.
- [Published results](docs/published.md): diagnostics from digitised group means.
- [Other models and scores](docs/other-models.md): scikit-learn models, scores from any method.
- [Simulations](docs/simulations.md): the simulation study of §6.
- [API reference](docs/api.md): generated from the docstrings; readable only in the built site.

## Layout

| Module        | Paper           | Contents                                                                  |
| ------------- | --------------- | ------------------------------------------------------------------------- |
| `core`        | §2              | group codes and means, weighted centring, correlation and R², annotation rank |
| `design`      | §3.3, Box 1     | `describe`, `r2_null`, `reduce_annotations`                                |
| `scores`      | Results 1, 3a   | between-group share η², `decompose_score`                                  |
| `methods`     | §3.2–3.3        | `pls`, `cca` (optionally row-weighted), `lda`, computed through the group means |
| `diagnostics` | §5, Result 2b   | group shares, effective numbers, the lens and its null, `leave_out`        |
| `weights`     | §4.1            | variance components, optimal and exchangeable weights, effective groups    |
| `similarity`  | §3.4, Results 4a–4c | typicality and pattern, `similarity_decomposition`, `compare_similarity`, `distance_similarity` |
| `inference`   | §4.2–4.5, Result 5 | `correlation_test`, `pls_test`, `cca_test`, `estimator_test`, `permutation_test`, `pairing_ratio`, `pairing_terms`, `size_check` |
| `validation`  | §4.5            | `leave_one_group_out`, `pca_regression`, scikit-learn models               |
| `simulate`    | §6.1            | `simulate`, `lognormal_sizes`                                              |

`simulations/` holds the simulation study of the paper. `tests/` checks the
identities against direct computation and pins the numbers of the paper's
toy figures. GitHub Actions run the tests, publish the documentation to
GitHub Pages and release to PyPI. AGENTS.md ("CI and releasing") lists the
steps.
