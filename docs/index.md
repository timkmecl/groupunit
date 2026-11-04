# groupunit

Many studies relate detailed measurements of individuals to variables that
exist only for the groups those individuals belong to. Text features of drug
experience reports are related to each drug's receptor profile, brain regions
to parcel-level atlas maps, and patients to characteristics of their hospital.
Because such a variable is the same for every member of a group, the data
hold one observation of it per group, however many individuals are measured.

`groupunit` implements the methods paper *The group is the unit: limits of
correlation, PLS and CCA with group-level variables* (Kmecl). It provides:

- **the quantities the paper derives**: between-group shares, group-level
  correlations, annotation rank and the annotation ratio κ, the lens;
- **the methods, written through group means**: PLS correlation, CCA and LDA;
- **diagnostics**: each group's share of a component, effective numbers of
  groups, the lens null, leaving out dominant groups, typicality versus
  pattern;
- **valid inference**: permutation tests that reassign annotations among
  groups, refitting what must be refitted, sequential tests of components, and
  weights that keep the tests valid when atypical profiles sit on large groups;
- **validation on new groups**: leave-one-group-out prediction against a mean
  baseline, with every selection step nested.

Every test resamples groups, never individuals.

## Installation

```bash
pip install -e .                  # NumPy and SciPy only
pip install -e ".[test]"          # adds pytest and scikit-learn for the test suite
pip install -e ".[docs]"          # adds MkDocs to build this documentation
```

Python 3.10 or newer is required.

## A first look

```python
import numpy as np
import groupunit as gu

# X: (N, p) features of individuals; labels: each individual's group;
# table: (M, q) annotations, one row per group, in the order of `group_names`
groups, names = gu.encode_groups(labels, order=group_names)
n = gu.group_sizes(groups, len(table))

print(gu.describe(groups, table))          # what the design can show at all
```

For a design with the dimensions of Suresh et al. (2026) the first lines read:

```text
individuals N = 4496, groups M = 21, annotations q = 19
annotation rank r = 19, annotation ratio kappa = r/(M-1) = 0.95
effective number of groups under size weights (Kish) = 10.8
at most 19 non-trivial PLS/CCA components
NEAR-SATURATED: the null mean of R² is close to 1, and PLS loadings are dominated by the lens; consider reducing the annotations.
```

The [workflow](guide.md) goes from there through analysis, diagnostics,
testing and validation. [Concepts](concepts.md) explains the ideas in a few
pages, and the [API reference](api.md) documents every function.

## Conventions

| Name     | Shape          | Meaning                                                                     |
| -------- | -------------- | --------------------------------------------------------------------------- |
| `X`      | (N, p)         | features of individuals                                                     |
| `groups` | (N,)           | integer group codes `0..M-1`; use `encode_groups` to make them from labels  |
| `Y`      | (M, q)         | annotation table, one row per group (the paper's Ỹ); row m is group m       |
| `n`      | (M,)           | group sizes                                                                 |
| `w`      | (M,)           | group weights; size weights `w = n` unless stated otherwise                 |
| `tbar`   | (M,) or (M, k) | group means of one or more component scores                                 |

Functions centre their inputs themselves: `X` over individuals, and group-level
arrays with the group weights. Results are returned as small read-only
dataclasses whose fields are documented in the [API reference](api.md).
