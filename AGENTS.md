# AGENTS.md

Guidance for coding agents working on this repository.

## What this is

`groupunit` is the companion package to the methods paper *The group is the unit:
limits of correlation, PLS and CCA with group-level variables* (Kmecl; Appendix B).
It analyses individual-level features `X` against annotations `Y` that exist only
per group. The paper's argument is that the group, not the individual, is the unit
of evidence, and the code must never contradict it:

- Permutation tests reassign rows of the annotation table among groups, never among individuals.
- Validation leaves groups out, never individuals.
- Anything data-dependent goes inside the permutation or validation loop.

## Layout

```
src/groupunit/   one module per paper section (see the table in README.md)
tests/           pytest; conftest.py holds a simulated design and the paper's toy data
examples/        walkthrough.py, the paper's checklist (Box 1) on simulated data
docs/            MkDocs site: hand-written guide pages + API reference generated from docstrings
simulations/     the paper's simulation study (Section 8); not part of the package
drafts/          drafts of paper text produced alongside the code
```

Module dependencies go one way: `core` → `scores`, `diagnostics`, `methods` →
`weights` → `inference` → `design`, `similarity`; `validation` stands alone.
Keep it acyclic.

## Commands

```bash
pip install -e ".[test]"
pytest
python examples/walkthrough.py
mkdocs build --strict              # needs ".[docs]"; must stay warning-free
python simulations/run_all.py --quick
```

Docstrings are the API reference: keep them in numpy style (`Attributes`,
`Returns` sections parse; a stray "Returns …" line inside `Parameters` breaks
the strict build). A new public function needs an entry in `docs/api.md`.

The suite runs in a few seconds. Keep it that way: small designs, few hundred
permutations, fixed seeds.

## CI and releasing

`.github/workflows/`:

- `tests.yml` runs pytest on Python 3.10–3.13 for every push to `main` and every pull request.
- `docs.yml` builds the site strictly on every push and pull request, and publishes it
  to GitHub Pages from `main`.
- `publish.yml` builds the sdist and wheel and uploads them with trusted publishing:
  to PyPI when a GitHub release is published, and to TestPyPI when run by hand.

The version lives only in `src/groupunit/__init__.py` (`__version__`); `pyproject.toml`
reads it. To release: bump `__version__`, commit, run `publish` by hand to check the
TestPyPI upload, then create a GitHub release tagged `v<version>`.

One-time setup, done by the repository owner, not by an agent:

1. GitHub: Settings → Pages → Source: GitHub Actions; Settings → Environments: create
   `pypi` and `testpypi`.
2. PyPI and TestPyPI: add a trusted publisher (pending publisher for a new project)
   with this repository, workflow `publish.yml` and the matching environment.
3. Optional: connect the repository to Zenodo so each release gets a DOI.

The package is MIT-licensed (`LICENSE`). The repository is
https://github.com/timkmecl/groupunit, and the docs publish to
https://timkmecl.github.io/groupunit/.

## Conventions

- **Notation follows the paper** (`groupunit/core.py` docstring): `X` (N, p), `groups` integer
  codes 0..M-1, `Y` the (M, q) group-level table (the paper's Ỹ), `n` group sizes, `w` group
  weights (default `w = n`), `tbar` group means of scores. Row m of `Y` is group m.
- **Functions centre their own inputs**: `X` over individuals, group-level arrays with `w`
  (`wcentre`). Don't require callers to pre-centre.
- **Group-level inputs wherever possible.** Diagnostics take `(tbar, Y, n)` so they work on
  published, digitised results. Methods compute through group means (the central identity
  XᵀY = X̄ᵀDỸ) and never form the N × q inherited annotation matrix.
- **Plain functions and NumPy arrays**; results are frozen dataclasses with documented fields.
  No classes for users to instantiate, no pandas dependency. Dependencies are NumPy and SciPy only.
- **scikit-learn is optional and never imported in `src/`.** Estimators are duck-typed
  (`fit`, `predict`, `transform`) and copied with `copy.deepcopy` before every fit
  (`estimator_test`, `estimator_model`). Keep the core methods (`pls`, `cca`, `lda`) as
  exact SVDs; don't replace them with scikit-learn's (see README, "Using scikit-learn
  models"). Tests that need scikit-learn live in `tests/test_sklearn.py` behind
  `pytest.importorskip`.
- **Docstrings cite the paper section or result** they implement, and say what the
  quantity means, not only how it is computed. Comments explain why, briefly.
- Accept a vector or a matrix of components where it is cheap (`(M,)` or `(M, k)`), and
  return the matching shape.
- Randomness takes an `rng` argument passed to `np.random.default_rng`.
- Permutation p-values use `inference.p_value` (counts ties, adds one). Family-wise
  correction is max-T (`max_t_p_value`, the default) or min-p (`min_p_value`), chosen with
  `correction=`. Max-T loses power when the cells' null spreads differ, such as selection
  statistics or components concentrated on different numbers of groups; don't present
  it as the only option.
- Raise a clear `ValueError` when a quantity is undefined for the design (e.g. `cca_test`
  at κ = 1, `variance_components` with saturated annotations) rather than return a
  meaningless number.

## Tests

- New results get a test against **direct computation** (individual-level arrays, scipy,
  brute force), not only against the package's own formulas. `tests/test_identities.py`
  is the model.
- Numbers quoted in the paper's figures from the toy data are pinned in
  `tests/test_diagnostics.py::test_toy_numbers_of_figures_2_and_3`. If one changes,
  the paper or the code is wrong, so don't just update the expected value.
- Calibration tests (false-positive rates) use loose bounds and fixed seeds.

## Things that are easy to get wrong

- **Centring.** Group shares need `tbar` and `Y` centred with the analysis weights.
  Removing groups moves the centroid; `leave_out` re-centres (closed form in its docstring).
- **Saturation.** At r = M − 1, CCA is identical for every annotation table; any test
  of the annotations through CCA is meaningless.
- **Unequal group sizes.** Group permutation is exact only if every group's term in
  the statistic is equally variable under the null. With size weights, a large group
  with an atypical profile makes tests badly liberal, more so with more groups.
  `pairing_ratio` detects this; exchangeable weights fix it and are the default of
  `pls_test` and `correlation_test` (`permutation_weights`). Simulations that study
  size-weighted tests must pass `weights="size"` explicitly. Don't describe group
  permutation as unconditionally valid in docs or docstrings.
- **Invalid procedures** (resampling individuals, holding refitted components fixed)
  exist only in `simulations/procedures.py`, for comparison. Never add them to the package.
- **SVD shapes in deflation.** Sequential tests must deflate with full SVDs
  (`full_matrices=True`), or the permuted spaces lose the unpaired directions;
  this once made `cca_test` reject ~100% under the null when q ≠ p.
- **Pattern tests.** Shared typicality makes the raw residual test liberal. The default
  is standardised residuals; correlation/cosine similarities remain liberal (see
  `similarity.py` and the README).
- **Selection.** `pls_test` and `cca_test` refit the model but do not repeat upstream
  selection. Those steps belong in a `permutation_test` statistic.
