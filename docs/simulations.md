# Simulations

The `simulations/` folder holds the simulation study of §6. It is not part of
the installed package. Each script covers one question of the study (§6.2) and uses the
package for everything except the invalid procedures it compares against.
Those live in `simulations/procedures.py`, so the package never offers them.

```bash
pip install -e ".[simulations]"
python simulations/run_all.py            # the full study, about 20 minutes on 8 cores
python simulations/run_all.py --quick    # a tenth of the replicates and 99 permutations
python simulations/run_all.py s1 s5      # selected outcomes
python simulations/s5_dominance.py 50     # outcome 5 with 50 groups instead of 20
```

| Script                    | Outcome                                                                       |
| ------------------------- | ----------------------------------------------------------------------------- |
| `s1_false_positives.py`   | false-positive rates: resampling individuals vs permuting groups, with and without refitting |
| `s2_power.py`             | power of the valid tests at realistic numbers of groups, and power against the number of individuals per group |
| `s3_saturation.py`        | indifference of saturated CCA, and the Beta null of R²                         |
| `s4_lens.py`              | how closely null PLS loadings follow the lens; agreement between null analyses |
| `s5_dominance.py`         | one group with an extreme profile: shares, pairing ratio and test validity     |
| `s6_weighting.py`         | size, equal and optimal weights; accuracy of the effective-number formulas     |
| `s7_similarity.py`        | typicality and pattern: false positives and power of the pattern test          |
| `s8_validation.py`        | optimism of individual-split versus group-split cross-validation               |
| `s9_plsr.py`              | PLS regression (as in Suresh et al.): tests of its R² and its cross-validation; needs scikit-learn |
| `s1_precise.py`           | the valid tests of outcome 1 with 2,000 replicates and 999 permutations, against the binomial band |
| `s10_comparators.py`      | one score fixed in advance: OLS, a REML mixed model (Wald χ² and F with M − q − 1 df), the wild cluster bootstrap, and group permutation |
| `s11_highdim.py`          | p = 10, 100, 300 features, low-rank features, Student t noise                  |
| `s12_pairing.py`          | several atypical groups, profiles growing with group size, heavy tails; row-weighted CCA |
| `s13_size_check.py`       | whether `size_check` with estimated variance components predicts a test's size for the table at hand |
| `s14_lens_refit.py`       | the refitted PLS lens null with the nominal and the effective number of features |
| `s15_cca_power.py`        | power of row-weighted CCA against ordinary CCA and reweighted PLS, raw and size-adjusted |

`python simulations/run_all.py s1p s10` runs outcome 1 at high precision and
the comparators. Set `GROUPUNIT_SIM_RESULTS` to write results somewhere other
than `simulations/results/`, for example for a `--quick` trial run.

The design is in `simulations/common.py` (`Design`). Within-group variance is
fixed at σ²_w = 1, and the heterogeneity ratio h = τ² / (σ²_w / n̄) sets τ².
The between-group share η² of the features then follows from h and the mean
group size n̄, so it is reported rather than set. Group sizes are
deterministic log-normal quantiles with a chosen Kish effective number.
Seeds are fixed, so results are reproducible.

Results are written to `simulations/results/`: a summary table per outcome
(`.csv` and `.md`) and one row per replicate (`_replicates.csv`).
