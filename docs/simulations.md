# Simulations

The `simulations/` folder holds the simulation study of §8. It is not part of
the installed package. Each script covers one outcome of §8.2 and uses the
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

The design is in `simulations/common.py` (`Design`). Within-group variance is
fixed at σ²_w = 1, and the heterogeneity ratio h = τ² / (σ²_w / n̄) sets τ².
The between-group share η² of the features then follows from h and the mean
group size n̄, so it is reported rather than set. Group sizes are
deterministic log-normal quantiles with a chosen Kish effective number.
Seeds are fixed, so results are reproducible.

Results are written to `simulations/results/`: a summary table per outcome
(`.csv` and `.md`) and one row per replicate (`_replicates.csv`).
