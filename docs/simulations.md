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
| `s2_power.py`             | power of the valid tests (size and exchangeable weights; standard design and with a dominant within-group direction) at realistic numbers of groups, and power against the number of individuals per group |
| `s3_saturation.py`        | indifference of saturated CCA, and the Beta null of R²                         |
| `s4_lens.py`              | how closely null PLS loadings follow the lens; agreement between null analyses |
| `s5_dominance.py`         | one group with an extreme profile: shares, pairing ratio and test validity     |
| `s6_weighting.py`         | size, equal and optimal weights; accuracy of the effective-number formulas     |
| `s7_similarity.py`        | typicality and pattern: false positives and power of the pattern test          |
| `s8_validation.py`        | optimism of individual-split versus group-split cross-validation               |
| `s9_plsr.py`              | PLS regression (as in Suresh et al.): tests of its R² and its cross-validation; needs scikit-learn |
| `s1_precise.py`           | the valid tests of outcome 1 with 2,000 replicates and 999 permutations, against the binomial band |
| `s10_comparators.py`      | one score fixed in advance: OLS, a REML mixed model (Wald χ² and F with M − r − 1 df), the wild cluster bootstrap, and group permutation |
| `s11_highdim.py`          | p = 10, 100, 300 features, low-rank features, Student t noise                  |
| `s12_pairing.py`          | several atypical groups, profiles growing with group size, heavy tails; row-weighted CCA |
| `s13_size_check.py`       | whether `size_check` with estimated variance components predicts a test's size for the table at hand |
| `s14_lens_refit.py`       | the refitted PLS lens null: nominal and effective number of features, participation-ratio estimates of it, designs with group effects in fewer dimensions than the within-group noise, nulls drawn from (estimated) covariances, and nulls that permute the observed group means, with unequal and equal group sizes |
| `s15_cca_power.py`        | power of row-weighted CCA against ordinary CCA and reweighted PLS, raw and size-adjusted |
| `s16_mixed_df.py`         | the mixed model of outcome 10 on the same data sets, with Kenward–Roger and Satterthwaite degrees of freedom |

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

## Implementation details

These are the choices the code makes where the paper's description leaves room.

**Variance components (`groupunit.weights`).** For one score, σ²_w is the pooled
within-group variance with denominator N − M. τ² is a moment estimate from an
*unweighted* (equally weighted) least-squares fit of the M group means: on the
annotations, or, with no annotations (r = 0), on an intercept alone, so around the
unweighted mean of the group means. The expected sampling part
σ²_w Σ (1 − h_mm) / n_m (h_mm the leverages) is subtracted from the residual sum of
squares, the result is divided by the residual degrees of freedom, and τ² is
truncated at zero. Over several features (`pooled_variance_components`, used for
exchangeable weights), σ²_w and τ² are estimated for each feature without
annotations, τ² truncated at zero *per feature*, and both are averaged over
features. Exchangeable weights are estimated in every simulated data set this
way; pairing ratios use the true components.

**Test statistics (`groupunit.inference`).** All p-values are
(1 + #{null ≥ observed}) / (1 + n_perm).

- Correlation tests: for each score–annotation cell, |Σ_m w_m t̃_m y_m| divided by
  quantities that do not change under permutation, which is equivalent to the
  absolute weighted group-level correlation. With size weights the statistic is
  η · r_n, the individual-level correlation. The family-wise p-value is max-T
  over cells (the maximum of the cell statistics in each permutation); the
  simulations report the smallest family-wise p-value.
- PLS: the largest singular value of X̄ᵀ W Ỹ (group means and annotations centred
  with the weights w), recomputed after every permutation. Component c is tested
  after removing components 1 … c − 1 from both sides.
- CCA: the largest canonical correlation, the largest singular value of PᵀQ with P
  and Q orthonormal bases of the (weighted) feature and annotation spaces, with
  full-SVD deflation for later components.
- Wild cluster bootstrap: one joint cluster-robust (CR1) Wald statistic for all r
  annotation coefficients of an OLS fit on the inherited annotations; Rademacher
  weights per group on residuals from the intercept-only fit; the observed Wald
  statistic is compared with its bootstrap distribution.
- Mixed model: REML for a random intercept, using the exact factorisation of the
  likelihood into the group means and the within-group sum of squares. Wald χ²(r),
  the same statistic over r against F(r, M − r − 1), and (outcome 16) Kenward–Roger
  and Satterthwaite (Fai–Cornelius) F tests, both of which reduce to
  F(r, M − r − 1) with equal group sizes.

**Effect size.** "Group-level R² = s" (`Design.signal`) refers to the *observed*
group means: annotation 1 drives features 1 and 2 with β² = s / (1 − s) ·
(τ² + σ²_w / n̄), so that, at the mean group size and with unit-variance
annotations, β² / (β² + τ² + σ²_w / n̄) = s. Figure 7 (power against group size)
instead fixes R² = 0.5 for the *population* means and converts it for each n̄.

**Rounding.** The number of annotations is r = max(1, round(κ(M − 1))) with
Python's round-half-to-even, so κ = 0.5 gives r = 4, 10, 24 and 50 at M = 10, 20,
50 and 100 (κ̃ = 0.44, 0.53, 0.49, 0.51). Group sizes are n̄ · exp(σ z_m − σ²/2)
at the normal quantiles z_m = Φ⁻¹((m − 0.5)/M), rounded to integers (half to even)
and floored at 2, with σ found by bisection to reach the target Kish fraction.
The realised mean is below n̄ = 50: 46.7 at M = 20 and Kish 40% (N = 933), 42.5
at M = 10 and 49.2 at M = 100; with Kish 75%, 48.8–49.9.

**Low-rank features (outcomes 11 and 14).** In each data set, a p × k standard
normal matrix is orthonormalised (QR), transposed and scaled by √(p/k), giving a
map A with A Aᵀ = (p/k) I. Group effects and within-group noise are drawn in the
k latent dimensions and mapped by A, so the total variance equals that of p
isotropic features; the relation acts in latent dimensions 1–2. Isotropic noise of
variance 0.01 is added to every feature to keep the features of full rank. CCA is
scale-invariant, so dimensions that contain only this noise count as much as the others.
In outcome 14's mixed designs, group effects occupy the first k_b of the k_w
latent dimensions of the noise.

**Typicality and pattern (outcome 7, Appendix C).** Profiles have 30 dimensions.
"Shared typicality only": each group gets a scale u_m = exp(N(0, 0.55²)); the
feature profiles are c + u_m z_m and the annotation profiles c′ + u′_m z′_m, with
u′_m = u_m · exp(N(0, 0.15²)) and independent standard normal directions z_m, z′_m
(and independent centres c, c′). The tables thus share how far out each group is
and nothing about which groups resemble which. Each similarity matrix is split by
least squares into s_ij = μ + a_i + a_j + e_ij over the off-diagonal entries. The
standardised pattern test divides the residuals by both groups' residual scales,
e_ij / (s_i s_j) with s_i² = Σ_j e_ij² / (M − 1), and correlates them over the
M(M − 1)/2 pairs; groups are permuted jointly over rows and columns of the second
matrix, with a one-sided p-value.

