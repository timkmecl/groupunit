# Figures

All figures of the methods paper, drawn with the package itself so that the figures and the code
cannot disagree. Section numbers follow the streamlined paper (simulation design in §6.1).

```bash
python figures/build_all.py               # every figure as SVG (about 20 s); PNGs too if cairosvg is installed
cd figures && npm install && node svg_to_pdf.js   # vector PDFs for the LaTeX build (needs Playwright)
```

Code lives in this folder; every generated file (SVG, PNG, PDF) goes to `output/`, so nothing
there should be edited by hand. The scripts write there through `OUT` in `svg.py`, from whatever
folder they are run.

`build_all.py` imports `groupunit` from `../src` when it is not installed. Figures 1–3 and 6 are
computed from toy data, Figures 4–5 re-simulate the §6.1 design on every run, and Figures 7–8 read
saved simulation results from `simulations/results/`. Every figure is reproducible from fixed seeds.

| Script | Output (in `output/`) | Content | Data |
| --- | --- | --- | --- |
| `fig1_data_structure.py` | `figure1_methods.svg`, `figure1_appraisal.svg` | Data structure: many individuals, few groups; what PLS and CCA see. Generic-first and psychedelic-first label versions. | invented toy values |
| `fig2_dominance.py` | `figure2_dominance.svg` | A component's loading as a sum of group terms; shares; effective number of groups; leave-dominant-out (re-centred); two-group contrast. | `toy.py` |
| `fig3_lens.py` | `figure3_lens.svg` | The lens: random group means through the annotation table and group sizes; eigenvalues and participation ratio; congruence null. | `toy.py` |
| `fig4_cca_saturation.py` | `figure4_cca_saturation.svg` | Saturated CCA, simulated as in §6.1 at M = 21. | simulated |
| `fig5_regimes.py` | `figure5_regimes.svg` | Regimes, simulated as in §6.1 at M = 21, with the Beta null under precision and size weights. | simulated |
| `fig6_typicality.py` | `figure6_typicality.svg` | Typicality and pattern, with distance similarities and standardised residuals. | toy, fixed seed |
| `fig7_pairing.py` | `figure7_pairing.svg` | False-positive rate of group permutation tests against the pairing ratio (outcome 5, M = 20 and 50). | `s5_dominance_replicates.csv`, `s5_dominance_M50_replicates.csv` |
| `fig8_group_size_matrix.py` | `figure8_group_size_matrix.svg` | Figure 8: power against individuals per group and number of groups, as two matrices (exchangeable and size weights). | `s2_group_size_replicates.csv` |
| `fig8_group_size.py` | `figure8_group_size.svg` | The same data as lines (supplementary version). | `s2_group_size_replicates.csv` |
| `toy.py`, `svg.py` | — | Shared toy dataset (groups A–H × annotations a1–a8) and the small SVG writer. | |

To regenerate the simulation results that Figures 7–8 read (a few minutes):

```bash
python simulations/run_all.py s5 s2
python simulations/s5_dominance.py 50     # the M = 50 run for Figure 7
```

All matrix values, report excerpts and drug-level numbers in Figures 1–3 and 6 are illustrative toy
data, except the Figure 1 sample-size bars (N = 4,496, M = 21, Kish M_eff = 10.8 from Suresh et al.
2026). Figures 2, 3 and 6 use generic labels so toy values are not read as findings about real
drugs; Figure 1 keeps drug names with invented quotes. Editing the toy data in `toy.py` updates
Figures 2 and 3 consistently. Figures 4–8 use the Okabe–Ito colours (blue size weights, orange
equal, green exchangeable), with identity also carried by line style, marker and direct labels.

Without cairosvg, render PNGs with a headless browser, for example:

```bash
msedge --headless --force-device-scale-factor=2 --window-size=1400,600 --screenshot=output/figure7_pairing.png output/figure7_pairing.svg
```
