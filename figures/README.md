# Figures

Figures that changed or were added after this session's findings, drawn with
the package itself so that the figures and the code cannot disagree. Figures
1–3 are built by the paper's own figure scripts.

| Script | Output | Content |
| --- | --- | --- |
| `fig4_cca_saturation.py` | `figure4_cca_saturation.svg`, `.png` | Saturated CCA, simulated as in §6.1 at M = 21 (replaces the original Figure 4). |
| `fig5_regimes.py` | `figure5_regimes.svg`, `.png` | Regimes, simulated as in §6.1 at M = 21, with the Beta null under precision and size weights (replaces the original Figure 5). |
| `fig6_typicality.py` | `figure6_typicality.svg`, `.png` | Typicality and pattern, with distance similarities and standardised residuals (replaces the original Figure 6). |
| `fig7_pairing.py` | `figure7_pairing.svg`, `.png` | False-positive rate of group permutation tests against the pairing ratio, from simulation outcome 5 at M = 20 and 50. |
| `fig8_group_size.py` | `figure8_group_size.svg`, `.png` | Power against individuals per group and number of groups, from simulation outcome 2 (line version). |
| `fig8_group_size_matrix.py` | `figure8_group_size_matrix.svg`, `.png` | The same data as two annotated matrices (alternative version). |

```bash
python figures/fig4_cca_saturation.py
python figures/fig5_regimes.py
python figures/fig6_typicality.py
python simulations/run_all.py s5          # once, if simulations/results/ is missing
python simulations/s5_dominance.py 50     # the M = 50 run
python figures/fig7_pairing.py
python simulations/run_all.py s2          # once, for Figure 8
python figures/fig8_group_size.py
python figures/fig8_group_size_matrix.py
```

Both use the same small SVG writer as Figures 1–5 (`svg.py`), so they match
the paper's other figures. `fig6_typicality.py` keeps the original Figure 6's
drawing code; only its data come from the package. Figure 7 uses the
Okabe–Ito colours of Figures 2–4 (blue size weights, orange equal, green
exchangeable). They pass a colourblind-safety check, and identity is also
carried by line style, marker and direct labels.

The scripts need pandas (Figure 7) and write a PNG preview only if cairosvg
is installed. Without it, render the SVGs with a headless browser, for
example:

```bash
msedge --headless --force-device-scale-factor=2 --window-size=1400,600 --screenshot=figure7_pairing.png figure7_pairing.svg
```
