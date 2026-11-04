# Figures

Figures that changed or were added after this session's findings, drawn with
the package itself so that the figures and the code cannot disagree. The
other figures (1–5) are built by the paper's own figure scripts.

| Script | Output | Content |
| --- | --- | --- |
| `fig6_typicality.py` | `figure6_typicality.svg`, `.png` | Typicality and pattern, with distance similarities and standardised residuals (replaces the original Figure 6). |
| `fig7_pairing.py` | `figure7_pairing.svg`, `.png` | False-positive rate of group permutation tests against the pairing ratio, from simulation outcome 5. |

```bash
python figures/fig6_typicality.py
python simulations/run_all.py s5      # once, if simulations/results/ is missing
python figures/fig7_pairing.py
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
