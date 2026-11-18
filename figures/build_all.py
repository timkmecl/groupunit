"""Build all figures of the paper as SVG in figures/output/, plus PNG previews if cairosvg is installed.

Usage (from any folder):
    python figures/build_all.py

Figures 1-3 and C1 are computed from toy data (toy.py, or fixed seeds).
Figures 4-5 simulate the design of Section 6.1 on every run (seconds to a minute).
Figures 6-7 read the saved simulation results in ../simulations/results/
(s5_dominance*_replicates.csv, s2_group_size_replicates.csv); to regenerate those, run
`python simulations/run_all.py s5 s2` and `python simulations/s5_dominance.py 50` (minutes).

Requires numpy, scipy and pandas; groupunit is imported from ../src if it is not installed.
PNG export needs `pip install cairosvg`.
"""
import glob
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))   # groupunit, if not installed

SCRIPTS = ["fig1_data_structure.py", "fig2_dominance.py", "fig3_lens.py",
           "fig4_cca_saturation.py", "fig5_regimes.py", "fig6_pairing.py",
           "fig7_power.py", "fig7_power_lines.py", "figC1_typicality.py"]

for script in SCRIPTS:
    print(f"running {script}")
    runpy.run_path(script, run_name="__main__")
    os.chdir(HERE)

try:
    import cairosvg
    for f in sorted(glob.glob(os.path.join("output", "figure*.svg"))):
        cairosvg.svg2png(url=f, write_to=f[:-4] + ".png", output_width=2800)
        print(f"  {f[:-4]}.png")
except ImportError:
    print("cairosvg not installed: SVGs only")
