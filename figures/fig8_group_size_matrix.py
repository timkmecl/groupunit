"""Figure 8, matrix version: power by number of groups and individuals per group.

An alternative to `fig8_group_size.py` with the same data (simulation outcome 2,
second part: τ² = 0.02, population R² = 0.5, κ = 0.5, Kish 40%; 300 data sets
per cell). Two matrices, rows the number of groups M and columns the mean
group size n̄, with the power printed in each cell.

Panels
  a  PLS test with exchangeable weights; the two cells outlined in orange have equal power
     with 20,000 and 2,500 individuals in total
  b  the same test with size weights

Cells with power of at least 0.8 are outlined.

    python simulations/run_all.py s2      # if missing
    python figures/fig8_group_size_matrix.py
"""
import os

import numpy as np
import pandas as pd

from svg import INK, INK2, SVG, save_png

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "simulations", "results")
ALPHA = 0.05

df = pd.read_csv(os.path.join(RESULTS, "s2_group_size_replicates.csv"))
GROUPS = sorted(df.M.unique())
SIZES = sorted(df.nbar.unique())


def matrix(test):
    rate = (df[test] < ALPHA).groupby([df.M, df.nbar]).mean()
    return np.array([[rate[(M, n)] for n in SIZES] for M in GROUPS])


def ramp(v):
    """Light grey-blue to deep blue for 0..1, as Figure 5a."""
    c0, c1 = np.array([238, 242, 246]), np.array([22, 64, 96])
    c = (c0 + (c1 - c0) * v ** 1.3).astype(int)
    return f"rgb({c[0]},{c[1]},{c[2]})"


CW, CH = 66, 62          # cell size


def panel(s, x0, letter, title, sub, P, rings=()):
    s.label(x0 + 20, 40, letter, title)
    s.text(x0 + 46, 62, sub, 12, INK2)
    gx, gy = x0 + 120, 110
    for i, M in enumerate(GROUPS[::-1]):                 # most groups at the top
        row = len(GROUPS) - 1 - i
        y = gy + i * CH
        s.text(gx - 10, y + CH / 2 + 4, f"{M}", 11.5, INK2, anchor="end")
        for j, n in enumerate(SIZES):
            v, x = P[row, j], gx + j * CW
            s.rect(x, y, CW - 2, CH - 2, ramp(v))
            if v >= 0.8:
                s.rect(x + 1.5, y + 1.5, CW - 5, CH - 5, "none", stroke=INK, sw=2)
            s.text(x + (CW - 2) / 2, y + CH / 2 + 1, f"{v:.2f}", 13, "#ffffff" if v > 0.55 else INK,
                   anchor="middle", weight="700")
            s.text(x + (CW - 2) / 2, y + CH / 2 + 16, f"N = {M * n:,}", 9.5,
                   "#dfe7ee" if v > 0.55 else INK2, anchor="middle")
            if (M, n) in rings:
                s.rect(x - 3, y - 3, CW + 4, CH + 4, "none", stroke="#E69F00", sw=3.5, rx=8)
    for j, n in enumerate(SIZES):
        s.text(gx + j * CW + (CW - 2) / 2, gy + len(GROUPS) * CH + 16, f"{n:,}", 11.5, INK2, anchor="middle")
    s.text(gx + len(SIZES) * CW / 2, gy + len(GROUPS) * CH + 38, "individuals per group n̄", 11.5, INK2,
           anchor="middle")
    s.text(gx - 44, gy + len(GROUPS) * CH / 2, "number of groups M", 11.5, INK2, anchor="middle")
    s.o[-1] = s.o[-1].replace("<text ", f'<text transform="rotate(-90 {gx - 44} {gy + len(GROUPS) * CH / 2})" ', 1)
    return gx, gy


exch, size = matrix("PLS: exchangeable"), matrix("PLS: size weights")

W, H = 1400, 500
s = SVG(W, H)
gx, gy = panel(s, 0, "a", "Power grows with groups, not individuals", "PLS test, exchangeable weights", exch,
               rings=((20, 1000), (100, 25)))
panel(s, 640, "b", "Size weights waste the extra individuals", "the same test with size weights", size)

# colour scale
lx = 1290
for i in range(100):
    s.rect(lx, gy + 200 - i * 2, 16, 2.2, ramp(i / 99))
s.text(lx + 22, gy + 204, "0", 10.5, INK2)
s.text(lx + 22, gy + 6, "1", 10.5, INK2)
s.text(lx, gy - 10, "power", 11, INK2)
s.rect(lx, gy + 222, 16, 16, "none", stroke=INK, sw=2)
s.text(lx + 22, gy + 234, "≥ 0.8", 10.5, INK2)

ny = 428
s.text(82, ny, "Orange: equal power (about 0.5) with 20 groups of 1,000 and with 100 groups of 25, eight times "
               "fewer individuals.", 12, INK)
s.text(82, ny + 18, "Simulated data with a fixed population: groups differ by τ² = 0.02, and their population means "
                    "relate to the annotations with R² = 0.5 (κ = 0.5, Kish 40%).", 12, INK)
s.text(82, ny + 36, f"{len(df):,} data sets, 300 per cell (standard error at most 0.03). N: individuals in total.",
       12, INK)

os.chdir(HERE)
s.save("figure8_group_size_matrix.svg")
save_png("figure8_group_size_matrix.svg")
