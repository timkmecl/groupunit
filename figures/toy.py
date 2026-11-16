"""Shared toy dataset and SVG helpers for Figures 2 and 3 (methods paper).

All values are invented for illustration and use generic labels (groups A-H,
annotations a1-a8), so they are not read as findings about real drugs.
"""
import os

import numpy as np

from svg import OUT

# ---------------------------------------------------------------- toy data
# Generic labels: groups A-H, annotations a1-a8. Four large coloured groups (A-D) carry
# most individuals; four small grey groups (E-H) show that a few groups dominate.
DRUGS = ["Group A", "Group B", "Group C", "Group D", "Group E", "Group F", "Group G", "Group H"]
SHORT = {d: d.split()[-1] for d in DRUGS}
COLOR = {"Group A": "#0072B2", "Group B": "#E69F00", "Group C": "#CC79A7", "Group D": "#009E73"}
GREY_DRUG = "#a7b0ba"
RECEPTORS = [f"a{j}" for j in range(1, 9)]

N_REPORTS = np.array([90, 60, 120, 80, 14, 10, 8, 6], float)

# group x annotation table (scaled 0-1, 0 = absent). Chosen so the lens spreads over
# two to three directions (participation ratio about 2.4), as in real designs.
Y_RAW = np.array([
    [.9, .8, .7, 1.0, .5, .3, .2, .1],   # A: broad, a1-a4
    [.4, .2, .8, .7, .9, .8, .1, .2],    # B: a3-a6
    [.4, .2, .3, .6, .3, .3, .9, .1],    # C: weak overall, peak at a7
    [.5, .1, .3, .8, .4, .9, .0, 1.0],   # D: a6 and a8
    [.5, .3, .2, .9, .6, .3, .4, .1],    # E
    [.3, .1, .3, .7, .5, .2, .2, .0],    # F
    [.4, .5, .1, .6, .4, .4, .0, .3],    # G
    [.2, .0, .2, .5, .3, .1, .5, .0],    # H
])

# group means of one component's scores (a "D versus C" component)
T_RAW = np.array([.5, .3, -2.0, 1.6, .2, -.3, .4, -.1])


def color(d):
    return COLOR.get(d, GREY_DRUG)


def wcentre(a, w):
    """Centre rows of a (vector or matrix) with weights w."""
    a = np.asarray(a, float)
    mu = (w[:, None] * a).sum(0) / w.sum() if a.ndim == 2 else (w * a).sum() / w.sum()
    return a - mu


def kish(w):
    w = np.asarray(w, float)
    return w.sum() ** 2 / (w ** 2).sum()


def decompose(Y, t, n):
    """Drug terms n_m * t_m * y_m, the loading q (their sum), shares and M_eff."""
    Yc, tc = wcentre(Y, n), wcentre(t, n)
    terms = (n * tc)[:, None] * Yc          # M x q
    q = terms.sum(0)
    v = q / np.linalg.norm(q)
    s = terms @ v                            # projected contributions
    abs_share = np.abs(s) / np.abs(s).sum()
    m_eff = np.abs(s).sum() ** 2 / (s ** 2).sum()
    return dict(Yc=Yc, tc=tc, terms=terms, q=q, s=s, share=s / s.sum(),
                abs_share=abs_share, m_eff=m_eff)


def lens(Y, n):
    Yc = wcentre(Y, n)
    L = Yc.T @ np.diag(n ** 2) @ Yc
    ev, evec = np.linalg.eigh(L)
    order = np.argsort(ev)[::-1]
    ev, evec = ev[order], evec[:, order]
    ev = np.clip(ev, 0, None)
    nu = ev.sum() ** 2 / (ev ** 2).sum()
    return Yc, ev, evec, nu


def random_loadings(Y, n, k, rng):
    Yc = wcentre(Y, n)
    out = []
    for _ in range(k):
        t = wcentre(rng.standard_normal(len(n)), n)
        q = Yc.T @ (n * t)
        out.append(q / np.linalg.norm(q))
    return np.array(out)


def tucker(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------- SVG helpers
INK, INK2, MUTED, RULE, ACCENT = "#1b2128", "#4d5966", "#8a96a3", "#d7dde3", "#1d5d7f"
SANS = "Helvetica, Arial, sans-serif"
SERIF = "Georgia, serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []
        self.o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">')
        self.o.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                      'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
                      f'fill="{INK2}"/></marker></defs>')
        self.o.append(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')

    def text(self, x, y, s, size=12, fill=INK, anchor="start", weight="400", style="normal", fam=SANS):
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                      f'font-weight="{weight}" font-style="{style}" font-family="{fam}">{s}</text>')

    def rect(self, x, y, w, h, fill, opacity=1, stroke="none", sw=1, rx=0):
        if h < 0:
            y, h = y + h, -h
        self.o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" '
                      f'fill-opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, x1, y1, x2, y2, stroke=INK2, sw=1, dash=None, arrow=False, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = ' marker-end="url(#ah)"' if arrow else ""
        self.o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                      f'stroke-width="{sw}" stroke-opacity="{opacity}"{d}{m}/>')

    def path(self, d, stroke=INK2, sw=1, fill="none", opacity=1):
        self.o.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-opacity="{opacity}"/>')

    def circle(self, x, y, r, fill, opacity=1, stroke="none"):
        self.o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}"/>')

    def label(self, x, y, letter, title):
        self.text(x, y, letter, 20, weight="700")
        self.text(x + 26, y, title, 16, weight="700")

    def save(self, path):
        """Write the SVG; a bare file name goes into OUT."""
        self.o.append("</svg>")
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, path), "w", encoding="utf-8") as f:
            f.write("\n".join(self.o))
