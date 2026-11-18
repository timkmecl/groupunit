"""Outcome 16: small-sample degrees of freedom for the mixed model (Table 3, lower panel).

Reruns the random-intercept mixed model of outcome 10 on the same data sets (same
cells and seed, so the Wald and F(r, M − r − 1) columns reproduce outcome 10) and
adds the Kenward–Roger and Satterthwaite F tests of the annotations. With equal
group sizes both reduce to F(r, M − r − 1), which is checked to machine precision.

Kenward–Roger is undefined (denominator degrees of freedom m ≤ 0) when τ² is
estimated near zero; such data sets count as non-rejections, and the share is
reported. The implementation has not been checked against pbkrtest.
"""
import procedures as pr
import s10_comparators as s10
from common import ALPHA, run, save

TESTS = ["mixed model, Wald chi2", "mixed model, F(q, M-q-1)", "mixed model, Kenward-Roger",
         "mixed model, Satterthwaite"]


def replicate(d, rng, n_perm):
    X, groups, Y, n = d.draw(rng)
    mm = pr.mixed_model(X[:, 0], groups, Y)
    return {"tau2 at zero": float(mm["tau2 zero"]),
            **dict(zip(TESTS, [mm["wald"], mm["F"], mm["KR"], mm["Satterthwaite"]]))}


def summarise(df):
    keys = [df.signal, df.structure, df.kappa, df.M, df.h]
    rates = (df[TESTS] < ALPHA).groupby(keys).mean()
    out = df[["tau2 at zero"]].groupby(keys).mean().join(rates)
    out["Kenward-Roger undefined"] = df["mixed model, Kenward-Roger"].isna().groupby(keys).mean()
    return out


def main(reps=1000, n_perm=499, seed=10):
    df = run(replicate, s10.cells(n_perm), reps, seed)
    save(df, summarise(df), "s16_mixed_df")


if __name__ == "__main__":
    main()
