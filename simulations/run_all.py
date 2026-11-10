"""Run the simulations of Section 6.

    python simulations/run_all.py            # full run (about 20 minutes on 8 cores)
    python simulations/run_all.py --quick    # a tenth of the replicates, 99 permutations
    python simulations/run_all.py s1 s7      # selected outcomes only
    python simulations/run_all.py s1p s10    # the high-precision rerun of outcome 1, the comparators

Summaries are written to simulations/results/<name>.csv and .md, with one
row per replicate in <name>_replicates.csv.
"""
import argparse
import time

import s1_false_positives
import s2_power
import s3_saturation
import s4_lens
import s5_dominance
import s6_weighting
import s7_similarity
import s8_validation
import s9_plsr
import s1_precise
import s10_comparators
import s11_highdim
import s12_pairing
import s13_size_check
import s14_lens_refit
import s15_cca_power

OUTCOMES = {
    "s1": (s1_false_positives, 500),
    "s2": (s2_power, 300),
    "s3": (s3_saturation, 300),
    "s4": (s4_lens, 300),
    "s5": (s5_dominance, 500),
    "s6": (s6_weighting, 2000),
    "s7": (s7_similarity, 500),
    "s8": (s8_validation, 300),
    "s9": (s9_plsr, 200),         # needs scikit-learn
    "s1p": (s1_precise, 2000),    # 999 permutations in a full run
    "s10": (s10_comparators, 1000),
    "s11": (s11_highdim, 500),
    "s12": (s12_pairing, 500),
    "s13": (s13_size_check, 40),
    "s14": (s14_lens_refit, 300),
    "s15": (s15_cca_power, 300),
}
N_PERM = {"s1p": 999, "s10": 499}     # outcomes that need more permutations than the default 199

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outcomes", nargs="*", default=list(OUTCOMES))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    for name in args.outcomes:
        module, reps = OUTCOMES[name]
        start = time.perf_counter()
        module.main(reps=max(20, reps // 10) if args.quick else reps,
                    n_perm=99 if args.quick else N_PERM.get(name, 199))
        print(f"[{name} done in {time.perf_counter() - start:.0f} s]")
