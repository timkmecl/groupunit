"""Run the simulations of Section 8.

    python simulations/run_all.py            # full run (about 20 minutes on 8 cores)
    python simulations/run_all.py --quick    # a tenth of the replicates, 99 permutations
    python simulations/run_all.py s1 s7      # selected outcomes only

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

OUTCOMES = {
    "s1": (s1_false_positives, 500),
    "s2": (s2_power, 300),
    "s3": (s3_saturation, 300),
    "s4": (s4_lens, 300),
    "s5": (s5_dominance, 300),
    "s6": (s6_weighting, 2000),
    "s7": (s7_similarity, 500),
    "s8": (s8_validation, 300),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outcomes", nargs="*", default=list(OUTCOMES))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    for name in args.outcomes:
        module, reps = OUTCOMES[name]
        start = time.perf_counter()
        module.main(reps=max(20, reps // 10) if args.quick else reps, n_perm=99 if args.quick else 199)
        print(f"[{name} done in {time.perf_counter() - start:.0f} s]")
