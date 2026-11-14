"""Welch's t-test between two configurations of one experiment (uses sweep.py output).

    python compare.py budget swarm10x50it swarm50x10it         # compares TEST accuracy (default)
    python compare.py loss bce mse f1                          # compare any metric: test, f1,
                                                               # precision, recall, train_loss, train
p < 0.05 suggests the difference in the chosen metric is unlikely to be chance given the
number of runs. With only ~10 runs per setting, report it as supporting evidence, not proof.
"""
import csv
import sys

from scipy.stats import ttest_ind_from_stats


def main():
    exp, a, b = sys.argv[1:4]
    key = sys.argv[4] if len(sys.argv) > 4 else "test"   # test | f1 | precision | recall | train_loss | train
    rows = {r["name"]: r for r in csv.DictReader(open(f"results/{exp}_summary.csv"))}
    for n in (a, b):
        if n not in rows:
            sys.exit(f"'{n}' not found. Available: {list(rows)}")
    s = [(float(rows[n][f"{key}_mean"]), float(rows[n][f"{key}_sd"]), int(rows[n]["runs"])) for n in (a, b)]
    t, p = ttest_ind_from_stats(s[0][0], s[0][1], s[0][2], s[1][0], s[1][1], s[1][2], equal_var=False)
    print(f"{key}  {a}: {s[0][0]:.4f}±{s[0][1]:.4f}   {b}: {s[1][0]:.4f}±{s[1][1]:.4f}")
    print(f"Welch t = {t:.3f}, p = {p:.4f}  ->  " + ("significant at 0.05" if p < 0.05 else "not significant at 0.05"))


if __name__ == "__main__":
    main()
