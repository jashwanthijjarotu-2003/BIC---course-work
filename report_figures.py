"""Redraws Figures 1-3 of the report from the outputs saved in the notebook.
Usage: python report_figures.py BIC_coursework-2.ipynb
"""
import json, re, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = json.load(open(sys.argv[1]))["cells"]
txt = "\n".join("".join("".join(o.get("text", "")) for o in c.get("outputs", [])) for c in N if c["cell_type"] == "code")
rows = {}
for m in re.finditer(r"^(\w+)\s+(\S+(?:, \d+\])?\S*)\s+loss ([\d.]+)\s+test ([\d.]+)±([\d.]+)", txt, re.M):
    rows.setdefault(m.group(1), {})[m.group(2)] = tuple(map(float, m.group(3, 4, 5)))

BLUE, ORANGE, AQUA, GREY = "#257de0", "#f47230", "#00b894", "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#999", "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True})

def eb(ax, x, y, s, c, lab=None, **k):
    ax.errorbar(x, y, yerr=s, color=c, lw=2, marker="o", ms=4, capsize=2, elinewidth=1, label=lab, **k)

# ---- Figure 1: architecture
A = rows["arch"]
hid = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[8]", "[4, 4]", "[8, 8]"]
lab = ["1", "2", "3", "4", "5", "6", "8", "4-4", "8-8"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
x = np.arange(len(hid))
for act, c, off in (("logistic", ORANGE, -0.12), ("relu", BLUE, 0), ("tanh", AQUA, 0.12)):
    eb(a1, x + off, [A[f"{h}-{act}"][1] for h in hid], [A[f"{h}-{act}"][2] for h in hid], c, act)
    a2.plot(x, [A[f"{h}-{act}"][0] for h in hid], color=c, lw=2, marker="o", ms=4, label=act)
for a, t in ((a1, "Test accuracy (mean ± sd)"), (a2, "Best training loss (MSE)")):
    a.set_xticks(x); a.set_xticklabels(lab); a.set_xlabel("hidden neurons (layers)"); a.set_title(t, fontsize=10.5)
    a.axvline(6.5, color=GREY, lw=0.8, ls=":")
a1.set_ylim(0.92, 1.0); a1.legend(frameon=False, fontsize=9.5, loc="lower left")
fig.tight_layout(); fig.savefig("results/report_f1_arch.png", dpi=220); plt.close(fig)

# ---- Figure 2: budget
B = rows["budget"]
sw = [3, 4, 5, 8, 10, 12, 15, 20, 25, 50, 100]
keys = [f"swarm{s}x{500 // s}it" for s in sw]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
eb(a1, sw, [B[k][1] for k in keys], [B[k][2] for k in keys], BLUE)
a2.plot(sw, [B[k][0] for k in keys], color=BLUE, lw=2, marker="o", ms=4)
for a, t in ((a1, "Test accuracy (mean ± sd)"), (a2, "Best training loss (MSE)")):
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks([3, 5, 8, 12, 20, 50, 100]); a.set_xticklabels([3, 5, 8, 12, 20, 50, 100])
    a.set_xlabel("swarm size (iterations = 500 / swarm)"); a.set_title(t, fontsize=10.5)
a1.set_ylim(0.90, 1.01)
fig.tight_layout(); fig.savefig("results/report_f2_budget.png", dpi=220); plt.close(fig)

# ---- Figure 3: coefficients
C = rows["coef"]
def runpy(name, vals):
    out = []
    for v in vals:
        blk = txt.split(f"== {name} {v} ==")[1]
        m = re.search(r"test accuracy\s+: ([\d.]+) \+/- ([\d.]+)", blk); out.append((float(m.group(1)), float(m.group(2))))
    return out
al = [0.0, 0.3, 0.5, 0.7, 0.9, 1.2]; ep = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
AL, EP = runpy("alpha", al), runpy("epsilon", ep)
bg = ["b0.5_g2.5", "b1.0_g2.0", "b1.5_g1.5", "b2.0_g1.0", "b2.5_g0.5"]
de = ["delta0.0", "delta0.5", "delta1.0", "delta2.0"]
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
eb(ax[0], range(5), [C[k][1] for k in bg], [C[k][2] for k in bg], BLUE)
ax[0].set_xticks(range(5)); ax[0].set_xticklabels(["0.5/2.5", "1/2", "1.5/1.5", "2/1", "2.5/0.5"], rotation=40, ha="right", rotation_mode="anchor", fontsize=9)
ax[0].set_xlabel("β / γ")
eb(ax[1], [0, 0.5, 1, 2], [C[k][1] for k in de], [C[k][2] for k in de], BLUE); ax[1].set_xlabel("δ")
eb(ax[2], al, [a for a, _ in AL], [s for _, s in AL], BLUE); ax[2].set_xlabel("α")
eb(ax[3], ep, [a for a, _ in EP], [s for _, s in EP], BLUE); ax[3].set_xlabel("ε")
ax[0].set_ylim(0.96, 1.0); ax[0].set_ylabel("test accuracy")
fig.tight_layout(); fig.savefig("results/report_f3_coef.png", dpi=220); plt.close(fig)
print("Saved report figures to results/")
