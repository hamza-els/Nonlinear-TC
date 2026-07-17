"""Guide-bias prescription: the paper's "proportional" vs our exact inverse.

The paper (Sec. III) sets the guide biases merely PROPORTIONAL to the target
activations, b0_i = k * A_i, and accepts that the guide's long-time
activations are then not equal to A (its reasoning: a classifier only needs
the hierarchy of output activations, not their values).  We have instead been
inverting the fixed-point relation, b0_i = 2 J2 A_i + 4 J4 A_i^3, so the guide
relaxes to exactly A -- motivated by regression, where the value IS the answer.

This runs both prescriptions over the same 5 seeds with everything else
identical (tf, K, architecture, beta), and compares.

Usage:
    python experiments/experiment_guide_mode.py [n_seeds]
"""

import os
import sys

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import target, TARGET_FREQ
from thermo_student import TF
from train_gd import run

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODES = ["exact", "proportional"]
COLORS = {"exact": "tab:green", "proportional": "tab:red"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"config: TARGET_FREQ={TARGET_FREQ} tf={TF} seeds={N_SEEDS} "
          f"modes={MODES}", flush=True)

    res = {m: {"rmse": [], "bias2": [], "var": [], "c": [], "curve": None}
           for m in MODES}
    for mode in MODES:
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            student, teacher, st = run(seed=seed, guide_mode=mode)
            res[mode]["rmse"].append(st["rmse"])
            res[mode]["bias2"].append(st["bias2"])
            res[mode]["var"].append(st["var"])
            res[mode]["c"].append(st["readout_c"])
            print(f">>> {mode:12s} seed={seed}  rmse={st['rmse']:.4f}  "
                  f"bias2={st['bias2']:.2e}  var={st['var']:.3f}  "
                  f"|w|={st['readout_c']:.2f}", flush=True)
        # keep the best seed's output curve for the figure
        best = int(np.argmin(res[mode]["rmse"]))
        torch.manual_seed(best)
        student, teacher, st = run(seed=best, guide_mode=mode)
        z = torch.linspace(0.0, 1.0, 250, device=device)
        with torch.no_grad():
            y = student.sample_outputs(z, M=100, seed=7).mean(dim=1)
        res[mode]["curve"] = (z.cpu().numpy(), y.cpu().numpy(),
                              target(z).cpu().numpy(), best, st["rmse"])

    # --- figure --------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(N_SEEDS)
    w = 0.38
    for i, mode in enumerate(MODES):
        a1.bar(x + (i - 0.5) * w, res[mode]["rmse"], w, color=COLORS[mode],
               alpha=0.85, label=f"{mode} (median "
                                 f"{np.median(res[mode]['rmse']):.4f})")
    a1.set_xticks(x); a1.set_xticklabels([f"seed {s}" for s in x])
    a1.set_yscale("log"); a1.set_ylabel("RMSE vs cos (M=256, K=250)")
    a1.set_title("per-seed accuracy by guide-bias prescription", fontsize=11)
    a1.legend(frameon=False, fontsize=9)

    for mode in MODES:
        zc, y, y0, best, r = res[mode]["curve"]
        a2.plot(zc, y, color=COLORS[mode], lw=1.0, alpha=0.9,
                label=f"{mode}: best seed {best} (RMSE {r:.4f})")
    a2.plot(zc, y0, "k-", lw=1.5, label="target")
    a2.set_xlabel("z"); a2.set_ylabel("y(z)")
    a2.set_title("best computer of each, M=100", fontsize=11)
    a2.legend(frameon=False, fontsize=9, loc="lower center")

    fig.suptitle(f"guide biases: paper's b0 proportional to A  vs  exact "
                 f"inverse b0 = 2A + 4A^3   (tf={TF:g}, {N_SEEDS} seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(GDIR, "fig_guide_mode.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    print("\n=== summary ===")
    for mode in MODES:
        r = np.array(res[mode]["rmse"])
        print(f"  {mode:12s} best {r.min():.4f}  median {np.median(r):.4f}  "
              f"worst {r.max():.4f}  |  median var "
              f"{np.median(res[mode]['var']):.3f}  median |w| "
              f"{np.median(res[mode]['c']):.2f}")


if __name__ == "__main__":
    main()
