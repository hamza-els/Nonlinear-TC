"""Teacher activation: plain tanh vs a fit to the true neuron activation.

Both papers note the thermodynamic computer's effective activation "is
different in detail to the hyperbolic tangent function of the neural network".
That mismatch is baked into every target we hand the student.  Here the
teacher's tanh is replaced by a 3-parameter fit to the TRUE finite-time
activation of an isolated neuron at our operating point (beta=10, tf=0.40):

    sigma(x) = a * (x^2/(x^2+c)) * cbrt(x) + tanh(d * x)
    a = 0.3524, c = 222.0131, d = 0.2469      (RMS 0.0081 vs 0.1132 for tanh)

Everything else is held identical (architecture, tf, K, beta, guides), and
both variants are run over the same 5 seeds.

Usage:
    python experiments/experiment_activation.py [n_seeds]
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

from digital_net import target, set_activation, HIDDEN
from thermo_student import TF
from train_gd import run

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODES = ["tanh", "thermo"]
COLORS = {"tanh": "tab:red", "thermo": "tab:green"}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"config: tf={TF} seeds={N_SEEDS} activations={MODES}", flush=True)

    res = {m: {"rmse": [], "var": [], "c": [], "tmse": [], "amax": [],
               "curve": None} for m in MODES}
    for mode in MODES:
        set_activation(mode)
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            student, teacher, st = run(seed=seed)
            z = torch.linspace(0.0, 1.0, 250, device=device)
            with torch.no_grad():
                _, A = teacher.activations(z)
            res[mode]["rmse"].append(st["rmse"])
            res[mode]["var"].append(st["var"])
            res[mode]["c"].append(st["readout_c"])
            res[mode]["tmse"].append(st["teacher_mse"])
            res[mode]["amax"].append(A.abs().max().item())
            print(f">>> {mode:7s} seed={seed}  rmse={st['rmse']:.4f}  "
                  f"var={st['var']:.3f}  |w|={st['readout_c']:.2f}  "
                  f"teacher_mse={st['teacher_mse']:.1e}  "
                  f"max|A|={A.abs().max().item():.2f}", flush=True)
        best = int(np.argmin(res[mode]["rmse"]))
        torch.manual_seed(best)
        student, teacher, st = run(seed=best)
        with torch.no_grad():
            y = student.sample_outputs(z, M=100, seed=7).mean(dim=1)
        res[mode]["curve"] = (z.cpu().numpy(), y.cpu().numpy(),
                              target(z).cpu().numpy(), best, st["rmse"])

    # --- figure --------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(N_SEEDS); w = 0.38
    for i, mode in enumerate(MODES):
        a1.bar(x + (i - 0.5) * w, res[mode]["rmse"], w, color=COLORS[mode],
               alpha=0.85, label=f"{mode} teacher (median "
                                 f"{np.median(res[mode]['rmse']):.4f})")
    a1.set_xticks(x); a1.set_xticklabels([f"seed {s}" for s in x])
    a1.set_yscale("log"); a1.set_ylabel("student RMSE vs cos (M=256, K=250)")
    a1.set_title("per-seed student accuracy by teacher activation",
                 fontsize=11)
    a1.legend(frameon=False, fontsize=9)

    for mode in MODES:
        zc, y, y0, best, r = res[mode]["curve"]
        a2.plot(zc, y, color=COLORS[mode], lw=1.0, alpha=0.9,
                label=f"{mode}: best seed {best} (RMSE {r:.4f})")
    a2.plot(zc, y0, "k-", lw=1.5, label="target")
    a2.set_xlabel("z"); a2.set_ylabel("y(z)")
    a2.set_title("best student of each, M=100", fontsize=11)
    a2.legend(frameon=False, fontsize=9, loc="lower center")

    fig.suptitle("teacher activation: tanh  vs  fitted thermodynamic-neuron "
                 r"activation $a\frac{x^2}{x^2+c}\sqrt[3]{x}+\tanh(dx)$"
                 f"   (tf={TF:g}, {N_SEEDS} seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(GDIR, "fig_activation.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    print("\n=== summary ===")
    for mode in MODES:
        r = np.array(res[mode]["rmse"])
        print(f"  {mode:7s} best {r.min():.4f}  median {np.median(r):.4f}  "
              f"worst {r.max():.4f}  |  median var "
              f"{np.median(res[mode]['var']):.3f}  median |w| "
              f"{np.median(res[mode]['c']):.2f}  |  teacher mse "
              f"{np.median(res[mode]['tmse']):.1e}  max|A| "
              f"{np.median(res[mode]['amax']):.2f}")


if __name__ == "__main__":
    main()
