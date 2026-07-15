"""Experiment O: how low can the clock go?  Train at 20 different tf values.

For each trained observation time tf in {0.1, 0.2, ..., 2.0} (same
architecture, same seed) we train a fresh computer, record its best RMSE at
its own clock time, and sweep phi over observation times (parameters fixed).
The result is a 4 x 5 grid of phi-vs-observation-time panels, one per trained
tf, each labeled with its trained clock (dotted vertical line) and best RMSE.

Usage:
    python experiments/experiment_tf_scan.py [seed]
      -> ../../Graphs/gd_graphs/fig_tf_scan_grid.png + tf_scan_results.npz
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import target
from train_gd import run

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

TF_LIST = [round(0.1 * k, 1) for k in range(1, 21)]      # 0.1 ... 2.0
SWEEP = np.logspace(-1, 1, 21)                           # observation times
M_SWEEP = 128                                            # samples per point


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    results = []            # (tf, rmse, phi_curve)

    for tf in TF_LIST:
        torch.manual_seed(SEED)
        student, teacher, stats = run(seed=SEED, tf=tf)
        z = torch.linspace(0.0, 1.0, 101, device=device)
        y0 = target(z)
        phi = []
        for t_obs in SWEEP:
            with torch.no_grad():
                pred = student.sample_outputs(z, M=M_SWEEP, tf=float(t_obs),
                                              seed=0).mean(dim=1)
            phi.append(torch.mean((pred - y0) ** 2).item())
        results.append((tf, stats["rmse"], phi))
        print(f">>> tf={tf:.1f}  rmse={stats['rmse']:.3f}  "
              f"phi_min={min(phi):.2e}  ({time.time()-t0:.0f}s)", flush=True)

    # --- grid figure ---------------------------------------------------------
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True, sharey=True)
    for ax, (tf, rmse, phi) in zip(axes.flat, results):
        ax.plot(SWEEP, phi, "-", color="tab:green", lw=1.1)
        ax.axvline(tf, color="0.4", ls=":", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$t_f$={tf:.1f}   RMSE {rmse:.3f}", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("observation time")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\phi$")
    fig.suptitle(f"phi vs observation time for computers trained at 20 clock "
                 f"times (seed {SEED}, M={M_SWEEP}; dotted line = trained "
                 f"$t_f$)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = os.path.join(GDIR, "fig_tf_scan_grid.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")

    np.savez(os.path.join(_GD_ROOT, "runs", "tf_scan_results.npz"),
             tf=np.array([r[0] for r in results]),
             rmse=np.array([r[1] for r in results]),
             sweep=SWEEP,
             phi=np.array([r[2] for r in results]), seed=SEED)
    print("\n=== best clock times ===")
    for tf, rmse, _ in sorted(results, key=lambda r: r[1])[:5]:
        print(f"  tf={tf:.1f}  rmse={rmse:.3f}")


if __name__ == "__main__":
    main()
