"""Output curves of the digital teacher and GD-trained student, panel-(d) style.

Trains the teacher + student with the default pipeline settings (fixed seed),
then plots y(z) on 250 z-points against the target cos(2 pi z):

  left   : digital teacher net (deterministic -- a "single sample" IS the curve)
  middle : student, M = 1  -- every point is a single stochastic trajectory
  right  : student, M = 20 -- every point is the average of 20 fresh trajectories

Mirrors the style of ga_training/plot_fig2.py panel (d) (black target,
tab:green thermo output; tab:blue for the digital teacher) so GA and GD
figures are directly comparable.

Usage:
    python plot_output_samples.py    # -> ../../Graphs/gd_graphs/fig_output_samples.png
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import target
from train_gd import run

OUT_PATH = "../../Graphs/gd_graphs/fig_output_samples.png"


def main():
    torch.manual_seed(0)
    student, teacher, stats = run()

    K = 250
    z = torch.linspace(0.0, 1.0, K)
    with torch.no_grad():
        yt = teacher(z)                                              # (K,)
        y1 = student.sample_outputs(z, M=1, seed=123).reshape(-1)    # (K,)
        y20 = student.sample_outputs(z, M=20, seed=456).mean(dim=1)  # (K,)
    y0 = target(z)

    zn, y0n = z.numpy(), y0.numpy()
    panels = [(yt.numpy(), "tab:blue", "teacher", "teacher (digital)"),
              (y1.numpy(), "tab:green", "y(z), M=1", "student, M = 1"),
              (y20.numpy(), "tab:green", "y(z), M=20", "student, M = 20")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (y, color, label, title) in zip(axes, panels):
        rmse = float(np.sqrt(np.mean((y - y0n) ** 2)))
        ax.plot(zn, y0n, "k-", lw=1.5, label="target")
        ax.plot(zn, y, color=color, lw=0.7, alpha=0.9, label=label)
        ax.set_xlabel("z")
        ax.set_title(f"{title}   (RMSE {rmse:.3f})")
        ax.legend(frameon=False, loc="lower center")
    axes[0].set_ylabel("y(z)")
    axes[0].set_ylim(-3.0, 3.0)
    fig.suptitle("teacher vs GD-trained student output, 250 z-points",
                 fontsize=13)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
