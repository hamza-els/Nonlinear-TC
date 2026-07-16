"""Output curves of the digital teacher and GD-trained student, panel-(d) style.

Trains the teacher + student with the default pipeline settings (fixed seed),
then plots y(z) on 250 z-points against the target cos(2 pi z):

  panel 1 : digital teacher net (deterministic -- a "single sample" IS the curve)
  panel 2 : student, M = 1   -- every point is a single stochastic trajectory
  panel 3 : student, M = 20  -- every point averages 20 fresh trajectories
  panel 4 : student, M = 100 -- every point averages 100 fresh trajectories

Mirrors the style of ga_training/plot_fig2.py panel (d) (black target,
tab:green thermo output; tab:blue for the digital teacher) so GA and GD
figures are directly comparable.

Usage:
    python plots/plot_output_samples.py    # -> Graphs/gd_graphs/fig_output_samples.png
"""

import os
import sys

# core modules (digital_net, thermo_student, train_gd) live one level up
_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import target
from train_gd import run
from thermo_student import TF

OUT_PATH = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs",
                        "fig_output_samples.png")


def plot_output(student, teacher, out_path=OUT_PATH,
                suptitle="teacher vs GD-trained student output, 250 z-points",
                tf=None):
    """Draw the 4-panel output figure for an already-trained (student, teacher).

    tf: observation time for the student rollouts (defaults to module TF)."""
    K = 250
    device = student.b.device
    tf = tf if tf is not None else TF
    z = torch.linspace(0.0, 1.0, K, device=device)
    with torch.no_grad():
        yt = teacher(z)                                              # (K,)
        y1 = student.sample_outputs(z, M=1, seed=123, tf=tf).reshape(-1)    # (K,)
        y20 = student.sample_outputs(z, M=20, seed=456, tf=tf).mean(dim=1)  # (K,)
        y100 = student.sample_outputs(z, M=100, seed=789, tf=tf).mean(dim=1)
    y0 = target(z)

    zn, y0n = z.cpu().numpy(), y0.cpu().numpy()
    panels = [(yt.cpu().numpy(), "tab:blue", "teacher", "teacher (digital)"),
              (y1.cpu().numpy(), "tab:green", "y(z), M=1", "student, M = 1"),
              (y20.cpu().numpy(), "tab:green", "y(z), M=20", "student, M = 20"),
              (y100.cpu().numpy(), "tab:green", "y(z), M=100",
               "student, M = 100")]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
    for ax, (y, color, label, title) in zip(axes, panels):
        rmse = float(np.sqrt(np.mean((y - y0n) ** 2)))
        ax.plot(zn, y0n, "k-", lw=1.5, label="target")
        ax.plot(zn, y, color=color, lw=0.7, alpha=0.9, label=label)
        ax.set_xlabel("z")
        ax.set_title(f"{title}   (RMSE {rmse:.3f})")
        ax.legend(frameon=False, loc="lower center")
    axes[0].set_ylabel("y(z)")
    axes[0].set_ylim(-3.0, 3.0)
    fig.suptitle(suptitle, fontsize=13)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    student, teacher, stats = run(seed=seed)
    plot_output(student, teacher)


if __name__ == "__main__":
    main()
