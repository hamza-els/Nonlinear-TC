"""Output of the GA-trained computers at M = 1, 20, 100 samples per input.

GA counterpart of gd_training/plots/plot_output_samples.py: for each trained run we
draw S = 100 fresh reset-sampled trajectories per z and show the readout
built from the first 1, the first 20, and all 100 of them, against the
target cos(2 pi z).  Same styling as the GD figure (black target, tab:green
output) so the two methods can be compared panel-for-panel.

Usage:
    python plot_output_samples_ga.py
      -> ../../Graphs/computer_graphs/fig_output_samples_<name>.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genetic_algorithm import load_run, sample_output_pool

GDIR = "../../Graphs/computer_graphs"
RUNS = [("runs/run_reg.npz", "regular"),
        ("runs/run_low_var.npz", "low_var")]
K = 250
MS = (1, 20, 100)


def plot_run(path, name, device):
    params, _, cfg = load_run(path)
    z = np.linspace(0.0, 1.0, K)
    y0 = np.cos(2.0 * np.pi * z)

    Y = sample_output_pool(
        params, z, S=max(MS), m_chunk=int(cfg["m_chunk"]),
        tf=float(cfg["tf"]), dt=float(cfg["dt"]),
        beta=float(cfg["beta"]), mu=float(cfg["mu"]), device=device,
    )                                                    # (K, S)

    fig, axes = plt.subplots(1, len(MS), figsize=(14, 4), sharey=True)
    for ax, M in zip(axes, MS):
        y = Y[:, :M].mean(axis=1)
        rmse = float(np.sqrt(np.mean((y - y0) ** 2)))
        ax.plot(z, y0, "k-", lw=1.5, label="target")
        ax.plot(z, y, color="tab:green", lw=0.7, alpha=0.9,
                label=f"y(z), M={M}")
        ax.set_xlabel("z")
        ax.set_title(f"M = {M}   (RMSE {rmse:.3f})")
        ax.legend(frameon=False, loc="lower center")
    axes[0].set_ylabel("y(z)")
    axes[0].set_ylim(-3.0, 3.0)
    fig.suptitle(f"GA-trained computer ({name}), {K} z-points", fontsize=13)
    fig.tight_layout()
    out = f"{GDIR}/fig_output_samples_{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path, name in RUNS:
        plot_run(path, name, device)


if __name__ == "__main__":
    main()
