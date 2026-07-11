"""Error vs number of samples M, for the three trained computers.

Analogue of the paper's Fig. 2(f)/S7(e), but overlaying the three variance
conditions (regular / low-var / high-var). For each M on the grid we simulate
n_runs * M fresh reset-samples, split them into n_runs disjoint runs of M
samples, average each run to an M-sample estimate y_M(z), and measure its MSE
against cos(2 pi z). The mean over the runs is the average error at that M;
their spread is drawn as a +/- 1 std band. Every point is fully re-simulated --
nothing is shared between M values or between runs, so each point is a genuine
independent experiment (at the price of Sum_M n_runs*M trajectories per
computer, dominated by the largest M).

This levels the comparison from a sampling perspective: the regular computer has
low bias but noisy single-shot output (error falls steeply with M), while the
variance-penalized computers trade bias for a tighter readout (they start lower
/ fall less steeply). A computer that collapsed toward a constant output sits
near its bias floor at all M.

Usage:
    python plot_error_vs_M.py            # -> ../../Graphs/computer_graphs/fig_error_vs_M.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genetic_algorithm import load_run, sample_output_pool


def error_vs_M(params, cfg, M_grid, n_runs=100, K=250, device="cpu"):
    """Mean and std (over n_runs fresh runs) of the MSE loss phi at each M.

    For each M, n_runs * M brand-new reset-samples are simulated and split into
    n_runs disjoint groups of M; no sample is reused between runs or between
    points on the M grid."""
    z = np.linspace(0.0, 1.0, K)
    y0 = np.cos(2.0 * np.pi * z)

    mean_err, std_err = [], []
    for M in M_grid:
        M = int(M)
        Y = sample_output_pool(
            params, z, S=n_runs * M, m_chunk=int(cfg["m_chunk"]),
            tf=float(cfg["tf"]), dt=float(cfg["dt"]),
            beta=float(cfg["beta"]), mu=float(cfg["mu"]), device=device,
        )                                                    # (K, n_runs * M)
        y_M = Y.reshape(K, n_runs, M).mean(axis=2)           # (K, n_runs)
        phi = ((y0[:, None] - y_M) ** 2).mean(axis=0)        # (n_runs,) loss per run
        mean_err.append(phi.mean())
        std_err.append(phi.std())
        print(f"    M={M:5d}  phi={mean_err[-1]:.4f}", flush=True)
    return np.array(mean_err), np.array(std_err)


def plot(out_path="../../Graphs/computer_graphs/fig_error_vs_M.png", n_runs=100,
         M_max=1000, K=250, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    M_grid = np.unique(np.round(np.logspace(0, np.log10(M_max), 18)).astype(int))

    # high_var is omitted: its collapsed constant output sits flat at its bias
    # floor (~0.5) for all M, which only compresses the interesting curves.
    runs = [
        ("runs/run_reg.npz", "regular", "tab:green"),
        ("runs/run_low_var.npz", "low var", "tab:blue"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for path, label, color in runs:
        print(f"{label} ...", flush=True)
        params, _, cfg = load_run(path)
        mean_err, std_err = error_vs_M(params, cfg, M_grid, n_runs=n_runs,
                                       K=K, device=device)
        ax.plot(M_grid, mean_err, "-o", ms=3, color=color, label=label)
        ax.fill_between(M_grid, np.clip(mean_err - std_err, 1e-4, None),
                        mean_err + std_err, color=color, alpha=0.18)
        print(f"{label:8s}  phi(M=1)={mean_err[0]:.3f}  phi(M={M_grid[-1]})={mean_err[-1]:.4f}")

    ax.set_yscale("log")
    ax.set_xlabel("M  (samples averaged per readout)")
    ax.set_ylabel(r"average error  $\phi = \langle (y_0 - y_M)^2 \rangle_z$")
    ax.set_title(f"error vs samples ({n_runs} fresh runs per M)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot()
