"""Error vs number of samples M, for the three trained computers.

Analogue of the paper's Fig. 2(f)/S7(e), but overlaying the three variance
conditions (regular / low-var / high-var). For each trained run we draw a pool
of independent single-shot outputs; then for a grid of M we form `n_runs`
disjoint groups of M samples, average each group to an M-sample estimate y_M(z),
and measure its MSE against cos(2 pi z). The mean over the groups is the average
error at that M; their spread is drawn as a +/- 1 std band.

This levels the comparison a sampling perspective: the regular computer has low
bias but noisy single-shot output (error falls steeply with M), while the
variance-penalized computers trade bias for a tighter readout (they start lower
/ fall less steeply). The high-var computer, which collapsed toward a constant,
sits near its bias floor at all M.

Usage:
    python plot_error_vs_M.py            # -> ../Graphs/fig_error_vs_M.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genetic_algorithm import load_run, sample_output_pool


def error_vs_M(params, cfg, M_grid, n_runs=250, K=250, pool=None,
               device="cpu", seed=0):
    """Mean and std (over n_runs runs) of the MSE loss phi at each M in M_grid.

    We simulate one pool of `pool` independent single-shot outputs per computer
    (real reset-samples), then form each run by averaging M samples drawn (with
    replacement) from that pool -- a bootstrap over the pool. This gives an
    unbiased mean error for any M without simulating n_runs * M fresh
    trajectories per point (which for M=500, n_runs=250 would be 125k sims each)."""
    z = np.linspace(0.0, 1.0, K)
    y0 = np.cos(2.0 * np.pi * z)
    M_max = int(M_grid.max())
    S = int(pool or max(8000, 16 * M_max))       # pool of single-shot outputs
    Y = sample_output_pool(
        params, z, S=S, m_chunk=int(cfg["m_chunk"]),
        tf=float(cfg["tf"]), dt=float(cfg["dt"]),
        beta=float(cfg["beta"]), mu=float(cfg["mu"]), device=device,
    )                                            # (K, S)

    rng = np.random.default_rng(seed)
    mean_err, std_err = [], []
    for M in M_grid:
        M = int(M)
        idx = rng.integers(0, S, size=(n_runs, M))          # bootstrap indices
        y_M = Y[:, idx].mean(axis=2)                         # (K, n_runs)
        phi = ((y0[:, None] - y_M) ** 2).mean(axis=0)        # (n_runs,) loss per run
        mean_err.append(phi.mean())
        std_err.append(phi.std())
    return np.array(mean_err), np.array(std_err)


def plot(out_path="../Graphs/fig_error_vs_M.png", n_runs=250, M_max=500,
         K=250, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    M_grid = np.unique(np.round(np.logspace(0, np.log10(M_max), 18)).astype(int))

    runs = [
        ("run_reg.npz", "regular", "tab:green"),
        ("run_low_var.npz", "low var", "tab:blue"),
        ("run_high_var.npz", "high var", "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for path, label, color in runs:
        params, _, cfg = load_run(path)
        mean_err, std_err = error_vs_M(params, cfg, M_grid, n_runs=n_runs,
                                       K=K, device=device)
        ax.plot(M_grid, mean_err, "-o", ms=3, color=color, label=label)
        ax.fill_between(M_grid, np.clip(mean_err - std_err, 1e-4, None),
                        mean_err + std_err, color=color, alpha=0.18)
        print(f"{label:8s}  phi(M=1)={mean_err[0]:.3f}  phi(M={M_grid[-1]})={mean_err[-1]:.4f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M  (samples averaged per readout)")
    ax.set_ylabel(r"average error  $\phi = \langle (y_0 - y_M)^2 \rangle_z$")
    ax.set_title(f"error vs samples ({n_runs} runs per M)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    plot()
