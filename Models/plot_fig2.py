"""Reproduce Fig. 2(c) and 2(d) for the nonlinear thermodynamic computer.

(c) loss phi vs evolutionary time n, log-scale.
(d) trained output y(z) vs input z (recomputed from the saved weights, averaged
    over M samples), against the target cos(2 pi z).

Usage:
    python plot_fig2.py [run.npz]      # default: trained_weights.npz
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genetic_algorithm import load_run, output_curve


def plot(run_path, out_path="fig2_cd.png", M_eval=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    params, history, cfg = load_run(run_path)

    fig, (axc, axd) = plt.subplots(1, 2, figsize=(10, 4))

    # --- (c) loss vs evolutionary time -----------------------------------
    axc.plot(np.arange(len(history)), history, color="tab:green", lw=1.0)
    axc.set_yscale("log")
    axc.set_xlabel("n")
    axc.set_ylabel(r"$\phi$")
    axc.set_title("(c) loss")

    # --- (d) trained output y(z) vs target -------------------------------
    z = np.linspace(0.0, 1.0, 250)
    axd.plot(z, np.cos(2.0 * np.pi * z), "k-", lw=1.5, label="target")
    m = int(M_eval or cfg["M"])
    y = output_curve(
        params, z, M=m, m_chunk=int(cfg["m_chunk"]),
        tf=float(cfg["tf"]), dt=float(cfg["dt"]),
        beta=float(cfg["beta"]), mu=float(cfg["mu"]), device=device,
    )
    axd.plot(z, y, color="tab:green", lw=1.0, label="y(z)")
    axd.set_xlabel("z")
    axd.set_ylabel("y(z)")
    axd.set_ylim(-1.3, 1.3)
    axd.set_title("(d) output")
    axd.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "trained_weights.npz"
    plot(run)


if __name__ == "__main__":
    main()
