"""Reproduce Fig. 2(c)/(d) for the nonlinear thermodynamic computer, plus a
third panel for the var-free (bias) loss.

(c) combined loss phi vs evolutionary time n, log-scale.
(d) trained output y(z) vs input z (recomputed from the saved weights, averaged
    over M samples), against the target cos(2 pi z).
(e) pure bias loss (mean_z bias^2) vs n, log-scale. With var_weight > 0 the
    combined loss (c) is dominated by the variance penalty, so (e) is what shows
    how well the computer actually fits the target. Requires a run trained with
    bias_history logging; older runs show a placeholder here.

Usage:
    python plot_fig2.py                 # regenerate all three runs into
                                        #   ../Graphs/computer_graphs/
    python plot_fig2.py runs/run.npz    # a single run -> fig2_cde.png
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genetic_algorithm import load_run, output_curve


def plot(run_path, out_path="fig2_cde.png", M_eval=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    params, history, cfg = load_run(run_path)
    raw = np.load(run_path)
    bias_history = raw["bias_history"] if "bias_history" in raw.files else None
    history = np.asarray(history)
    n = np.arange(len(history))

    fig, (axc, axd, axe) = plt.subplots(1, 3, figsize=(15, 4))

    # --- (c) combined loss vs evolutionary time --------------------------
    axc.plot(n, history, color="tab:green", lw=1.0)
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

    # --- (e) pure bias loss vs evolutionary time -------------------------
    axe.set_xlabel("n")
    axe.set_ylabel(r"$\phi_{\mathrm{bias}}$")
    axe.set_title("(e) bias loss")
    if bias_history is not None:
        bias_history = np.asarray(bias_history)
        axe.plot(n, bias_history, color="tab:purple", lw=1.0)
        axe.set_yscale("log")
        # Share y-limits with (c) so the gap (the variance penalty) is legible.
        lo = float(min(np.nanmin(history), np.nanmin(bias_history)))
        hi = float(max(np.nanmax(history), np.nanmax(bias_history)))
        for ax in (axc, axe):
            ax.set_ylim(0.8 * lo, 1.2 * hi)
    else:
        axe.text(0.5, 0.5, "bias_history\nnot recorded",
                 ha="center", va="center", transform=axe.transAxes,
                 color="0.5")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    if len(sys.argv) > 1:
        plot(sys.argv[1])
        return
    runs = [("runs/run_reg.npz", "regular"),
            ("runs/run_low_var.npz", "low_var"),
            ("runs/run_high_var.npz", "high_var")]
    for path, name in runs:
        plot(path, out_path=f"../Graphs/computer_graphs/fig2_{name}.png")


if __name__ == "__main__":
    main()
