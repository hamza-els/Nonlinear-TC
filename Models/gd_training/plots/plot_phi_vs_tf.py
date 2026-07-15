"""Loss phi vs observation time tf for the trained GD student.

Analog of Whitelam & Casert's phi-vs-tf panel (arXiv:2412.17183 Fig. 2(e)):
the computer is trained to produce its answer AT tf = 1, so the loss
phi(tf) = mean_z (y0 - y_M)^2, evaluated with the trained parameters and the
tf=1-fitted readout at other observation times, shows a sharp notch at the
trained clock time: observe too early and the answer has not formed, too
late and the state drifts off toward equilibrium.

No retraining -- the seed-1 champion is rebuilt deterministically and only
sampled at each tf.

Usage:
    python plots/plot_phi_vs_tf.py [seed]   # -> Graphs/gd_graphs/fig_phi_vs_tf.png
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
from thermo_student import TF
from train_gd import run

OUT_PATH = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs",
                        f"fig_phi_vs_tf_tf{TF:g}.png")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
M = 256


def main():
    torch.manual_seed(SEED)
    student, teacher, stats = run(seed=SEED)
    device = student.b.device
    z = torch.linspace(0.0, 1.0, 101, device=device)
    y0 = target(z)

    # log grid plus extra linear-spaced points so the linear panel is smooth
    tf_grid = np.unique(np.round(np.concatenate(
        [np.logspace(-1, 1, 25), np.linspace(1.5, 9.5, 9)]), 4))
    phi = []
    for tf in tf_grid:
        with torch.no_grad():
            pred = student.sample_outputs(z, M=M, tf=float(tf),
                                          seed=0).mean(dim=1)
        phi.append(torch.mean((pred - y0) ** 2).item())
        print(f"  tf={tf:7.3f}  phi={phi[-1]:.4e}", flush=True)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax in (axl, axr):
        ax.plot(tf_grid, phi, "-", color="tab:green", lw=1.3)
        ax.set_yscale("log")
        ax.set_xlabel(r"$t_{\mathrm{f}}$")
    axl.set_xscale("log")
    axl.set_ylabel(r"$\phi$")
    axl.set_title("log time axis", fontsize=10)
    axr.set_title("linear time axis", fontsize=10)
    fig.suptitle(rf"loss vs observation time (trained at $t_f={TF:g}$, M={M})",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
