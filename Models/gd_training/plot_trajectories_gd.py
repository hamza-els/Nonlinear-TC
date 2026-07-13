"""How closely do the student's nodes follow their idealized trajectories?

Analog of the paper's Fig. 2(b): node activations x_i(t) of the trained
thermodynamic student (solid, sample-averaged over M trajectories) against the
idealized teacher trajectories they were trained to reproduce (dashed).

Panels 1-3: trajectories at z = 0.25, 0.5, 0.75 for the output node and the
three hidden nodes with the largest mean |target| (same nodes / colors in
every panel, so a line can be followed across z).

Panel 4: per-node closeness over a z-grid -- the RMS deviation
sqrt(mean_{t,z} (<x_i(t)> - x_ideal_i(t))^2) for each of the 33 nodes.

Usage:
    python plot_trajectories_gd.py   # -> ../../Graphs/gd_graphs/fig_trajectories.png
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import HIDDEN, N
from thermo_student import idealized_trajectory, TF, DT
from train_gd import run, student_targets

OUT_PATH = "../../Graphs/gd_graphs/fig_trajectories.png"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    student, teacher, stats = run(seed=seed)

    device = student.b.device

    # --- trajectories at three representative inputs ----------------------
    z3 = torch.tensor([0.25, 0.5, 0.75], device=device)
    A3 = student_targets(teacher, z3)                       # (3, N)
    ideal3 = idealized_trajectory(A3)                       # (K+1, 3, N)
    with torch.no_grad():
        steps, mean3 = student.mean_trajectory(z3, M=200, record_every=10,
                                               seed=0)     # (T, 3, N)
    mean3 = mean3.cpu()
    ideal3_sub = ideal3[steps].cpu()                        # (T, 3, N)
    t_frac = steps.numpy() * DT / TF

    # Track the same nodes in every panel: output + 3 largest-|target| hidden.
    hid_rank = A3[:, :HIDDEN].abs().mean(dim=0).argsort(descending=True)
    nodes = [int(hid_rank[0]), int(hid_rank[1]), int(hid_rank[2]), N - 1]
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:green"]
    names = [f"hidden {i}" for i in nodes[:3]] + ["output"]

    # --- per-node RMS deviation over a z-grid -----------------------------
    zg = torch.linspace(0.0, 1.0, 21, device=device)
    Ag = student_targets(teacher, zg)
    idealg = idealized_trajectory(Ag)
    with torch.no_grad():
        steps_g, meang = student.mean_trajectory(zg, M=100, record_every=20,
                                                 seed=1)
    dev = (meang - idealg[steps_g]) ** 2                    # (T, 21, N)
    node_rms = dev.mean(dim=(0, 1)).sqrt().cpu().numpy()    # (N,)

    # --- figure ------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, zi in zip(axes.flat[:3], range(3)):
        for node, color, name in zip(nodes, colors, names):
            lw = 1.8 if node == N - 1 else 1.1
            ax.plot(t_frac, mean3[:, zi, node].numpy(), "-", color=color,
                    lw=lw, label=name)
            ax.plot(t_frac, ideal3_sub[:, zi, node].numpy(), "--", color=color,
                    lw=lw * 0.8)
        ax.set_xlabel(r"$t/t_f$")
        ax.set_ylabel(r"$x_i(t)$")
        ax.set_title(f"z = {z3[zi].item():.2f}")
    axes.flat[0].legend(frameon=False, fontsize=8, loc="upper left")

    axd = axes.flat[3]
    bar_colors = ["tab:blue"] * HIDDEN + ["tab:green"]
    axd.bar(np.arange(N), node_rms, color=bar_colors, width=0.8)
    axd.set_xlabel("node index")
    axd.set_ylabel(r"RMS$_{t,z}\,(\langle x_i\rangle - x_i^{(0)})$")
    axd.set_title("per-node deviation from ideal (output in green)")

    fig.suptitle("student node trajectories vs idealized teacher trajectories"
                 "\n(solid = student mean over 200 samples, dashed = ideal)",
                 fontsize=12)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved {OUT_PATH}")
    print(f"mean node RMS deviation: {node_rms.mean():.4f}   "
          f"output node: {node_rms[-1]:.4f}")


if __name__ == "__main__":
    main()
