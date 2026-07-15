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
    python plots/plot_trajectories_gd.py   # -> Graphs/gd_graphs/fig_trajectories.png
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

from digital_net import HIDDEN, N, N_OUT
from thermo_student import idealized_trajectory, TF, DT
from train_gd import run, student_targets

OUT_PATH = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs",
                        "fig_trajectories.png")


def plot_trajectories(student, teacher, out_path=OUT_PATH,
                      suptitle="student node trajectories vs idealized "
                               "teacher trajectories", gains=None,
                      target_fn=None):
    """Draw the trajectory-comparison figure for a trained (student, teacher).

    gains: optional per-node (N,) multipliers applied to the targets (for
    students trained on gain-calibrated targets).
    target_fn: optional callable z -> (B, N) targets, replacing
    student_targets entirely (for pointwise-corrected targets).
    Returns (mean_node_rms, output_node_rms)."""
    device = student.b.device
    get_A = target_fn or (lambda zz: student_targets(teacher, zz))

    # --- trajectories at three representative inputs ----------------------
    z3 = torch.tensor([0.25, 0.5, 0.75], device=device)
    A3 = get_A(z3)                                          # (3, N)
    if gains is not None:
        A3 = A3 * gains
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
    Ag = get_A(zg)
    if gains is not None:
        Ag = Ag * gains
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
    bar_colors = ["tab:blue"] * HIDDEN + ["tab:green"] * N_OUT
    axd.bar(np.arange(N), node_rms, color=bar_colors, width=0.8)
    axd.set_xlabel("node index")
    axd.set_ylabel(r"RMS$_{t,z}\,(\langle x_i\rangle - x_i^{(0)})$")
    axd.set_title("per-node deviation from ideal (outputs in green)")

    fig.suptitle(suptitle + "\n(solid = student mean over 200 samples, "
                 "dashed = ideal)", fontsize=12)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"mean node RMS deviation: {node_rms.mean():.4f}   "
          f"output node: {node_rms[-1]:.4f}")
    return float(node_rms.mean()), float(node_rms[-1])


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    student, teacher, stats = run(seed=seed)
    plot_trajectories(student, teacher)


if __name__ == "__main__":
    main()
