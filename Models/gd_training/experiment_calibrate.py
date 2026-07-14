"""Experiment G: per-node gain calibration of the idealized trajectories.

Pass 1 trains a student as usual and records its realized mean trajectories.
Each node's realized curve is regressed (through the origin) against its
ideal curve, giving a per-node gain g_i = <x_real . x_ideal> / <x_ideal^2>
over (t, z) -- the multiplier that "best fits the curves the student actually
generated".  Pass 2 rescales every node's target by g_i (so the idealized
trajectories point where the dynamics naturally goes) and retrains a fresh
student on the calibrated targets.

Usage:
    python experiment_calibrate.py [seed]
"""

import sys

import torch

from digital_net import target
from thermo_student import idealized_trajectory
from train_gd import run, student_targets, train_student, evaluate
from plot_output_samples import plot_output
from plot_trajectories_gd import plot_trajectories

GDIR = "../../Graphs/gd_graphs"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def main():
    # --- pass 1: ordinary training ------------------------------------------
    torch.manual_seed(SEED)
    student, teacher, stats = run(seed=SEED)
    print(f"pass 1: rmse {stats['rmse']:.3f}  bias2 {stats['bias2']:.3f}  "
          f"c {stats['readout_c']:.2f}")

    # regenerate the pass-1 trajectory figure with correctly-scaled ideals
    # (earlier fig_traj_polyinput.png drew ideals at act_scale=0.5 by mistake)
    plot_trajectories(student, teacher,
                      out_path=f"{GDIR}/fig_traj_polyinput.png",
                      suptitle=f"F: polynomial inputs (z, z^2, z^3), seed "
                               f"{SEED} (RMSE {stats['rmse']:.3f})")

    device = student.b.device
    z = torch.linspace(0.0, 1.0, 128, device=device)
    A = student_targets(teacher, z)                     # (B, N)
    ideal = idealized_trajectory(A)                     # (K+1, B, N)
    with torch.no_grad():
        steps, mean = student.mean_trajectory(z, M=100, record_every=20,
                                              seed=SEED + 10)
    ideal_s = ideal[steps]                              # (T, B, N)

    # --- per-node best-fit gain ---------------------------------------------
    num = (mean * ideal_s).sum(dim=(0, 1))
    den = (ideal_s * ideal_s).sum(dim=(0, 1)).clamp_min(1e-9)
    g = (num / den).clamp(0.25, 3.0)                    # (N,)
    print(f"gains g_i: min {g.min():.2f}  median {g.median():.2f}  "
          f"max {g.max():.2f}   (output node: {g[-1]:.2f})")
    A2 = A * g
    print(f"max |target|: {A.abs().max():.2f} -> {A2.abs().max():.2f}")

    # --- pass 2: retrain on calibrated targets -------------------------------
    student2, hist = train_student(A2, z, seed=SEED, device=device,
                                   verbose=False)
    print(f"pass 2: OM loss {hist[0]:.3e} -> {hist[-1]:.3e} "
          f"({len(hist)} rounds)")
    c = student2.fit_readout(z, target(z), M=256, seed=SEED + 1)
    ev = evaluate(student2, M=256, seed=SEED, device=device)
    print(f"pass 2: rmse {ev['rmse']:.3f}  bias2 {ev['bias2']:.3f}  "
          f"var {ev['var']:.3f}  c {c:.2f}  "
          f"range [{ev['pred_min']:+.2f}, {ev['pred_max']:+.2f}]")

    title = (f"G: gain-calibrated targets, seed {SEED} "
             f"(pass1 {stats['rmse']:.3f} -> pass2 {ev['rmse']:.3f})")
    plot_output(student2, teacher, out_path=f"{GDIR}/fig_teacher_calib.png",
                suptitle=title)
    plot_trajectories(student2, teacher, out_path=f"{GDIR}/fig_traj_calib.png",
                      suptitle=title, gains=g)


if __name__ == "__main__":
    main()
