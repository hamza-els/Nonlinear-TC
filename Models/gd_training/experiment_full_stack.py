"""Experiment I: 6 polynomial inputs + the full optimization stack.

Pipeline (everything that has individually proven out, composed):
  1. Teacher lottery over 10 seeds (T=0 guide) -- pick the best teacher.
  2. Retrain the winner's student on NOISY guide trajectories
     (beta = BETA, GUIDE_M realizations) -- quiets the output (single-shot).
  3. Gain-calibrate: per-node least-squares gain of the noisy student's
     realized mean curves vs its ideal curves; rescale targets by g_i.
  4. Final training: calibrated targets + noisy guides.  Figures + summary.

Usage:
    python experiment_full_stack.py
"""

import torch

from digital_net import target, N
from thermo_student import BETA, idealized_trajectory
from train_gd import run, student_targets, train_student, evaluate
from experiment_teacher_opt import sweep
from plot_output_samples import plot_output
from plot_trajectories_gd import plot_trajectories

GDIR = "../../Graphs/gd_graphs"
GUIDE_M = 4


def noisy_train(A, z, seed, device):
    student, hist = train_student(A, z, guide_beta=BETA, guide_M=GUIDE_M,
                                  rel_tol=0.0, seed=seed, device=device,
                                  verbose=False)
    c = student.fit_readout(z, target(z), M=256, seed=seed + 1)
    ev = evaluate(student, M=256, seed=seed, device=device)
    return student, c, ev


def main():
    # --- 1. lottery ---------------------------------------------------------
    s, student0, teacher, stats0 = sweep("I-poly6", range(10))

    device = student0.b.device
    z = torch.linspace(0.0, 1.0, 128, device=device)
    A = student_targets(teacher, z)

    # --- 2. noisy guides on the winner ---------------------------------------
    student_n, c_n, ev_n = noisy_train(A, z, s, device)
    print(f">>> noisy guide:        rmse {ev_n['rmse']:.3f}  "
          f"bias2 {ev_n['bias2']:.3f}  var {ev_n['var']:.3f}  c {c_n:.2f}")

    # --- 3. per-node gain calibration ----------------------------------------
    ideal = idealized_trajectory(A)
    with torch.no_grad():
        steps, mean = student_n.mean_trajectory(z, M=100, record_every=20,
                                                seed=s + 10)
    ideal_s = ideal[steps]
    num = (mean * ideal_s).sum(dim=(0, 1))
    den = (ideal_s * ideal_s).sum(dim=(0, 1)).clamp_min(1e-9)
    g = (num / den).clamp(0.25, 3.0)
    print(f">>> gains: min {g.min():.2f}  median {g.median():.2f}  "
          f"max {g.max():.2f}  (output {g[-1]:.2f})")

    # --- 4. final: calibrated targets + noisy guides --------------------------
    student_f, c_f, ev_f = noisy_train(A * g, z, s, device)
    print(f">>> noisy + calibrated: rmse {ev_f['rmse']:.3f}  "
          f"bias2 {ev_f['bias2']:.3f}  var {ev_f['var']:.3f}  c {c_f:.2f}")

    title = (f"I: 6 poly inputs, full stack, seed {s} "
             f"(lottery {stats0['rmse']:.3f} -> noisy {ev_n['rmse']:.3f} "
             f"-> +calib {ev_f['rmse']:.3f})")
    plot_output(student_f, teacher, out_path=f"{GDIR}/fig_teacher_poly6.png",
                suptitle=title)
    plot_trajectories(student_f, teacher,
                      out_path=f"{GDIR}/fig_traj_poly6.png",
                      suptitle=title, gains=g)

    print("\n=== summary (seed", s, ") ===")
    print(f"lottery (T=0 guide)   rmse {stats0['rmse']:.3f}")
    print(f"+ noisy guides        rmse {ev_n['rmse']:.3f}  var {ev_n['var']:.3f}")
    print(f"+ gain calibration    rmse {ev_f['rmse']:.3f}  var {ev_f['var']:.3f}")


if __name__ == "__main__":
    main()
