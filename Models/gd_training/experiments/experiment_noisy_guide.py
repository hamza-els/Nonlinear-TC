"""Experiment H: train on NOISY idealized trajectories.

The guide is the same biased noninteracting computer (b0 = 2A + 4A^3), but
simulated at the student's own temperature (beta = BETA) with GUIDE_M
stochastic realizations per input, instead of the deterministic T=0 path.
Rationale (verified in test_om_recovery check 4): noise explores state space,
breaking the OM null-space degeneracy, and the fit absorbs the <x^3>
rectification of finite-T mean dynamics.  The OM loss floor becomes ~N/2.

Usage:
    python experiments/experiment_noisy_guide.py [seed]
"""

import os
import sys

# core modules (digital_net, thermo_student, train_gd) live one level up
_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import torch

from digital_net import target, N
from thermo_student import BETA
from train_gd import run, student_targets, train_student, evaluate
from plots.plot_output_samples import plot_output
from plots.plot_trajectories_gd import plot_trajectories

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2
GUIDE_M = 4


def main():
    # --- baseline: deterministic guide (pass 1 of the usual pipeline) ------
    torch.manual_seed(SEED)
    student0, teacher, stats0 = run(seed=SEED)
    print(f"baseline (T=0 guide): rmse {stats0['rmse']:.3f}  "
          f"bias2 {stats0['bias2']:.3f}")

    # --- noisy-guide training ------------------------------------------------
    device = student0.b.device
    z = torch.linspace(0.0, 1.0, 128, device=device)
    A = student_targets(teacher, z)
    student, hist = train_student(A, z, guide_beta=BETA, guide_M=GUIDE_M,
                                  rel_tol=0.0, seed=SEED, device=device,
                                  verbose=False)
    print(f"noisy guide (beta={BETA:g}, M={GUIDE_M}): OM loss "
          f"{hist[0]:.3f} -> {hist[-1]:.3f}  (floor ~ N/2 = {N / 2})")

    c = student.fit_readout(z, target(z), M=256, seed=SEED + 1)
    ev = evaluate(student, M=256, seed=SEED, device=device)
    print(f"noisy guide: rmse {ev['rmse']:.3f}  bias2 {ev['bias2']:.3f}  "
          f"var {ev['var']:.3f}  c {c:.2f}  "
          f"range [{ev['pred_min']:+.2f}, {ev['pred_max']:+.2f}]")

    title = (f"H: noisy guide trajectories (beta={BETA:g}, M={GUIDE_M}), "
             f"seed {SEED} (T=0 guide {stats0['rmse']:.3f} -> "
             f"noisy {ev['rmse']:.3f})")
    plot_output(student, teacher, out_path=f"{GDIR}/fig_teacher_noisy.png",
                suptitle=title)
    plot_trajectories(student, teacher, out_path=f"{GDIR}/fig_traj_noisy.png",
                      suptitle=title)


if __name__ == "__main__":
    main()
