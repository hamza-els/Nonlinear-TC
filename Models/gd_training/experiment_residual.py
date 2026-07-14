"""Experiment J: pointwise residual correction of the idealized targets.

Iterated "aim off-center" scheme:
  1. Train a student (noisy guides) on targets A(z).
  2. Measure its realized mean activations at tf: realized_i(z) (M samples).
  3. Correct pointwise per (node, z):  A'(z) = A(z) + [A(z) - realized(z)]
     -- i.e. shift each target by minus the systematic miss.  Unlike gain
     calibration this can move targets whose ideal is zero (the output node's
     -0.2 offset at z = 0.25/0.75).
  4. Retrain on A'(z); optionally iterate.

Usage:
    python experiment_residual.py [seed] [iterations]
"""

import sys

import torch

from digital_net import target
from thermo_student import BETA
from train_gd import run, student_targets, train_student, evaluate
from plot_output_samples import plot_output
from plot_trajectories_gd import plot_trajectories

GDIR = "../../Graphs/gd_graphs"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
ITERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
GUIDE_M = 4


def noisy_train(A, z, seed, device):
    student, _ = train_student(A, z, guide_beta=BETA, guide_M=GUIDE_M,
                               rel_tol=0.0, seed=seed, device=device,
                               verbose=False)
    c = student.fit_readout(z, target(z), M=256, seed=seed + 1)
    ev = evaluate(student, M=256, seed=seed, device=device)
    return student, c, ev


def main():
    torch.manual_seed(SEED)
    _, teacher, stats0 = run(seed=SEED)   # baseline (T=0 guide) for reference
    device = next(teacher.parameters()).device
    z = torch.linspace(0.0, 1.0, 128, device=device)
    A0 = student_targets(teacher, z)      # (B, N) original targets

    history = [("T=0 guide baseline", stats0["rmse"])]

    A = A0
    students = []
    for it in range(ITERS + 1):           # it=0 trains on uncorrected A0
        student, c, ev = noisy_train(A, z, SEED, device)
        students.append(student)
        label = "noisy, uncorrected" if it == 0 else f"residual iter {it}"
        history.append((label, ev["rmse"]))
        print(f">>> {label}: rmse {ev['rmse']:.3f}  bias2 {ev['bias2']:.3f}  "
              f"var {ev['var']:.3f}  c {c:.2f}")
        if it == ITERS:
            break
        # pointwise correction from this student's realized mean activations
        with torch.no_grad():
            realized = student.simulate(z, M=512, seed=SEED + 20 + it)
        delta = A0 - realized              # miss w.r.t. the ORIGINAL ideals
        A = A + delta                      # accumulate correction
        print(f"    max |miss|: hidden {delta[:, :-1].abs().max():.3f}  "
              f"output {delta[:, -1].abs().max():.3f}")

    # target_fn for plots: corrected targets at arbitrary z, computed the same
    # way (original targets there + accumulated correction interpolated by
    # re-simulating the previous students).
    corr_students = students[:-1]

    def target_fn(zz):
        Az = student_targets(teacher, zz)
        Acorr = Az.clone()
        for st in corr_students:
            with torch.no_grad():
                r = st.simulate(zz, M=512, seed=SEED + 40)
            Acorr = Acorr + (Az - r)
        return Acorr

    final = students[-1]
    title = (f"J: pointwise residual correction x{ITERS}, seed {SEED} "
             f"(uncorrected {history[1][1]:.3f} -> {history[-1][1]:.3f})")
    plot_output(final, teacher, out_path=f"{GDIR}/fig_teacher_residual.png",
                suptitle=title)
    plot_trajectories(final, teacher,
                      out_path=f"{GDIR}/fig_traj_residual.png",
                      suptitle=title, target_fn=target_fn)

    print("\n=== summary (seed", SEED, ") ===")
    for label, r in history:
        print(f"{label:22s} rmse {r:.3f}")


if __name__ == "__main__":
    main()
