"""Teacher-optimization experiments: make the teacher easy for the student.

Three ways of picking/shaping the digital teacher, all with the same student
pipeline (2x32 teacher, tf = 1, beta = 10, act_scale = 1, post-hoc c):

  A. Lottery   : train N_LOTTERY teachers (seeds 0..N-1), train a student on
                 each, keep the teacher whose student evaluates best.
  B. Act-reg   : penalize hidden-activation magnitude during teacher training
                 (act_reg * mean(A^2)) so targets stay off the tanh rails.
  C. Weight dec: plain L2 weight decay on the teacher.

Each experiment saves its own 4-panel output figure for its best seed.

Each experiment saves an output figure AND a trajectory figure for its best
seed.  An optional filename suffix keeps runs at different settings apart:

Usage:
    python experiments/experiment_teacher_opt.py          # fig_teacher_<name>.png
    python experiments/experiment_teacher_opt.py _tf02    # fig_teacher_<name>_tf02.png
"""

import os
import sys

# core modules (digital_net, thermo_student, train_gd) live one level up
_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import torch

from train_gd import run
from plots.plot_output_samples import plot_output
from plots.plot_trajectories_gd import plot_trajectories

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
N_LOTTERY = 10
REG_SEEDS = (0, 1, 2)
ACT_REG = 1e-2
WD = 1e-3


def one(seed, **kw):
    torch.manual_seed(seed)
    student, teacher, stats = run(seed=seed, **kw)
    return student, teacher, stats


def sweep(label, seeds, **kw):
    best = None
    for s in seeds:
        student, teacher, stats = one(s, **kw)
        print(f">>> {label} seed={s}  rmse={stats['rmse']:.3f}  "
              f"bias2={stats['bias2']:.3f}  c={stats['readout_c']:.2f}")
        if best is None or stats["rmse"] < best[3]["rmse"]:
            best = (s, student, teacher, stats)
    s, student, teacher, stats = best
    print(f">>> {label} BEST seed={s}  rmse={stats['rmse']:.3f}")
    return best


def report(name, label, s, student, teacher, stats):
    """Save output + trajectory figures for an experiment's best seed."""
    title = f"{label} -- best seed {s} (RMSE {stats['rmse']:.3f})"
    plot_output(student, teacher,
                out_path=f"{GDIR}/fig_teacher_{name}{SUFFIX}.png",
                suptitle=title)
    plot_trajectories(student, teacher,
                      out_path=f"{GDIR}/fig_traj_{name}{SUFFIX}.png",
                      suptitle=title)


def main():
    # --- A: lottery over plain teachers ------------------------------------
    s, student, teacher, stats = sweep("A-lottery", range(N_LOTTERY))
    report("lottery", f"A: teacher lottery over {N_LOTTERY} seeds",
           s, student, teacher, stats)
    res_a = (s, stats)

    # --- B: activation-regularized teacher ---------------------------------
    s, student, teacher, stats = sweep("B-actreg", REG_SEEDS,
                                       teacher_act_reg=ACT_REG)
    report("actreg", f"B: activation-regularized teacher (act_reg={ACT_REG:g})",
           s, student, teacher, stats)
    res_b = (s, stats)

    # --- C: weight-decayed teacher ------------------------------------------
    s, student, teacher, stats = sweep("C-wd", REG_SEEDS, teacher_wd=WD)
    report("wd", f"C: weight-decay teacher (wd={WD:g})",
           s, student, teacher, stats)
    res_c = (s, stats)

    print("\n=== summary ===")
    for name, (s, st) in [("A lottery", res_a), ("B act-reg", res_b),
                          ("C weight-decay", res_c)]:
        print(f"{name:15s} best seed {s}  rmse {st['rmse']:.3f}  "
              f"bias2 {st['bias2']:.3f}  var {st['var']:.3f}  "
              f"teacher_mse {st['teacher_mse']:.1e}")


if __name__ == "__main__":
    main()
