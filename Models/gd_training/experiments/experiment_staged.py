"""Staged (first-layer-kill) architecture vs the standard all-to-all computer.

Trains, over the same seeds and teacher, a StagedStudent (first-layer
couplings switch off at kill_time) and a standard ThermoStudent baseline, and
compares.  Graphs the best staged computer (output samples + node
trajectories -- the trajectory figure shows the first-layer nodes going dead
after kill_time).

Default activation is thermo (tanh selectable).

Usage:
    python experiments/experiment_staged.py [n_seeds] [kill_time] [activation]
"""

import os
import sys

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import set_activation, target
from thermo_student import TF
from staged_student import StagedStudent, LAYER1
from train_gd import run
from plots.plot_output_samples import plot_output
from plots.plot_trajectories_gd import plot_trajectories

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
KILL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
ACT = sys.argv[3] if len(sys.argv) > 3 else "thermo"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_activation(ACT)
    print(f"config: activation={ACT} tf={TF:g} kill_time={KILL:g} "
          f"seeds={N_SEEDS} LAYER1={LAYER1}", flush=True)

    staged, base = [], []
    best = (None, None, 1e9)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        s_st, teacher, st_st = run(seed=seed, student_cls=StagedStudent,
                                   student_kw={"kill_time": KILL})
        torch.manual_seed(seed)
        s_bs, _, st_bs = run(seed=seed)          # standard all-to-all baseline
        staged.append(st_st["rmse"]); base.append(st_bs["rmse"])
        print(f">>> seed={seed}  staged rmse={st_st['rmse']:.4f} "
              f"var={st_st['var']:.3f}  |  baseline rmse={st_bs['rmse']:.4f} "
              f"var={st_bs['var']:.3f}", flush=True)
        if st_st["rmse"] < best[2]:
            best = (s_st, teacher, st_st["rmse"])

    student, teacher, _ = best
    st = {"rmse": best[2]}
    title = (f"staged computer ({ACT}, kill first layer at t={KILL:g}, "
             f"tf={TF:g}) -- best (RMSE {best[2]:.4f})")
    plot_output(student, teacher, out_path=f"{GDIR}/fig_teacher_staged.png",
                suptitle=title, tf=TF)
    plot_trajectories(student, teacher,
                      out_path=f"{GDIR}/fig_traj_staged.png",
                      suptitle=title, tf=TF)

    staged, base = np.array(staged), np.array(base)
    print("\n=== summary (RMSE over seeds) ===")
    print(f"  staged   : median {np.median(staged):.4f}  best {staged.min():.4f}"
          f"  worst {staged.max():.4f}")
    print(f"  baseline : median {np.median(base):.4f}  best {base.min():.4f}"
          f"  worst {base.max():.4f}")
    print(f"  staged wins per-seed: {int((staged < base).sum())}/{N_SEEDS}")


if __name__ == "__main__":
    main()
