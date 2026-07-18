"""Deep 4x8 teacher: cascaded layer-kill (staged) vs standard, per seed.

Teacher: WIDTH=8, DEPTH=4 (four hidden layers of 8, indices [0,8) [8,16)
[16,24) [24,32); output [32,40)).  Observation time tf = 0.6.

Staged schedule -- disconnect earlier layers one at a time as the run
proceeds:
    layer 1 (nodes 0-7)   dies at t = 0.25
    layer 2 (nodes 8-15)  dies at t = 0.4
    layer 3 (nodes 16-23) dies at t = 0.5
    sample at t = 0.6   (layer 4 + output run the final window alone)

For each seed both a StagedStudent (cascade) and a standard ThermoStudent
baseline are trained at tf=0.8 from the same teacher, and compared.

Usage:
    python experiments/experiment_staged_deep.py [n_seeds] [activation]
"""

import os
import sys

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import set_activation, target, WIDTH, DEPTH, HIDDEN, N
from staged_student import StagedStudent
from train_gd import run

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ACT = sys.argv[2] if len(sys.argv) > 2 else "thermo"
TF_DEEP = 0.6
SCHEDULE = [(0.25, 0, 8), (0.4, 8, 16), (0.5, 16, 24)]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_activation(ACT)
    assert (WIDTH, DEPTH) == (8, 4), f"expected 4x8 teacher, got {WIDTH}x{DEPTH}"
    print(f"config: activation={ACT} WIDTH={WIDTH} DEPTH={DEPTH} N={N} "
          f"tf={TF_DEEP} schedule={SCHEDULE} seeds={N_SEEDS}", flush=True)

    staged, base = [], []
    stv, bsv = [], []
    best = (None, None, 1e9)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        s_st, teacher, st = run(seed=seed, tf=TF_DEEP,
                                student_cls=StagedStudent,
                                student_kw={"kill_schedule": SCHEDULE})
        torch.manual_seed(seed)
        s_bs, _, bs = run(seed=seed, tf=TF_DEEP)      # standard baseline, same tf
        staged.append(st["rmse"]); base.append(bs["rmse"])
        stv.append(st["var"]); bsv.append(bs["var"])
        print(f">>> seed={seed}  staged rmse={st['rmse']:.4f} var={st['var']:.3f}"
              f"  |  baseline rmse={bs['rmse']:.4f} var={bs['var']:.3f}  "
              f"teacher_mse={st['teacher_mse']:.1e}", flush=True)
        if st["rmse"] < best[2]:
            best = (s_st, teacher, st["rmse"])

    staged, base = np.array(staged), np.array(base)
    stv, bsv = np.array(stv), np.array(bsv)

    # --- figure: per-seed staged vs baseline ---------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(N_SEEDS); w = 0.38
    a1.bar(x - w/2, base, w, color="tab:gray", label=f"baseline (median {np.median(base):.4f})")
    a1.bar(x + w/2, staged, w, color="tab:green", label=f"staged (median {np.median(staged):.4f})")
    a1.set_xticks(x); a1.set_xticklabels([f"seed {s}" for s in x])
    a1.set_yscale("log"); a1.set_ylabel("RMSE (M=256)")
    a1.set_title("per-seed: cascaded kill vs standard", fontsize=11)
    a1.legend(frameon=False, fontsize=9)
    a2.bar(x - w/2, bsv, w, color="tab:gray", label="baseline Var[y]")
    a2.bar(x + w/2, stv, w, color="tab:green", label="staged Var[y]")
    a2.set_xticks(x); a2.set_xticklabels([f"seed {s}" for s in x])
    a2.set_ylabel("Var[y]"); a2.set_title("per-seed variance", fontsize=11)
    a2.legend(frameon=False, fontsize=9)
    fig.suptitle(f"4x8 teacher, tf={TF_DEEP}: cascaded layer-kill "
                 f"(L1->0.25, L2->0.4, L3->0.5) vs standard  ({ACT}, {N_SEEDS} seeds)",
                 fontsize=12)
    fig.tight_layout(rect=(0,0,1,0.95))
    out = os.path.join(GDIR, "fig_staged_deep.png")
    fig.savefig(out, dpi=150); print(f"saved {out}")

    # --- trajectory of the best staged computer ------------------------------
    from plots.plot_trajectories_gd import plot_trajectories
    student, teacher, _ = best
    plot_trajectories(student, teacher,
                      out_path=os.path.join(GDIR, "fig_traj_staged_deep.png"),
                      suptitle=f"4x8 staged (cascade kill), tf={TF_DEEP} "
                               f"-- best (RMSE {best[2]:.4f})", tf=TF_DEEP)

    print("\n=== summary (RMSE over seeds) ===")
    print(f"  staged   : median {np.median(staged):.4f}  best {staged.min():.4f}  worst {staged.max():.4f}")
    print(f"  baseline : median {np.median(base):.4f}  best {base.min():.4f}  worst {base.max():.4f}")
    print(f"  staged wins per-seed: {int((staged < base).sum())}/{N_SEEDS}")


if __name__ == "__main__":
    main()
