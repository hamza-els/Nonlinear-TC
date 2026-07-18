"""Optimal observation time for the deep 4x8 architecture (standard, non-staged).

Sweeps the trained clock tf in {0.2, 0.3, ..., 1.0}, 5 seeds each, and records
the student's RMSE and Var[y] at its own clock (M=256, K=250).  The teacher
depends only on the seed, so it is trained ONCE per seed and reused across all
tf values -- 5 teachers + 45 students in all.

Requires digital_net at WIDTH=8, DEPTH=4.  Uses the global default
    activation (now thermo; call set_activation("tanh") to override).

Usage:
    python experiments/experiment_tf_sweep_4x8.py          # sweep (+figure)
    python experiments/experiment_tf_sweep_4x8.py plot     # figure from npz
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import (set_activation, train_teacher, target, WIDTH, DEPTH,
                         ACTIVATION)
from train_gd import student_targets, train_student, evaluate

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
NPZ = os.path.join(_GD_ROOT, "runs", "tf_sweep_4x8_results.npz")

TF_LIST = [round(0.2 + 0.1 * k, 1) for k in range(9)]     # 0.2 .. 1.0
SEEDS = [0, 1, 2, 3, 4]
K_TRAIN = 250
M_EVAL = 256


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert (WIDTH, DEPTH) == (8, 4), f"expected 4x8 teacher, got {WIDTH}x{DEPTH}"
    t0 = time.time()
    print(f"config: activation={ACTIVATION} WIDTH={WIDTH} DEPTH={DEPTH} "
          f"tfs={TF_LIST} seeds={SEEDS} K={K_TRAIN} M_eval={M_EVAL} "
          f"device={device} ({len(TF_LIST)*len(SEEDS)} students)", flush=True)

    nt, ns = len(TF_LIST), len(SEEDS)
    rmse = np.full((nt, ns), np.nan)
    var = np.full((nt, ns), np.nan)

    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        teacher = train_teacher(epochs=3000, K=256, seed=seed, device=device,
                                verbose=False)
        z = torch.linspace(0.0, 1.0, K_TRAIN, device=device)
        A = student_targets(teacher, z)
        for ti, tf in enumerate(TF_LIST):
            student, _ = train_student(A, z, tf=tf, seed=seed, device=device,
                                       verbose=False)
            student.fit_readout(z, target(z), M=M_EVAL, seed=seed + 1, tf=tf)
            ev = evaluate(student, M=M_EVAL, seed=seed, device=device, tf=tf)
            rmse[ti, si] = ev["rmse"]; var[ti, si] = ev["var"]
            print(f">>> tf={tf:<4g} seed={seed}  rmse={ev['rmse']:.4f}  "
                  f"var={ev['var']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(NPZ), exist_ok=True)
        np.savez(NPZ, tf=np.array(TF_LIST), seeds=np.array(SEEDS),
                 rmse=rmse, var=var, activation=ACTIVATION)
        print(f"checkpoint: seed {seed} -> {NPZ}", flush=True)

    med = np.nanmedian(rmse, axis=1)
    print("\n=== RMSE per tf (median [best]) ===")
    for ti, tf in enumerate(TF_LIST):
        print(f"  tf={tf:<4g}  median {med[ti]:.4f}  best {np.nanmin(rmse[ti]):.4f}")
    print(f"optimal tf (median): {TF_LIST[int(np.argmin(med))]}  "
          f"(median RMSE {med.min():.4f})")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(NPZ)
    tf, rmse, var = d["tf"], d["rmse"], d["var"]
    ns = rmse.shape[1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for si in range(ns):
        a1.plot(tf, rmse[:, si], "-", lw=0.7, alpha=0.35, color="tab:green")
    a1.plot(tf, np.nanmedian(rmse, 1), "-o", color="tab:green", ms=5,
            label="median")
    a1.plot(tf, np.nanmin(rmse, 1), "--", lw=1.2, color="darkgreen",
            label="best of 5")
    a1.fill_between(tf, np.nanmin(rmse, 1), np.nanmax(rmse, 1),
                    color="tab:green", alpha=0.12)
    a1.set_yscale("log"); a1.set_xlabel(r"trained clock time $t_f$")
    a1.set_ylabel("RMSE (M=256, K=250)")
    best_tf = tf[int(np.nanargmin(np.nanmedian(rmse, 1)))]
    a1.axvline(best_tf, color="0.5", ls=":", lw=0.9)
    a1.set_title(f"accuracy vs $t_f$ (median optimum {best_tf:g})", fontsize=11)
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.2)

    for si in range(ns):
        a2.plot(tf, var[:, si], "-", lw=0.7, alpha=0.35, color="tab:blue")
    a2.plot(tf, np.nanmedian(var, 1), "-o", color="tab:blue", ms=5,
            label="median Var[y]")
    a2.set_xlabel(r"trained clock time $t_f$"); a2.set_ylabel("Var[y]")
    a2.set_title("variance vs $t_f$", fontsize=11)
    a2.legend(frameon=False, fontsize=9); a2.grid(alpha=0.2)

    act = str(d["activation"]) if "activation" in d.files else "tanh"
    fig.suptitle(f"4x8 teacher ({act}), standard architecture: optimal "
                 f"observation time ({ns} seeds per $t_f$)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(GDIR, "fig_tf_sweep_4x8.png")
    fig.savefig(out, dpi=150); print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        scan()
        try:
            plot()
        except Exception as e:
            print(f"figure skipped ({type(e).__name__}: {e})")
