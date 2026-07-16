"""Experiment O: clock-time scan -- 31 trained tf values x 5 seeds.

For each trained observation time tf in {0.05, 0.1, 0.2, ..., 3.0} and each
seed in 0..4 (same architecture throughout) we train a fresh computer, record
its RMSE at its own clock (M=256, standard protocol), and sweep phi over
observation times with parameters frozen.  Panels show all 5 seed curves on a
LINEAR observation-time axis (log phi), dotted line at the trained clock,
title = best RMSE across seeds.

Cluster-ready: results are checkpointed to runs/tf_scan_results.npz after
every seed; the figure is drawn last with a lazily imported matplotlib and is
optional.  To (re)draw the figure locally from a saved npz:

    python experiments/experiment_tf_scan.py plot

Usage:
    python experiments/experiment_tf_scan.py          # full scan (+figure)
    python experiments/experiment_tf_scan.py plot     # figure from saved npz
"""

import os
import sys
import time

# core modules (digital_net, thermo_student, train_gd) live one level up
_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import target, train_teacher
from train_gd import student_targets, train_student, evaluate, save_student

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
NPZ = os.path.join(_GD_ROOT, "runs", "tf_scan_results.npz")

SEEDS = [0, 1, 2, 3, 4]
TF_LIST = [0.05] + [round(0.1 * k, 2) for k in range(1, 31)]   # 0.05, 0.1..3.0
# linear sweep grid, augmented with every trained clock so each panel's notch
# is sampled exactly; shared by all computers (rectangular phi array)
SWEEP = np.unique(np.round(np.concatenate(
    [np.linspace(0.05, 6.0, 25), np.array(TF_LIST)]), 4))
K_TRAIN = 250
M_EVAL = 256      # standard reporting protocol for the per-clock RMSE
M_SWEEP = 128     # samples per sweep point


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"config: seeds={SEEDS} n_tf={len(TF_LIST)} tf_max={TF_LIST[-1]} "
          f"sweep_pts={len(SWEEP)} K={K_TRAIN} M_eval={M_EVAL} "
          f"M_sweep={M_SWEEP} device={device}", flush=True)

    nt, ns = len(TF_LIST), len(SEEDS)
    rmse = np.full((nt, ns), np.nan)
    phi = np.full((nt, ns, len(SWEEP)), np.nan)

    for si, seed in enumerate(SEEDS):
        # teacher depends only on the seed -- train once, reuse for all tf
        torch.manual_seed(seed)
        teacher = train_teacher(epochs=3000, K=256, seed=seed, device=device,
                                verbose=False)
        z_train = torch.linspace(0.0, 1.0, K_TRAIN, device=device)
        A = student_targets(teacher, z_train)
        z_eval = torch.linspace(0.0, 1.0, 250, device=device)
        y0 = target(z_eval)

        for ti, tf in enumerate(TF_LIST):
            student, _ = train_student(A, z_train, tf=tf, seed=seed,
                                       device=device, verbose=False)
            student.fit_readout(z_train, target(z_train), M=M_EVAL,
                                seed=seed + 1, tf=tf)
            ev = evaluate(student, M=M_EVAL, seed=seed, device=device, tf=tf)
            rmse[ti, si] = ev["rmse"]
            save_student(student, f"tfscan_tf{tf:g}_seed{seed}",
                         meta={"experiment": "tf_scan", "tf": tf,
                               "seed": seed, "K": K_TRAIN,
                               "rmse": ev["rmse"], "bias2": ev["bias2"],
                               "var": ev["var"]})
            with torch.no_grad():
                for j, t_obs in enumerate(SWEEP):
                    pred = student.sample_outputs(
                        z_eval, M=M_SWEEP, tf=float(t_obs), seed=0).mean(dim=1)
                    phi[ti, si, j] = torch.mean((pred - y0) ** 2).item()
            print(f">>> tf={tf:<5g} seed={seed}  rmse={ev['rmse']:.3f}  "
                  f"phi_min={np.nanmin(phi[ti, si]):.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # checkpoint after every completed seed (cluster-safe)
        os.makedirs(os.path.dirname(NPZ), exist_ok=True)
        np.savez(NPZ, tf=np.array(TF_LIST), seeds=np.array(SEEDS),
                 sweep=SWEEP, rmse=rmse, phi=phi)
        print(f"checkpoint: seed {seed} done -> {NPZ}", flush=True)

    print("\n=== best clock times (best across seeds) ===")
    best = np.nanmin(rmse, axis=1)
    for ti in np.argsort(best)[:5]:
        print(f"  tf={TF_LIST[ti]:g}  best rmse={best[ti]:.3f}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(NPZ)
    tf_list, sweep, rmse, phi = d["tf"], d["sweep"], d["rmse"], d["phi"]
    nt = len(tf_list)
    ncols, nrows = 7, int(np.ceil(nt / 7))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows),
                             sharex=True, sharey=True)
    for ax in axes.flat[nt:]:
        ax.set_visible(False)
    for ti, ax in enumerate(axes.flat[:nt]):
        for si in range(phi.shape[1]):
            if np.all(np.isnan(phi[ti, si])):
                continue
            ax.plot(sweep, phi[ti, si], "-", color="tab:green",
                    lw=0.9, alpha=0.5)
        ax.axvline(tf_list[ti], color="0.4", ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_xlim(0.0, sweep.max())
        ax.set_title(f"$t_f$={tf_list[ti]:g}   "
                     f"best {np.nanmin(rmse[ti]):.3f}", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("observation time")
    for row in axes:
        row[0].set_ylabel(r"$\phi$")
    n_seeds = phi.shape[1]
    fig.suptitle(f"phi vs observation time (linear axis), {nt} trained clock "
                 f"times x {n_seeds} seeds (dotted line = trained $t_f$; "
                 f"title = best RMSE across seeds)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(GDIR, "fig_tf_scan_grid.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        scan()
        try:
            plot()
        except Exception as e:
            print(f"figure skipped ({type(e).__name__}: {e}); run "
                  f"'python experiments/experiment_tf_scan.py plot' locally")
