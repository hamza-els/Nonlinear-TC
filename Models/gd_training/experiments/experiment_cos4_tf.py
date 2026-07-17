"""Harder target: cos(4 pi z) at three clock times, 5 seeds each.

Same architecture as the cos(2 pi z) work (6 polynomial input channels, 64
hidden + 8 output nodes, all-to-all hidden/hidden-output/output-output,
beta=10), but the target has TWO periods on z in [0, 1] instead of one --
a harder function for the computer's finite-time dynamics to express.

For each tf in {0.5, 1.0, 1.5} and seed in 0..4 we train a fresh computer,
record its RMSE at its own clock (M=256, K=250), and sweep phi over
observation times with parameters frozen.  The figure shows the three clocks
side by side, one translucent curve per seed, dotted line at the trained tf.

Requires digital_net.TARGET_FREQ = 4.0.

Usage:
    python experiments/experiment_cos4_tf.py          # scan (+figure)
    python experiments/experiment_cos4_tf.py plot     # figure from saved npz
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import target, train_teacher, TARGET_FREQ
from train_gd import student_targets, train_student, evaluate, save_student

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
NPZ = os.path.join(_GD_ROOT, "runs", "cos4_tf_results.npz")

SEEDS = [0, 1, 2, 3, 4]
TF_LIST = [0.5, 1.0, 1.5]
SWEEP = np.unique(np.round(np.concatenate(
    [np.linspace(0.05, 4.0, 30), np.array(TF_LIST)]), 4))
K_TRAIN = 250
M_EVAL = 256
M_SWEEP = 128


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"config: TARGET_FREQ={TARGET_FREQ} seeds={SEEDS} tfs={TF_LIST} "
          f"sweep_pts={len(SWEEP)} K={K_TRAIN} M_eval={M_EVAL} "
          f"M_sweep={M_SWEEP} device={device}", flush=True)

    nt, ns = len(TF_LIST), len(SEEDS)
    rmse = np.full((nt, ns), np.nan)
    phi = np.full((nt, ns, len(SWEEP)), np.nan)

    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        teacher = train_teacher(epochs=3000, K=256, seed=seed, device=device,
                                verbose=False)
        z_train = torch.linspace(0.0, 1.0, K_TRAIN, device=device)
        A = student_targets(teacher, z_train)
        z_eval = torch.linspace(0.0, 1.0, 250, device=device)
        y0 = target(z_eval)
        tmse = torch.mean((teacher(z_eval) - y0) ** 2).item()
        print(f"  teacher seed {seed}: fit mse {tmse:.2e}", flush=True)

        for ti, tf in enumerate(TF_LIST):
            student, _ = train_student(A, z_train, tf=tf, seed=seed,
                                       device=device, verbose=False)
            student.fit_readout(z_train, target(z_train), M=M_EVAL,
                                seed=seed + 1, tf=tf)
            ev = evaluate(student, M=M_EVAL, seed=seed, device=device, tf=tf)
            rmse[ti, si] = ev["rmse"]
            save_student(student, f"cos4_tf{tf:g}_seed{seed}",
                         meta={"experiment": "cos4_tf", "target_freq":
                               TARGET_FREQ, "tf": tf, "seed": seed,
                               "K": K_TRAIN, "rmse": ev["rmse"],
                               "bias2": ev["bias2"], "var": ev["var"]})
            with torch.no_grad():
                for j, t_obs in enumerate(SWEEP):
                    pred = student.sample_outputs(
                        z_eval, M=M_SWEEP, tf=float(t_obs), seed=0).mean(dim=1)
                    phi[ti, si, j] = torch.mean((pred - y0) ** 2).item()
            print(f">>> tf={tf:<4g} seed={seed}  rmse={ev['rmse']:.4f}  "
                  f"var={ev['var']:.3f}  phi_min={np.nanmin(phi[ti, si]):.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

        os.makedirs(os.path.dirname(NPZ), exist_ok=True)
        np.savez(NPZ, tf=np.array(TF_LIST), seeds=np.array(SEEDS),
                 sweep=SWEEP, rmse=rmse, phi=phi, target_freq=TARGET_FREQ)

    print("\n=== summary (rmse per tf) ===")
    for ti, tf in enumerate(TF_LIST):
        print(f"  tf={tf:<4g} best {np.nanmin(rmse[ti]):.4f}  "
              f"median {np.nanmedian(rmse[ti]):.4f}  "
              f"worst {np.nanmax(rmse[ti]):.4f}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(NPZ)
    tf_list, sweep, rmse, phi = d["tf"], d["sweep"], d["rmse"], d["phi"]
    freq = float(d["target_freq"]) if "target_freq" in d.files else 4.0
    nt = len(tf_list)
    fig, axes = plt.subplots(1, nt, figsize=(5.0 * nt, 4.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ti, ax in enumerate(axes):
        for si in range(phi.shape[1]):
            if np.all(np.isnan(phi[ti, si])):
                continue
            ax.plot(sweep, phi[ti, si], "-", color="tab:green", lw=1.0,
                    alpha=0.55, label=f"seed {si}" if ti == 0 else None)
        ax.axvline(tf_list[ti], color="0.4", ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlim(0.0, sweep.max())
        ax.set_xlabel("observation time")
        ax.set_title(f"$t_f$={tf_list[ti]:g}   best RMSE "
                     f"{np.nanmin(rmse[ti]):.4f}   "
                     f"median {np.nanmedian(rmse[ti]):.4f}", fontsize=10)
    axes[0].set_ylabel(r"$\phi$")
    fig.suptitle(rf"target $\cos({freq:g}\pi z)$: $\phi$ vs observation time, "
                 rf"{phi.shape[1]} seeds per clock (dotted = trained $t_f$)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(GDIR, "fig_cos4_tf_grid.png")
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
            print(f"figure skipped ({type(e).__name__}: {e})")
