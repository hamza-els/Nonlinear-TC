"""phi-vs-observation-time grid for EVEN-KILL STAGED 4x8 computers, per clock.

Same idea as experiment_thermo_tf.py (one panel per trained clock time tf, one
translucent phi(t_obs) curve per seed, dotted line at the trained tf), but every
student here is a cascaded STAGED 4x8 computer with EVENLY spaced kills: the
first three hidden layers ([0,8) [8,16) [16,24)) are disconnected at
0.25 tf, 0.50 tf, 0.75 tf and the output is read at tf.

Sweeps tf from 0.5 to 3.0 in steps of 0.05 (51 clocks), 10 seeds each -> 510
staged students.  For each we train at tf, record RMSE / Var at the trained
clock, then sweep the observation time and store phi(t_obs) = mean_z (y0 - y)^2.
Because the kills fire at ABSOLUTE times (0.25 tf ...), reading out earlier than
tf means fewer kills have fired -- exactly the "read the same computer at a
different clock" notch the phi-vs-tf panel shows.

Requires digital_net at WIDTH=8, DEPTH=4 (export TC_WIDTH=8 TC_DEPTH=4).
Thermo activation (the global default).  Checkpoints per seed.

Usage:
    python experiments/experiment_staged_tf_grid.py          # scan (+figure)
    python experiments/experiment_staged_tf_grid.py plot     # figure from npz
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import (target, train_teacher, set_activation, TARGET_FREQ,
                         WIDTH, DEPTH)
from staged_student import StagedStudent
from train_gd import student_targets, train_student, evaluate

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs", "tf_graphs")
NPZ = os.path.join(_GD_ROOT, "runs", "staged_tf_grid_results.npz")

SEEDS = list(range(10))
TF_LIST = [round(0.5 + 0.05 * k, 2) for k in range(51)]   # 0.5 .. 3.0
# the three earlier hidden layers of the 4x8 teacher, killed in order
LAYER_RANGES = [(0, 8), (8, 16), (16, 24)]
EVEN = (0.25, 0.25, 0.25, 0.25)              # evenly spaced kill windows
# observation-time grid for the phi sweep (shared across panels; each panel
# marks its own trained tf with a dotted line)
SWEEP = np.round(np.linspace(0.05, 3.5, 36), 4)
K_TRAIN = 250
M_EVAL = 256
M_SWEEP = 96


def schedule(windows, tf):
    """(w0..w3) window fractions -> kill_schedule at cumulative absolute times."""
    cum = np.cumsum(windows)[:3] * tf
    return [(float(t), lo, hi) for t, (lo, hi) in zip(cum, LAYER_RANGES)]


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_activation("thermo")
    assert (WIDTH, DEPTH) == (8, 4), f"expected 4x8 teacher, got {WIDTH}x{DEPTH}"
    t0 = time.time()
    print(f"config: activation=thermo staged=even-kill WIDTH={WIDTH} "
          f"DEPTH={DEPTH} TARGET_FREQ={TARGET_FREQ} seeds={len(SEEDS)} "
          f"tfs={len(TF_LIST)} ({TF_LIST[0]}..{TF_LIST[-1]}) "
          f"sweep_pts={len(SWEEP)} M_eval={M_EVAL} M_sweep={M_SWEEP} "
          f"device={device}  -> {len(TF_LIST)*len(SEEDS)} students", flush=True)

    nt, ns = len(TF_LIST), len(SEEDS)
    rmse = np.full((nt, ns), np.nan)
    var = np.full((nt, ns), np.nan)
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
        print(f"  teacher seed {seed}: fit mse {tmse:.2e} "
              f"({time.time()-t0:.0f}s)", flush=True)

        for ti, tf in enumerate(TF_LIST):
            sched = schedule(EVEN, tf)
            student, _ = train_student(A, z_train, tf=tf, seed=seed,
                                       device=device, verbose=False,
                                       student_cls=StagedStudent,
                                       student_kw={"kill_schedule": sched})
            student.fit_readout(z_train, target(z_train), M=M_EVAL,
                                seed=seed + 1, tf=tf)
            ev = evaluate(student, M=M_EVAL, seed=seed, device=device, tf=tf)
            rmse[ti, si] = ev["rmse"]
            var[ti, si] = ev["var"]
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
                 sweep=SWEEP, rmse=rmse, var=var, phi=phi,
                 target_freq=TARGET_FREQ, activation="thermo",
                 kill="even", windows=np.array(EVEN))
        print(f"checkpoint: seed {seed} -> {NPZ}  ({time.time()-t0:.0f}s)",
              flush=True)

    print("\n=== summary (rmse per tf: best / median / worst) ===")
    for ti, tf in enumerate(TF_LIST):
        print(f"  tf={tf:<4g} best {np.nanmin(rmse[ti]):.4f}  "
              f"median {np.nanmedian(rmse[ti]):.4f}  "
              f"worst {np.nanmax(rmse[ti]):.4f}  |  median var "
              f"{np.nanmedian(var[ti]):.3f}")
    best_ti = int(np.nanargmin(np.nanmedian(rmse, axis=1)))
    print(f"optimal tf (median RMSE): {TF_LIST[best_ti]}  "
          f"({np.nanmedian(rmse[best_ti]):.4f})")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(NPZ)
    tf_list, sweep, rmse, phi = d["tf"], d["sweep"], d["rmse"], d["phi"]
    nt = len(tf_list)
    ncols = 6
    nrows = int(np.ceil(nt / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ti in range(nt):
        ax = axes[ti]
        for si in range(phi.shape[1]):
            if np.all(np.isnan(phi[ti, si])):
                continue
            ax.plot(sweep, phi[ti, si], "-", color="tab:green", lw=1.0,
                    alpha=0.5)
        ax.axvline(tf_list[ti], color="0.4", ls=":", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlim(0.0, sweep.max())
        ax.set_title(f"$t_f$={tf_list[ti]:g}  best "
                     f"{np.nanmin(rmse[ti]):.3f} / med "
                     f"{np.nanmedian(rmse[ti]):.3f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for k in range(nt, len(axes)):        # blank the unused slots
        axes[k].axis("off")
    for k in range(nt - ncols, nt):       # x-labels on the bottom row of data
        if 0 <= k < nt:
            axes[k].set_xlabel("observation time", fontsize=8)
    for r in range(nrows):
        axes[r * ncols].set_ylabel(r"$\phi$", fontsize=9)
    fig.suptitle("even-kill staged 4x8 computer, target $\\cos(2\\pi z)$: "
                 f"$\\phi$ vs observation time, {phi.shape[1]} seeds per clock "
                 "(kills @ 0.25/0.5/0.75 $t_f$; dotted = trained $t_f$)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(GDIR, exist_ok=True)
    out = os.path.join(GDIR, "fig_staged_tf_grid.png")
    fig.savefig(out, dpi=140)
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
