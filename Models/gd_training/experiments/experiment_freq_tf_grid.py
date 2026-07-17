"""Target frequency x clock time: a 5 x 5 grid, 5 seeds per cell.

Rows    -- target functions cos(f * pi * z) for f in {1, 2, 3, 4, 5}
           (1 = half a period on z in [0,1] ... 5 = two and a half periods;
           higher f is a harder function for the finite-time dynamics).
Columns -- trained clock times tf in {0.4, 0.8, 1.2, 1.6, 2.0}.
Cells   -- 5 seeds each: a fresh teacher + student per (f, tf, seed), with
           phi swept over observation times at frozen parameters.

Architecture is held fixed throughout (6 polynomial input channels, 64 hidden
+ 8 output nodes, all-to-all hidden/hidden-output/output-output, beta=10,
exact-inverse guide biases).  125 computers in all.

Cluster-ready: results checkpoint to runs/freq_tf_grid_results.npz after every
(frequency, seed); the figure is drawn last with a lazily imported matplotlib
and is optional.  To (re)draw the figure locally from a saved npz:

    python experiments/experiment_freq_tf_grid.py plot

Usage:
    python experiments/experiment_freq_tf_grid.py          # scan (+figure)
    python experiments/experiment_freq_tf_grid.py plot     # figure only
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

import digital_net
from digital_net import target, train_teacher, set_target_freq
from train_gd import student_targets, train_student, evaluate, save_student

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
NPZ = os.path.join(_GD_ROOT, "runs", "freq_tf_grid_results.npz")

FREQS = [1.0, 2.0, 3.0, 4.0, 5.0]          # rows: cos(f * pi * z)
TF_LIST = [0.4, 0.8, 1.2, 1.6, 2.0]        # columns: trained clock times
SEEDS = [0, 1, 2, 3, 4]
# shared observation-time grid (linear), augmented with every trained clock
SWEEP = np.unique(np.round(np.concatenate(
    [np.linspace(0.05, 5.0, 30), np.array(TF_LIST)]), 4))
K_TRAIN = 250
M_EVAL = 256      # standard reporting protocol
M_SWEEP = 128     # samples per sweep point


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"config: freqs={FREQS} tfs={TF_LIST} seeds={SEEDS} "
          f"sweep_pts={len(SWEEP)} K={K_TRAIN} M_eval={M_EVAL} "
          f"M_sweep={M_SWEEP} device={device} "
          f"({len(FREQS)*len(TF_LIST)*len(SEEDS)} computers)", flush=True)

    nf, nt, ns = len(FREQS), len(TF_LIST), len(SEEDS)
    rmse = np.full((nf, nt, ns), np.nan)
    var = np.full((nf, nt, ns), np.nan)
    phi = np.full((nf, nt, ns, len(SWEEP)), np.nan)
    tmse = np.full((nf, ns), np.nan)

    for fi, freq in enumerate(FREQS):
        set_target_freq(freq)               # switches the task globally
        for si, seed in enumerate(SEEDS):
            # teacher depends on (freq, seed) only -- train once, reuse per tf
            torch.manual_seed(seed)
            teacher = train_teacher(epochs=3000, K=256, seed=seed,
                                    device=device, verbose=False)
            z_train = torch.linspace(0.0, 1.0, K_TRAIN, device=device)
            A = student_targets(teacher, z_train)
            z_eval = torch.linspace(0.0, 1.0, 250, device=device)
            y0 = target(z_eval)
            tmse[fi, si] = torch.mean((teacher(z_eval) - y0) ** 2).item()
            print(f"  teacher f={freq:g} seed={seed}: fit mse "
                  f"{tmse[fi, si]:.2e}", flush=True)

            for ti, tf in enumerate(TF_LIST):
                student, _ = train_student(A, z_train, tf=tf, seed=seed,
                                           device=device, verbose=False)
                student.fit_readout(z_train, target(z_train), M=M_EVAL,
                                    seed=seed + 1, tf=tf)
                ev = evaluate(student, M=M_EVAL, seed=seed, device=device,
                              tf=tf)
                rmse[fi, ti, si] = ev["rmse"]
                var[fi, ti, si] = ev["var"]
                save_student(student, f"freqgrid_f{freq:g}_tf{tf:g}_seed{seed}",
                             meta={"experiment": "freq_tf_grid",
                                   "target_freq": freq, "tf": tf, "seed": seed,
                                   "K": K_TRAIN, "rmse": ev["rmse"],
                                   "bias2": ev["bias2"], "var": ev["var"]})
                with torch.no_grad():
                    for j, t_obs in enumerate(SWEEP):
                        pred = student.sample_outputs(
                            z_eval, M=M_SWEEP, tf=float(t_obs),
                            seed=0).mean(dim=1)
                        phi[fi, ti, si, j] = torch.mean(
                            (pred - y0) ** 2).item()
                print(f">>> f={freq:g} tf={tf:<4g} seed={seed}  "
                      f"rmse={ev['rmse']:.4f}  var={ev['var']:.3f}  "
                      f"phi_min={np.nanmin(phi[fi, ti, si]):.2e}  "
                      f"({time.time()-t0:.0f}s)", flush=True)

            # checkpoint after every (frequency, seed)
            os.makedirs(os.path.dirname(NPZ), exist_ok=True)
            np.savez(NPZ, freqs=np.array(FREQS), tf=np.array(TF_LIST),
                     seeds=np.array(SEEDS), sweep=SWEEP, rmse=rmse, var=var,
                     phi=phi, teacher_mse=tmse)
            print(f"checkpoint: f={freq:g} seed={seed} -> {NPZ}", flush=True)

    print("\n=== best RMSE per (frequency, clock) ===")
    header = "        " + "".join(f"tf={t:<8g}" for t in TF_LIST)
    print(header)
    for fi, freq in enumerate(FREQS):
        cells = "".join(f"{np.nanmin(rmse[fi, ti]):<11.4f}"
                        for ti in range(nt))
        print(f"cos{freq:g}pi  {cells}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(NPZ)
    freqs, tf_list, sweep = d["freqs"], d["tf"], d["sweep"]
    rmse, phi = d["rmse"], d["phi"]
    nf, nt = len(freqs), len(tf_list)

    # sharey per row: each target frequency has its own phi scale
    fig, axes = plt.subplots(nf, nt, figsize=(3.6 * nt, 2.9 * nf),
                             sharex=True, sharey="row")
    axes = np.atleast_2d(axes)
    for fi in range(nf):
        for ti in range(nt):
            ax = axes[fi, ti]
            for si in range(phi.shape[2]):
                if np.all(np.isnan(phi[fi, ti, si])):
                    continue
                ax.plot(sweep, phi[fi, ti, si], "-", color="tab:green",
                        lw=0.9, alpha=0.55)
            ax.axvline(tf_list[ti], color="0.4", ls=":", lw=0.9)
            ax.set_yscale("log")
            ax.set_xlim(0.0, sweep.max())
            best = np.nanmin(rmse[fi, ti])
            med = np.nanmedian(rmse[fi, ti])
            ax.set_title(rf"$t_f$={tf_list[ti]:g}   best {best:.3f} / "
                         rf"med {med:.3f}", fontsize=9)
        axes[fi, 0].set_ylabel(rf"$\cos({freqs[fi]:g}\pi z)$" "\n" r"$\phi$",
                               fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("observation time")

    fig.suptitle(r"$\phi$ vs observation time: target frequency (rows) $\times$ "
                 r"trained clock $t_f$ (columns), "
                 f"{phi.shape[2]} seeds per cell "
                 r"(dotted = trained $t_f$; rows share a $\phi$ scale)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(GDIR, "fig_freq_tf_grid.png")
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
                  f"'python experiments/experiment_freq_tf_grid.py plot' "
                  f"locally")
