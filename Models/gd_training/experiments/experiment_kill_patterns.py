"""Which kill-timing PATTERN works best for the cascaded staged 4x8 computer?

Three kills disconnect the first three hidden layers of the 4x8 teacher
(layers [0,8) [8,16) [16,24); survivors [24,32) + output).  Their placement
carves the run into four windows W0..W3 (before k1, k1->k2, k2->k3, k3->tf).
This sweeps the *shape* of that window schedule, over a couple of clock times
and a few seeds, against a standard (non-staged) baseline trained at the same
tf.

Patterns (window fractions of tf, summing to 1; kills at the cumulative
positions of the first three):
  equal              (0.25,0.25,0.25,0.25) -- kills evenly spaced.
  long_tight_long    (0.375,0.125,0.125,0.375) -- long opening + long final
                     window, the two middle kills bunched.
  tight_longer_long  (0.1,0.2,0.3,0.4) -- quick first kill, then progressively
                     more time, longest final window.

Default activation thermo.  For each (tf, seed): one baseline + one staged
student per pattern, all from the same teacher.

Usage:
    python experiments/experiment_kill_patterns.py          # run (+figure)
    python experiments/experiment_kill_patterns.py plot     # figure from npz
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
from staged_student import StagedStudent
from train_gd import student_targets, train_student, evaluate

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
NPZ = os.path.join(_GD_ROOT, "runs", "kill_patterns_results.npz")

PATTERNS = {
    "equal":             (0.25, 0.25, 0.25, 0.25),
    "long_tight_long":   (0.375, 0.125, 0.125, 0.375),
    "tight_longer_long": (0.1, 0.2, 0.3, 0.4),
}
PNAMES = list(PATTERNS)
TFS = [0.4, 0.6, 0.8]
SEEDS = [0, 1, 2, 3]
K_TRAIN = 250
M_EVAL = 256
# first three hidden layers of the 4x8 teacher get killed, in order
LAYER_RANGES = [(0, 8), (8, 16), (16, 24)]


def schedule(windows, tf):
    """(w0,w1,w2,w3) window fractions -> kill_schedule at cumulative times."""
    cum = np.cumsum(windows)[:3] * tf         # kill times for the 3 layers
    return [(float(t), lo, hi) for t, (lo, hi) in zip(cum, LAYER_RANGES)]


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_activation("thermo")
    assert (WIDTH, DEPTH) == (8, 4), f"expected 4x8 teacher, got {WIDTH}x{DEPTH}"
    t0 = time.time()
    print(f"config: activation={ACTIVATION} patterns={PNAMES} tfs={TFS} "
          f"seeds={SEEDS} device={device}", flush=True)

    npat, ntf, ns = len(PATTERNS), len(TFS), len(SEEDS)
    rmse = np.full((npat, ntf, ns), np.nan)      # staged
    var = np.full((npat, ntf, ns), np.nan)
    b_rmse = np.full((ntf, ns), np.nan)          # baseline (per tf, seed)
    b_var = np.full((ntf, ns), np.nan)

    for si, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        teacher = train_teacher(epochs=3000, K=256, seed=seed, device=device,
                                verbose=False)
        z = torch.linspace(0.0, 1.0, K_TRAIN, device=device)
        A = student_targets(teacher, z)
        for ti, tf in enumerate(TFS):
            # baseline (no staging) once per (tf, seed)
            sb, _ = train_student(A, z, tf=tf, seed=seed, device=device,
                                  verbose=False)
            sb.fit_readout(z, target(z), M=M_EVAL, seed=seed + 1, tf=tf)
            eb = evaluate(sb, M=M_EVAL, seed=seed, device=device, tf=tf)
            b_rmse[ti, si] = eb["rmse"]; b_var[ti, si] = eb["var"]
            print(f"    baseline tf={tf:<4g} seed={seed} rmse={eb['rmse']:.4f}",
                  flush=True)
            for pi, pname in enumerate(PNAMES):
                sched = schedule(PATTERNS[pname], tf)
                ss, _ = train_student(A, z, tf=tf, seed=seed, device=device,
                                      verbose=False, student_cls=StagedStudent,
                                      student_kw={"kill_schedule": sched})
                ss.fit_readout(z, target(z), M=M_EVAL, seed=seed + 1, tf=tf)
                ev = evaluate(ss, M=M_EVAL, seed=seed, device=device, tf=tf)
                rmse[pi, ti, si] = ev["rmse"]; var[pi, ti, si] = ev["var"]
                kt = ", ".join(f"{t:.2f}" for t, _, _ in sched)
                print(f">>> {pname:17s} tf={tf:<4g} seed={seed} "
                      f"rmse={ev['rmse']:.4f} var={ev['var']:.3f} "
                      f"kills@[{kt}]  ({time.time()-t0:.0f}s)", flush=True)
        os.makedirs(os.path.dirname(NPZ), exist_ok=True)
        np.savez(NPZ, patterns=np.array(PNAMES), tfs=np.array(TFS),
                 seeds=np.array(SEEDS), rmse=rmse, var=var,
                 b_rmse=b_rmse, b_var=b_var,
                 windows=np.array([PATTERNS[p] for p in PNAMES]))
        print(f"checkpoint: seed {seed} -> {NPZ}", flush=True)

    print("\n=== median RMSE (staged / baseline) per pattern x tf ===")
    for ti, tf in enumerate(TFS):
        bm = np.nanmedian(b_rmse[ti])
        print(f"  tf={tf}:  baseline {bm:.4f}")
        for pi, pn in enumerate(PNAMES):
            pm = np.nanmedian(rmse[pi, ti])
            print(f"     {pn:17s} {pm:.4f}  ({pm/bm:.2f}x baseline)")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(NPZ)
    pats = [str(p) for p in d["patterns"]]; tfs = d["tfs"]
    rmse, b_rmse = d["rmse"], d["b_rmse"]
    npat, ntf, ns = rmse.shape

    fig, axes = plt.subplots(1, ntf, figsize=(6.2 * ntf, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(npat)
    for ti, ax in enumerate(axes):
        med = np.nanmedian(rmse[:, ti], axis=1)
        ax.bar(x, med, 0.6, color="tab:green", alpha=0.8, label="staged median")
        for pi in range(npat):                     # per-seed dots
            ax.scatter([pi]*ns, rmse[pi, ti], s=18, color="0.25", zorder=3)
        bm = np.nanmedian(b_rmse[ti])
        ax.axhline(bm, color="tab:red", ls="--", lw=1.2,
                   label=f"baseline median {bm:.4f}")
        ax.set_yscale("log"); ax.set_xticks(x)
        ax.set_xticklabels(pats, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"$t_f$ = {tfs[ti]:g}", fontsize=11)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("RMSE (M=256, K=250)")
    fig.suptitle("Kill-timing patterns for the cascaded staged 4x8 computer "
                 f"(thermo, {ns} seeds; dots = seeds, bar = median)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(GDIR, "fig_kill_patterns.png")
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
