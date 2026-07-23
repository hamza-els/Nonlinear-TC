"""Local, lightweight version of the MNIST tf scan: fewer seeds, tf <= 1.0, so
it finishes in ~30 min on one GPU (for when the cluster queue is backed up).

Same locked config (thermo teacher + proportional k=100), but 2 seeds and no
tf > 1.0.  Writes to its own npz/figure so it never collides with the full
cluster run (tf_scan_mnist_results.npz).

Usage:
    python run_local_tf_scan.py            # scan (+figure)
    python run_local_tf_scan.py plot       # figure from the local npz
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import experiment_tf_scan_mnist as m

# --- local overrides -------------------------------------------------------
m.TF_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
m.SEEDS = [0, 1]
m.NPZ = os.path.join(_HERE, "runs", "tf_scan_mnist_local.npz")
FIG = os.path.join(_HERE, "..", "..", "Graphs", "computer_graphs",
                   "fig_tf_scan_mnist_local.png")


def plot_local():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(m.NPZ)
    tf, acc = d["tf"], d["acc"]
    ns = acc.shape[1]
    med = np.nanmedian(acc, 1); best = np.nanmax(acc, 1); worst = np.nanmin(acc, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for si in range(ns):
        a1.plot(tf, acc[:, si], "-", lw=0.7, alpha=0.35, color="tab:purple")
    a1.fill_between(tf, worst, best, color="tab:purple", alpha=0.12)
    a1.plot(tf, med, "-o", color="tab:purple", ms=4, label="median")
    bi = int(np.nanargmax(med)); a1.axvline(tf[bi], color="0.5", ls=":", lw=0.9)
    a1.set_xlabel(r"observation time $t_f$"); a1.set_ylabel("top-1 test accuracy")
    a1.set_title(f"accuracy vs $t_f$ (optimum {tf[bi]:g}, median {med[bi]:.4f})",
                 fontsize=11)
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.2)
    for si in range(ns):
        a2.plot(tf, 1 - acc[:, si], "-", lw=0.7, alpha=0.35, color="tab:red")
    a2.plot(tf, 1 - med, "-o", color="tab:red", ms=4, label="median error")
    a2.set_yscale("log"); a2.set_xlabel(r"observation time $t_f$")
    a2.set_ylabel("test error (1 - acc)"); a2.set_title("error vs $t_f$", fontsize=11)
    a2.legend(frameon=False, fontsize=9); a2.grid(alpha=0.2, which="both")
    ta = float(d["teacher_acc"]) if "teacher_acc" in d.files else 0.0
    fig.suptitle(f"MNIST thermo computer (LOCAL: {ns} seeds, $t_f$<=1.0): thermo "
                 f"teacher {ta:.3f}, proportional k={d['k_hidden']:g}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150); print(f"saved {FIG}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_local()
    else:
        m.scan()
        try:
            plot_local()
        except Exception as e:
            print(f"figure skipped ({type(e).__name__}: {e})")
