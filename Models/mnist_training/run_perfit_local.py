"""Small local per-tf-refit scan: tf in {0.2,0.4,0.6,0.8,1.0}, 1 seed each.
Each tf refits the thermo activation to the real neuron at beta=1,tf, retrains
the teacher, then trains one student.  ~15 min on one GPU.  Distinct outputs."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import experiment_tf_scan_mnist_perfit as m

m.TF_LIST = [0.2, 0.4, 0.6, 0.8, 1.0]
m.SEEDS = [0]
m.NPZ = os.path.join(_HERE, "runs", "tf_scan_mnist_perfit_local.npz")
FIG = os.path.join(_HERE, "..", "..", "Graphs", "computer_graphs",
                   "fig_tf_scan_mnist_perfit_local.png")


def plot_local():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(m.NPZ)
    tf, acc = d["tf"], d["acc"][:, 0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(tf, acc, "-o", color="teal", ms=5, label="per-tf refit")
    # overlay the fixed-teacher local scan if present, for comparison
    fx = os.path.join(_HERE, "runs", "tf_scan_mnist_local.npz")
    if os.path.exists(fx):
        e = np.load(fx)
        a1.plot(e["tf"], np.nanmedian(e["acc"], 1), "-s", color="tab:purple",
                ms=4, alpha=0.7, label="fixed teacher (ACT_C=13.53)")
    a1.set_xlabel(r"observation time $t_f$"); a1.set_ylabel("top-1 test accuracy")
    a1.set_title("accuracy vs $t_f$", fontsize=11)
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.2)
    a2.plot(tf, d["act_a"], "-o", color="tab:orange", ms=5, label="ACT_A")
    a2b = a2.twinx(); a2b.plot(tf, d["act_c"], "-s", color="tab:blue", ms=5)
    a2.set_xlabel(r"observation time $t_f$")
    a2.set_ylabel("ACT_A", color="tab:orange")
    a2b.set_ylabel("ACT_C", color="tab:blue")
    a2.set_title("refit constants vs $t_f$", fontsize=11); a2.grid(alpha=0.2)
    fig.suptitle("MNIST thermo computer: PER-TF REFIT teacher (LOCAL, 1 seed, "
                 "beta=1, proportional k=100)", fontsize=12)
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
