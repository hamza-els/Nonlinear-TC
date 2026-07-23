"""tf scan with a PER-TF REFIT thermo teacher.

Same as experiment_tf_scan_mnist, but the teacher's thermo activation is not
fixed: at each tf we probe the real isolated neuron at beta=1, tf (see
thermo_activation_fit) and refit (ACT_A, ACT_C), retrain the teacher with that
activation, and only then train students at that tf.  So the guide targets are
the finite-time activations a real neuron would actually show when read at that
clock -- not an inherited fixed curve.

Records, per tf: student test accuracy (per seed), the fitted (ACT_A, ACT_C),
and the refit teacher's accuracy.  Guide is proportional k=100 (locked).

Usage:
    python experiment_tf_scan_mnist_perfit.py          # scan (+figure)
    python experiment_tf_scan_mnist_perfit.py plot     # figure from the npz
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from teacher_net import load_mnist, train_teacher
from train_mnist import teacher_activations, accuracy
from thermo_activation_fit import fit_thermo_activation
from experiment_tf_scan_mnist import train_one   # reuses the locked guide config

TF_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
SEEDS = list(range(10))
TEACHER_EPOCHS = 300
M_EVAL = 10

NPZ = os.path.join(_HERE, "runs", "tf_scan_mnist_perfit_results.npz")


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, Xte, yte = load_mnist(device)
    nt, ns = len(TF_LIST), len(SEEDS)
    acc = np.full((nt, ns), np.nan)
    act_a = np.full(nt, np.nan); act_c = np.full(nt, np.nan)
    tacc = np.full(nt, np.nan)
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        if d["acc"].shape == acc.shape:
            acc = d["acc"].copy(); act_a = d["act_a"].copy()
            act_c = d["act_c"].copy(); tacc = d["teacher_acc"].copy()
            print(f"resuming: {int(np.isfinite(acc).sum())}/{acc.size} done",
                  flush=True)
    t0 = time.time()
    print(f"config: PER-TF REFIT teacher (beta=1) | teacher_epochs="
          f"{TEACHER_EPOCHS} guide=proportional k=100 M_eval={M_EVAL} "
          f"tfs={nt} seeds={ns} device={device}", flush=True)

    for ti, tf in enumerate(TF_LIST):
        if np.all(np.isfinite(acc[ti])):
            continue
        # 1) refit the activation to the real neuron at this tf
        a, c, info = fit_thermo_activation(tf, beta=1.0, device=device)
        act_a[ti], act_c[ti] = a, c
        # 2) retrain the teacher with that activation
        teacher, ta = train_teacher(epochs=TEACHER_EPOCHS, activation="thermo",
                                    act_a=a, act_c=c, seed=0, device=device,
                                    verbose=False)
        tacc[ti] = ta
        Ah, Ao = teacher_activations(teacher, Xtr)
        print(f"tf={tf:<5g}: fit ACT_A={a:.4f} ACT_C={c:.3f} (rms {info['rms']:.4f}"
              f") | teacher {ta:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        # 3) train students at this tf
        for si, seed in enumerate(SEEDS):
            if np.isfinite(acc[ti, si]):
                continue
            student = train_one(Ah, Ao, Xtr, tf, seed, device)
            av = accuracy(student, Xte, yte, M=M_EVAL, tf=tf, seed=seed)
            acc[ti, si] = av
            print(f">>> tf={tf:<5g} seed={seed}  acc={av:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            os.makedirs(os.path.dirname(NPZ), exist_ok=True)
            np.savez(NPZ, tf=np.array(TF_LIST), seeds=np.array(SEEDS), acc=acc,
                     act_a=act_a, act_c=act_c, teacher_acc=tacc,
                     teacher_epochs=TEACHER_EPOCHS, m_eval=M_EVAL)

    med = np.nanmedian(acc, axis=1)
    print("\n=== accuracy per tf (refit teacher) ===")
    for ti, tf in enumerate(TF_LIST):
        print(f"  tf={tf:<5g}  acc {med[ti]:.4f}  ACT_A={act_a[ti]:.4f} "
              f"ACT_C={act_c[ti]:.3f}  teacher {tacc[ti]:.4f}")
    bi = int(np.nanargmax(med))
    print(f"optimal tf: {TF_LIST[bi]}  ({med[bi]:.4f})")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(NPZ)
    tf, acc = d["tf"], d["acc"]
    ns = acc.shape[1]
    med = np.nanmedian(acc, 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for si in range(ns):
        a1.plot(tf, acc[:, si], "-", lw=0.7, alpha=0.35, color="teal")
    a1.plot(tf, med, "-o", color="teal", ms=4, label="accuracy")
    a1.set_xlabel(r"observation time $t_f$"); a1.set_ylabel("top-1 test accuracy")
    a1.set_title("accuracy vs $t_f$ (per-tf refit teacher)", fontsize=11)
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.2)
    a2.plot(tf, d["act_a"], "-o", color="tab:orange", ms=4, label="ACT_A")
    a2b = a2.twinx()
    a2b.plot(tf, d["act_c"], "-s", color="tab:blue", ms=4, label="ACT_C")
    a2.set_xlabel(r"observation time $t_f$")
    a2.set_ylabel("ACT_A", color="tab:orange")
    a2b.set_ylabel("ACT_C", color="tab:blue")
    a2.set_title("refit activation constants vs $t_f$", fontsize=11)
    a2.grid(alpha=0.2)
    fig.suptitle("MNIST thermo computer, PER-TF REFIT thermo teacher (beta=1, "
                 f"proportional k=100, {ns} seeds)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(_HERE, "..", "..", "Graphs", "computer_graphs",
                       "fig_tf_scan_mnist_perfit.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
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
