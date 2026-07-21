"""tf scan for the MNIST thermodynamic computer: test accuracy vs observation
time, training a fresh student AT each tf, 10 seeds each.

Locked configuration (best of the teacher x guide comparison, 93.8% at tf=0.2):
    - teacher: THERMO-activation 784-32-32-10 (teacher_thermo.pt, 97.06%)
    - guide:   PROPORTIONAL, k_hidden = k_output = 100 (paper form; the thermo
               teacher's naturally-large activations put this in the +/-4 regime)
    - student: the 74-node all-to-all ThermoMNIST, OM-GD from theta = 0, kT = 1.

For each tf in TF_LIST and seed in SEEDS a student is trained (EPOCHS epochs)
with observation time = tf, then its top-1 test accuracy is measured (M_EVAL
reset samples per digit, full 10k).  The MNIST analog of the cosine phi-vs-tf
scans: it locates the observation time that best classifies.

Checkpoints after every (tf, seed) to runs/tf_scan_mnist_results.npz and RESUMES
(skips finished cells), so the run survives a wall limit.

Usage:
    python experiment_tf_scan_mnist.py          # scan (+figure)
    python experiment_tf_scan_mnist.py plot     # figure from the npz
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from teacher_net import load_mnist, load_teacher
from thermo_mnist import ThermoMNIST, guide_trajectory
from train_mnist import teacher_activations, accuracy

TF_LIST = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5]
SEEDS = list(range(10))
EPOCHS = 15
BATCH = 256
LR = 1e-2
GUIDE_MODE = "proportional"
K_HIDDEN = 100.0
K_OUTPUT = 100.0
TEACHER_ACT = "thermo"
M_EVAL = 10

NPZ = os.path.join(_HERE, "runs", "tf_scan_mnist_results.npz")


def train_one(Ah, Ao, Xtr, tf, seed, device):
    """Train one ThermoMNIST from theta=0 at observation time tf."""
    torch.manual_seed(seed)
    student = ThermoMNIST(scale=0.0).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    ntr = Xtr.shape[0]
    for _ in range(EPOCHS):
        perm = torch.randperm(ntr, device=device)
        for i in range(0, ntr, BATCH):
            idx = perm[i:i + BATCH]
            traj = guide_trajectory(Ah[idx], Ao[idx], tf=tf, guide_mode=GUIDE_MODE,
                                    k_hidden=K_HIDDEN, k_output=K_OUTPUT)
            loss = student.om_loss(traj, Xtr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return student


def scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr, Xte, yte = load_mnist(device)
    teacher, tacc = load_teacher(activation=TEACHER_ACT, device=device)
    Ah, Ao = teacher_activations(teacher, Xtr)
    nt, ns = len(TF_LIST), len(SEEDS)
    acc = np.full((nt, ns), np.nan)
    if os.path.exists(NPZ):                       # resume
        d = np.load(NPZ)
        if d["acc"].shape == acc.shape:
            acc = d["acc"].copy()
            print(f"resuming: {int(np.isfinite(acc).sum())}/{acc.size} cells done",
                  flush=True)
    t0 = time.time()
    print(f"config: teacher={TEACHER_ACT}({tacc:.4f}) guide={GUIDE_MODE} "
          f"k={K_HIDDEN} epochs={EPOCHS} M_eval={M_EVAL} tfs={len(TF_LIST)} "
          f"seeds={ns} device={device}", flush=True)

    for ti, tf in enumerate(TF_LIST):
        for si, seed in enumerate(SEEDS):
            if np.isfinite(acc[ti, si]):
                continue
            student = train_one(Ah, Ao, Xtr, tf, seed, device)
            a = accuracy(student, Xte, yte, M=M_EVAL, tf=tf, seed=seed)
            acc[ti, si] = a
            print(f">>> tf={tf:<5g} seed={seed}  acc={a:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            os.makedirs(os.path.dirname(NPZ), exist_ok=True)
            np.savez(NPZ, tf=np.array(TF_LIST), seeds=np.array(SEEDS), acc=acc,
                     teacher_acc=tacc, guide_mode=GUIDE_MODE, k_hidden=K_HIDDEN,
                     k_output=K_OUTPUT, epochs=EPOCHS, m_eval=M_EVAL)

    med = np.nanmedian(acc, axis=1)
    print("\n=== test accuracy per tf (median [best]) ===")
    for ti, tf in enumerate(TF_LIST):
        print(f"  tf={tf:<5g}  median {med[ti]:.4f}  best {np.nanmax(acc[ti]):.4f}")
    bi = int(np.nanargmax(med))
    print(f"optimal tf (median acc): {TF_LIST[bi]}  ({med[bi]:.4f})")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(NPZ)
    tf, acc = d["tf"], d["acc"]
    ns = acc.shape[1]
    med = np.nanmedian(acc, 1); best = np.nanmax(acc, 1); worst = np.nanmin(acc, 1)
    err = 1.0 - acc

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for si in range(ns):
        a1.plot(tf, acc[:, si], "-", lw=0.6, alpha=0.3, color="tab:purple")
    a1.fill_between(tf, worst, best, color="tab:purple", alpha=0.12)
    a1.plot(tf, med, "-o", color="tab:purple", ms=4, label="median")
    a1.plot(tf, best, "--", lw=1.1, color="indigo", label=f"best of {ns}")
    bi = int(np.nanargmax(med)); a1.axvline(tf[bi], color="0.5", ls=":", lw=0.9)
    a1.set_xlabel(r"observation time $t_f$"); a1.set_ylabel("top-1 test accuracy")
    a1.set_title(f"accuracy vs $t_f$ (optimum {tf[bi]:g}, median {med[bi]:.4f})",
                 fontsize=11)
    a1.legend(frameon=False, fontsize=9); a1.grid(alpha=0.2)

    for si in range(ns):
        a2.plot(tf, err[:, si], "-", lw=0.6, alpha=0.3, color="tab:red")
    a2.plot(tf, np.nanmedian(err, 1), "-o", color="tab:red", ms=4, label="median error")
    a2.set_yscale("log"); a2.set_xlabel(r"observation time $t_f$")
    a2.set_ylabel("test error (1 - acc)"); a2.grid(alpha=0.2, which="both")
    a2.set_title("error vs $t_f$", fontsize=11); a2.legend(frameon=False, fontsize=9)

    ta = float(d["teacher_acc"]) if "teacher_acc" in d.files else 0.0
    fig.suptitle(f"MNIST thermodynamic computer: optimal observation time "
                 f"(thermo teacher {ta:.3f}, proportional k={d['k_hidden']:g}, "
                 f"{ns} seeds/$t_f$)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(_HERE, "..", "..", "Graphs", "computer_graphs",
                       "fig_tf_scan_mnist.png")
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
