"""Onsager-Machlup gradient-descent training of the MNIST thermodynamic computer.

Loads the trained teacher (teacher_net.py), records its activations for every
digit, and trains the ThermoMNIST student to reproduce them by minimizing the
OM action of the idealized guide trajectories (paper Sec. II).  Training is
minibatch GD (the paper's online Eqs. 6-7 in batched form); inference accuracy
on the test set is reported each epoch (argmax of the 10 output nodes, averaged
over M reset samples).  Start is theta = 0 (the paper's default).

Usage:
    python train_mnist.py [epochs] [guide_mode] [k_hidden] [k_output] [tf]
      epochs     default 20
      guide_mode "proportional" (paper) | "exact"
      k_hidden   hidden guide scale (b0 ~ k_hidden * A), default 4.0
      k_output   output guide scale (b0 ~ k_output * (2A-1)), default 4.0
      tf         observation time, default 0.2
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch

from teacher_net import load_mnist, load_teacher
from thermo_mnist import ThermoMNIST, guide_trajectory, TF, DT, MU, KT

RUNS_DIR = os.path.join(_HERE, "runs")

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GUIDE_MODE = sys.argv[2] if len(sys.argv) > 2 else "proportional"
K_HIDDEN = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
K_OUTPUT = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
TRAIN_TF = float(sys.argv[5]) if len(sys.argv) > 5 else TF

BATCH = 256
LR = 1e-2
M_EVAL = 16          # reset samples per digit at inference
EVAL_N = 2000        # test digits scored each epoch (full 10k at the end)


@torch.no_grad()
def teacher_activations(teacher, X, chunk=10000):
    hs, os_ = [], []
    for i in range(0, X.shape[0], chunk):
        h, o = teacher.activations(X[i:i + chunk])
        hs.append(h); os_.append(o)
    return torch.cat(hs), torch.cat(os_)


@torch.no_grad()
def accuracy(student, X, y, M=M_EVAL, tf=TRAIN_TF, chunk=2000, seed=0):
    correct = 0
    for i in range(0, X.shape[0], chunk):
        pred = student.logits(X[i:i + chunk], M=M, tf=tf, seed=seed).argmax(1)
        correct += (pred == y[i:i + chunk]).sum().item()
    return correct / X.shape[0]


def train(epochs=EPOCHS, batch=BATCH, lr=LR, guide_mode=GUIDE_MODE,
          k_hidden=K_HIDDEN, k_output=K_OUTPUT, tf=TRAIN_TF, seed=0,
          teacher_act="tanh", eval_n=EVAL_N, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, Xte, yte = load_mnist(device)
    teacher, tacc = load_teacher(activation=teacher_act, device=device)
    Ah, Ao = teacher_activations(teacher, Xtr)             # (60000,64),(60000,10)
    print(f"teacher[{teacher_act}] loaded (test_acc {tacc:.4f}); activations "
          f"{tuple(Ah.shape)}/{tuple(Ao.shape)}  |A_hidden| max "
          f"{Ah.abs().max():.2f}", flush=True)

    student = ThermoMNIST(scale=0.0, seed=seed).to(device)  # theta = 0 start
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    ntr = Xtr.shape[0]
    # report the guide's actual activation magnitude at these scales
    g = guide_trajectory(Ah[:256], Ao[:256], tf=tf, guide_mode=guide_mode,
                         k_hidden=k_hidden, k_output=k_output)[-1]
    print(f"student: {student.n_params():,} params | epochs={epochs} batch={batch} "
          f"lr={lr} guide={guide_mode} k_h={k_hidden} k_o={k_output} tf={tf} | "
          f"guide |x(tf)| max {g.abs().max():.2f} (hidden {g[:, :64].abs().max():.2f}"
          f" / output {g[:, 64:].abs().max():.2f}) device={device}", flush=True)

    t0 = time.time()
    for ep in range(epochs):
        student.train()
        perm = torch.randperm(ntr, device=device)
        run_loss = 0.0
        for i in range(0, ntr, batch):
            idx = perm[i:i + batch]
            traj = guide_trajectory(Ah[idx], Ao[idx], tf=tf, guide_mode=guide_mode,
                                    k_hidden=k_hidden, k_output=k_output)
            loss = student.om_loss(traj, Xtr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item()
        acc = accuracy(student, Xte[:eval_n], yte[:eval_n], tf=tf)
        print(f"  epoch {ep:3d}  OM {run_loss/(ntr//batch):.4e}  "
              f"test_acc[{eval_n}] {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    full_acc = accuracy(student, Xte, yte, tf=tf)
    print(f"final test accuracy (10k, M={M_EVAL}): {full_acc:.4f}", flush=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR,
                        f"student_{guide_mode}_kh{k_hidden:g}_ko{k_output:g}_tf{tf:g}.pt")
    torch.save({"state_dict": student.state_dict(), "test_acc": full_acc,
                "guide_mode": guide_mode, "k_hidden": k_hidden,
                "k_output": k_output, "tf": tf, "epochs": epochs}, path)
    print(f"saved {path}", flush=True)
    return student, full_acc


if __name__ == "__main__":
    train()
