"""Cross-check: load the paper's released weights into our ThermoMNIST and
confirm they reproduce the reported MNIST test accuracy.

The trained parameters are from the paper's official code,
github.com/swhitelam/thermo_gd (reference/parameters.dat, 63,143 values).  The
test digits come from torchvision (identical to the repo's mnist_test.dat) and
are normalized by teacher_net.load_mnist, which now replicates the C++
normalization exactly.  Running our Langevin dynamics (kT=1, tf=0.2) on these
weights should give ~91.8% (M=1) / ~92.0% (M=10), matching the paper
(91.7% single-trajectory / 92.0% over 10) and the C++ (91.76%).

Parameter layout in parameters.dat (thermo.cpp bond order, then per-spin
biases): ii(1568) ih(50176) hh(2016) ho(640) oo(45) io(7840) | biases(858).
Only the 74 evolving spins' bonds/biases affect the dynamics; the frozen
input<->input bonds and input biases are skipped.  The bias enters the C++
force as -b, so our (dV/dx = ... - b ...) convention needs b = -param(bias).

Usage:
    python verify_paper_weights.py
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch

from teacher_net import load_mnist
from thermo_mnist import ThermoMNIST

PARAMS = os.path.join(_HERE, "reference", "parameters.dat")

# bond-class offsets (cumulative), and biases start = number_of_bonds
IH0, HH0, HO0, OO0, IO0, NB = 1568, 51744, 53760, 54400, 54445, 62285


def load_paper_weights(net, device):
    p = np.array(open(PARAMS).read().split(), dtype=np.float64)
    assert p.size == 63143, p.size
    W = np.zeros((784, 74), dtype=np.float32)
    W[:, 0:64] = p[IH0:IH0 + 784 * 64].reshape(784, 64)     # input -> hidden
    W[:, 64:74] = p[IO0:IO0 + 784 * 10].reshape(784, 10)    # input -> output
    Jhh = np.zeros((64, 64), dtype=np.float32); k = HH0
    for i in range(64):
        for j in range(i):
            Jhh[i, j] = Jhh[j, i] = p[k]; k += 1
    Jho = p[HO0:HO0 + 64 * 10].reshape(64, 10).astype(np.float32)
    Joo = np.zeros((10, 10), dtype=np.float32); k = OO0
    for i in range(10):
        for j in range(i):
            Joo[i, j] = Joo[j, i] = p[k]; k += 1
    b = np.zeros(74, dtype=np.float32)                       # sign flip: b = -param
    for h in range(64):
        b[h] = -p[NB + 784 + h]
    for o in range(10):
        b[64 + o] = -p[NB + 784 + 64 + o]
    with torch.no_grad():
        net.W.copy_(torch.tensor(W, device=device))
        net.Jhh_raw.copy_(torch.tensor(Jhh, device=device))   # already sym
        net.Jho.copy_(torch.tensor(Jho, device=device))
        net.Joo_raw.copy_(torch.tensor(Joo, device=device))
        net.b.copy_(torch.tensor(b, device=device))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ThermoMNIST(scale=0.0).to(device)
    load_paper_weights(net, device)
    _, _, Xte, yte = load_mnist(device)
    print(f"loaded paper weights; test set {tuple(Xte.shape)}  device={device}")
    for M in (1, 10):
        torch.manual_seed(0)
        correct = 0; t0 = time.time()
        for i in range(0, Xte.shape[0], 2000):
            pred = net.logits(Xte[i:i + 2000], M=M, seed=i).argmax(1)
            correct += (pred == yte[i:i + 2000]).sum().item()
        acc = correct / Xte.shape[0]
        print(f"M={M:2d}: top-1 accuracy {acc:.4f}   ({time.time()-t0:.0f}s)   "
              f"[paper 0.917/0.920, C++ 0.9176]")


if __name__ == "__main__":
    main()
