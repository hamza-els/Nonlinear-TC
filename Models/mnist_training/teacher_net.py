"""Teacher #1 for the MNIST thermodynamic computer (Whitelam, arXiv:2509.15324).

A fully connected 784-32-32-10 network with tanh hidden activations and a
softmax output, trained by SGD to classify MNIST -- the reference model whose
node activations the thermodynamic computer (the student) will later be trained
to reproduce.  Paper settings (Sec. III): SGD lr 0.2, weight decay 1e-4, batch
256, 300 epochs -> 97.3% test accuracy.

The N = 74 recorded activations per digit are, in order:
    hidden : the 64 tanh activations (layer-1's 32 then layer-2's 32), in
             [-1, 1] -- guide biases for the student's hidden units are set
             proportional to these (b0 ~ A_i);
    output : the 10 softmax class probabilities, in [0, 1] -- guide biases for
             the student's output units use the shifted form 2 A_i - 1 (+1 for
             the recognized class, -1 otherwise), exactly as the paper.

Input preprocessing matches the computer's input (paper Sec. III): pixels are
standardized to ~zero mean / unit variance (global MNIST mean/std) and then
L2-normalized per digit so every image has the same norm.  The teacher is
trained on these same inputs so its activations correspond to what the student
will see.

Usage:
    python teacher_net.py [epochs]      # train (default 300) -> runs/teacher.pt
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
RUNS_DIR = os.path.join(_HERE, "runs")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Architecture ----------------------------------------------------------
N_IN = 784
H1 = 32
H2 = 32
N_HIDDEN = H1 + H2          # 64 hidden activations recorded as targets
N_OUT = 10
N_NODES = N_HIDDEN + N_OUT  # 74 thermodynamic degrees of freedom to reproduce


# Paper's fixed MNIST standardization constants (thermo.cpp read_mnist):
# mnist_mu = 0.13066, and sigma = sqrt(E[x^2] - mu^2) with E[x^2] = 0.112003.
MNIST_MU = 0.13066
MNIST_SIGMA = (0.112003 - MNIST_MU ** 2) ** 0.5    # 0.30811


def _paper_normalize(X):
    """Standardize with the paper's fixed MNIST mean/std, then scale EVERY image
    to the same (maximum) L2 norm across the set -- exactly thermo.cpp's
    normalize_digits_by_l2 (scale = max_norm / this_norm), NOT unit norm."""
    X = (X - MNIST_MU) / MNIST_SIGMA
    norms = X.norm(dim=1)
    return X * (norms.max() / norms).unsqueeze(1)


def load_mnist(device="cpu"):
    """Return (Xtr, ytr, Xte, yte): flattened, standardized, max-L2-normalized
    exactly as the paper's thermo.cpp (verified to reproduce its weights'
    accuracy).  Each set is scaled to its own max L2 norm, as in the C++."""
    from torchvision import datasets
    tr = datasets.MNIST(DATA_DIR, train=True, download=True)
    te = datasets.MNIST(DATA_DIR, train=False, download=True)
    Xtr = _paper_normalize(tr.data.float().div(255.0).reshape(-1, N_IN))
    Xte = _paper_normalize(te.data.float().div(255.0).reshape(-1, N_IN))
    return (Xtr.to(device), tr.targets.to(device),
            Xte.to(device), te.targets.to(device))


# "thermo" hidden activation: a fit to the TRUE finite-time activation of an
# isolated thermodynamic neuron (from gd_training/digital_net.py) --
#   sigma(x) = ACT_A*(x^2/(x^2+ACT_C))*cbrt(x) + tanh(x)
# captures the linear-small / cube-root-large response tanh cannot.  Evaluated
# via the stable identity (x^2/(x^2+c))*cbrt(x) == sign(x)*|x|^(7/3)/(x^2+c) so
# the gradient stays finite at x=0.  Unbounded, so hidden activations can exceed
# +/-1 (unlike tanh) -- closer to the student neuron's own response.
ACT_A, ACT_C = 0.5617, 13.53


def thermo_activation(x):
    return (ACT_A * torch.sign(x) * x.abs().pow(7.0 / 3.0) / (x ** 2 + ACT_C)
            + torch.tanh(x))


def _hidden_act(x, kind):
    if kind == "tanh":
        return torch.tanh(x)
    if kind == "thermo":
        return thermo_activation(x)
    raise ValueError(f"unknown activation {kind!r}")


class TeacherMLP(nn.Module):
    """784-32-32-10, softmax output; hidden activation "tanh" or "thermo"."""

    def __init__(self, activation="tanh"):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(N_IN, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.out = nn.Linear(H2, N_OUT)

    def forward(self, x):
        """Class logits (B, 10) -- for cross-entropy training."""
        h1 = _hidden_act(self.fc1(x), self.activation)
        h2 = _hidden_act(self.fc2(h1), self.activation)
        return self.out(h2)

    def activations(self, x):
        """Return (hidden, out): the reference activations A_i for the student.

        hidden : (B, 64) hidden activations [layer1(32) | layer2(32)] -- in
                 [-1, 1] for tanh, unbounded (cube-root) for thermo.
        out    : (B, 10) softmax class probabilities, in [0, 1].
        """
        h1 = _hidden_act(self.fc1(x), self.activation)
        h2 = _hidden_act(self.fc2(h1), self.activation)
        out = F.softmax(self.out(h2), dim=1)
        return torch.cat([h1, h2], dim=1), out


def accuracy(model, X, y, chunk=10000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            pred = model(X[i:i + chunk]).argmax(1)
            correct += (pred == y[i:i + chunk]).sum().item()
    return correct / X.shape[0]


def train_teacher(epochs=300, lr=0.2, weight_decay=1e-4, batch=256, seed=0,
                  momentum=0.9, activation="tanh", device=None, verbose=True):
    """Train the teacher to classify MNIST (paper Sec. III: SGD lr 0.2, weight
    decay 1e-4, batch 256, 300 epochs).

    Adds cosine learning-rate decay (lr 0.2 -> 0 over the run) and momentum on
    top of the paper's plain SGD -- the standard "learning-rate decay and such"
    that stabilizes the final epochs at the larger max-L2 input scale (fixed
    lr=0.2 there plateaus a couple points low)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = load_mnist(device)
    model = TeacherMLP(activation=activation).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ntr = Xtr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr, device=device)
        for i in range(0, ntr, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep % max(1, epochs // 20) == 0 or ep == epochs - 1):
            acc = accuracy(model, Xte, yte)
            print(f"  epoch {ep:4d}  loss {loss.item():.4f}  test_acc {acc:.4f}"
                  f"  lr {sched.get_last_lr()[0]:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    final_acc = accuracy(model, Xte, yte)
    print(f"final test accuracy: {final_acc:.4f}  ({time.time()-t0:.0f}s)",
          flush=True)
    return model, final_acc


def save_teacher(model, acc, path=None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = path or os.path.join(RUNS_DIR, f"teacher_{model.activation}.pt")
    torch.save({"state_dict": model.state_dict(), "arch": [N_IN, H1, H2, N_OUT],
                "activation": model.activation, "test_acc": acc}, path)
    print(f"saved {path}  (test_acc {acc:.4f})")
    return path


def load_teacher(path=None, activation="tanh", device="cpu"):
    path = path or os.path.join(RUNS_DIR, f"teacher_{activation}.pt")
    d = torch.load(path, map_location=device)
    model = TeacherMLP(activation=d.get("activation", activation)).to(device)
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model, d.get("test_acc")


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    activation = sys.argv[2] if len(sys.argv) > 2 else "tanh"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training MNIST teacher: 784-{H1}-{H2}-{N_OUT} {activation}, "
          f"{epochs} epochs, device={device}", flush=True)
    model, acc = train_teacher(epochs=epochs, activation=activation, device=device)
    save_teacher(model, acc)


if __name__ == "__main__":
    main()
