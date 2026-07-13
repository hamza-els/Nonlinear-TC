"""Digital deterministic teacher network for the cosine task.

This is "teacher #1" in Whitelam, arXiv:2509.15324 (Fig. 1): a standard,
deterministic feed-forward neural network trained by gradient descent to
approximate a target function.  Here the target is y0(z) = cos(2 pi z) on
z in [0, 1] (a regression task) rather than MNIST classification.

The network has WIDTH=4, DEPTH=2 (two hidden layers of four tanh units),
giving HIDDEN = 8 hidden neurons plus a single linear output neuron.  The
per-neuron activations A_i (hidden + output) are the targets that the
thermodynamic student is later trained to reproduce (see thermo_student.py).
A shallow teacher keeps the target activations computationally "early"
functions of z, which the student's finite-time dynamics can reach (deep-layer
activations were the hardest to mimic -- see the per-node deviation figure).
"""

import torch
import torch.nn as nn

# --- Architecture ---------------------------------------------------------
WIDTH = 32
DEPTH = 2                      # number of hidden layers
HIDDEN = WIDTH * DEPTH         # 32 hidden neurons total
N_OUT = 1                      # single scalar output (the cosine prediction)
N = HIDDEN + N_OUT             # total non-input degrees of freedom (= 33)


def target(z):
    """Target function y0(z) = cos(2 pi z)."""
    return torch.cos(2.0 * torch.pi * z)


class DigitalCosineNet(nn.Module):
    """Deterministic MLP: z -> [tanh hidden layers] -> linear output.

    forward(z) returns the scalar prediction.  activations(z) additionally
    returns every neuron's activation, concatenated as (hidden..., output),
    which is the N-vector A_i used to build the idealized teacher trajectory.
    """

    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        dims = [1] + [width] * depth
        self.hidden = nn.ModuleList(
            nn.Linear(dims[l], dims[l + 1]) for l in range(depth)
        )
        self.out = nn.Linear(width, N_OUT)

    def activations(self, z):
        """Return (y, A) for input z of shape (B,).

        y : (B,)        scalar prediction (== output activation)
        A : (B, N)      all neuron activations, order [hidden..., output].
        The hidden activations are post-tanh; the output is linear.
        """
        h = z.reshape(-1, 1)
        acts = []
        for layer in self.hidden:
            h = torch.tanh(layer(h))
            acts.append(h)
        y = self.out(h).reshape(-1)          # linear output neuron
        acts.append(y.reshape(-1, 1))
        A = torch.cat(acts, dim=1)           # (B, N)
        return y, A

    def forward(self, z):
        return self.activations(z)[0]


def train_teacher(width=WIDTH, depth=DEPTH, K=256, epochs=4000, lr=1e-2,
                  weight_decay=0.0, seed=0, device="cpu", verbose=True):
    """Fit the digital net to cos(2 pi z) on K evenly spaced points in [0, 1].

    Full-batch Adam; deterministic given the seed.  Returns the trained model.
    """
    torch.manual_seed(seed)
    model = DigitalCosineNet(width, depth).to(device)
    z = torch.linspace(0.0, 1.0, K, device=device)
    y0 = target(z)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for ep in range(epochs):
        opt.zero_grad()
        y = model(z)
        loss = torch.mean((y - y0) ** 2)
        loss.backward()
        opt.step()
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  teacher epoch {ep:5d}  mse {loss.item():.3e}")
    return model


def main():
    # Tiny self-test: train briefly and report fit quality + activation shapes.
    model = train_teacher(epochs=500, verbose=True)
    z = torch.linspace(0.0, 1.0, 11)
    with torch.no_grad():
        y, A = model.activations(z)
    mse = torch.mean((y - target(z)) ** 2).item()
    print(f"final mse on 11-point grid: {mse:.3e}")
    print(f"activation matrix A shape: {tuple(A.shape)}  (expect (11, {N}))")


if __name__ == "__main__":
    main()
