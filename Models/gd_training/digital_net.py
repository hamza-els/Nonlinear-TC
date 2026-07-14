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
N_OUT = 8                      # tanh output neurons, linearly combined to y
N = HIDDEN + N_OUT             # total non-input degrees of freedom
N_IN = 6                       # input channels: features(z) = (z, ..., z^6)


def features(z):
    """Input features phi(z) = (z, z^2, ..., z^N_IN), shape (B, N_IN).

    Multiple frozen input channels raise the rank of the field the student
    can apply (sum_k W_k phi_k(z) instead of W z), letting it exert
    nonlinear-in-z, non-monotone forces from t = 0 -- the scalar-input analog
    of the paper's 784 pixel channels.  Both teacher and student see them.
    """
    return torch.stack([z ** k for k in range(1, N_IN + 1)], dim=-1)


def target(z):
    """Target function y0(z) = cos(2 pi z)."""
    return torch.cos(2.0 * torch.pi * z)


class DigitalCosineNet(nn.Module):
    """Deterministic MLP: z -> [tanh hidden layers] -> linear output.

    forward(z) returns the scalar prediction.  activations(z) additionally
    returns every neuron's activation, concatenated as (hidden..., output),
    which is the N-vector A_i used to build the idealized teacher trajectory.
    """

    def __init__(self, width=WIDTH, depth=DEPTH, dropout=0.0):
        super().__init__()
        dims = [N_IN] + [width] * depth
        self.hidden = nn.ModuleList(
            nn.Linear(dims[l], dims[l + 1]) for l in range(depth)
        )
        self.out = nn.Linear(width, N_OUT)
        # Learned linear combination of the N_OUT tanh output neurons into the
        # scalar prediction y.  The thermodynamic student mimics the 8 output
        # activations; this readout (or a post-hoc LS refit of it) turns its
        # 8 mean output-node values into y.
        self.readout = nn.Linear(N_OUT, 1, bias=False)
        # Dropout on hidden activations (active in train() mode only): forces
        # a distributed representation in which no single neuron is
        # load-bearing -- useful here because the thermodynamic student tracks
        # each activation target imperfectly, which is exactly the perturbed-
        # unit regime dropout trains robustness against.
        self.drop = nn.Dropout(dropout)

    def activations(self, z):
        """Return (y, A) for input z of shape (B,).

        y : (B,)        scalar prediction = readout of the 8 output neurons
        A : (B, N)      all neuron activations, order [hidden..., outputs].
        Hidden and output activations are post-tanh.
        """
        h = features(z.reshape(-1))
        acts = []
        for layer in self.hidden:
            h = self.drop(torch.tanh(layer(h)))
            acts.append(h)
        o = torch.tanh(self.out(h))          # (B, N_OUT) output activations
        acts.append(o)
        y = self.readout(o).reshape(-1)      # learned linear combination
        A = torch.cat(acts, dim=1)           # (B, N)
        return y, A

    def forward(self, z):
        return self.activations(z)[0]


def train_teacher(width=WIDTH, depth=DEPTH, K=256, epochs=4000, lr=1e-2,
                  weight_decay=0.0, act_reg=0.0, sat_reg=0.0, sat_thresh=0.8,
                  dropout=0.0, seed=0, device="cpu", verbose=True):
    """Fit the digital net to cos(2 pi z) on K evenly spaced points in [0, 1].

    Full-batch Adam; deterministic given the seed.  Returns the trained model.

    act_reg > 0 adds an L2 penalty act_reg * mean(A_hidden^2) on the hidden
    activations: it keeps them away from the tanh saturation rails, which
    makes them cheaper targets for the thermodynamic student (guide fields
    grow as 2A + 4A^3).

    sat_reg > 0 adds a SATURATION penalty sat_reg * mean(relu(|A| -
    sat_thresh)^2) on the hidden activations: unlike act_reg it leaves
    mid-range amplitudes untouched and only pushes units off the tanh rails.
    Motivated by the lottery correlation (2026-07-13): student RMSE correlates
    positively with the teacher's saturated-unit fraction (+0.47, n=10).
    """
    torch.manual_seed(seed)
    model = DigitalCosineNet(width, depth, dropout=dropout).to(device)
    z = torch.linspace(0.0, 1.0, K, device=device)
    y0 = target(z)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        y, A = model.activations(z)
        loss = torch.mean((y - y0) ** 2)
        if act_reg:
            loss = loss + act_reg * A[:, :HIDDEN].pow(2).mean()
        if sat_reg:
            excess = (A[:, :HIDDEN].abs() - sat_thresh).clamp_min(0.0)
            loss = loss + sat_reg * excess.pow(2).mean()
        loss.backward()
        opt.step()
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"  teacher epoch {ep:5d}  mse {loss.item():.3e}")
    # eval() disables dropout: the activations handed to the student as
    # targets are the deterministic full-network ones.
    model.eval()
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
