import torch
import torch.nn as nn

# --- Network architecture -------------------------------------------------
# Four layers of 8 neurons feeding a single output neuron.
LAYERS = [8, 8, 8, 8, 1]
N = sum(LAYERS)                          # total neurons (= 33)
OFFSETS = torch.cumsum(torch.tensor([0] + LAYERS), 0)  # start index of each layer
INPUT_IDX = torch.arange(OFFSETS[0], OFFSETS[1])        # layer-0 neurons (take input z)
OUTPUT_IDX = torch.arange(OFFSETS[-2], OFFSETS[-1])     # final output neuron(s)

# Intrinsic neuron couplings J = (J2, J3, J4); (1, 0, 1) -> nonlinear neuron.
J_INTRINSIC = (1.0, 0.0, 1.0)


class ThermoNet(nn.Module):
    """A thermodynamic computer (Eq. S8/S9) as a PyTorch module.

    The adjustable parameters theta = {W} u {b} u {Jij} u {f} are stored as
    nn.Parameters, so the whole simulation is differentiable and the network
    can be trained either by genetic algorithm (no grad) or by gradient
    descent / backprop-through-time (autograd).
    """

    def __init__(self, scale=1e-2):
        super().__init__()
        # Input weights coupling z into the first layer.
        self.W = nn.Parameter(scale * torch.randn(LAYERS[0]))
        # Per-neuron biases b_i (the bias in U_J(x_i, b_i)).
        self.b = nn.Parameter(scale * torch.randn(N))
        # Bilinear coupling matrices between adjacent layers.
        self.Jmats = nn.ParameterList([
            nn.Parameter(scale * torch.randn(LAYERS[l], LAYERS[l + 1]))
            for l in range(len(LAYERS) - 1)
        ])
        # Output weights coupling the output neuron(s) to the scalar y.
        self.f = nn.Parameter(scale * torch.randn(len(OUTPUT_IDX)))

    def coupling_matrix(self):
        """Assemble the symmetric N x N bilinear coupling matrix Jc from the
        per-layer matrices. Nonzero only between adjacent layers."""
        Jc = self.W.new_zeros(N, N)
        for l, Wmat in enumerate(self.Jmats):
            a0, a1 = OFFSETS[l], OFFSETS[l + 1]
            b0, b1 = OFFSETS[l + 1], OFFSETS[l + 2]
            Jc[a0:a1, b0:b1] = Wmat
            Jc[b0:b1, a0:a1] = Wmat.t()
        return Jc

    def simulate(self, z, M=1000, tf=1.0, dt=1e-3, beta=10.0, mu=1.0):
        """Reset-sampling Langevin simulation. Returns <x_i(z)>_r of shape (K, N).

            dV/dx_i = (2 J2 x_i + 3 J3 x_i^2 + 4 J4 x_i^3)   [intrinsic]
                      - b_i                                    [bias]
                      + sum_j Jc[i,j] x_j                      [bilinear coupling]
                      + W_i z   (input-layer neurons only)     [external input]
        """
        J2, J3, J4 = J_INTRINSIC
        z = torch.atleast_1d(torch.as_tensor(z, dtype=self.W.dtype, device=self.W.device))
        K = z.shape[0]

        Jc = self.coupling_matrix()

        # External field W_i z on the input-layer neurons, zero elsewhere: (K, N).
        ext = self.W.new_zeros(K, N)
        ext[:, INPUT_IDX] = z[:, None] * self.W[None, :]

        kT = 1.0 / beta
        noise_amp = (2.0 * mu * kT * dt) ** 0.5
        n_steps = int(round(tf / dt))

        # State for every (input, sample) pair: (K, M, N), all reset to zero.
        x = self.W.new_zeros(K, M, N)
        for _ in range(n_steps):
            dVdx = (
                (2.0 * J2 * x + 3.0 * J3 * x ** 2 + 4.0 * J4 * x ** 3)
                - self.b[None, None, :]
                + x @ Jc
                + ext[:, None, :]
            )
            x = x - mu * dVdx * dt + noise_amp * torch.randn_like(x)

        return x.mean(dim=1)  # average over the M samples -> (K, N)

    def forward(self, z, **kw):
        """Scalar output y(z) = sum_{i in outputs} f_i <x_i(z)>_r."""
        mean_x = self.simulate(z, **kw)
        return mean_x[:, OUTPUT_IDX] @ self.f


# --- Target and loss ------------------------------------------------------
def target(z):
    """Target function y_0(z) = cos(2 pi z)."""
    return torch.cos(2.0 * torch.pi * z)


def loss_fn(model, K=250, **kw):
    """Mean-squared-error loss phi over K evenly-spaced points on [0, 1]."""
    z = torch.arange(K, dtype=model.W.dtype, device=model.W.device) / (K - 1)
    y = model(z, **kw)
    return torch.mean((target(z) - y) ** 2)


def main():
    # Build the model and evaluate the loss once with random parameters.
    torch.manual_seed(0)
    model = ThermoNet()
    loss = loss_fn(model, K=250, M=100)
    print(f"loss (random params): {loss.item():.4f}")


if __name__ == "__main__":
    main()
