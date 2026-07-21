"""Thermodynamic computer (the student) for MNIST, per Whitelam arXiv:2509.15324.

The scale-up of gd_training's ThermoStudent to the paper's image-classification
architecture (Sec. III):

    - 64 hidden + 10 output = 74 fluctuating degrees of freedom x_i (same count
      as the teacher's hidden+output nodes);
    - the 784 MNIST pixels are FROZEN input variables coupled to every node
      (hidden and output) through trainable weights W (784 x 74) -- the paper's
      input<->hidden and input<->output bonds;
    - node<->node couplings: hidden<->hidden (Jhh, symmetric), hidden<->output
      (Jho), and output<->output (Joo, symmetric) -- all bilinear/bidirectional;
    - the intrinsic neuron is the native double well (J2=J4, force 2x+4x^3);
    - inference: show a digit, run the Langevin dynamics to the observation time
      tf, read the 10 output nodes' mean activations, predict argmax (paper: the
      most-positive output unit).

Training (Onsager-Machlup GD, Sec. II) reproduces the teacher's activations
without ever integrating the student: for each digit an idealized noninteracting
T=0 guide trajectory is built whose biases are set from the teacher activations
A (hidden b0 ~ A_i, output b0 ~ 2A_i-1), and theta = {b, W, Jhh, Jho, Joo} is
moved to maximize the likelihood (minimize the OM action) that the interacting
student would have generated that trajectory.

Usage (see train_mnist.py for the full training driver):
    from thermo_mnist import ThermoMNIST, guide_trajectory
"""

import torch
import torch.nn as nn

from teacher_net import N_IN, N_HIDDEN, N_OUT, N_NODES   # 784, 64, 10, 74

# --- Physics constants (paper Sec. III: J2 = J4 = kT = 1) -------------------
# Verified against the paper's released weights (github.com/swhitelam/thermo_gd):
# at kT = 1 those weights reproduce 91.8% (M=1) / 92.0% (M=10) test accuracy.
# gd_training's cosine work used BETA = 10 "cold units"; the MNIST computer is
# kT = 1, so BETA = 1 here.
J2, J4 = 1.0, 1.0
BETA = 1.0
KT = 1.0 / BETA
MU = 1.0
TF = 0.2            # paper's observation time tf = 1/5 (units of 1/mu)
DT = 1e-3

HIDDEN = N_HIDDEN   # 64
N = N_NODES         # 74
OUTPUT_IDX = torch.arange(HIDDEN, N)


def _intrinsic_grad(x):
    """Intrinsic part of dV/dx_i: 2 J2 x + 4 J4 x^3 (the native thermo neuron)."""
    return 2.0 * J2 * x + 4.0 * J4 * x ** 3


@torch.no_grad()
def guide_trajectory(A_hidden, A_out, tf=TF, dt=DT, mu=MU, guide_mode="proportional",
                     k_hidden=1.0, k_output=1.0):
    """Build teacher #2: the idealized noninteracting (Jij=0), T=0 trajectory.

    A_hidden : (B, 64) teacher hidden activations (tanh, in [-1, 1]).
    A_out    : (B, 10) teacher output activations (softmax, in [0, 1]).

    The output guide uses the paper's class-encoding shift S = 2 A_out - 1 (in
    [-1, 1]): +1 for the recognized class, -1 otherwise.

    Hidden and output are scaled SEPARATELY (k_hidden, k_output) so the guide's
    long-time activations can be pushed into the paper's +/-3..4 range (Fig 2b);
    at k=1 the proportional guide only reaches ~+/-0.4, too weak to separate
    above the kT=1 noise.  NOTE the cube in 2x+4x^3 means proportional biases
    saturate: b0=4 -> x~0.8, not 4.  Use "exact" mode to hit a target directly.

    guide_mode:
      "proportional" (paper, Sec. III): b0 = k * target, so the guide's
          long-time activations are only CORRELATED with the targets (nonlinear
          intrinsic response).  target = A_hidden (hidden), S (output).
      "exact": invert 2x+4x^3 = b0 so the T=0 guide relaxes to EXACTLY k*target
          -- b0 = 2 s + 4 s^3 with s = k*target.  Here k_hidden=4 really does
          drive hidden activations to +/-4.

    Returns the trajectory (n_steps+1, B, 74).
    """
    S = 2.0 * A_out - 1.0                                   # output class encoding
    scaled = torch.cat([k_hidden * A_hidden, k_output * S], dim=1)   # (B, 74)
    if guide_mode == "proportional":
        b0 = scaled                                        # biases propto target
    elif guide_mode == "exact":
        b0 = 2.0 * J2 * scaled + 4.0 * J4 * scaled ** 3    # relax exactly to scaled
    else:
        raise ValueError(f"unknown guide_mode {guide_mode!r}")
    n_steps = int(round(tf / dt))
    x = torch.zeros_like(b0)
    traj = [x]
    for _ in range(n_steps):
        x = x + mu * (-(_intrinsic_grad(x) - b0)) * dt     # noninteracting, T=0
        traj.append(x)
    return torch.stack(traj, dim=0)


class ThermoMNIST(nn.Module):
    """Interacting thermodynamic computer, theta = {b, W, Jhh, Jho, Joo}.

    Pixels (784) couple to all 74 nodes via W; nodes couple via Jhh (hidden-
    hidden), Jho (hidden-output), Joo (output-output).  Start from theta = 0
    (scale=0, the paper's default) or a small random init (scale > 0).
    """

    def __init__(self, scale=0.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def init(*shape):
            return scale * torch.randn(*shape, generator=g) if scale \
                else torch.zeros(*shape)

        self.b = nn.Parameter(init(N))                     # biases, all nodes
        self.W = nn.Parameter(init(N_IN, N))               # 784 -> 74 input bonds
        self.Jhh_raw = nn.Parameter(init(HIDDEN, HIDDEN))  # hidden <-> hidden
        self.Jho = nn.Parameter(init(HIDDEN, N_OUT))       # hidden <-> output
        self.Joo_raw = nn.Parameter(init(N_OUT, N_OUT))    # output <-> output

    @staticmethod
    def _sym0(Jraw):
        J = 0.5 * (Jraw + Jraw.t())
        return J - torch.diag(torch.diag(J))

    def coupling(self):
        """Assemble the symmetric 74 x 74 coupling matrix with zero diagonal."""
        Jc = self.b.new_zeros(N, N)
        Jc[:HIDDEN, :HIDDEN] = self._sym0(self.Jhh_raw)
        Jc[:HIDDEN, HIDDEN:] = self.Jho
        Jc[HIDDEN:, :HIDDEN] = self.Jho.t()
        Jc[HIDDEN:, HIDDEN:] = self._sym0(self.Joo_raw)
        return Jc

    def input_field(self, pixels):
        """(B, 74) external field on every node: pixels @ W."""
        return pixels @ self.W

    def dVdx(self, x, pixels):
        """dV/dx_i (Eq. 10) for state x (..., B, 74) and pixels (B, 784)."""
        return _intrinsic_grad(x) - self.b + x @ self.coupling() \
            + self.input_field(pixels)

    def om_loss(self, traj, pixels, dt=DT, mu=MU, kT=KT):
        """Onsager-Machlup action (Eq. 5) over a fixed idealized trajectory.

        traj : (K+1, B, 74) guide trajectory (constant w.r.t. theta).
        pixels : (B, 784) inputs that produced traj.
        """
        x = traj[:-1]
        dx = traj[1:] - traj[:-1]
        drift = self.dVdx(x, pixels)
        resid = dx + mu * drift * dt
        action = resid ** 2 / (4.0 * mu * kT * dt)
        return action.sum(dim=-1).mean()

    def _rollout(self, pixels, M=16, tf=TF, dt=DT, beta=BETA, mu=MU, seed=None):
        """Integrate the finite-T Langevin dynamics from x=0; final (B, M, 74)."""
        B = pixels.shape[0]
        Jc = self.coupling()
        inp = self.input_field(pixels).reshape(B, 1, N)
        noise_amp = (2.0 * mu * (1.0 / beta) * dt) ** 0.5
        n_steps = int(round(tf / dt))
        gen = torch.Generator(device=pixels.device).manual_seed(seed) \
            if seed is not None else None
        x = torch.zeros(B, M, N, dtype=self.b.dtype, device=pixels.device)
        for _ in range(n_steps):
            dVdx = _intrinsic_grad(x) - self.b + x @ Jc + inp
            noise = torch.randn(x.shape, generator=gen, dtype=x.dtype,
                                device=x.device)
            x = x - mu * dVdx * dt + noise_amp * noise
        return x

    @torch.no_grad()
    def logits(self, pixels, M=16, tf=TF, **kw):
        """Mean output-node activations <x_out(tf)>, shape (B, 10) -- class scores."""
        x = self._rollout(pixels, M=M, tf=tf, **kw)
        return x.mean(dim=1)[:, OUTPUT_IDX]

    @torch.no_grad()
    def predict(self, pixels, **kw):
        """Predicted class = argmax output node (paper: most-positive output)."""
        return self.logits(pixels, **kw).argmax(dim=1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    # Self-test: build a guide from random teacher activations, check the OM
    # action is finite and differentiable for every parameter, and that a
    # rollout produces sane output scores.
    torch.manual_seed(0)
    B = 8
    A_hidden = torch.tanh(torch.randn(B, N_HIDDEN))
    A_out = torch.softmax(torch.randn(B, N_OUT), dim=1)
    pixels = torch.randn(B, N_IN)
    traj = guide_trajectory(A_hidden, A_out)
    print(f"guide trajectory {tuple(traj.shape)}  (expect ({int(TF/DT)+1}, {B}, {N}))")

    net = ThermoMNIST(scale=1e-2)
    print(f"ThermoMNIST params: {net.n_params():,}  "
          f"(pixel-node {N_IN*N:,} + Jhh {HIDDEN*(HIDDEN-1)//2:,} + Jho "
          f"{HIDDEN*N_OUT:,} + Joo {N_OUT*(N_OUT-1)//2} + b {N})")
    Jc = net.coupling()
    assert torch.allclose(Jc, Jc.t()) and Jc.diagonal().abs().max() == 0.0
    loss = net.om_loss(traj, pixels)
    loss.backward()
    for name, p in net.named_parameters():
        print(f"  OM grad |d{name:8s}| = {p.grad.norm().item():.3e}")
    print(f"OM loss {loss.item():.4e}")
    scores = net.logits(pixels, M=8)
    print(f"output scores {tuple(scores.shape)}  argmax {net.predict(pixels).tolist()}")


if __name__ == "__main__":
    main()
