"""Thermodynamic-computer student, trained by gradient descent.

Implements the "student" of Whitelam, arXiv:2509.15324 (Sec. II), specialized
to the cosine-regression architecture:

    - HIDDEN = 32 hidden neurons, all-to-all coupled among themselves (Jhh),
    - the input z couples to EVERY neuron (hidden and output) via a trainable
      W_i -- the paper (Sec. III) likewise has trainable couplings between the
      input data and both the hidden and the output nodes; the direct
      z -> output coupling gives the output a first-order drive from t = 0,
    - each hidden neuron couples to the single output neuron via a learned
      scalar Jho_i,
    - the prediction is the output neuron's mean activation <x_out(tf)>,
      optionally times a learned readout scale c (fitted post-hoc; c is not
      part of the thermodynamic dynamics).

The N = 33 degrees of freedom evolve under overdamped Langevin dynamics with
potential (paper Eq. 2)

    V_theta(x) = sum_i (J2 x_i^2 + J4 x_i^4)      [intrinsic, fixed]
               - sum_i b_i x_i                     [trainable biases]
               + sum_{i<j} Jij x_i x_j             [trainable couplings]
               + z * sum_{i in hidden} W_i x_i     [trainable input coupling]

so the drift is  -mu dV/dx_i  with

    dV/dx_i = 2 J2 x_i + 4 J4 x_i^3 - b_i + sum_j Jc[i,j] x_j + W_i z.   (Eq.10)

Energy/time conventions follow the paper: J2 = J4 = kBT (kT = 1, energies in
units of kBT), mu = 1, dt = 1e-3, and tf = 1.0 (see the note at the TF
constant for why not the paper's 1/5).
The input z is a frozen variable coupled through W (paper: input acts as a set
of biases via the trainable interactions).

Training does NOT integrate the student.  Instead we build a fixed idealized
teacher trajectory (noninteracting, T=0) whose activations relax to the target
values A_i -- the digital net's hidden activations for the hidden nodes, and
the GROUND-TRUTH output cos(2 pi z) for the output node -- and minimize the
negative-log path probability (Onsager-Machlup action, Eq. 5) with which the
*student* would generate it.  Because the trajectory is fixed, the action is a
closed-form function of theta and PyTorch autograd yields exactly the analytic
gradients of Eqs. 6-10.
"""

import torch
import torch.nn as nn

from digital_net import N, HIDDEN, N_OUT

# --- Physics constants -----------------------------------------------------
# Paper convention (Sec. III): J2 = J4 = kBT, energies displayed in units of
# kBT. We set kT = 1 and J2 = J4 = 1, i.e. J2 = J4 = kBT exactly. NOTE: this
# deviates from ga_training/cosine_training_torch.py, which uses beta = 10
# (10x stiffer wells relative to noise) -- GA/GD losses are not directly
# comparable after this change.
J2, J4 = 1.0, 1.0
BETA = 1.0
KT = 1.0 / BETA
MU = 1.0
# Observation time (units of 1/mu). The paper uses tf = 1/5, but its inputs
# couple directly to the output nodes; here z reaches the output only through
# the hidden layer, and at tf = 0.2 the output never activates (tested:
# predictions ~ 0). tf = 1.0 (~2 relaxation times) works markedly better.
TF = 1.0
DT = 1e-3

# The output neuron is the last degree of freedom; its activation is the
# prediction.  Hidden neurons occupy indices [0, HIDDEN).
OUTPUT_IDX = torch.arange(HIDDEN, N)


def _intrinsic_grad(x):
    """Intrinsic part of dV/dx_i: 2 J2 x + 4 J4 x^3."""
    return 2.0 * J2 * x + 4.0 * J4 * x ** 3


@torch.no_grad()
def idealized_trajectory(A, tf=TF, dt=DT, mu=MU):
    """Build "teacher #2": the idealized trajectory whose activations relax to A.

    Uses a NONINTERACTING (Jij = 0), zero-temperature computer whose guide
    biases are chosen so the fixed point equals A_i exactly.  The steady state
    of x_dot = -mu(2 J2 x + 4 J4 x^3 - b0) satisfies 2 J2 x + 4 J4 x^3 = b0, so
    setting  b0_i = 2 J2 A_i + 4 J4 A_i^3  makes x_i relax to exactly A_i.
    (The paper sets b0 merely proportional to A and accepts the nonlinearity;
    the exact inverse is a small improvement that helps the regression task.)

    A    : (B, N) target activations, one row per input.
    Returns traj of shape (n_steps + 1, B, N), starting from x = 0.
    """
    b0 = 2.0 * J2 * A + 4.0 * J4 * A ** 3          # (B, N)
    n_steps = int(round(tf / dt))
    x = torch.zeros_like(A)
    traj = [x]
    for _ in range(n_steps):
        drift = -(_intrinsic_grad(x) - b0)         # noninteracting, T = 0
        x = x + mu * drift * dt
        traj.append(x)
    return torch.stack(traj, dim=0)                # (K+1, B, N)


class ThermoStudent(nn.Module):
    """Interacting thermodynamic computer, theta = {b, W, Jhh, Jho}.

    Connectivity (see module docstring): z -> hidden (W), hidden <-> hidden
    all-to-all (Jhh, symmetric, zero diagonal), hidden <-> output (Jho).
    All couplings are bilinear and bidirectional, as in the paper.
    """

    def __init__(self, scale=0.0, seed=0):
        # Paper: training starts from theta = 0 (scale=0.0, the default).
        # Pass scale > 0 for a small random init instead.
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        def init(*shape):
            if scale:
                return scale * torch.randn(*shape, generator=g)
            return torch.zeros(*shape)

        self.b = nn.Parameter(init(N))              # biases, all nodes
        self.W = nn.Parameter(init(N))              # z -> node couplings (all)
        self.Jhh_raw = nn.Parameter(init(HIDDEN, HIDDEN))  # hidden <-> hidden
        self.Jho = nn.Parameter(init(HIDDEN))       # hidden <-> output scalars
        # Post-hoc readout scale c: y = c * <x_out(tf)>.  Not a thermodynamic
        # parameter (it lives outside the dynamics); fitted by fit_readout().
        self.register_buffer("readout", torch.ones(()))

    def coupling(self):
        """Assemble the symmetric N x N coupling matrix with zero diagonal."""
        Jhh = 0.5 * (self.Jhh_raw + self.Jhh_raw.t())
        Jhh = Jhh - torch.diag(torch.diag(Jhh))
        Jc = self.b.new_zeros(N, N)
        Jc[:HIDDEN, :HIDDEN] = Jhh
        Jc[:HIDDEN, HIDDEN:] = self.Jho.reshape(HIDDEN, N_OUT)
        Jc[HIDDEN:, :HIDDEN] = self.Jho.reshape(N_OUT, HIDDEN)
        return Jc

    def input_field(self, z):
        """(B, N) external field W_i z on every node (hidden and output)."""
        return z.reshape(-1, 1) * self.W.reshape(1, N)

    def dVdx(self, x, z):
        """dV/dx_i (Eq. 10) for state x (..., B, N) and input z (B,)."""
        return (_intrinsic_grad(x)
                - self.b
                + x @ self.coupling()
                + self.input_field(z))

    def om_loss(self, traj, z, dt=DT, mu=MU, kT=KT):
        """Onsager-Machlup action (Eq. 5) summed over the fixed trajectory.

        traj : (K+1, B, N) idealized trajectory (a constant w.r.t. theta).
        z    : (B,) inputs that produced traj.
        Returns the mean per-(step, input) action -- the loss to minimize.
        """
        x = traj[:-1]                     # (K, B, N) states before each step
        dx = traj[1:] - traj[:-1]         # (K, B, N) observed increments
        drift = self.dVdx(x, z)           # student's predicted -force/mu
        resid = dx + mu * drift * dt      # noise the student would have needed
        action = resid ** 2 / (4.0 * mu * kT * dt)
        return action.sum(dim=-1).mean()  # sum over neurons, mean over K and B

    def _rollout(self, z, M=256, tf=TF, dt=DT, beta=BETA, mu=MU, seed=None,
                 record_every=None):
        """Integrate the finite-temperature Langevin dynamics from x = 0.

        Returns the final state x(tf) of shape (B, M, N), or -- when
        record_every=k is given -- the states at t = 0, k dt, 2k dt, ...
        (final step always included) stacked as (T, B, M, N).
        """
        z = torch.atleast_1d(torch.as_tensor(z, dtype=self.b.dtype,
                                             device=self.b.device))
        B = z.shape[0]
        Jc = self.coupling()
        inp = self.input_field(z).reshape(B, 1, N)
        kT = 1.0 / beta
        noise_amp = (2.0 * mu * kT * dt) ** 0.5
        n_steps = int(round(tf / dt))
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.b.device).manual_seed(seed)
        x = torch.zeros(B, M, N, dtype=self.b.dtype, device=self.b.device)
        rec = [x] if record_every else None
        for step in range(1, n_steps + 1):
            dVdx = _intrinsic_grad(x) - self.b + x @ Jc + inp
            noise = torch.randn(x.shape, generator=gen, dtype=x.dtype,
                                device=x.device)
            x = x - mu * dVdx * dt + noise_amp * noise
            if record_every and (step % record_every == 0 or step == n_steps):
                rec.append(x)
        return torch.stack(rec, dim=0) if record_every else x

    def simulate(self, z, **kw):
        """Sample-averaged activations <x_i(tf)> of shape (B, N)."""
        return self._rollout(z, **kw).mean(dim=1)

    def mean_trajectory(self, z, M=200, record_every=10, **kw):
        """Sample-averaged trajectory <x(t)>, shape (T, B, N), plus the step
        indices (in units of dt) each row corresponds to."""
        traj = self._rollout(z, M=M, record_every=record_every, **kw).mean(dim=2)
        n_steps = traj.shape[0]
        # reconstruct recorded step indices: 0, k, 2k, ..., plus the final step
        tf = kw.get("tf", TF)
        dt = kw.get("dt", DT)
        total = int(round(tf / dt))
        steps = list(range(0, total + 1, record_every))
        if steps[-1] != total:
            steps.append(total)
        assert len(steps) == n_steps
        return torch.tensor(steps), traj

    def sample_outputs(self, z, **kw):
        """Per-sample readout y = c * x_out(tf), shape (B, M)."""
        x = self._rollout(z, **kw)
        return self.readout * x[:, :, OUTPUT_IDX].reshape(x.shape[0], -1)

    def predict(self, z, **kw):
        """Scalar prediction y = c * <x_out(tf)>."""
        mean_x = self.simulate(z, **kw)
        return self.readout * mean_x[:, OUTPUT_IDX].reshape(-1)

    @torch.no_grad()
    def fit_readout(self, z, y0, **kw):
        """Least-squares fit of the readout scale c to ground truth y0(z).

        Runs the stochastic dynamics once on the grid z and sets
        c = <x_out . y0> / <x_out . x_out>.  Returns the fitted c.
        """
        xout = self.simulate(z, **kw)[:, OUTPUT_IDX].reshape(-1)
        c = (xout @ y0) / (xout @ xout).clamp_min(1e-12)
        self.readout.copy_(c)
        return c.item()


def main():
    # Tiny self-test: build a trajectory from random targets and check that
    # (a) it relaxes to the targets, (b) the OM loss is differentiable for
    # every parameter, and (c) the coupling matrix has the right structure.
    torch.manual_seed(0)
    B = 5
    A = torch.tanh(torch.randn(B, N))          # plausible activation targets
    traj = idealized_trajectory(A, tf=1.0, dt=1e-3)
    relax_err = (traj[-1] - A).abs().max().item()
    print(f"idealized trajectory shape {tuple(traj.shape)}; "
          f"max |x(tf) - A| = {relax_err:.2e}")

    z = torch.linspace(0.0, 1.0, B)
    student = ThermoStudent(scale=1e-2)        # random init so grads flow
    Jc = student.coupling()
    assert torch.allclose(Jc, Jc.t()) and Jc.diagonal().abs().max() == 0.0
    loss = student.om_loss(traj, z)
    loss.backward()
    for name, p in student.named_parameters():
        print(f"  OM grad |d{name}| = {p.grad.norm().item():.3e}")
    print(f"OM loss {loss.item():.4e}")


if __name__ == "__main__":
    main()
