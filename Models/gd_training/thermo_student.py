"""Thermodynamic-computer student, trained by gradient descent.

Implements the "student" of Whitelam, arXiv:2509.15324 (Sec. II), specialized
to the cosine-regression architecture:

    - HIDDEN = 32 hidden neurons, all-to-all coupled among themselves (Jhh),
    - the input enters as N_IN frozen feature channels phi(z) = (z, z^2, z^3),
      each coupled to EVERY neuron (hidden and output) via trainable W_k,i --
      the paper (Sec. III) likewise couples its input data (784 pixels) to
      both the hidden and the output nodes; multiple channels give the
      computer a rank-N_IN, nonlinear-in-z field family from t = 0,
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

from digital_net import N, HIDDEN, N_OUT, N_IN, features

# --- Physics constants -----------------------------------------------------
# "Cold" units: J2 = J4 = 1 with kT = 0.1 (beta = 10, matching ga_training).
# This runs the computer 10x colder relative to its well depth than the
# paper's J2 = J4 = kBT convention: equilibrium fluctuations sigma^2 drop from
# ~0.4 to ~0.05, taming both single-shot readout noise and the <x^3>
# noise-rectification that pulls the mean dynamics off the T=0 ideal path
# (stiffening correction 12 sigma^2: ~250% at kT=1 -> ~30% here).
J2, J4 = 1.0, 1.0
BETA = 10.0
KT = 1.0 / BETA
MU = 1.0
# Observation time (units of 1/mu). tf = 0.40 is the location of the
# secondary phi(tf) valley of the tf=1-trained champion (fine-scanned
# 2026-07-14: its transient crosses the answer at 0.40 with phi ~ 9e-4)
# -- testing whether training AT that crossing time works outright.
# History: tf = 1.0 was the long-standing default (~2 relaxation times);
# the paper's tf = 0.2 fails for hidden-mediated scalar-input architectures
# but became near-viable with polynomial input channels.
TF = 0.40
DT = 1e-3

# The output neuron is the last degree of freedom; its activation is the
# prediction.  Hidden neurons occupy indices [0, HIDDEN).
OUTPUT_IDX = torch.arange(HIDDEN, N)


def _intrinsic_grad(x):
    """Intrinsic part of dV/dx_i: 2 J2 x + 4 J4 x^3."""
    return 2.0 * J2 * x + 4.0 * J4 * x ** 3


@torch.no_grad()
def idealized_trajectory(A, tf=TF, dt=DT, mu=MU, beta=None, M=1, seed=None):
    """Build "teacher #2": the idealized trajectory whose activations relax to A.

    Uses a NONINTERACTING (Jij = 0) computer whose guide biases are chosen so
    the T=0 fixed point equals A_i exactly.  The steady state of
    x_dot = -mu(2 J2 x + 4 J4 x^3 - b0) satisfies 2 J2 x + 4 J4 x^3 = b0, so
    setting  b0_i = 2 J2 A_i + 4 J4 A_i^3  makes x_i relax to exactly A_i.
    (The paper sets b0 merely proportional to A and accepts the nonlinearity;
    the exact inverse is a small improvement that helps the regression task.)

    beta=None (default): zero-temperature, deterministic guide -- returns
    (n_steps + 1, B, N), starting from x = 0.

    beta=<float>, M=<int>: NOISY guide -- the same biased noninteracting
    computer simulated at temperature 1/beta, M independent realizations per
    input.  Returns (n_steps + 1, B, M, N).  Training on noisy guides breaks
    the OM null-space degeneracy (noise explores state space; see
    test_om_recovery check 4) and bakes the <x^3> noise-rectification of the
    finite-T mean dynamics into the fit.

    A : (B, N) target activations, one row per input.
    """
    b0 = 2.0 * J2 * A + 4.0 * J4 * A ** 3          # (B, N)
    n_steps = int(round(tf / dt))
    if beta is None:
        x = torch.zeros_like(A)
        noise_amp = 0.0
        gen = None
    else:
        b0 = b0[:, None, :]                        # (B, 1, N) broadcast over M
        x = A.new_zeros(A.shape[0], M, A.shape[1])
        noise_amp = (2.0 * mu * (1.0 / beta) * dt) ** 0.5
        gen = None
        if seed is not None:
            gen = torch.Generator(device=A.device).manual_seed(seed)
    traj = [x]
    for _ in range(n_steps):
        drift = -(_intrinsic_grad(x) - b0)         # noninteracting
        x = x + mu * drift * dt
        if noise_amp:
            x = x + noise_amp * torch.randn(x.shape, generator=gen,
                                            dtype=x.dtype, device=x.device)
        traj.append(x)
    return torch.stack(traj, dim=0)


class ThermoStudent(nn.Module):
    """Interacting thermodynamic computer, theta = {b, W, Jhh, Jho}.

    Connectivity (see module docstring): z -> hidden (W), hidden <-> hidden
    all-to-all (Jhh, symmetric, zero diagonal), hidden <-> output (Jho).
    All couplings are bilinear and bidirectional, as in the paper.
    """

    def __init__(self, scale=0.0, seed=0, oo_couplings=True):
        # Paper: training starts from theta = 0 (scale=0.0, the default).
        # Pass scale > 0 for a small random init instead.
        # oo_couplings=False removes the output <-> output couplings (they
        # exist in the paper's published code; flag for ablation).
        super().__init__()
        self.oo_couplings = oo_couplings
        g = torch.Generator().manual_seed(seed)

        def init(*shape):
            if scale:
                return scale * torch.randn(*shape, generator=g)
            return torch.zeros(*shape)

        self.b = nn.Parameter(init(N))              # biases, all nodes
        # Input couplings: one row per frozen input channel phi_k(z), one
        # column per node (hidden and output), as in the paper's pixel bonds.
        self.W = nn.Parameter(init(N_IN, N))
        self.Jhh_raw = nn.Parameter(init(HIDDEN, HIDDEN))  # hidden <-> hidden
        self.Jho = nn.Parameter(init(HIDDEN, N_OUT))       # hidden <-> output
        # Output <-> output couplings: present in the paper's published code
        # (thermo_gd set_interactions has an oo bond class).
        self.Joo_raw = nn.Parameter(init(N_OUT, N_OUT))
        # Post-hoc readout weights: y = <x_out(tf)> . w  (N_OUT-vector).  Not
        # thermodynamic parameters (outside the dynamics); fit_readout() sets
        # them by least squares against the ground truth.
        self.register_buffer("readout", torch.ones(N_OUT) / N_OUT)

    @staticmethod
    def _sym0(Jraw):
        J = 0.5 * (Jraw + Jraw.t())
        return J - torch.diag(torch.diag(J))

    def coupling(self):
        """Assemble the symmetric N x N coupling matrix with zero diagonal."""
        Jc = self.b.new_zeros(N, N)
        Jc[:HIDDEN, :HIDDEN] = self._sym0(self.Jhh_raw)
        Jc[:HIDDEN, HIDDEN:] = self.Jho
        Jc[HIDDEN:, :HIDDEN] = self.Jho.t()
        if self.oo_couplings:
            Jc[HIDDEN:, HIDDEN:] = self._sym0(self.Joo_raw)
        return Jc

    def input_field(self, z):
        """(B, N) external field sum_k W_k,i phi_k(z) on every node."""
        return features(z.reshape(-1)) @ self.W

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
        """Per-sample readout y = x_out(tf) . w, shape (B, M)."""
        x = self._rollout(z, **kw)
        return x[:, :, OUTPUT_IDX] @ self.readout

    def predict(self, z, **kw):
        """Scalar prediction y = <x_out(tf)> . w."""
        mean_x = self.simulate(z, **kw)
        return mean_x[:, OUTPUT_IDX] @ self.readout

    @torch.no_grad()
    def fit_readout(self, z, y0, **kw):
        """Least-squares fit of the readout weights w to ground truth y0(z).

        Runs the stochastic dynamics once on the grid z and solves
        min_w ||Xout w - y0||^2 for the (N_OUT,) weight vector.  Returns
        ||w|| as a scalar summary (logged as 'c' in run stats).
        """
        Xout = self.simulate(z, **kw)[:, OUTPUT_IDX]          # (B, N_OUT)
        w = torch.linalg.lstsq(Xout, y0.reshape(-1, 1)).solution.reshape(-1)
        self.readout.copy_(w)
        return w.norm().item()


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
