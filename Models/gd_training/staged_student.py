"""Staged thermodynamic computer: disconnect layers on a timed schedule.

Identical connectivity, parameters, teacher and guides to ThermoStudent, but
the coupling matrix is time-staged.  A kill_schedule is a list of
(time, lo, hi) events: at t = time, every coupling Jij touching a node in
[lo, hi) is set to zero for the rest of the run (that layer is electrically
disconnected; its input coupling W is left intact).  Kills are cumulative --
once a layer is dead it stays dead.  Sampling happens at tf as usual.

Examples
  single first-layer kill (the default, LAYER1 = first WIDTH hidden nodes):
      StagedStudent(kill_time=0.2)
  cascade of the first three layers of a 4x8 teacher:
      StagedStudent(kill_schedule=[(0.2,0,8),(0.4,8,16),(0.6,16,24)])

Because the guide trajectory is noninteracting, the targets are unchanged --
the student learns the same target activations under the staged dynamics (the
OM loss and the rollout both apply the time-dependent coupling).
"""

import torch

from digital_net import WIDTH
from thermo_student import (ThermoStudent, _intrinsic_grad, N, HIDDEN,
                            BETA, KT, MU, TF, DT)

# The first hidden layer occupies node indices [0, LAYER1); it is what the
# single-kill default disconnects.
LAYER1 = WIDTH


class StagedStudent(ThermoStudent):
    """ThermoStudent whose couplings switch off layer-by-layer on a schedule."""

    def __init__(self, scale=0.0, seed=0, oo_couplings=True,
                 kill_time=0.2, kill_schedule=None):
        super().__init__(scale=scale, seed=seed, oo_couplings=oo_couplings)
        # normalise to a sorted list of (time, lo, hi)
        self.kill_schedule = sorted(
            kill_schedule if kill_schedule is not None
            else [(kill_time, 0, LAYER1)], key=lambda e: e[0])

    def _phase_couplings(self, dt):
        """Return (start_steps, [Jc_phase, ...]).

        Phase p is active for steps in [start_steps[p], start_steps[p+1]);
        its coupling has the cumulatively-dead nodes' rows/cols zeroed.
        """
        Jc = self.coupling()
        times = sorted({t for t, _, _ in self.kill_schedule})
        start_steps = [0] + [int(round(t / dt)) for t in times]
        phases = [Jc]                                  # phase 0: nothing dead
        for t in times:
            # fresh mask each phase: all nodes killed at or before time t
            dead = torch.zeros(N, dtype=torch.bool, device=Jc.device)
            for et, lo, hi in self.kill_schedule:
                if et <= t:
                    dead[lo:hi] = True
            # out-of-place (masked_fill) so autograd never sees a mutated mask
            Jp = Jc.masked_fill(dead.view(N, 1), 0.0).masked_fill(
                dead.view(1, N), 0.0)
            phases.append(Jp)
        return start_steps, phases

    def om_loss(self, traj, z, dt=DT, mu=MU, kT=KT):
        x = traj[:-1]                     # (K, B, N)
        dx = traj[1:] - traj[:-1]
        K = x.shape[0]
        start, phases = self._phase_couplings(dt)
        coup = torch.empty_like(x)
        for p, Jp in enumerate(phases):
            lo = min(start[p], K)
            hi = min(start[p + 1], K) if p + 1 < len(start) else K
            if hi > lo:
                coup[lo:hi] = x[lo:hi] @ Jp
        drift = _intrinsic_grad(x) - self.b + coup + self.input_field(z)
        resid = dx + mu * drift * dt
        action = resid ** 2 / (4.0 * mu * kT * dt)
        return action.sum(dim=-1).mean()

    def _rollout(self, z, M=256, tf=TF, dt=DT, beta=BETA, mu=MU, seed=None,
                 record_every=None):
        z = torch.atleast_1d(torch.as_tensor(z, dtype=self.b.dtype,
                                             device=self.b.device))
        B = z.shape[0]
        start, phases = self._phase_couplings(dt)
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
            # coupling for the state BEFORE this step (time (step-1)*dt)
            k = step - 1
            phase = sum(1 for st in start[1:] if k >= st)
            Jc = phases[phase]
            dVdx = _intrinsic_grad(x) - self.b + x @ Jc + inp
            noise = torch.randn(x.shape, generator=gen, dtype=x.dtype,
                                device=x.device)
            x = x - mu * dVdx * dt + noise_amp * noise
            if record_every and (step % record_every == 0 or step == n_steps):
                rec.append(x)
        return torch.stack(rec, dim=0) if record_every else x


def main():
    torch.manual_seed(0)
    from thermo_student import idealized_trajectory
    B = 5
    A = torch.tanh(torch.randn(B, N))
    sched = [(0.2, 0, 8), (0.4, 8, 16), (0.6, 16, 24)]
    traj = idealized_trajectory(A, tf=0.8)
    s = StagedStudent(scale=1e-2, kill_schedule=sched)
    start, phases = s._phase_couplings(DT)
    dead_counts = [(P[:, :].abs().sum(1) == 0).sum().item() for P in phases]
    print(f"N={N} HIDDEN={HIDDEN}  schedule={sched}")
    print(f"phase start steps: {start}  (of {int(round(0.8/DT))})")
    print(f"fully-dead nodes per phase: {dead_counts}  (expect 0,8,16,24)")
    loss = s.om_loss(traj, torch.linspace(0, 1, B), dt=DT)
    loss.backward()
    print(f"staged OM loss {loss.item():.4e}; grad ok "
          f"(|dJhh|={s.Jhh_raw.grad.norm():.3e})")
    with torch.no_grad():
        out = s.simulate(torch.linspace(0, 1, B), M=8, tf=0.8, seed=0)
    print(f"staged rollout ok, output shape {tuple(out[:, HIDDEN:].shape)}")


if __name__ == "__main__":
    main()
