"""Fit the thermo activation to the REAL isolated-neuron response at (beta, tf).

The "thermo" teacher activation is a fit to what a single isolated thermodynamic
neuron actually does at the observation time.  Instead of the inherited cosine-
work constants (fit at beta=10, tf=0.40), this probes the neuron at the MNIST
operating point (beta=1) and, crucially, AT EACH tf -- so the teacher's targets
reflect how relaxed the neuron is when read at that clock.

Probe: an isolated unit x (no couplings) under
    dx = -mu (2 J2 x + 4 J4 x^3 - I) dt + sqrt(2 mu kT dt) eta,   x(0)=0,
integrated to tf for a grid of frozen input fields I; the activation curve is
sigma_true(I) = <x(tf)>.  We fit the field-unit form
    sigma(I) = A * sign(I)|I|^(7/3)/(I^2 + C) + tanh(g I)
(the stable identity for A (I^2/(I^2+C)) cbrt(I)), then gain-normalize to the
teacher's unit-slope form sigma(x)=ACT_A (x^2/(x^2+ACT_C)) cbrt(x)+tanh(x) via
gain = 1/g:  ACT_A = A * gain^(1/3),  ACT_C = C / gain^2.

Usage:
    from thermo_activation_fit import fit_thermo_activation
    ACT_A, ACT_C, info = fit_thermo_activation(tf, beta=1.0, device="cuda")
"""

import numpy as np
import torch

J2, J4, MU = 1.0, 1.0, 1.0
DT = 1e-3


@torch.no_grad()
def probe_activation(tf, beta=1.0, I_max=40.0, n_I=161, M=4000, dt=DT,
                     seed=0, device="cpu"):
    """Isolated-neuron activation sigma_true(I)=<x(tf)>, shape (n_I,)."""
    I = torch.linspace(-I_max, I_max, n_I, device=device)
    n_steps = int(round(tf / dt))
    noise_amp = (2.0 * MU * (1.0 / beta) * dt) ** 0.5
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.zeros(n_I, M, device=device)
    Ib = I[:, None]
    for _ in range(n_steps):
        drift = 2.0 * J2 * x + 4.0 * J4 * x ** 3 - Ib
        x = x - MU * drift * dt + noise_amp * torch.randn(
            x.shape, generator=gen, device=device)
    return I, x.mean(dim=1)


def _sigma_field(I, A, C, g):
    return (A * torch.sign(I) * I.abs().pow(7.0 / 3.0) / (I ** 2 + C)
            + torch.tanh(g * I))


def fit_thermo_activation(tf, beta=1.0, I_max=40.0, n_I=161, M=4000,
                          steps=1500, seed=0, device=None):
    """Return (ACT_A, ACT_C, info) for the isolated neuron at (beta, tf)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    I, sig = probe_activation(tf, beta=beta, I_max=I_max, n_I=n_I, M=M,
                              seed=seed, device=device)
    # nonlinear LS fit of (A, C, g) via Adam; params kept positive by softplus
    rawA = torch.tensor(np.log(np.expm1(0.35)), device=device, requires_grad=True)
    rawC = torch.tensor(np.log(np.expm1(200.0)), device=device, requires_grad=True)
    rawg = torch.tensor(np.log(np.expm1(0.25)), device=device, requires_grad=True)
    opt = torch.optim.Adam([rawA, rawC, rawg], lr=0.05)
    sp = torch.nn.functional.softplus
    for _ in range(steps):
        opt.zero_grad()
        pred = _sigma_field(I, sp(rawA), sp(rawC), sp(rawg))
        loss = ((pred - sig) ** 2).mean()
        loss.backward(); opt.step()
    A, C, g = float(sp(rawA)), float(sp(rawC)), float(sp(rawg))
    gain = 1.0 / g
    ACT_A = A * gain ** (1.0 / 3.0)
    ACT_C = C / gain ** 2
    rms = float(((_sigma_field(I, sp(rawA), sp(rawC), sp(rawg)) - sig) ** 2)
                .mean().sqrt())
    info = {"A": A, "C": C, "g": g, "gain": gain, "rms": rms,
            "sigma_max": float(sig.abs().max()), "tf": tf, "beta": beta}
    return ACT_A, ACT_C, info


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"per-tf isolated-neuron activation fit (beta=1, device={dev})")
    print(f"{'tf':>6}{'ACT_A':>9}{'ACT_C':>9}{'gain':>8}{'|sig|max':>9}{'rmsFit':>9}")
    for tf in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5):
        A, C, info = fit_thermo_activation(tf, beta=1.0, device=dev)
        print(f"{tf:6.2f}{A:9.4f}{C:9.3f}{info['gain']:8.2f}"
              f"{info['sigma_max']:9.3f}{info['rms']:9.4f}")
    print("\nreference (cosine work, beta=10 tf=0.40): ACT_A=0.5617 ACT_C=13.53")
