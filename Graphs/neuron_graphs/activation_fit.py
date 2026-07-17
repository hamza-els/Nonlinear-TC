"""The thermodynamic neuron's TRUE activation function, and fits to it.

Motivation: both papers note that "the effective activation function of the
thermodynamic computer is different in detail to the hyperbolic tangent
function of the neural network".  Our teacher uses tanh, so its activations
are targets the thermodynamic units cannot natively realize -- a built-in
teacher-student mismatch.  If we instead give the teacher an activation that
matches what a thermodynamic neuron actually does, its activations become
natively reachable.

An isolated neuron with U(x, I) = J2 x^2 + J4 x^4 - I x has:
  - small I:  x -> I / (2 J2)              (linear)
  - large I:  x -> (I / (4 J4))^(1/3)      (CUBE ROOT -- tanh saturates instead)

This script computes three "true" curves over a wide input range,

  1. finite-time  <x(tf)> at the operating point (beta=10, tf=0.40) -- the one
     that actually matters, since the computer is read out of equilibrium;
  2. equilibrium  <x>_0 at beta=10;
  3. T=0 fixed point (exact root of 4 J4 x^3 + 2 J2 x = I),

and least-squares fits several closed forms to curve (1).

Usage:
    python activation_fit.py            # -> activation_fit.png + report
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit

from neuron_dynamics import equilibrium_activation

J2, J4 = 1.0, 1.0
BETA, TF, DT, MU = 10.0, 0.40, 1e-3, 1.0
M = 20000                      # noise realizations per input
I_MAX = 40.0                   # "a little farther": x reaches ~2.1 here
I = np.linspace(-I_MAX, I_MAX, 321)


def finite_time_activation(I, tf=TF, beta=BETA, M=M):
    """<x(tf)> for an isolated neuron started at x=0, averaged over M runs."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Iv = torch.tensor(I, dtype=torch.float32, device=dev)[:, None]
    x = torch.zeros(len(I), M, device=dev)
    amp = (2.0 * MU * (1.0 / beta) * DT) ** 0.5
    g = torch.Generator(device=dev).manual_seed(0)
    for _ in range(int(round(tf / DT))):
        dU = 2.0 * J2 * x + 4.0 * J4 * x ** 3 - Iv
        x = x - MU * dU * DT + amp * torch.randn(x.shape, generator=g,
                                                 device=dev)
    return x.mean(dim=1).cpu().numpy()


def fixed_point(I):
    """Exact T=0 fixed point: real root of 4 J4 x^3 + 2 J2 x - I = 0 (Cardano)."""
    p = 2.0 * J2 / (4.0 * J4)          # x^3 + p x = q
    q = np.asarray(I, dtype=float) / (4.0 * J4)
    disc = np.sqrt(q ** 2 / 4.0 + p ** 3 / 27.0)
    return np.cbrt(q / 2.0 + disc) + np.cbrt(q / 2.0 - disc)


# --- candidate closed forms (all odd in x) ---------------------------------
def cbrt(x):
    return np.cbrt(x)


def f_tanh(x, a, b):
    """What the teacher uses now: saturating."""
    return a * np.tanh(b * x)


def f_user(x, a, c, b, d):
    """User's proposal: (x^2/(x^2+c)) * cbrt(x) blended with tanh."""
    return a * (x ** 2 / (x ** 2 + c)) * cbrt(x) + b * np.tanh(d * x)


def f_interp(x, a, c):
    """Physically-motivated interpolant: linear -> cube root.

    a*x / (1 + (|x|/c)^(2/3)):  ~a x at small |x|;  ~a c^(2/3) x^(1/3) at large.
    """
    return a * x / (1.0 + (np.abs(x) / c) ** (2.0 / 3.0))


def f_softplus_cbrt(x, a, c):
    """cbrt of a soft-thresholded argument: a * cbrt(x) * (x^2/(x^2+c))."""
    return a * cbrt(x) * (x ** 2 / (x ** 2 + c))


def f_cardano(x, a, s):
    """Exact T=0 fixed point with a free scale and input gain."""
    return a * fixed_point(s * x)


CANDIDATES = [
    ("tanh (current teacher)      a*tanh(b x)", f_tanh, [1.0, 1.0]),
    ("user: (x^2/(x^2+c))cbrt(x)+b tanh(d x)", f_user, [1.0, 20.0, 1.0, 1.0]),
    ("interp: a x/(1+(|x|/c)^(2/3))", f_interp, [0.5, 1.0]),
    ("cbrt gate: a cbrt(x) x^2/(x^2+c)", f_softplus_cbrt, [1.0, 1.0]),
    ("Cardano T=0 root: a*fp(s x)", f_cardano, [1.0, 1.0]),
]


def main():
    print(f"computing true activations (beta={BETA}, tf={TF}, M={M}) ...")
    y_ft = finite_time_activation(I)
    y_eq = equilibrium_activation((J2, 0.0, J4), I, beta=BETA)
    y_fp = fixed_point(I)
    print(f"  finite-time range: [{y_ft.min():.2f}, {y_ft.max():.2f}]  "
          f"at I=+-{I_MAX:g}")

    # --- fit each candidate to the finite-time curve ------------------------
    results = []
    for name, fn, p0 in CANDIDATES:
        try:
            popt, _ = curve_fit(fn, I, y_ft, p0=p0, maxfev=200000)
            resid = fn(I, *popt) - y_ft
            rms = float(np.sqrt(np.mean(resid ** 2)))
            mx = float(np.max(np.abs(resid)))
            results.append((rms, mx, name, fn, popt))
        except Exception as e:
            print(f"  fit failed for {name}: {e}")
    results.sort(key=lambda r: r[0])

    print("\n=== fits to the finite-time activation (beta=10, tf=0.40) ===")
    for rms, mx, name, _, popt in results:
        ps = ", ".join(f"{p:.4f}" for p in popt)
        print(f"  RMS {rms:.4f}  max {mx:.4f}   {name}\n"
              f"      params: [{ps}]")

    # --- figure --------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))

    a1.plot(I, y_ft, "k-", lw=2.0, label=f"TRUE finite-time (beta={BETA:g}, "
                                        f"tf={TF:g})")
    a1.plot(I, y_eq, color="0.55", ls="--", lw=1.2,
            label=f"equilibrium (beta={BETA:g})")
    a1.plot(I, y_fp, color="0.55", ls=":", lw=1.2, label="T=0 fixed point")
    for rms, mx, name, fn, popt in results[:3]:
        a1.plot(I, fn(I, *popt), lw=1.1, alpha=0.9,
                label=f"{name.split(':')[0]} (RMS {rms:.3f})")
    a1.set_xlabel("total input I"); a1.set_ylabel(r"$\langle x(t_f)\rangle$")
    a1.set_title("true activation vs fits", fontsize=11)
    a1.legend(frameon=False, fontsize=8)

    for rms, mx, name, fn, popt in results:
        a2.plot(I, fn(I, *popt) - y_ft, lw=1.0,
                label=f"{name.split(':')[0]} (RMS {rms:.3f})")
    a2.axhline(0, color="k", lw=0.8, ls=":")
    a2.set_xlabel("total input I"); a2.set_ylabel("fit - true")
    a2.set_title("residuals", fontsize=11)
    a2.legend(frameon=False, fontsize=8)

    fig.suptitle("thermodynamic neuron activation: linear at small I, "
                 "CUBE-ROOT at large I (tanh saturates)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("activation_fit.png", dpi=150)
    print("\nsaved activation_fit.png")


if __name__ == "__main__":
    main()
