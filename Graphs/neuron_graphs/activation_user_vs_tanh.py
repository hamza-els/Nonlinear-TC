"""Clean two-way comparison: the proposed activation vs plain tanh.

Fits both to the TRUE finite-time activation of an isolated thermodynamic
neuron at the operating point (beta=10, tf=0.40) and plots only those two,
with residuals.  See activation_fit.py for the full candidate bake-off.

Usage:
    python activation_user_vs_tanh.py   # -> activation_user_vs_tanh.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from activation_fit import (finite_time_activation, f_user, f_tanh,
                            I, BETA, TF)


def main():
    print(f"computing true activation (beta={BETA:g}, tf={TF:g}) ...")
    y = finite_time_activation(I)

    p_user, _ = curve_fit(f_user, I, y, p0=[1.0, 20.0, 1.0, 1.0], maxfev=200000)
    p_tanh, _ = curve_fit(f_tanh, I, y, p0=[1.0, 1.0], maxfev=200000)
    y_user, y_tanh = f_user(I, *p_user), f_tanh(I, *p_tanh)
    rms = lambda r: float(np.sqrt(np.mean(r ** 2)))
    r_user, r_tanh = rms(y_user - y), rms(y_tanh - y)

    lab_user = (rf"$\frac{{x^2}}{{x^2+{p_user[1]:.0f}}}\sqrt[3]{{x}}\cdot"
                rf"{p_user[0]:.3f} + {p_user[2]:.3f}\tanh({p_user[3]:.3f}x)$")
    lab_tanh = rf"${p_tanh[0]:.3f}\tanh({p_tanh[1]:.3f}x)$"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))

    a1.plot(I, y, "k-", lw=2.4, label="TRUE activation", zorder=3)
    a1.plot(I, y_user, color="tab:green", lw=1.6,
            label=f"proposed  (RMS {r_user:.4f})")
    a1.plot(I, y_tanh, color="tab:red", lw=1.6, ls="--",
            label=f"tanh  (RMS {r_tanh:.4f})")
    a1.set_xlabel("total input $I$")
    a1.set_ylabel(r"$\langle x(t_{\mathrm{f}})\rangle$")
    a1.set_title("activation function", fontsize=11)
    a1.legend(frameon=False, fontsize=10, loc="upper left")

    a2.axhline(0, color="k", lw=0.9, ls=":")
    a2.plot(I, y_user - y, color="tab:green", lw=1.4,
            label=f"proposed  (RMS {r_user:.4f}, max {np.abs(y_user-y).max():.3f})")
    a2.plot(I, y_tanh - y, color="tab:red", lw=1.4, ls="--",
            label=f"tanh  (RMS {r_tanh:.4f}, max {np.abs(y_tanh-y).max():.3f})")
    a2.set_xlabel("total input $I$")
    a2.set_ylabel("fit $-$ true")
    a2.set_title(f"residuals  ({r_tanh/r_user:.0f}x better)", fontsize=11)
    a2.legend(frameon=False, fontsize=9)

    fig.suptitle(f"proposed activation  {lab_user}   vs   tanh  {lab_tanh}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("activation_user_vs_tanh.png", dpi=150)
    print(f"proposed: RMS {r_user:.4f}  params {np.round(p_user, 4)}")
    print(f"tanh    : RMS {r_tanh:.4f}  params {np.round(p_tanh, 4)}")
    print("saved activation_user_vs_tanh.png")


if __name__ == "__main__":
    main()
