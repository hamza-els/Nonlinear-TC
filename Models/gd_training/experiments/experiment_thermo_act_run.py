"""Full graph set for a thermodynamic computer trained from a THERMO-activation
teacher.

Trains teachers whose neurons use the fitted thermodynamic-neuron activation

    sigma(x) = 0.5617 * (x^2/(x^2+13.53)) * cbrt(x) + tanh(x)

(the true finite-time neuron response at beta=10, tf=0.40, rescaled to the
teacher's input range), trains a student on each, picks the best seed, and
produces the standard three figures: output samples (M=1/20/100), node
trajectories vs their ideals, and phi vs observation time (as t/tf, so the
trained clock sits at 1).

Usage:
    python experiments/experiment_thermo_act_run.py [n_seeds]
"""

import os
import sys

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from digital_net import target, set_activation
from thermo_student import TF
from train_gd import run
from plots.plot_output_samples import plot_output
from plots.plot_trajectories_gd import plot_trajectories

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
M_SWEEP = 256


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_activation("thermo")
    print(f"config: activation=thermo tf={TF:g} seeds={N_SEEDS}", flush=True)

    # --- train a computer per seed, keep the best ---------------------------
    best = (None, None, None, 1e9)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        student, teacher, st = run(seed=seed)
        print(f">>> thermo seed={seed}  rmse={st['rmse']:.4f}  "
              f"bias2={st['bias2']:.2e}  var={st['var']:.3f}  "
              f"|w|={st['readout_c']:.2f}  tmse={st['teacher_mse']:.1e}",
              flush=True)
        if st["rmse"] < best[3]:
            best = (student, teacher, st, st["rmse"])
    student, teacher, st, _ = best
    seed = int(np.argmin([st["rmse"]]))  # for the title only
    print(f"\nbest: rmse {st['rmse']:.4f}  bias2 {st['bias2']:.2e}  "
          f"var {st['var']:.3f}", flush=True)

    title = (f"thermo-activation teacher, tf={TF:g} "
             f"(RMSE {st['rmse']:.4f}, K=250)")

    # --- 1. output samples (the MSE figure) ---------------------------------
    plot_output(student, teacher,
                out_path=f"{GDIR}/fig_teacher_thermoact.png", suptitle=title)

    # --- 2. trajectories vs ideals ------------------------------------------
    plot_trajectories(student, teacher,
                      out_path=f"{GDIR}/fig_traj_thermoact.png",
                      suptitle=title)

    # --- 3. phi vs t/tf ------------------------------------------------------
    z = torch.linspace(0.0, 1.0, 250, device=device)
    y0 = target(z)
    ratio = np.unique(np.round(np.concatenate(
        [np.logspace(-1, 1, 25), [1.0]]), 4))          # t/tf
    phi = []
    for r in ratio:
        with torch.no_grad():
            pred = student.sample_outputs(z, M=M_SWEEP, tf=float(r * TF),
                                          seed=0).mean(dim=1)
        phi.append(torch.mean((pred - y0) ** 2).item())
        print(f"  t/tf={r:6.3f}  phi={phi[-1]:.4e}", flush=True)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax in (axl, axr):
        ax.plot(ratio, phi, "-", color="tab:green", lw=1.3)
        ax.axvline(1.0, color="0.5", ls=":", lw=0.9)
        ax.set_yscale("log")
        ax.set_xlabel(r"$t/t_{\mathrm{f}}$")
    axl.set_xscale("log")
    axl.set_ylabel(r"$\phi$")
    axl.set_title("log time axis", fontsize=10)
    axr.set_title("linear time axis", fontsize=10)
    fig.suptitle(rf"thermo-activation computer: $\phi$ vs $t/t_f$ "
                 rf"(trained at $t_f$={TF:g}, M={M_SWEEP})", fontsize=11)
    fig.tight_layout()
    out = f"{GDIR}/fig_phi_vs_tf_thermoact.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")
    i = int(np.argmin(phi))
    print(f"phi minimum {phi[i]:.3e} at t/tf = {ratio[i]:.3f}")


if __name__ == "__main__":
    main()
