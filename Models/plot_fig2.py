"""Reproduce Fig. 2(c)/(d) for the nonlinear thermodynamic computer, plus a
third panel for the var-free (bias) loss.

(c) combined loss phi vs evolutionary time n, log-scale.
(d) trained output y(z) vs input z (recomputed from the saved weights, averaged
    over M samples), against the target cos(2 pi z).
(e) pure bias loss (mean_z bias^2) vs n, log-scale. With var_weight > 0 the
    combined loss (c) is dominated by the variance penalty, so (e) is what shows
    how well the computer actually fits the target. Requires a run trained with
    bias_history logging; older runs show a placeholder here.
(f) per-sample output variance (mean_z Var[y]) vs n. Uses the run's recorded
    "var_history" when present (works for any var_weight, including 0). Older
    runs without it fall back to the derivation from the two logged histories:
    for the "squared" loss, phi = bias^2 + var_weight*Var, so Var =
    (loss_history - bias_history) / var_weight -- which needs var_weight > 0;
    otherwise a placeholder is shown.

Usage:
    python plot_fig2.py                 # regenerate all three runs into
                                        #   ../Graphs/computer_graphs/
    python plot_fig2.py runs/run.npz    # a single run -> fig2_cdef.png
"""

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from genetic_algorithm import load_run, output_curve


def plot(run_path, out_path="fig2_cdef.png", M_eval=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    params, history, cfg = load_run(run_path)
    raw = np.load(run_path)
    bias_history = raw["bias_history"] if "bias_history" in raw.files else None
    history = np.asarray(history)
    n = np.arange(len(history))

    fig, (axc, axd, axe, axf) = plt.subplots(1, 4, figsize=(20, 4))

    # --- (c) combined loss vs evolutionary time --------------------------
    axc.plot(n, history, color="tab:green", lw=1.0)
    axc.set_yscale("log")
    axc.set_xlabel("n")
    axc.set_ylabel(r"$\phi$")
    axc.set_title("(c) loss")

    # --- (d) trained output y(z) vs target -------------------------------
    z = np.linspace(0.0, 1.0, 250)
    axd.plot(z, np.cos(2.0 * np.pi * z), "k-", lw=1.5, label="target")
    m = int(M_eval or cfg["M"])
    y = output_curve(
        params, z, M=m, m_chunk=int(cfg["m_chunk"]),
        tf=float(cfg["tf"]), dt=float(cfg["dt"]),
        beta=float(cfg["beta"]), mu=float(cfg["mu"]), device=device,
    )
    axd.plot(z, y, color="tab:green", lw=1.0, label="y(z)")
    axd.set_xlabel("z")
    axd.set_ylabel("y(z)")
    axd.set_ylim(-1.3, 1.3)
    axd.set_title("(d) output")
    axd.legend(frameon=False)

    # --- (e) pure bias loss vs evolutionary time -------------------------
    axe.set_xlabel("n")
    axe.set_ylabel(r"$\phi_{\mathrm{bias}}$")
    axe.set_title("(e) bias loss")
    if bias_history is not None:
        bias_history = np.asarray(bias_history)
        axe.plot(n, bias_history, color="tab:purple", lw=1.0)
        axe.set_yscale("log")
        # Share y-limits with (c) so the gap (the variance penalty) is legible.
        lo = float(min(np.nanmin(history), np.nanmin(bias_history)))
        hi = float(max(np.nanmax(history), np.nanmax(bias_history)))
        for ax in (axc, axe):
            ax.set_ylim(0.8 * lo, 1.2 * hi)
    else:
        axe.text(0.5, 0.5, "bias_history\nnot recorded",
                 ha="center", va="center", transform=axe.transAxes,
                 color="0.5")

    # --- (f) per-sample output variance vs evolutionary time -------------
    # Prefer the recorded var_history; older runs fall back to the identity
    # phi = bias^2 + var_weight*Var (loss_mode "squared"), i.e. the variance of
    # the selected-best computer is (loss - bias) / var_weight per generation.
    axf.set_xlabel("n")
    axf.set_ylabel(r"$\mathrm{Var}[y]$")
    axf.set_title("(f) output variance")
    vw = float(cfg.get("var_weight", 0.0))
    mode = str(cfg.get("loss_mode", "squared"))
    var_history = raw["var_history"] if "var_history" in raw.files else None
    if var_history is None and bias_history is not None and vw > 0 and mode == "squared":
        var_history = (history - bias_history) / vw
    if var_history is not None:
        axf.plot(n, np.clip(var_history, 1e-12, None), color="tab:orange", lw=1.0)
        axf.set_yscale("log")
    else:
        reason = ("var_weight = 0:\nvariance not in loss" if vw == 0
                  else "bias_history\nnot recorded" if bias_history is None
                  else f'loss_mode "{mode}":\nnot decomposable')
        axf.text(0.5, 0.5, reason, ha="center", va="center",
                 transform=axf.transAxes, color="0.5")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    if len(sys.argv) > 1:
        plot(sys.argv[1])
        return
    runs = [("runs/run_reg.npz", "regular"),
            ("runs/run_low_var.npz", "low_var"),
            ("runs/run_high_var.npz", "high_var")]
    for path, name in runs:
        plot(path, out_path=f"../Graphs/computer_graphs/fig2_{name}.png")


if __name__ == "__main__":
    main()
