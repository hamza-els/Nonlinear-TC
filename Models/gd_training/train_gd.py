"""End-to-end gradient-descent training of the thermodynamic computer.

Pipeline (Whitelam, arXiv:2509.15324, Fig. 1) for the cosine task:

  1. Train the digital deterministic teacher net to fit cos(2 pi z).
  2. For a batch of inputs z, build the target activations A(z): the teacher's
     32 HIDDEN activations for the hidden nodes, and the GROUND-TRUTH output
     cos(2 pi z) for the single output node.  Build the idealized (teacher #2)
     trajectories that relax to these targets.
  3. Train the interacting thermodynamic student by gradient descent, minimizing
     the Onsager-Machlup action so the student would generate those trajectories.
  4. Fit the post-hoc readout scale c, then evaluate: run the stochastic
     student forward and compare y = c * <x_out(tf)> to cos(2 pi z).

Defaults here are deliberately small for a quick local sanity check; scale up
K / epochs (and run on a GPU) for a real fit -- see run_config() notes.
"""

import os
import time
from datetime import datetime

import numpy as np
import torch

from digital_net import train_teacher, target, N, HIDDEN
from thermo_student import ThermoStudent, idealized_trajectory, TF, DT

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "runs", "weights")


def save_student(student, label, meta=None):
    """Persist all student tensors (incl. readout) with a provenance label.

    Files land in runs/weights/ as <timestamp>_<label>.pt so successive
    iterations of the same experiment never overwrite each other.  meta is an
    arbitrary dict (config, stats) stored alongside.  Returns the path.
    """
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(WEIGHTS_DIR, f"{stamp}_{label}.pt")
    torch.save({"state_dict": {k: v.cpu() for k, v in
                               student.state_dict().items()},
                "meta": meta or {}}, path)
    return path


def load_student(path, device="cpu"):
    """Rebuild a ThermoStudent from a save_student() file: (student, meta)."""
    d = torch.load(path, map_location=device)
    student = ThermoStudent().to(device)
    student.load_state_dict(d["state_dict"])
    return student, d.get("meta", {})


def student_targets(teacher, z, act_scale=1.0):
    # NOTE: default must match run()'s act_scale default -- plot scripts and
    # calibration call this bare and must see the same targets training used.
    """Target activations A (B, N): the teacher's own activations, all nodes.

    With N_OUT = 8 tanh output neurons the student mimics the teacher's
    output-layer activations directly (like the paper's MNIST setup); the
    ground truth enters only through the linear readout of the 8 output
    nodes, fitted post-hoc.  Targets are scaled by act_scale.
    """
    with torch.no_grad():
        _, A_all = teacher.activations(z)
    return act_scale * A_all


def train_student(A, z, rounds=30, rel_tol=1e-5, tf=TF, dt=DT,
                  guide_beta=None, guide_M=4, oo_couplings=True,
                  seed=0, device="cpu", verbose=True):
    """Fit a ThermoStudent to reproduce the idealized trajectories for (A, z).

    Because the trajectory is fixed, the OM action is an (ill-conditioned)
    convex quadratic in theta, so full-batch LBFGS converges in a handful of
    rounds where first-order Adam stalls for thousands of epochs.  Rounds stop
    early once the relative per-round improvement drops below rel_tol.

    guide_beta=<float> switches to NOISY guide trajectories (guide_M
    realizations per input at temperature 1/guide_beta); the OM loss then has
    a statistical floor of N/2, so pass rel_tol=0 to disable early stopping.

    A : (B, N) target activations; z : (B,) inputs.  Returns (student, history).
    """
    A = A.to(device)
    z = z.to(device)
    if guide_beta is None:
        traj = idealized_trajectory(A, tf=tf, dt=dt)  # (K+1, B, N), no grad
    else:
        traj = idealized_trajectory(A, tf=tf, dt=dt, beta=guide_beta,
                                    M=guide_M, seed=seed + 100)
        T_, B_, M_, N_ = traj.shape
        traj = traj.reshape(T_, B_ * M_, N_)          # fold realizations
        z = z.repeat_interleave(M_)                   # matching inputs
    student = ThermoStudent(seed=seed, oo_couplings=oo_couplings).to(device)
    opt = torch.optim.LBFGS(student.parameters(), max_iter=40, history_size=30,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = student.om_loss(traj, z, dt=dt)
        loss.backward()
        return loss

    history = []
    for r in range(rounds):
        loss = opt.step(closure)
        history.append(loss.item())
        if verbose:
            print(f"  student LBFGS round {r}  OM loss {loss.item():.4e}")
        if len(history) > 1 and abs(history[-2] - history[-1]) \
                < rel_tol * abs(history[-2]):
            if verbose:
                print(f"  converged (relative improvement < {rel_tol:g})")
            break
    return student, history


def evaluate(student, K=250, M=256, seed=0, device="cpu", tf=TF):
    """Evaluate the stochastic student on a z-grid with one rollout.

    Returns a dict of stats: rmse (of the M-sample-averaged prediction),
    bias2 = mean_z (E[y] - y0)^2, var = mean_z Var[y] (per-sample readout
    variance), pred range, plus the raw curves (z, pred, y0).
    """
    z = torch.linspace(0.0, 1.0, K, device=device)
    with torch.no_grad():
        Y = student.sample_outputs(z, M=M, seed=seed, tf=tf)  # (K, M)
    pred = Y.mean(dim=1)
    y0 = target(z)
    return {
        "rmse": torch.sqrt(torch.mean((pred - y0) ** 2)).item(),
        "bias2": torch.mean((pred - y0) ** 2).item(),
        "var": Y.var(dim=1).mean().item(),
        "pred_min": pred.min().item(),
        "pred_max": pred.max().item(),
        "z": z.cpu(), "pred": pred.cpu(), "y0": y0.cpu(),
    }


def run(teacher_epochs=3000, K=250, student_rounds=30, act_scale=1.0,
        teacher_wd=0.0, teacher_act_reg=0.0, teacher_sat_reg=0.0,
        teacher_dropout=0.0, tf=TF, oo_couplings=True,
        eval_M=256, device=None, save_path=None, seed=0):
    """Full teacher -> student pipeline. Returns (student, teacher, stats)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[1/3] training digital teacher ...  (device: {device})")
    teacher = train_teacher(epochs=teacher_epochs, K=max(K, 256),
                            weight_decay=teacher_wd, act_reg=teacher_act_reg,
                            sat_reg=teacher_sat_reg, dropout=teacher_dropout,
                            seed=seed, device=device)
    z = torch.linspace(0.0, 1.0, K, device=device)
    A = student_targets(teacher, z, act_scale=act_scale)
    tmse = torch.mean((teacher(z) - target(z)) ** 2).item()
    t_teacher = time.time() - t0

    print("[2/3] training thermodynamic student by gradient descent ...")
    t1 = time.time()
    student, history = train_student(A, z, rounds=student_rounds,
                                     tf=tf, oo_couplings=oo_couplings,
                                     seed=seed, device=device)
    t_student = time.time() - t1

    print("[3/3] fitting readout scale (post-training) and evaluating ...")
    t2 = time.time()
    # c is fitted AFTER training, from a stochastic run of the frozen student;
    # it never enters the OM training itself.
    c = student.fit_readout(z, target(z), M=eval_M, seed=seed + 1, tf=tf)
    ev = evaluate(student, M=eval_M, seed=seed, device=device, tf=tf)
    t_eval = time.time() - t2

    n_params = sum(p.numel() for p in student.parameters())
    stats = {
        "act_scale": act_scale,
        "teacher_mse": tmse, "om_first": history[0], "om_final": history[-1],
        "readout_c": c, "rmse": ev["rmse"], "bias2": ev["bias2"],
        "var": ev["var"], "pred_min": ev["pred_min"], "pred_max": ev["pred_max"],
        "n_params": n_params, "t_teacher": t_teacher, "t_student": t_student,
        "t_eval": t_eval,
    }

    print("\n--- run summary " + "-" * 44)
    print(f"teacher fit mse     {tmse:.3e}            ({t_teacher:.1f}s)")
    print(f"OM loss             {history[0]:.3e} -> {history[-1]:.3e}  "
          f"({len(history)} LBFGS rounds, {t_student:.1f}s)")
    print(f"readout scale c     {c:.4f}")
    print(f"RMSE vs cos         {ev['rmse']:.3e}            "
          f"(M={eval_M}, {t_eval:.1f}s)")
    print(f"bias^2              {ev['bias2']:.3e}   mean_z (E[y]-y0)^2")
    print(f"Var[y]              {ev['var']:.3e}   mean_z per-sample variance")
    print(f"pred range          [{ev['pred_min']:+.3f}, {ev['pred_max']:+.3f}]"
          f"   (target [-1, +1])")
    print(f"trainable params    {n_params}")
    oo_tag = "" if oo_couplings else "_nooo"
    wpath = save_student(student, f"run_seed{seed}_tf{tf:g}_K{K}{oo_tag}",
                         meta={**stats, "tf": tf, "K": K, "seed": seed,
                               "oo_couplings": oo_couplings})
    print(f"weights saved       {os.path.basename(wpath)}")
    print("-" * 60)

    if save_path:
        np.savez(save_path,
                 z=ev["z"].numpy(), pred=ev["pred"].numpy(),
                 y0=ev["y0"].numpy(), history=np.asarray(history),
                 **{k: v for k, v in stats.items()})
        print(f"saved run to {save_path}")
    return student, teacher, stats


if __name__ == "__main__":
    # End-to-end run; ~1-2 min on CPU with the defaults below. Raise K,
    # teacher_epochs, student_rounds and eval_M for a higher-quality fit.
    torch.manual_seed(0)
    run(teacher_epochs=3000, K=250, eval_M=256)
