"""Large seed sweep: N seeds each of the tanh and thermo teacher activations.

Trains a fresh teacher+student for every (activation, seed) at the current
champion config (tf, K, beta, exact guides) and records a rich set of
per-seed metrics, so downstream analysis can test what predicts a good
student -- and whether the thermo activation really gives a tighter, safer
distribution than tanh (the n=10 hint: tanh worst 0.057 vs thermo worst 0.021).

Metrics per (activation, seed):
  rmse, bias2, var, readout_c   -- student accuracy / noise / readout gain
  teacher_mse                   -- how well the teacher fit the target
  max_abs_A, mean_abs_A         -- target activation magnitudes
  sat_frac                      -- fraction of (node, z) with |A| > 0.9
  node_dev_mean, node_dev_out   -- trackability: RMS(student mean - guide
                                   ideal) over (t, z), averaged over all nodes
                                   / for the output nodes only

Headless and cluster-ready: results checkpoint to runs/seed_sweep_results.npz
after every 5 seeds; no figures (analyse locally from the npz).

Usage:
    python experiments/experiment_seed_sweep.py [n_seeds]     # default 100
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import set_activation, HIDDEN
from thermo_student import idealized_trajectory, TF
from train_gd import run, student_targets

NPZ = os.path.join(_GD_ROOT, "runs", "seed_sweep_results.npz")
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ACTS = ["tanh", "thermo"]

FIELDS = ["rmse", "bias2", "var", "readout_c", "teacher_mse",
          "max_abs_A", "mean_abs_A", "sat_frac",
          "node_dev_mean", "node_dev_out"]


def node_deviation(student, teacher, device):
    """RMS(student mean - guide ideal) over (t, z): mean over all nodes, and
    over the output nodes only.  Same probe used in the activation A/B."""
    zg = torch.linspace(0.0, 1.0, 21, device=device)
    A = student_targets(teacher, zg)
    ideal = idealized_trajectory(A, tf=TF)
    with torch.no_grad():
        steps, mean = student.mean_trajectory(zg, M=100, record_every=20,
                                              seed=1, tf=TF)
    dev = ((mean - ideal[steps]) ** 2).mean(dim=(0, 1)).sqrt()   # (N,)
    return float(dev.mean()), float(dev[HIDDEN:].mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"config: n_seeds={N_SEEDS} acts={ACTS} tf={TF:g} device={device} "
          f"({N_SEEDS * len(ACTS)} trainings)", flush=True)

    data = {a: {f: np.full(N_SEEDS, np.nan) for f in FIELDS} for a in ACTS}

    for act in ACTS:
        set_activation(act)
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            student, teacher, st = run(seed=seed)
            z = torch.linspace(0.0, 1.0, 250, device=device)
            with torch.no_grad():
                _, A = teacher.activations(z)
            nd_mean, nd_out = node_deviation(student, teacher, device)
            row = {
                "rmse": st["rmse"], "bias2": st["bias2"], "var": st["var"],
                "readout_c": st["readout_c"], "teacher_mse": st["teacher_mse"],
                "max_abs_A": A.abs().max().item(),
                "mean_abs_A": A.abs().mean().item(),
                "sat_frac": (A.abs() > 0.9).float().mean().item(),
                "node_dev_mean": nd_mean, "node_dev_out": nd_out,
            }
            for f in FIELDS:
                data[act][f][seed] = row[f]
            print(f">>> {act:7s} seed={seed:3d}  rmse={st['rmse']:.4f}  "
                  f"var={st['var']:.3f}  ndev={nd_mean:.4f}  "
                  f"maxA={row['max_abs_A']:.2f}  ({time.time()-t0:.0f}s)",
                  flush=True)

            if (seed + 1) % 5 == 0 or seed == N_SEEDS - 1:
                os.makedirs(os.path.dirname(NPZ), exist_ok=True)
                save = {}
                for a in ACTS:
                    for f in FIELDS:
                        save[f"{a}_{f}"] = data[a][f]
                np.savez(NPZ, n_seeds=N_SEEDS, acts=np.array(ACTS), **save)

    print("\n=== summary (median [best, worst]) ===")
    for act in ACTS:
        r = data[act]["rmse"]
        r = r[~np.isnan(r)]
        print(f"  {act:7s} rmse: median {np.median(r):.4f}  "
              f"best {r.min():.4f}  worst {r.max():.4f}  "
              f"IQR {np.percentile(r,75)-np.percentile(r,25):.4f}  "
              f"var-median {np.nanmedian(data[act]['var']):.4f}")


if __name__ == "__main__":
    main()
