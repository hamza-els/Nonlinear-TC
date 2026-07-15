"""Experiment N: GA fine-tuning of the GD-trained thermodynamic computer.

The OM/GD method optimizes a proxy (mimicry of the tanh teacher's activations
along guide trajectories), not the task loss.  Here we warm-start an elitist
genetic algorithm at the GD solution theta_GD and let it minimize the ACTUAL
objective -- the MSE of the sampled readout against cos(2 pi z), optionally
plus a variance penalty -- through the full stochastic dynamics.  This is the
hybrid pipeline suggested in the conclusions of the GD paper.

Design choices:
  - Common random numbers: every candidate in a generation sees identical
    noise realizations, so selection compares parameters, not luck.  Fresh
    noise is drawn each generation to prevent overfitting one noise sample.
  - Per-candidate least-squares readout refit at every fitness evaluation
    (cheap, and removes 8 dimensions from the search).
  - Elitist selection with fan-in-scaled Gaussian mutations, as in the GA
    repo; mutation scale is small (fine-tuning, not exploration).
  - Elites survive unmutated, so the GD baseline can never be lost.

Usage:
    python experiments/experiment_ga_finetune.py [seed] [generations] [var_weight]
"""

import os
import sys
import time

# core modules (digital_net, thermo_student, train_gd) live one level up
_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import target, features, N, HIDDEN, N_OUT, N_IN
from thermo_student import ThermoStudent, BETA, MU, TF, DT
from train_gd import run, evaluate
# NOTE: plotting is imported lazily in main() -- matplotlib may not exist on
# headless cluster nodes, and the figure is optional (npz is saved first).

GDIR = os.path.join(_GD_ROOT, "..", "..", "Graphs", "gd_graphs")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
GENS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
VAR_WEIGHT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

P = 50          # population size
N_ELITE = 5
MUT = 2e-2      # base mutation scale, divided by sqrt(fan-in) per tensor
K = 256         # z-grid for fitness
M_FIT = 1000    # samples per (candidate, z) during fitness evaluation
# Common random numbers: share one noise draw across all P candidates per
# step, so selection compares parameters rather than luck.  Worth it when the
# fitness noise floor Var/M_FIT is comparable to the bias differences being
# selected on (e.g. M_FIT = 64); at M_FIT ~ 1000 the floor is far below the
# signal and independent noise (CRN = False) matches the original GA protocol.
CRN = True
# Samples are simulated in chunks of M_CHUNK to bound GPU memory (the state
# tensor is P x K x M_CHUNK x N); fitness statistics are accumulated exactly
# across chunks, so results are independent of the chunk size.
M_CHUNK = 500

FANIN = {"b": 1.0, "W": float(N_IN), "Jhh": float(HIDDEN),
         "Jho": (HIDDEN + N_OUT) / 2.0, "Joo": float(N_OUT)}


def sym0(J):
    """Symmetrize with zero diagonal, batched over leading dims."""
    J = 0.5 * (J + J.transpose(-1, -2))
    eye = torch.eye(J.shape[-1], device=J.device, dtype=J.dtype)
    return J * (1.0 - eye)


def population_fitness(pop, phi, y0, gen_seed, device):
    """Fitness of all P candidates with shared (common) noise.

    pop: dict of batched tensors (P, ...).  Returns (fitness (P,), bias (P,),
    var (P,)) where fitness = bias + VAR_WEIGHT * var.
    """
    Pn = pop["b"].shape[0]
    Jc = pop["b"].new_zeros(Pn, N, N)
    Jc[:, :HIDDEN, :HIDDEN] = sym0(pop["Jhh"])
    Jc[:, :HIDDEN, HIDDEN:] = pop["Jho"]
    Jc[:, HIDDEN:, :HIDDEN] = pop["Jho"].transpose(-1, -2)
    Jc[:, HIDDEN:, HIDDEN:] = sym0(pop["Joo"])

    ext = torch.einsum("ki,pin->pkn", phi, pop["W"])       # (P, K, N)
    b = pop["b"][:, None, None, :]                          # (P, 1, 1, N)
    ext = ext[:, :, None, :]                                # (P, K, 1, N)

    kT = 1.0 / BETA
    noise_amp = (2.0 * MU * kT * DT) ** 0.5
    n_steps = int(round(TF / DT))
    gen = torch.Generator(device=device).manual_seed(gen_seed)

    # accumulate exact first/second moments of the output nodes over m-chunks
    S1 = pop["b"].new_zeros(Pn, K, N_OUT)                   # sum of xout
    S2 = pop["b"].new_zeros(Pn, K, N_OUT, N_OUT)            # sum of outer prods
    m_done = 0
    while m_done < M_FIT:
        mc = min(M_CHUNK, M_FIT - m_done)
        x = pop["b"].new_zeros(Pn, K, mc, N)
        for _ in range(n_steps):
            dV = (2.0 * x + 4.0 * x ** 3 - b
                  + torch.einsum("pkmn,pnj->pkmj", x, Jc) + ext)
            # CRN=True: one draw shared by every candidate (broadcast over P);
            # CRN=False: independent noise per candidate, as in the original GA.
            shape = (K, mc, N) if CRN else (Pn, K, mc, N)
            eta = torch.randn(shape, generator=gen, device=device)
            x = x - MU * dV * DT + noise_amp * eta
        xout = x[..., HIDDEN:]                              # (P, K, mc, 8)
        S1 += xout.sum(dim=2)
        S2 += torch.einsum("pkmo,pkmq->pkoq", xout, xout)
        m_done += mc

    Xmean = S1 / M_FIT                                      # (P, K, 8)
    B = y0.reshape(1, K, 1).expand(Pn, K, 1)
    w = torch.linalg.lstsq(Xmean, B).solution               # (P, 8, 1)
    pred = (Xmean @ w).squeeze(-1)                          # (P, K)
    bias = ((pred - y0) ** 2).mean(dim=1)                   # (P,)
    Cov = S2 / M_FIT - Xmean[..., :, None] * Xmean[..., None, :]
    wv = w.squeeze(-1)                                      # (P, 8)
    var = torch.einsum("pkoq,po,pq->pk", Cov, wv, wv).mean(dim=1)
    return bias + VAR_WEIGHT * var, bias, var


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"config: seed={SEED} gens={GENS} var_weight={VAR_WEIGHT:g} | "
          f"P={P} elites={N_ELITE} K={K} M_FIT={M_FIT} M_CHUNK={M_CHUNK} "
          f"CRN={CRN} MUT={MUT:g} | tf={TF:g} beta={BETA:g} dt={DT:g}",
          flush=True)

    # --- GD warm start -------------------------------------------------------
    torch.manual_seed(SEED)
    student, teacher, stats = run(seed=SEED)
    print(f"GD baseline (seed {SEED}): rmse {stats['rmse']:.4f}  "
          f"bias2 {stats['bias2']:.2e}  var {stats['var']:.3f}")

    theta0 = {"b": student.b.detach(), "W": student.W.detach(),
              "Jhh": sym0(student.Jhh_raw.detach()),
              "Jho": student.Jho.detach(),
              "Joo": sym0(student.Joo_raw.detach())}

    z = torch.linspace(0.0, 1.0, K, device=device)
    phi = features(z)                                       # (K, N_IN)
    y0 = target(z)

    # --- initial population: elites = exact GD copies ------------------------
    g = torch.Generator(device=device).manual_seed(SEED + 1000)

    def mutate(base):
        out = {}
        for k, v in base.items():
            sig = MUT / np.sqrt(FANIN[k])
            out[k] = v + sig * torch.randn(v.shape, generator=g, device=device)
        return out

    def stack(cands):
        return {k: torch.stack([c[k] for c in cands]) for k in theta0}

    cands = [dict(theta0) for _ in range(N_ELITE)] + \
            [mutate(theta0) for _ in range(P - N_ELITE)]
    pop = stack(cands)

    history = []
    t_last, n_last = time.time(), 0
    for n in range(GENS):
        fit, bias, var = population_fitness(pop, phi, y0, SEED + 2000 + n,
                                            device)
        order = torch.argsort(fit)
        best = order[0].item()
        history.append((fit[best].item(), bias[best].item(), var[best].item()))
        if n % 10 == 0 or n == GENS - 1:
            now = time.time()
            spg = (now - t_last) / max(1, n - n_last)
            t_last, n_last = now, n
            print(f"  gen {n:4d}  fitness {history[-1][0]:.3e}  "
                  f"bias {history[-1][1]:.3e}  var {history[-1][2]:.3f}  "
                  f"{spg:7.2f} s/gen  ({now - t0:.0f}s total)", flush=True)
        elites = [{k: pop[k][i].clone() for k in pop}
                  for i in order[:N_ELITE].tolist()]
        cands = [dict(e) for e in elites] + \
                [mutate(elites[i % N_ELITE]) for i in range(P - N_ELITE)]
        pop = stack(cands)

    # --- independent evaluation of the GA champion ---------------------------
    champ = elites[0]
    tuned = ThermoStudent().to(device)
    with torch.no_grad():
        tuned.b.copy_(champ["b"])
        tuned.W.copy_(champ["W"])
        tuned.Jhh_raw.copy_(champ["Jhh"])   # already symmetric, sym0 is no-op
        tuned.Jho.copy_(champ["Jho"])
        tuned.Joo_raw.copy_(champ["Joo"])
    tuned.fit_readout(z, y0, M=256, seed=SEED + 1)
    ev = evaluate(tuned, M=256, seed=SEED + 7, device=device)
    print(f"\nGA-finetuned:  rmse {ev['rmse']:.4f}  bias2 {ev['bias2']:.2e}  "
          f"var {ev['var']:.3f}")
    print(f"GD baseline:   rmse {stats['rmse']:.4f}  "
          f"bias2 {stats['bias2']:.2e}  var {stats['var']:.3f}")

    runs_dir = os.path.join(_GD_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    tag = (f"seed{SEED}_g{GENS}_vw{VAR_WEIGHT:g}_M{M_FIT}_"
           f"{'crn' if CRN else 'nocrn'}")
    np.savez(os.path.join(runs_dir, f"run_ga_finetune_{tag}.npz"),
             **{f"p_{k}": v.cpu().numpy() for k, v in champ.items()},
             history=np.asarray(history), seed=SEED, gens=GENS,
             var_weight=VAR_WEIGHT, tf=TF, beta=BETA,
             crn=CRN, P=P, n_elite=N_ELITE, K=K, m_fit=M_FIT,
             m_chunk=M_CHUNK, mut=MUT)
    try:
        from plots.plot_output_samples import plot_output
        plot_output(tuned, teacher,
                    out_path=f"{GDIR}/fig_teacher_gafinetune_{tag}.png",
                    suptitle=f"N: GA fine-tune of GD solution, seed {SEED}, "
                             f"{GENS} gens, var_weight={VAR_WEIGHT:g} "
                             f"(GD {stats['rmse']:.3f} -> GA {ev['rmse']:.3f})")
    except Exception as e:
        print(f"figure skipped ({type(e).__name__}: {e}); "
              f"regenerate locally from the saved npz")


if __name__ == "__main__":
    main()
