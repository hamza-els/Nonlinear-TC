"""Pure GA from scratch on the gd-architecture thermodynamic computer.

Same physical computer as the GD/OM student (thermo_student.ThermoStudent):
  - 32 hidden + 8 output neurons, ALL-TO-ALL coupled (Jhh hidden-hidden, Jho
    hidden-output, Joo output-output -- the outputs are fully connected too),
  - 6 polynomial input channels phi(z) = (z, z^2, ..., z^6) driving every node,
  - native thermodynamic neuron (intrinsic force 2x + 4x^3),
  - 8-output linear readout, refit by least squares each fitness evaluation.
This is the WIDTH=8/DEPTH=4 shape (N = 40); run with TC_WIDTH=8 TC_DEPTH=4.

Unlike experiment_ga_finetune, there is NO teacher and NO gradient descent
anywhere: the population is initialized at RANDOM and an elitist GA minimizes
the ACTUAL task loss -- mean_z (cos(2 pi z) - y(z))^2 (+ var_weight * Var[y]) --
through the full stochastic dynamics.  The fitness is identical to the
fine-tune's (common random numbers, per-candidate LS readout, chunked sampling);
only the initialization differs.

Usage:
    python experiments/experiment_ga_scratch.py [gens] [var_weight] [seed] [tf]
"""

import os
import sys
import time

_GD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _GD_ROOT)

import numpy as np
import torch

from digital_net import target, features, N, HIDDEN, N_OUT, N_IN, WIDTH, DEPTH
from thermo_student import ThermoStudent, BETA, MU, DT
from train_gd import evaluate

GENS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
VAR_WEIGHT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
TF = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

# --- GA hyperparameters ----------------------------------------------------
P = 50           # population size
N_ELITE = 5
INIT_STD = 0.1   # base init std (variance 1e-2, as in the GA repo)
MUT = 0.1        # base mutation std; both are divided by sqrt(fan-in) per tensor
K = 250          # z-grid for fitness
M_FIT = 1000     # reset samples per (candidate, z) during fitness
M_CHUNK = 1000   # sample-chunk size (caps the P x K x M_CHUNK x N state tensor)
CRN = True       # common random numbers across candidates within a generation
CKPT_EVERY = 100

# Feedforward-ish fan-in per tensor, for scaling both init and mutation so no
# neuron's incoming field starts (or steps) O(fan-in) too large.
FANIN = {"b": 1.0, "W": float(N_IN), "Jhh": float(HIDDEN),
         "Jho": (HIDDEN + N_OUT) / 2.0, "Joo": float(N_OUT)}


def sym0(J):
    """Symmetrize (last two dims) and zero the diagonal; batched-safe."""
    Js = 0.5 * (J + J.transpose(-1, -2))
    return Js - torch.diag_embed(torch.diagonal(Js, dim1=-2, dim2=-1))


def population_fitness(pop, phi, y0, gen_seed, device):
    """Fitness of all P candidates (shared noise).  Returns (fitness, bias, var),
    each (P,); fitness = bias + VAR_WEIGHT * var.  Physics identical to the
    fine-tune's fitness: native 2x+4x^3 neuron, all-to-all coupling, 6-poly
    input field, per-candidate least-squares readout."""
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

    S1 = pop["b"].new_zeros(Pn, K, N_OUT)
    S2 = pop["b"].new_zeros(Pn, K, N_OUT, N_OUT)
    m_done = 0
    while m_done < M_FIT:
        mc = min(M_CHUNK, M_FIT - m_done)
        x = pop["b"].new_zeros(Pn, K, mc, N)
        for _ in range(n_steps):
            dV = (2.0 * x + 4.0 * x ** 3 - b
                  + torch.einsum("pkmn,pnj->pkmj", x, Jc) + ext)
            shape = (K, mc, N) if CRN else (Pn, K, mc, N)
            eta = torch.randn(shape, generator=gen, device=device)
            x = x - MU * dV * DT + noise_amp * eta
        xout = x[..., HIDDEN:]                              # (P, K, mc, 8)
        S1 += xout.sum(dim=2)
        S2 += torch.einsum("pkmo,pkmq->pkoq", xout, xout)
        m_done += mc

    Xmean = S1 / M_FIT                                      # (P, K, 8)
    Bt = y0.reshape(1, K, 1).expand(Pn, K, 1)
    w = torch.linalg.lstsq(Xmean, Bt).solution              # (P, 8, 1)
    pred = (Xmean @ w).squeeze(-1)                          # (P, K)
    bias = ((pred - y0) ** 2).mean(dim=1)                   # (P,)
    Cov = S2 / M_FIT - Xmean[..., :, None] * Xmean[..., None, :]
    wv = w.squeeze(-1)                                      # (P, 8)
    var = torch.einsum("pkoq,po,pq->pk", Cov, wv, wv).mean(dim=1)
    return bias + VAR_WEIGHT * var, bias, var


def init_population(device, gen):
    """P random computers as batched tensors (leading axis P), fan-in-scaled."""
    def r(*shape, fan):
        return (INIT_STD / np.sqrt(fan)) * torch.randn(
            P, *shape, generator=gen, device=device)
    return {"b":   r(N, fan=FANIN["b"]),
            "W":   r(N_IN, N, fan=FANIN["W"]),
            "Jhh": r(HIDDEN, HIDDEN, fan=FANIN["Jhh"]),
            "Jho": r(HIDDEN, N_OUT, fan=FANIN["Jho"]),
            "Joo": r(N_OUT, N_OUT, fan=FANIN["Joo"])}


def select_and_breed(pop, fit, mut_std, gen):
    """Elitist: keep the n_elite best unmutated, fill the rest with fan-in-scaled
    Gaussian mutants of the elite."""
    order = torch.argsort(fit)
    elite = order[:N_ELITE]
    n_child = P - N_ELITE
    per = -(-n_child // N_ELITE)
    parents = torch.cat([elite, elite.repeat_interleave(per)[:n_child]])
    new = {k: v[parents].clone() for k, v in pop.items()}
    ch = slice(N_ELITE, P)
    for k in new:
        new[k][ch] += mut_std[k] * torch.randn(
            new[k][ch].shape, generator=gen, device=fit.device)
    return new


def to_student(champ, device):
    """Rebuild a ThermoStudent from a champion param dict (as in the fine-tune)."""
    s = ThermoStudent().to(device)
    with torch.no_grad():
        s.b.copy_(champ["b"]); s.W.copy_(champ["W"])
        s.Jhh_raw.copy_(champ["Jhh"]); s.Jho.copy_(champ["Jho"])
        s.Joo_raw.copy_(champ["Joo"])
    return s


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert (WIDTH, DEPTH) == (8, 4), f"expected 4x8 (N=40), got {WIDTH}x{DEPTH}"
    t0 = time.time()
    print(f"config: SCRATCH GA on gd-arch  seed={SEED} gens={GENS} "
          f"var_weight={VAR_WEIGHT:g} tf={TF:g} | N={N} HIDDEN={HIDDEN} "
          f"N_OUT={N_OUT} (all-to-all, {N_IN}-poly input) | P={P} elites={N_ELITE} "
          f"K={K} M_FIT={M_FIT} CRN={CRN} INIT={INIT_STD} MUT={MUT} | "
          f"beta={BETA:g} dt={DT:g} device={device}", flush=True)

    z = torch.linspace(0.0, 1.0, K, device=device)
    phi = features(z)                                      # (K, N_IN)
    y0 = target(z)
    mut_std = {k: MUT / np.sqrt(FANIN[k]) for k in FANIN}
    g = torch.Generator(device=device).manual_seed(SEED + 1000)

    runs_dir = os.path.join(_GD_ROOT, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    tag = f"scratch_4x8_vw{VAR_WEIGHT:g}_tf{TF:g}_seed{SEED}_g{GENS}"
    result_path = os.path.join(runs_dir, f"run_ga_{tag}.npz")
    ckpt_path = os.path.join(runs_dir, f"ckpt_ga_{tag}.pt")

    def save_result(champ, history):
        s = to_student(champ, device)
        s.fit_readout(z, y0, M=256, seed=SEED + 1, tf=TF)
        ev = evaluate(s, M=256, seed=SEED + 7, device=device, tf=TF)
        np.savez(result_path,
                 **{f"p_{k}": v.detach().cpu().numpy() for k, v in champ.items()},
                 history=np.asarray(history), seed=SEED, gens=GENS,
                 var_weight=VAR_WEIGHT, tf=TF, beta=BETA, K=K, m_fit=M_FIT,
                 p=P, n_elite=N_ELITE, init_std=INIT_STD, mut=MUT, crn=CRN,
                 width=WIDTH, depth=DEPTH)
        return ev

    # --- init or resume ------------------------------------------------------
    pop = init_population(device, g)
    history = []
    start = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        pop = {k: v.to(device) for k, v in ck["pop"].items()}
        history = list(ck["history"]); start = int(ck["gen"])
        print(f"resuming from {ckpt_path}: gen {start}/{GENS}", flush=True)

    t_last, n_last = time.time(), start
    for n in range(start, GENS):
        fit, bias, var = population_fitness(pop, phi, y0, SEED + 2000 + n, device)
        # a candidate whose dynamics blew up (NaN/inf) must never win selection
        fit = torch.nan_to_num(fit, nan=float("inf"), posinf=float("inf"))
        order = torch.argsort(fit); best = order[0].item()
        history.append((fit[best].item(), bias[best].item(), var[best].item()))
        if n % 10 == 0 or n == GENS - 1:
            now = time.time()
            spg = (now - t_last) / max(1, n - n_last); t_last, n_last = now, n
            print(f"  gen {n:4d}  fitness {history[-1][0]:.4e}  "
                  f"bias {history[-1][1]:.4e}  var {history[-1][2]:.3f}  "
                  f"{spg:6.2f} s/gen  ({now - t0:.0f}s)", flush=True)
        pop = select_and_breed(pop, fit, mut_std, g)
        if (n + 1) % CKPT_EVERY == 0 and n + 1 < GENS:
            champ = {k: pop[k][0].clone() for k in pop}     # elite 0 = current best
            torch.save({"pop": {k: v.cpu() for k, v in pop.items()},
                        "gen": n + 1, "history": history}, ckpt_path)
            save_result(champ, history)
            print(f"  checkpoint @gen {n+1} -> {ckpt_path}", flush=True)

    champ = {k: pop[k][0].clone() for k in pop}             # best survives as elite 0
    ev = save_result(champ, history)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"\nGA-from-scratch done: rmse {ev['rmse']:.4f}  bias2 {ev['bias2']:.2e}  "
          f"var {ev['var']:.3f}  -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
