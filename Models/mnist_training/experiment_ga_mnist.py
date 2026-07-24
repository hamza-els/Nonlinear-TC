"""GA fine-tuning of the per-tf-refit MNIST thermodynamic computer.

OM-GD trains the computer on a PROXY (mimicry of the teacher's activations along
guide trajectories).  Here we warm-start an elitist genetic algorithm at that
solution and let it minimize the ACTUAL task objective through the full
stochastic dynamics:

    fitness = CE(<x_out(tf)>, label) + var_weight * mean Var[x_out]

i.e. cross-entropy on the mean output-node activations (the same 10 numbers the
computer's argmax prediction reads) plus an optional penalty on the per-sample
output variance -- the quieter the computer, the fewer reset samples it needs.

Self-contained: it refits the thermo activation to the real neuron at (beta=1,
tf), retrains the teacher with it, OM-GD trains the student (the warm start),
then runs the GA.  Fresh random training digits each generation with common
random numbers within a generation, so selection compares parameters not luck.

Usage:
    python experiment_ga_mnist.py [generations] [var_weight] [tf] [seed]
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F

from teacher_net import load_mnist, train_teacher, N_IN
from thermo_mnist import (ThermoMNIST, HIDDEN, N, N_OUT, BETA, MU, DT)
from train_mnist import teacher_activations, accuracy
from thermo_activation_fit import fit_thermo_activation
from experiment_tf_scan_mnist import train_one

GENS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
VAR_WEIGHT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
TF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

# --- GA hyperparameters ----------------------------------------------------
P = 65            # population size (65 candidate computers per generation)
N_ELITE = 10      # elite lineages carried over unmutated
MUT = 1e-1        # base mutation std (5x up from 2e-2): probe how far
                  # random-mutation fine-tuning can push a 63k-param model
# Fitness = each computer's cross-entropy over the ENTIRE 60k-digit TRAIN set
# (no subsampling), so it is the true full-train error and identical every
# generation -- selection compares parameters, never the luck of a batch.  The
# TEST set is untouched (used only for the final accuracy report).
K_FIT = 60000     # = all of the MNIST train split (full-dataset fitness)
M_FIT = 1         # reset samples per digit (raise for a variance penalty)
K_CHUNK = 2500    # digit-chunk, bounds the P x K_CHUNK x M x N state tensor
CRN = True        # common random numbers within a generation
TEACHER_EPOCHS = 300
M_EVAL = 10
CKPT_EVERY = 25

# Fan-in for mutation scaling.  The true all-to-all in-degrees (W: 784, Jhh: 64)
# divide the mutation by sqrt(784) ~ 28, making W steps ~7e-4 -- far too small to
# move a 63k-parameter computer, which is why the GA sat still.  Instead we scale
# by the PREVIOUS-LAYER width of the digital teacher this computer imitates
# (784-32-32-10), i.e. 32 for the couplings, so steps are ~5x larger.
FANIN_REF = 32.0
FANIN = {"b": 1.0, "W": FANIN_REF, "Jhh": FANIN_REF,
         "Jho": FANIN_REF, "Joo": float(N_OUT)}


def sym0(J):
    Js = 0.5 * (J + J.transpose(-1, -2))
    return Js - torch.diag_embed(torch.diagonal(Js, dim1=-2, dim2=-1))


def population_fitness(pop, pixels, labels, gen_seed, device):
    """(fitness, ce, var) each (P,) for all candidates on this digit batch."""
    Pn = pop["b"].shape[0]
    K = pixels.shape[0]
    Jc = pop["b"].new_zeros(Pn, N, N)
    Jc[:, :HIDDEN, :HIDDEN] = sym0(pop["Jhh"])
    Jc[:, :HIDDEN, HIDDEN:] = pop["Jho"]
    Jc[:, HIDDEN:, :HIDDEN] = pop["Jho"].transpose(-1, -2)
    Jc[:, HIDDEN:, HIDDEN:] = sym0(pop["Joo"])
    b = pop["b"][:, None, None, :]
    noise_amp = (2.0 * MU * (1.0 / BETA) * DT) ** 0.5
    n_steps = int(round(TF / DT))
    gen = torch.Generator(device=device).manual_seed(gen_seed)

    s1 = pop["b"].new_zeros(Pn, K, N_OUT)
    s2 = pop["b"].new_zeros(Pn, K, N_OUT)
    for k0 in range(0, K, K_CHUNK):
        px = pixels[k0:k0 + K_CHUNK]
        kc = px.shape[0]
        ext = torch.einsum("ki,pin->pkn", px, pop["W"])[:, :, None, :]
        x = pop["b"].new_zeros(Pn, kc, M_FIT, N)
        for _ in range(n_steps):
            dV = (2.0 * x + 4.0 * x ** 3 - b
                  + torch.einsum("pkmn,pnj->pkmj", x, Jc) + ext)
            shape = (kc, M_FIT, N) if CRN else (Pn, kc, M_FIT, N)
            eta = torch.randn(shape, generator=gen, device=device)
            x = x - MU * dV * DT + noise_amp * eta
        out = x[..., HIDDEN:]                              # (P, kc, M, 10)
        s1[:, k0:k0 + kc] = out.mean(dim=2)                # (P, kc, 10)
        s2[:, k0:k0 + kc] = out.var(dim=2, unbiased=False)  # per-sample variance

    ce = torch.stack([F.cross_entropy(s1[p], labels) for p in range(Pn)])
    var = s2.mean(dim=(1, 2))
    return ce + VAR_WEIGHT * var, ce, var


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    print(f"config: GA on MNIST thermo computer | gens={GENS} "
          f"var_weight={VAR_WEIGHT:g} tf={TF:g} seed={SEED} | P={P} "
          f"elites={N_ELITE} K_FIT={K_FIT} M_FIT={M_FIT} CRN={CRN} MUT={MUT:g} "
          f"| kT={1.0/BETA:g} device={device}", flush=True)

    Xtr, ytr, Xte, yte = load_mnist(device)

    # --- warm start: per-tf refit teacher -> OM-GD student -------------------
    a, c, info = fit_thermo_activation(TF, beta=1.0, device=device)
    print(f"refit activation @tf={TF:g}: ACT_A={a:.4f} ACT_C={c:.3f} "
          f"(rms {info['rms']:.4f})", flush=True)
    teacher, tacc = train_teacher(epochs=TEACHER_EPOCHS, activation="thermo",
                                  act_a=a, act_c=c, seed=0, device=device,
                                  verbose=False)
    print(f"teacher (thermo, refit): {tacc:.4f}", flush=True)
    Ah, Ao = teacher_activations(teacher, Xtr)
    student = train_one(Ah, Ao, Xtr, TF, SEED, device)
    gd_acc = accuracy(student, Xte, yte, M=M_EVAL, tf=TF, seed=SEED)
    print(f"OM-GD warm start: test acc {gd_acc:.4f}  ({time.time()-t0:.0f}s)",
          flush=True)

    theta0 = {"b": student.b.detach(), "W": student.W.detach(),
              "Jhh": sym0(student.Jhh_raw.detach()),
              "Jho": student.Jho.detach(),
              "Joo": sym0(student.Joo_raw.detach())}

    g = torch.Generator(device=device).manual_seed(SEED + 1000)
    mut_std = {k: MUT / np.sqrt(FANIN[k]) for k in FANIN}

    def mutate(base):
        return {k: v + mut_std[k] * torch.randn(v.shape, generator=g,
                                                device=device)
                for k, v in base.items()}

    def stack(cands):
        return {k: torch.stack([cd[k] for cd in cands]) for k in theta0}

    pop = stack([dict(theta0) for _ in range(N_ELITE)]
                + [mutate(theta0) for _ in range(P - N_ELITE)])

    runs_dir = os.path.join(_HERE, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    tag = f"ga_mnist_tf{TF:g}_vw{VAR_WEIGHT:g}_seed{SEED}_g{GENS}"
    result_path = os.path.join(runs_dir, f"{tag}.npz")
    ckpt_path = os.path.join(runs_dir, f"ckpt_{tag}.pt")

    def save_result(champ, history, ga_acc):
        np.savez(result_path,
                 **{f"p_{k}": v.detach().cpu().numpy() for k, v in champ.items()},
                 history=np.asarray(history), tf=TF, var_weight=VAR_WEIGHT,
                 seed=SEED, gens=GENS, gd_acc=gd_acc, ga_acc=ga_acc,
                 teacher_acc=tacc, act_a=a, act_c=c, p=P, k_fit=K_FIT,
                 m_fit=M_FIT, mut=MUT)

    history, start = [], 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        pop = {k: v.to(device) for k, v in ck["pop"].items()}
        history = list(ck["history"]); start = int(ck["gen"])
        print(f"resuming from gen {start}/{GENS}", flush=True)

    elites = [{k: pop[k][i].clone() for k in pop} for i in range(N_ELITE)]
    # ONE fixed evaluation set, drawn once from the TRAIN split and reused every
    # generation, so fitness is comparable across generations (the test set is
    # kept untouched for reporting -- optimizing on it would be leakage).
    gsel = torch.Generator(device=device).manual_seed(12345)
    fit_idx = torch.randperm(Xtr.shape[0], generator=gsel,
                             device=device)[:K_FIT]
    Xfit, yfit = Xtr[fit_idx], ytr[fit_idx]
    if VAR_WEIGHT > 0 and M_FIT < 2:
        print(f"WARNING: var_weight={VAR_WEIGHT:g} but M_FIT={M_FIT} -- the "
              f"per-sample variance of a single sample is identically 0, so the "
              f"penalty does nothing.  Set M_FIT >= 2 for variance runs.",
              flush=True)

    t_last, n_last = time.time(), start
    for n in range(start, GENS):
        fit, ce, var = population_fitness(pop, Xfit, yfit,
                                          SEED + 2000 + n, device)
        fit = torch.nan_to_num(fit, nan=float("inf"), posinf=float("inf"))
        order = torch.argsort(fit); best = order[0].item()
        history.append((fit[best].item(), ce[best].item(), var[best].item()))
        if n % 5 == 0 or n == GENS - 1:
            now = time.time()
            spg = (now - t_last) / max(1, n - n_last); t_last, n_last = now, n
            print(f"  gen {n:4d}  fitness {history[-1][0]:.4f}  "
                  f"CE {history[-1][1]:.4f}  var {history[-1][2]:.4f}  "
                  f"{spg:6.2f} s/gen  ({now-t0:.0f}s)", flush=True)
        elites = [{k: pop[k][i].clone() for k in pop}
                  for i in order[:N_ELITE].tolist()]
        pop = stack([dict(e) for e in elites]
                    + [mutate(elites[i % N_ELITE]) for i in range(P - N_ELITE)])
        if (n + 1) % CKPT_EVERY == 0 and n + 1 < GENS:
            torch.save({"pop": {k: v.cpu() for k, v in pop.items()},
                        "gen": n + 1, "history": history}, ckpt_path)
            save_result(elites[0], history, float("nan"))
            print(f"  checkpoint @gen {n+1}", flush=True)

    champ = elites[0]
    tuned = ThermoMNIST(scale=0.0).to(device)
    with torch.no_grad():
        tuned.b.copy_(champ["b"]); tuned.W.copy_(champ["W"])
        tuned.Jhh_raw.copy_(champ["Jhh"]); tuned.Jho.copy_(champ["Jho"])
        tuned.Joo_raw.copy_(champ["Joo"])
    ga_acc = accuracy(tuned, Xte, yte, M=M_EVAL, tf=TF, seed=SEED)
    save_result(champ, history, ga_acc)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print(f"\nGA-finetuned:  test acc {ga_acc:.4f}   (M={M_EVAL})")
    print(f"OM-GD baseline: test acc {gd_acc:.4f}")
    print(f"-> {result_path}", flush=True)


if __name__ == "__main__":
    main()
