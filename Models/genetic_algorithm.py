import warnings

import numpy as np
import torch

from cosine_training_torch import (
    LAYERS, N, OFFSETS, INPUT_IDX, OUTPUT_IDX, J_INTRINSIC, target,
)

# Integer layer boundaries (OFFSETS entries are 0-d tensors).
OFF = [int(o) for o in OFFSETS]
N_OUT = len(OUTPUT_IDX)
N_INPUTS = 1  # cosine task has a single external input z

# GA / init hyperparameters.
INIT_STD = 1e-2   # parameters initialized from N(0, INIT_STD)
MUT_STD = 1e-2    # base mutation size, before fan-in scaling


# --- Per-neuron fan-in (for scaling mutations) ----------------------------
def neuron_fanin():
    """In-degree of each neuron (adjacent layer sizes + inputs), shape (N,)."""
    fanin = torch.zeros(N)
    L = len(LAYERS)
    for l in range(L):
        prev_sz = LAYERS[l - 1] if l > 0 else 0
        next_sz = LAYERS[l + 1] if l < L - 1 else 0
        c = prev_sz + next_sz + (N_INPUTS if l == 0 else 0)
        fanin[OFF[l]:OFF[l + 1]] = c
    return fanin


def mutation_std():
    """Per-parameter mutation std = MUT_STD * C^{-1/2}, C = fan-in (paper S3 B).
    Returns a dict keyed like a population (b, f, W, J)."""
    fanin = neuron_fanin()
    std = {
        "b": MUT_STD * torch.ones(N),                       # C = 1
        "f": MUT_STD * torch.ones(N_OUT),                   # C = 1
        "W": MUT_STD / fanin[INPUT_IDX].sqrt(),             # C = fan-in of input neuron
    }
    std_J = []
    for l in range(len(LAYERS) - 1):
        fi = fanin[OFF[l]:OFF[l + 1]]            # this layer's neurons
        fj = fanin[OFF[l + 1]:OFF[l + 2]]        # next layer's neurons
        C = 0.5 * (fi[:, None] + fj[None, :])    # (s_l, s_{l+1})
        std_J.append(MUT_STD / C.sqrt())
    std["J"] = std_J
    return std


# --- Population representation --------------------------------------------
def init_population(P, device="cpu"):
    """P computers as batched tensors (leading axis P) drawn from N(0, INIT_STD).
    Keys: W (P,LAYERS[0]), b (P,N), f (P,N_OUT), J list of (P,s_l,s_{l+1})."""
    g = {
        "W": INIT_STD * torch.randn(P, LAYERS[0], device=device),
        "b": INIT_STD * torch.randn(P, N, device=device),
        "f": INIT_STD * torch.randn(P, N_OUT, device=device),
        "J": [
            INIT_STD * torch.randn(P, LAYERS[l], LAYERS[l + 1], device=device)
            for l in range(len(LAYERS) - 1)
        ],
    }
    return g


def coupling_matrices(pop):
    """Batched symmetric coupling matrices (P, N, N) from the per-layer blocks."""
    P = pop["b"].shape[0]
    Jc = pop["b"].new_zeros(P, N, N)
    for l, Wmat in enumerate(pop["J"]):          # Wmat: (P, s_l, s_{l+1})
        a0, a1 = OFF[l], OFF[l + 1]
        b0, b1 = OFF[l + 1], OFF[l + 2]
        Jc[:, a0:a1, b0:b1] = Wmat
        Jc[:, b0:b1, a0:a1] = Wmat.transpose(1, 2)
    return Jc


# --- One Euler-Maruyama step (hot loop) -----------------------------------
def _langevin_step(x, Jc, field4, c1, c3, c4, mudt, noise_amp):
    """One overdamped-Langevin update, physics identical to the original loop.

    field4 = (external input - bias) broadcast to (P,K,1,N); c1,c3,c4 are the
    prefactors 2*J2, 3*J3, 4*J4; mudt = mu*dt. The J3 (cubic-in-U) term is
    skipped when c3 == 0 (our default (1,0,1) neuron)."""
    coupling = torch.einsum("pkmn,pnj->pkmj", x, Jc)
    x2 = x * x
    drift = c1 * x + c4 * (x2 * x) + coupling + field4
    if c3 != 0.0:
        drift = drift + c3 * x2
    return x - mudt * drift + noise_amp * torch.randn_like(x)


_compiled_step = None       # lazily-built torch.compile version of _langevin_step
_COMPILE_ENABLED = True      # set False if compilation fails once, to stop retrying


def _integrate(make_x, Jc, field4, consts, n_steps, use_compile):
    """Run n_steps of the update from a fresh zero state. Uses a torch.compile'd
    step on CUDA when available, falling back to eager if compilation fails."""
    global _compiled_step, _COMPILE_ENABLED
    step = _langevin_step
    if use_compile and _COMPILE_ENABLED:
        if _compiled_step is None:
            _compiled_step = torch.compile(_langevin_step)
        step = _compiled_step

    x = make_x()
    try:
        for _ in range(n_steps):
            x = step(x, Jc, field4, *consts)
        return x
    except Exception as e:                       # inductor/triton unavailable, etc.
        if step is _langevin_step:
            raise
        warnings.warn(f"torch.compile step failed ({type(e).__name__}: {e}); "
                      "falling back to eager execution")
        _COMPILE_ENABLED = False
        x = make_x()
        for _ in range(n_steps):
            x = _langevin_step(x, Jc, field4, *consts)
        return x


# --- Batched simulation + loss for the whole population -------------------
@torch.no_grad()
def population_loss(pop, z, M=128, m_chunk=None, var_weight=0.0,
                    loss_mode="squared", tf=1.0, dt=1e-3, beta=10.0, mu=1.0,
                    compile_step=True):
    """Reset-sampling loss of all P computers, shape (P,).

    Runs the Langevin dynamics to tf and averages the output over M samples
    (processed in m_chunk batches to cap memory). var_weight (~1/M_inf) weights a
    per-sample output-variance penalty; loss_mode "squared" -> MSE + var_weight*Var,
    "rms" -> mean_z sqrt(bias^2 + var_weight*Var). compile_step torch.compile's the
    hot integration step on CUDA (falls back to eager if unavailable)."""
    J2, J3, J4 = J_INTRINSIC
    P = pop["b"].shape[0]
    K = z.shape[0]
    if m_chunk is None or m_chunk > M:
        m_chunk = M
    f = pop["f"]

    Jc = coupling_matrices(pop)                  # (P, N, N)

    # Constant per-neuron field = external input (W_i z on layer 0) minus bias.
    # Precomputed once and broadcast over the sample axis: (P, K, 1, N).
    ext = pop["b"].new_zeros(P, K, N)
    ext[:, :, INPUT_IDX] = z[None, :, None] * pop["W"][:, None, :]
    field4 = (ext - pop["b"][:, None, :]).unsqueeze(2)   # (P, K, 1, N)

    noise_amp = (2.0 * mu * (1.0 / beta) * dt) ** 0.5
    n_steps = int(round(tf / dt))
    # Folded constants passed to the step (avoids recomputing every iteration).
    consts = (2.0 * J2, 3.0 * J3, 4.0 * J4, mu * dt, noise_amp)
    use_compile = compile_step and pop["b"].is_cuda

    # Accumulate, over chunks of samples, the summed per-sample output Y and its
    # square, so both the mean output and its variance fall out at the end. Peak
    # state tensor is only (P, K, m_chunk, N) at a time.
    sum_y = pop["b"].new_zeros(P, K)             # sum_alpha Y^(alpha)
    sum_y2 = pop["b"].new_zeros(P, K)            # sum_alpha (Y^(alpha))^2
    remaining = M
    while remaining > 0:
        m = min(m_chunk, remaining)              # (population, input, sample, neuron)
        x = _integrate(lambda: pop["b"].new_zeros(P, K, m, N),
                       Jc, field4, consts, n_steps, use_compile)
        Y = (x[:, :, :, OUTPUT_IDX] * f[:, None, None, :]).sum(dim=-1)  # (P, K, m)
        sum_y += Y.sum(dim=2)
        sum_y2 += (Y ** 2).sum(dim=2)
        remaining -= m

    y = sum_y / M                                # mean output, (P, K)
    bias2 = (target(z)[None, :] - y) ** 2        # (P, K)
    var = (sum_y2 / M - y ** 2).clamp_min(0.0)   # per-sample output variance (P, K)
    if loss_mode == "rms":
        # expected RMS error of a readout: sqrt(bias^2 + var/M_inf) per input,
        # averaged over inputs (var_weight plays the role of 1/M_inf).
        return torch.sqrt(bias2 + var_weight * var).mean(dim=1)
    if loss_mode == "squared":
        return bias2.mean(dim=1) + var_weight * var.mean(dim=1)
    raise ValueError(f"unknown loss_mode {loss_mode!r}")


# --- Genetic algorithm ----------------------------------------------------
@torch.no_grad()
def select_and_breed(pop, losses, std, n_elite=5):
    """Keep the n_elite lowest-loss computers, clone them back to size P, and
    mutate every clone (fan-in-scaled std). Returns the next-gen population."""
    P = losses.shape[0]
    elite = torch.argsort(losses)[:n_elite]                # indices of best
    # Each elite produces P / n_elite offspring.
    parents = elite.repeat_interleave(P // n_elite)[:P]    # (P,) parent index per slot

    new = {
        "W": pop["W"][parents].clone(),
        "b": pop["b"][parents].clone(),
        "f": pop["f"][parents].clone(),
        "J": [Wmat[parents].clone() for Wmat in pop["J"]],
    }
    # Mutate: add C^{-1/2} N(0, MUT_STD) to every parameter.
    new["W"] += std["W"] * torch.randn_like(new["W"])
    new["b"] += std["b"] * torch.randn_like(new["b"])
    new["f"] += std["f"] * torch.randn_like(new["f"])
    for Wmat, sJ in zip(new["J"], std["J"]):
        Wmat += sJ * torch.randn_like(Wmat)
    return new


def resolve_device(device=None):
    """Pick the compute device: explicit choice, else CUDA if available."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def save_best(pop, losses, path):
    """Save the lowest-loss computer's weights to a .npz file."""
    b = int(losses.argmin())
    arrs = {"W": pop["W"][b].detach().cpu().numpy(),
            "b": pop["b"][b].detach().cpu().numpy(),
            "f": pop["f"][b].detach().cpu().numpy()}
    for l, Jm in enumerate(pop["J"]):
        arrs[f"J{l}"] = Jm[b].detach().cpu().numpy()
    np.savez(path, **arrs)


def train(generations=500, P=50, n_elite=5, K=250, M=128, device=None,
          save_path=None, checkpoint_every=50, **sim_kw):
    """Run the GA over K evenly-spaced inputs on [0, 1]. Each generation scores
    all P computers and breeds the best n_elite. **sim_kw forwards to
    population_loss (m_chunk, var_weight, loss_mode, tf, dt, beta, mu).

    If save_path is given, the current best computer's weights are written there
    every checkpoint_every generations and at the end (so a dropped session or
    wall-time limit doesn't lose the run). Returns (final population, history)."""
    device = resolve_device(device)
    print(f"training on {device}")
    z = torch.arange(K, device=device, dtype=torch.float32) / (K - 1)
    std = mutation_std()
    std = {"W": std["W"].to(device), "b": std["b"].to(device),
           "f": std["f"].to(device), "J": [s.to(device) for s in std["J"]]}

    pop = init_population(P, device=device)
    history = []
    for gen in range(generations):
        losses = population_loss(pop, z, M=M, **sim_kw)
        best = losses.min().item()
        history.append(best)
        if gen % 10 == 0:
            print(f"gen {gen:4d}   best loss {best:.4f}", flush=True)
        if save_path and (gen == generations - 1
                          or (checkpoint_every and gen % checkpoint_every == 0)):
            save_best(pop, losses, save_path)
        pop = select_and_breed(pop, losses, std, n_elite=n_elite)
    return pop, history


def main():
    # Small smoke test: confirm the GA runs and the loss is being driven down.
    pop, history = train(generations=20, P=20, K=32, M=16, tf=1.0)
    print(f"first {history[0]:.4f} -> last {history[-1]:.4f}")


if __name__ == "__main__":
    main()
