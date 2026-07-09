import time
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

# GA / init hyperparameters. The paper writes N(0, 10^-2); that is a *variance*,
# so the standard deviation is sqrt(1e-2) = 0.1 (using 0.01 as the std, as before,
# makes init/mutations ~10x too small and the GA never leaves the ~0.5 plateau).
INIT_STD = 0.1    # parameters initialized from N(0, std=0.1)  [variance 1e-2]
MUT_STD = 0.1     # base mutation std, before fan-in scaling   [variance 1e-2]


# --- Per-neuron fan-in (for scaling mutations) ----------------------------
def neuron_fanin():
    """Feedforward in-degree of each neuron, shape (N,): the number of signals
    feeding *into* it from the layer above (the previous layer's size, or the
    N_INPUTS external inputs for layer 0).

    The paper (S3 B) sets C = "the number of connections entering neuron j",
    citing LeCun-style fan-in, which counts a unit's inputs -- not its total
    undirected degree. We previously used total degree (prev + next + inputs),
    which made W mutations 3x and J mutations 1.2-1.7x smaller than this
    reading; that run escaped the ~0.5 loss plateau at gen ~870 vs the paper's
    ~300-400, consistent with plateau escape being mutation-scale limited."""
    fanin = torch.zeros(N)
    for l in range(len(LAYERS)):
        c = LAYERS[l - 1] if l > 0 else N_INPUTS
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
def _simulate_yvar(pop, z, M=128, m_chunk=None,
                   tf=1.0, dt=1e-3, beta=10.0, mu=1.0, compile_step=True):
    """Reset-sampling simulation of all P computers. Returns the mean output
    y(z) and the per-sample output variance, each shape (P, K).

    The M samples run in m_chunk batches to cap memory; compile_step torch.compiles
    the hot step on CUDA (falls back to eager if unavailable)."""
    J2, J3, J4 = J_INTRINSIC
    P = pop["b"].shape[0]
    K = z.shape[0]
    if m_chunk is None or m_chunk > M:
        m_chunk = M
    f = pop["f"]

    Jc = coupling_matrices(pop)                  # (P, N, N)

    # Constant per-neuron field = external input (W_i z on layer 0) minus bias.
    ext = pop["b"].new_zeros(P, K, N)
    ext[:, :, INPUT_IDX] = z[None, :, None] * pop["W"][:, None, :]
    field4 = (ext - pop["b"][:, None, :]).unsqueeze(2)   # (P, K, 1, N)

    noise_amp = (2.0 * mu * (1.0 / beta) * dt) ** 0.5
    n_steps = int(round(tf / dt))
    consts = (2.0 * J2, 3.0 * J3, 4.0 * J4, mu * dt, noise_amp)
    use_compile = compile_step and pop["b"].is_cuda

    # Accumulate summed per-sample output Y and its square over sample chunks;
    # peak state tensor is only (P, K, m_chunk, N) at a time.
    sum_y = pop["b"].new_zeros(P, K)
    sum_y2 = pop["b"].new_zeros(P, K)
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
    var = (sum_y2 / M - y ** 2).clamp_min(0.0)   # per-sample output variance (P, K)
    return y, var


@torch.no_grad()
def population_output(pop, z, **kw):
    """Mean output y(z) of each computer, shape (P, K). See _simulate_yvar."""
    return _simulate_yvar(pop, z, **kw)[0]


@torch.no_grad()
def population_loss(pop, z, var_weight=0.0, loss_mode="squared", **kw):
    """Reset-sampling loss of all P computers, shape (P,). var_weight (~1/M_inf)
    weights a per-sample output-variance penalty; loss_mode "squared" ->
    MSE + var_weight*Var, "rms" -> mean_z sqrt(bias^2 + var_weight*Var). Remaining
    kwargs (M, m_chunk, tf, dt, beta, mu, compile_step) go to _simulate_yvar."""
    y, var = _simulate_yvar(pop, z, **kw)
    bias2 = (target(z)[None, :] - y) ** 2        # (P, K)
    if loss_mode == "rms":
        return torch.sqrt(bias2 + var_weight * var).mean(dim=1)
    if loss_mode == "squared":
        return bias2.mean(dim=1) + var_weight * var.mean(dim=1)
    raise ValueError(f"unknown loss_mode {loss_mode!r}")


@torch.no_grad()
def population_loss_components(pop, z, var_weight=0.0, loss_mode="squared", **kw):
    """Like population_loss, but also returns the bias and variance parts -- all
    from one simulation. Returns (full, bias2, var_mean), each shape (P,):
      bias2    = mean_z (target - y)^2   -- var-free fidelity to the target,
      var_mean = mean_z Var[y]           -- per-sample output variance,
      full     = the selection loss (identical to population_loss): for "squared",
                 bias2 + var_weight*var_mean; for "rms", mean_z sqrt(bias^2 +
                 var_weight*Var).
    This lets train() log a var-free "bias_history" alongside the combined loss at
    no extra simulation cost (the var penalty otherwise hides the true bias when
    var_weight > 0)."""
    y, var = _simulate_yvar(pop, z, **kw)
    bias2_z = (target(z)[None, :] - y) ** 2      # (P, K)
    bias2 = bias2_z.mean(dim=1)                   # (P,)
    var_mean = var.mean(dim=1)                    # (P,)
    if loss_mode == "rms":
        full = torch.sqrt(bias2_z + var_weight * var).mean(dim=1)
    elif loss_mode == "squared":
        full = bias2 + var_weight * var_mean
    else:
        raise ValueError(f"unknown loss_mode {loss_mode!r}")
    return full, bias2, var_mean


def params_to_pop(params, device="cpu"):
    """Wrap a single computer's numpy weights (W, b, f, J list) into a P=1
    population of torch tensors, so the batched simulators can run on it."""
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)[None]
    return {"W": t(params["W"]), "b": t(params["b"]), "f": t(params["f"]),
            "J": [t(Jm) for Jm in params["J"]]}


@torch.no_grad()
def output_curve(params, z, M=1000, m_chunk=None, device="cpu", **kw):
    """Compute the trained computer's output y(z) over the inputs z (numpy), by
    reset-sampling averaging over M samples. Returns a numpy array."""
    pop = params_to_pop(params, device)
    zt = torch.as_tensor(z, dtype=torch.float32, device=device)
    y = population_output(pop, zt, M=M, m_chunk=m_chunk, **kw)
    return y[0].cpu().numpy()


@torch.no_grad()
def sample_output_pool(params, z, S, m_chunk=None, tf=1.0, dt=1e-3,
                       beta=10.0, mu=1.0, device="cpu", compile_step=False):
    """Draw S independent single-shot readouts of one trained computer.

    Unlike output_curve (which averages the samples into y(z)), this returns the
    raw per-sample outputs Y, shape (K, S), where Y[:, a] = sum_i f_i x_i^{(a)}(z, tf)
    for reset-sample a over the K inputs z (numpy). Averaging any M columns gives
    an M-sample estimate of y(z), so a caller can study how error falls with M
    without re-simulating. Samples run in m_chunk batches to cap memory."""
    J2, J3, J4 = J_INTRINSIC
    pop = params_to_pop(params, device)          # P = 1 population
    zt = torch.as_tensor(z, dtype=torch.float32, device=device)
    K = zt.shape[0]
    if m_chunk is None or m_chunk > S:
        m_chunk = S
    f = pop["f"]                                  # (1, N_OUT)

    Jc = coupling_matrices(pop)                   # (1, N, N)
    ext = pop["b"].new_zeros(1, K, N)
    ext[:, :, INPUT_IDX] = zt[None, :, None] * pop["W"][:, None, :]
    field4 = (ext - pop["b"][:, None, :]).unsqueeze(2)     # (1, K, 1, N)

    noise_amp = (2.0 * mu * (1.0 / beta) * dt) ** 0.5
    n_steps = int(round(tf / dt))
    consts = (2.0 * J2, 3.0 * J3, 4.0 * J4, mu * dt, noise_amp)
    use_compile = compile_step and pop["b"].is_cuda

    out = pop["b"].new_zeros(K, S)
    filled = 0
    while filled < S:
        m = min(m_chunk, S - filled)             # (population, input, sample, neuron)
        x = _integrate(lambda: pop["b"].new_zeros(1, K, m, N),
                       Jc, field4, consts, n_steps, use_compile)
        Y = (x[:, :, :, OUTPUT_IDX] * f[:, None, None, :]).sum(dim=-1)  # (1, K, m)
        out[:, filled:filled + m] = Y[0]
        filled += m
    return out.cpu().numpy()


# --- Multi-GPU evaluation -------------------------------------------------
def as_devices(devices):
    """Normalize the devices argument: None -> None; an int n -> the first n
    CUDA devices; a list/tuple -> itself (as strings)."""
    if devices is None:
        return None
    if isinstance(devices, int):
        return [f"cuda:{i}" for i in range(devices)]
    return [str(d) for d in devices]


def _shard_to(pop, sl, device):
    """Slice computers [sl] out of a population and move them to `device`."""
    return {
        "W": pop["W"][sl].to(device, non_blocking=True),
        "b": pop["b"][sl].to(device, non_blocking=True),
        "f": pop["f"][sl].to(device, non_blocking=True),
        "J": [Jm[sl].to(device, non_blocking=True) for Jm in pop["J"]],
    }


@torch.no_grad()
def population_loss_sharded(pop, z, devices, **kw):
    """Evaluate the population split across several GPUs, gathered to pop's
    device. Each shard's parameters (tiny) are copied to its GPU and its loss is
    launched without an intervening sync, so the GPUs' simulations overlap; the
    final gather is the only synchronization point."""
    P = pop["b"].shape[0]
    G = len(devices)
    main = pop["b"].device
    bounds = [round(i * P / G) for i in range(G + 1)]   # contiguous, uneven-safe

    partial = []
    for g, dev in enumerate(devices):
        sl = slice(bounds[g], bounds[g + 1])
        sub = _shard_to(pop, sl, dev)
        partial.append(population_loss(sub, z.to(dev, non_blocking=True), **kw))
    return torch.cat([p.to(main) for p in partial])       # (P,), gather + sync


# --- Genetic algorithm ----------------------------------------------------
@torch.no_grad()
def select_and_breed(pop, losses, std, n_elite=5):
    """Elitist selection: carry the n_elite lowest-loss computers over
    *unmutated*, and fill the remaining P - n_elite slots with mutated clones
    of the elite (fan-in-scaled std). Returns the next-gen population (size P),
    with the elite occupying the first n_elite slots.

    Experiment log: elitism was first tried with the old total-degree fan-in
    (W mutations 3x too small) and failed -- var-penalized runs locked onto the
    constant-output attractor (bias ~0.5, variance ~0) and never escaped. This
    is the retest with the corrected in-degree fan-in, to separate the effect
    of elitism from that of the mutation scale. The paper's scheme (S3 B)
    mutates every clone; if this retest also fails, revert to that."""
    P = losses.shape[0]
    elite = torch.argsort(losses)[:n_elite]                # indices of best
    n_child = P - n_elite                                  # mutated-offspring slots
    # Each elite produces ~n_child / n_elite offspring (ceil then trim, so the
    # population stays exactly size P even when n_child % n_elite != 0).
    per_parent = -(-n_child // n_elite)                    # ceil division
    child_parents = elite.repeat_interleave(per_parent)[:n_child]
    # Parent index per slot: elite first (kept as-is), then their offspring.
    parents = torch.cat([elite, child_parents])            # (P,)

    new = {
        "W": pop["W"][parents].clone(),
        "b": pop["b"][parents].clone(),
        "f": pop["f"][parents].clone(),
        "J": [Wmat[parents].clone() for Wmat in pop["J"]],
    }
    # Mutate offspring only (slots n_elite:); the elite (slots :n_elite) are
    # carried over unchanged. Mutation adds C^{-1/2} N(0, MUT_STD) per parameter.
    ch = slice(n_elite, P)
    new["W"][ch] += std["W"] * torch.randn_like(new["W"][ch])
    new["b"][ch] += std["b"] * torch.randn_like(new["b"][ch])
    new["f"][ch] += std["f"] * torch.randn_like(new["f"][ch])
    for Wmat, sJ in zip(new["J"], std["J"]):
        Wmat[ch] += sJ * torch.randn_like(Wmat[ch])
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


def save_run(path, pop, losses, history, config, bias_history=None,
             var_history=None):
    """Save a training run to a .npz: the best computer's weights (W, b, f, J*),
    the per-generation loss history, and config scalars (cfg_*). This is enough
    to reproduce the loss curve (Fig 2c) and recompute the output y(z) (Fig 2d).

    bias_history, if given, is the per-generation var-free loss (mean_z bias^2) of
    the selected-best computer, saved as "bias_history". With var_weight > 0 the
    combined loss_history is dominated by the variance penalty, so this is what to
    plot to see how well the computer actually fits the target over training.

    var_history, if given, is the per-generation per-sample output variance
    (mean_z Var[y]) of that same computer, saved as "var_history". Unlike the
    (loss - bias)/var_weight derivation, this is available even when
    var_weight = 0, where the variance never enters the loss."""
    b = int(losses.argmin())
    arrs = {"W": pop["W"][b].detach().cpu().numpy(),
            "b": pop["b"][b].detach().cpu().numpy(),
            "f": pop["f"][b].detach().cpu().numpy(),
            "loss_history": np.asarray(history, dtype=np.float64)}
    if bias_history is not None:
        arrs["bias_history"] = np.asarray(bias_history, dtype=np.float64)
    if var_history is not None:
        arrs["var_history"] = np.asarray(var_history, dtype=np.float64)
    for l, Jm in enumerate(pop["J"]):
        arrs[f"J{l}"] = Jm[b].detach().cpu().numpy()
    for k, v in config.items():
        arrs[f"cfg_{k}"] = np.asarray(v)
    np.savez(path, **arrs)


def load_run(path):
    """Load a run saved by save_run. Returns (params, history, config), where
    params is the numpy weights dict {W, b, f, J:[...]}, history is the loss
    array, and config is a dict of the saved scalars."""
    d = np.load(path)
    n_J = sum(1 for k in d.files if k[0] == "J" and k[1:].isdigit())
    params = {"W": d["W"], "b": d["b"], "f": d["f"],
              "J": [d[f"J{l}"] for l in range(n_J)]}
    config = {k[4:]: d[k] for k in d.files if k.startswith("cfg_")}
    return params, d["loss_history"], config


def train(generations=500, P=50, n_elite=5, K=250, M=128, device=None,
          devices=None, save_path=None, checkpoint_every=50, print_every=10, **sim_kw):
    """Run the GA over K evenly-spaced inputs on [0, 1]. Each generation scores
    all P computers and breeds the best n_elite. **sim_kw forwards to
    population_loss (m_chunk, var_weight, loss_mode, tf, dt, beta, mu).

    devices enables multi-GPU: pass an int n (use the first n GPUs) or a list of
    device strings; the population is sharded across them each generation. If
    None, a single device is used (device, else auto).

    Prints best loss + wall-time-per-generation every print_every generations
    (set print_every=1 for per-generation updates when speed-testing). If
    save_path is given, a run record (best weights + loss history + config) is
    checkpointed there every checkpoint_every generations and at the end; load it
    with load_run to reproduce Fig 2c/2d. The record also stores per-generation
    "bias_history" (var-free loss) and "var_history" (per-sample output variance)
    of the selected-best computer next to the combined "loss_history".
    Returns (final population, history)."""
    dev_list = as_devices(devices)
    if dev_list and len(dev_list) > 1:
        main = dev_list[0]
        print(f"training on {len(dev_list)} GPUs: {dev_list}")
    else:
        main = resolve_device(device if dev_list is None else dev_list[0])
        dev_list = None
        print(f"training on {main}")

    # Config captured with the run so the figures can be reproduced exactly.
    config = {"K": K, "M": M, "P": P,
              "generations": generations, "n_elite": n_elite,
              "tf": sim_kw.get("tf", 1.0), "dt": sim_kw.get("dt", 1e-3),
              "beta": sim_kw.get("beta", 10.0), "mu": sim_kw.get("mu", 1.0),
              "m_chunk": sim_kw.get("m_chunk") or M,
              "var_weight": sim_kw.get("var_weight", 0.0),
              "loss_mode": sim_kw.get("loss_mode", "squared"),
              "init_std": INIT_STD, "mut_std": MUT_STD,
              "layers": LAYERS,       # architecture (runs before 4x8 didn't record it: those are [8,8,8,8])
              "fanin": "in_degree",   # feedforward in-degree C (old runs: total degree)
              "breeding": "elitist"}  # elite kept unmutated (paper: mutate_all)

    z = torch.arange(K, device=main, dtype=torch.float32) / (K - 1)
    std = mutation_std()
    std = {"W": std["W"].to(main), "b": std["b"].to(main),
           "f": std["f"].to(main), "J": [s.to(main) for s in std["J"]]}

    pop = init_population(P, device=main)
    history, bias_history, var_history = [], [], []
    last_t, last_gen = time.time(), 0
    for gen in range(generations):
        # The single-device path also returns each computer's var-free bias and
        # output variance (from the same simulation); the multi-GPU sharded path
        # returns the combined loss only, so both are recorded as NaN there.
        if dev_list:
            losses = population_loss_sharded(pop, z, dev_list, M=M, **sim_kw)
            bias2 = var_mean = None
        else:
            losses, bias2, var_mean = population_loss_components(pop, z, M=M, **sim_kw)
        # A computer whose dynamics blew up (NaN/inf loss) must never be selected
        # as elite or saved as "best": map NaN -> inf so min/argmin/argsort all
        # rank it last instead of propagating NaN into the checkpoint.
        losses = torch.nan_to_num(losses, nan=float("inf"), posinf=float("inf"))
        best_idx = int(losses.argmin())
        best = losses[best_idx].item()
        history.append(best)
        # Var-free loss and output variance of that same selected-best computer
        # (NaN on the sharded path). With var_weight > 0 the combined `best` is
        # dominated by var_weight * var, far above the bias term.
        bias_history.append(float(bias2[best_idx]) if bias2 is not None
                            else float("nan"))
        var_history.append(float(var_mean[best_idx]) if var_mean is not None
                           else float("nan"))
        if gen % print_every == 0:
            now = time.time()
            per = (now - last_t) / max(gen - last_gen, 1)   # avg since last print
            print(f"gen {gen:4d}   best loss {best:.4f}   {per:6.2f} s/gen", flush=True)
            last_t, last_gen = now, gen
        if save_path and (gen == generations - 1
                          or (checkpoint_every and gen % checkpoint_every == 0)):
            save_run(save_path, pop, losses, history, config,
                     bias_history=bias_history, var_history=var_history)
        pop = select_and_breed(pop, losses, std, n_elite=n_elite)
    return pop, history


def main():
    # Small smoke test: confirm the GA runs and the loss is being driven down.
    pop, history = train(generations=20, P=20, K=32, M=16, tf=1.0)
    print(f"first {history[0]:.4f} -> last {history[-1]:.4f}")


if __name__ == "__main__":
    main()
