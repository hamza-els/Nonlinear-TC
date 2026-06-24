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
    """Number of connections entering each neuron: the sizes of the adjacent
    layers it couples to (bilinear coupling is bidirectional), plus the external
    inputs for input-layer neurons. Used to scale mutations (paper, Sec. S3 B).

    Takes no arguments (reads the module-level LAYERS / N_INPUTS).

    Returns
    -------
    fanin : (N,)     the in-degree of each neuron, indexed in the flat layout."""
    fanin = torch.zeros(N)
    L = len(LAYERS)
    for l in range(L):
        prev_sz = LAYERS[l - 1] if l > 0 else 0
        next_sz = LAYERS[l + 1] if l < L - 1 else 0
        c = prev_sz + next_sz + (N_INPUTS if l == 0 else 0)
        fanin[OFF[l]:OFF[l + 1]] = c
    return fanin


def mutation_std():
    """Per-parameter mutation standard deviations, std = MUT_STD * C^{-1/2}.

    C is the fan-in: 1 for biases b and output weights f; the receiving neuron's
    fan-in for input weights W; and the average of the two neurons' fan-ins for
    couplings J. Scaling by fan-in keeps the change to each neuron's total input
    of similar size across the network (paper, Sec. S3 B).

    Takes no arguments (reads the module-level architecture and MUT_STD).

    Returns
    -------
    dict with the same keys as a population (see init_population):
        "b" : (N,)            std for every neuron bias.
        "f" : (N_OUT,)        std for every output weight.
        "W" : (LAYERS[0],)    std for each input weight, scaled by its neuron's fan-in.
        "J" : list of (s_l, s_{l+1})  std for each coupling, scaled by the pair's fan-in.
    Each tensor matches a population tensor's trailing shape, so it broadcasts
    over the leading population axis during mutation."""
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
    """A population of P thermodynamic computers as batched tensors, all drawn
    from N(0, INIT_STD). Every tensor has a leading population axis of size P,
    so index p selects the parameters of the p-th computer.

    Parameters
    ----------
    P : int          number of computers in the population (the GA's gene pool).
    device : str     torch device the tensors live on ("cpu" or "cuda").

    Returns
    -------
    dict of batched parameters:
        "W" : (P, LAYERS[0])  input weights coupling z into the first layer.
        "b" : (P, N)          per-neuron biases.
        "f" : (P, N_OUT)      output weights mapping output neuron(s) -> y.
        "J" : list of (P, s_l, s_{l+1})  bilinear couplings between adjacent layers."""
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
    """Assemble the batched symmetric coupling matrices Jc of shape (P, N, N).

    Each computer's rectangular per-layer couplings pop["J"] are written into the
    full N x N matrix at the adjacent-layer blocks, and mirrored across the
    diagonal so Jc[p] is symmetric (Jc[i,j] = Jc[j,i]).

    Parameters
    ----------
    pop : dict       a population (see init_population); only pop["J"] and the
                     population size are used.

    Returns
    -------
    Jc : (P, N, N)   one symmetric coupling matrix per computer."""
    P = pop["b"].shape[0]
    Jc = pop["b"].new_zeros(P, N, N)
    for l, Wmat in enumerate(pop["J"]):          # Wmat: (P, s_l, s_{l+1})
        a0, a1 = OFF[l], OFF[l + 1]
        b0, b1 = OFF[l + 1], OFF[l + 2]
        Jc[:, a0:a1, b0:b1] = Wmat
        Jc[:, b0:b1, a0:a1] = Wmat.transpose(1, 2)
    return Jc


# --- Batched simulation + loss for the whole population -------------------
@torch.no_grad()
def population_loss(pop, z, M=128, tf=1.0, dt=1e-3, beta=10.0, mu=1.0):
    """Reset-sampling simulation of all P computers at once, scoring each against
    the target. Every trajectory starts from x = 0, evolves under overdamped
    Langevin dynamics to time tf, and the final activations are averaged over the
    M samples before computing the MSE loss.

    Parameters
    ----------
    pop : dict       the population of parameters (see init_population).
    z : (K,) tensor  the K input values to evaluate (here points on [0, 1]).
    M : int          samples (independent noisy replicas) averaged per input;
                     more M -> less noisy loss, more compute/memory.
    tf : float       observation time at which the output is read (units of mu^-1).
    dt : float       Euler-Maruyama timestep; n_steps = tf / dt.
    beta : float     inverse temperature 1/(kB T); sets the noise strength.
    mu : float       mobility (overall time constant of the dynamics).

    Returns
    -------
    losses : (P,)    the loss phi of each computer in the population."""
    J2, J3, J4 = J_INTRINSIC
    P = pop["b"].shape[0]
    K = z.shape[0]

    Jc = coupling_matrices(pop)                  # (P, N, N)

    # External field W_i z on input-layer neurons: (P, K, N), zero elsewhere.
    ext = pop["b"].new_zeros(P, K, N)
    ext[:, :, INPUT_IDX] = z[None, :, None] * pop["W"][:, None, :]

    kT = 1.0 / beta
    noise_amp = (2.0 * mu * kT * dt) ** 0.5
    n_steps = int(round(tf / dt))

    b = pop["b"]
    x = pop["b"].new_zeros(P, K, M, N)           # (population, input, sample, neuron)
    for _ in range(n_steps):
        coupling = torch.einsum("pkmn,pnj->pkmj", x, Jc)
        dVdx = (
            (2.0 * J2 * x + 3.0 * J3 * x ** 2 + 4.0 * J4 * x ** 3)
            - b[:, None, None, :]
            + coupling
            + ext[:, :, None, :]
        )
        x = x - mu * dVdx * dt + noise_amp * torch.randn_like(x)

    mean_x = x.mean(dim=2)                        # (P, K, N)
    y = (mean_x[:, :, OUTPUT_IDX] * pop["f"][:, None, :]).sum(dim=-1)  # (P, K)
    return ((target(z)[None, :] - y) ** 2).mean(dim=1)                  # (P,)


# --- Genetic algorithm ----------------------------------------------------
@torch.no_grad()
def select_and_breed(pop, losses, std, n_elite=5):
    """One generation of selection + reproduction: keep the n_elite lowest-loss
    computers, clone them to refill the population back to size P, then mutate
    every clone (mutations scaled per-parameter by fan-in).

    Parameters
    ----------
    pop : dict       the current population of parameters (see init_population).
    losses : (P,)    each computer's loss, from population_loss; lower is better.
    std : dict       per-parameter mutation std from mutation_std().
    n_elite : int    how many top computers survive and become parents; each
                     produces P / n_elite mutated offspring.

    Returns
    -------
    new : dict       the next-generation population, same structure as pop."""
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


def train(generations=500, P=50, n_elite=5, K=250, M=128, device=None, **sim_kw):
    """Run the genetic algorithm: repeatedly score the population and breed the
    survivors. The K evenly-spaced inputs z_j on [0, 1] and the fan-in mutation
    stds are set up once, then each generation scores all P computers and
    replaces them with mutated offspring of the best n_elite.

    Parameters
    ----------
    generations : int   number of GA iterations to run.
    P : int             population size (computers scored each generation).
    n_elite : int       survivors per generation (parents of the next).
    K : int             number of input points used to evaluate the loss.
    M : int             samples per input in the simulation (passed through).
    device : str        torch device ("cpu" or "cuda").
    **sim_kw            extra simulation kwargs forwarded to population_loss
                        (tf, dt, beta, mu).

    Returns
    -------
    pop : dict          the final population after `generations` iterations.
    history : list      best loss recorded at each generation."""
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
        pop = select_and_breed(pop, losses, std, n_elite=n_elite)
        if gen % 10 == 0:
            print(f"gen {gen:4d}   best loss {best:.4f}")
    return pop, history


def main():
    # Small smoke test: confirm the GA runs and the loss is being driven down.
    pop, history = train(generations=20, P=20, K=32, M=16, tf=1.0)
    print(f"first {history[0]:.4f} -> last {history[-1]:.4f}")


if __name__ == "__main__":
    main()
