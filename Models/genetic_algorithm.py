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
    inputs for input-layer neurons. Used to scale mutations (paper, Sec. S3 B)."""
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
    couplings J. Returned as a dict matching the population tensors' trailing
    shapes (broadcast over the population axis)."""
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
    from N(0, INIT_STD). Every tensor has a leading population axis of size P."""
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
    """Assemble the batched symmetric coupling matrices Jc of shape (P, N, N)."""
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
    """Reset-sampling simulation of all P computers at once. Returns the loss
    phi of each computer, shape (P,)."""
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
    """Keep the n_elite lowest-loss computers, clone them to refill the
    population, then mutate every clone (mutations scaled by fan-in)."""
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


def train(generations=500, P=50, n_elite=5, K=250, M=128, device="cpu", **sim_kw):
    """Run the genetic algorithm. Returns the trained population and the
    best-loss history."""
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
