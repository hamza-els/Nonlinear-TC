import numpy as np

# --- Network architecture -------------------------------------------------
# Four layers of 8 neurons feeding a single output neuron (the user's design).
LAYERS = [8, 8, 8, 8, 1]
N = sum(LAYERS)                                   # total neurons (= 33)
OFFSETS = np.cumsum([0] + LAYERS)                 # start index of each layer
INPUT_IDX = np.arange(OFFSETS[0], OFFSETS[1])     # layer-0 neurons (take input z)
OUTPUT_IDX = np.arange(OFFSETS[-2], OFFSETS[-1])  # final output neuron(s)

# Intrinsic neuron couplings J = (J2, J3, J4); (1, 0, 1) -> nonlinear neuron.


# --- Parameters -----------------------------------------------------------
def init_params(scale=1e-2):
    """Adjustable parameters theta = {W} u {b} u {Jij} u {f}, drawn from
    N(0, scale^2) as in the paper. Returns a dict.

    W : (8,)            input weights coupling z into the first layer.
    b : (N,)            per-neuron biases (the b_i in U_J(x_i, b_i)).
    Jmats : list        bilinear coupling matrices between adjacent layers,
                        Jmats[l] has shape (LAYERS[l], LAYERS[l+1]).
    f : (n_outputs,)    weights coupling the output neuron(s) to y.
    """
    return {
        "W": np.random.normal(scale=scale, size=LAYERS[0]),
        "b": np.random.normal(scale=scale, size=N),
        "Jmats": [
            np.random.normal(scale=scale, size=(LAYERS[l], LAYERS[l + 1]))
            for l in range(len(LAYERS) - 1)
        ],
        "f": np.random.normal(scale=scale, size=len(OUTPUT_IDX)),
    }


def coupling_matrix(Jmats):
    """Assemble the symmetric N x N bilinear coupling matrix from the per-layer
    matrices. Nonzero only between adjacent layers (all-to-all)."""
    Jc = np.zeros((N, N))
    for l, Wmat in enumerate(Jmats):
        a0, a1 = OFFSETS[l], OFFSETS[l + 1]
        b0, b1 = OFFSETS[l + 1], OFFSETS[l + 2]
        Jc[a0:a1, b0:b1] = Wmat
        Jc[b0:b1, a0:a1] = Wmat.T
    return Jc


# --- Simulation -----------------------------------------------------------
def simulate_network(params, z_values, M=1000, tf=1.0, dt=1e-3, beta=10.0, mu=1.0):
    """Reset-sampling simulation of the thermodynamic computer (Eq. S9).

    Runs M independent trajectories for each input z, all starting from x = 0,
    integrating overdamped Langevin dynamics with Euler-Maruyama up to time tf,
    then averages the final neuron activations over the M samples.

        dV/dx_i = (2 J2 x_i + 3 J3 x_i^2 + 4 J4 x_i^3)   [intrinsic]
                  - b_i                                    [bias]
                  + sum_j Jc[i,j] x_j                      [bilinear coupling]
                  + W_i z   (input-layer neurons only)     [external input]

    Returns
    -------
    mean_x : (K, N) array of reset-sampling averages <x_i(z)>_r.
    """
    J2, J3, J4 = (1.0, 0.0, 1.0)
    z = np.atleast_1d(np.asarray(z_values, dtype=float))
    K = z.size

    Jc = coupling_matrix(params["Jmats"])
    b = params["b"]

    # External field W_i z on the input-layer neurons, zero elsewhere: (K, N).
    ext = np.zeros((K, N))
    ext[:, INPUT_IDX] = z[:, None] * params["W"][None, :]

    kT = 1.0 / beta
    noise_amp = np.sqrt(2.0 * mu * kT * dt)
    n_steps = int(round(tf / dt))

    # State for every (input, sample) pair: (K, M, N), all reset to zero.
    x = np.zeros((K, M, N))
    for _ in range(n_steps):
        dVdx = (
            (2.0 * J2 * x + 3.0 * J3 * x ** 2 + 4.0 * J4 * x ** 3)
            - b[None, None, :]
            + x @ Jc
            + ext[:, None, :]
        )
        x = x - mu * dVdx * dt + noise_amp * np.random.normal(size=x.shape)

    return x.mean(axis=1)  # average over the M samples -> (K, N)


def network_output(params, z_values, **kw):
    """Scalar output y(z) = sum_{i in outputs} f_i <x_i(z)>_r."""
    mean_x = simulate_network(params, z_values, **kw)
    return mean_x[:, OUTPUT_IDX] @ params["f"]


# --- Target and loss ------------------------------------------------------
def target(z):
    """Target function y_0(z) = cos(2 pi z), Eq. (S13)."""
    return np.cos(2.0 * np.pi * z)


def loss(params, K=250, **kw):
    """Mean-squared-error loss phi, Eq. (S14), over K evenly-spaced points
    z_j = j/(K-1) on [0, 1]."""
    z = np.arange(K) / (K - 1)
    y = network_output(params, z, **kw)
    return np.mean((target(z) - y) ** 2)


def main():
    # Quick check that the simulation and loss run end-to-end with random params.
    params = init_params()
    phi = loss(params, K=250, M=100)
    print(f"loss (random params): {phi:.4f}")


if __name__ == "__main__":
    main()
