"""Animate a thermodynamic computer as it runs.

Runs a single Langevin trajectory for a fixed input z with a given set of
weights, and animates the network graph: each neuron is a node whose color
tracks its activation x_i(t) over time, and edges are drawn for the bilinear
couplings (colored / sized by weight, so the picture also shows the wiring).

This is intentionally simple -- it visualizes one run with one set of weights.
It can later be extended to compare weight sets or sweep inputs.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # save to file; no blocking window in this environment
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

from cosine_training_torch import LAYERS, J_INTRINSIC

# --- Architecture (recomputed here in plain numpy, decoupled from torch) ---
N = sum(LAYERS)
OFF = np.cumsum([0] + LAYERS)                 # layer boundaries in the flat array
LAYER_IDX = [np.arange(OFF[l], OFF[l + 1]) for l in range(len(LAYERS))]
INPUT_IDX = LAYER_IDX[0]
OUTPUT_IDX = LAYER_IDX[-1]
N_INPUTS = 1


# --- Weights --------------------------------------------------------------
def random_params(scale=0.5, seed=None):
    """A random set of weights, deliberately larger than the GA init scale so
    the dynamics are visibly active. Replace with trained weights later.

    Returns a dict:
        W : (LAYERS[0],)             input weights coupling z into layer 0.
        b : (N,)                     per-neuron biases.
        J : list of (s_l, s_{l+1})   bilinear couplings between adjacent layers.
        f : (len(OUTPUT_IDX),)       output weights (unused for the animation).
    """
    rng = np.random.default_rng(seed)
    return {
        "W": rng.normal(scale=scale, size=LAYERS[0]),
        "b": rng.normal(scale=scale, size=N),
        "J": [rng.normal(scale=scale, size=(LAYERS[l], LAYERS[l + 1]))
              for l in range(len(LAYERS) - 1)],
        "f": rng.normal(scale=scale, size=len(OUTPUT_IDX)),
    }


def coupling_matrix(J):
    """Symmetric N x N coupling matrix from the per-layer coupling blocks."""
    Jc = np.zeros((N, N))
    for l, Wmat in enumerate(J):
        a0, a1 = OFF[l], OFF[l + 1]
        b0, b1 = OFF[l + 1], OFF[l + 2]
        Jc[a0:a1, b0:b1] = Wmat
        Jc[b0:b1, a0:a1] = Wmat.T
    return Jc


# --- Simulation (records the full trajectory of every neuron) --------------
def simulate_trajectory(params, z, tf=1.0, dt=1e-3, beta=10.0, mu=1.0, seed=None):
    """One reset-sampling trajectory, recording x_i(t) for all neurons.

    Returns
    -------
    t : (n_steps+1,)        time axis.
    history : (n_steps+1, N) neuron activations at every step.
    """
    J2, J3, J4 = J_INTRINSIC
    rng = np.random.default_rng(seed)
    Jc = coupling_matrix(params["J"])
    b = params["b"]

    ext = np.zeros(N)
    ext[INPUT_IDX] = params["W"] * z       # external field W_i z on input layer

    n_steps = int(round(tf / dt))
    noise_amp = np.sqrt(2.0 * mu * (1.0 / beta) * dt)

    x = np.zeros(N)
    history = np.empty((n_steps + 1, N))
    history[0] = x
    for k in range(n_steps):
        dVdx = (2 * J2 * x + 3 * J3 * x ** 2 + 4 * J4 * x ** 3) - b + Jc @ x + ext
        x = x - mu * dVdx * dt + noise_amp * rng.standard_normal(N)
        history[k + 1] = x

    t = np.arange(n_steps + 1) * dt
    return t, history


# --- Layout ---------------------------------------------------------------
def node_positions(x_spacing=3.0, y_spread=4.0):
    """(x, y) position of every neuron: x = layer index (spread out by
    x_spacing to leave room for edge labels), y = centered vertical spread."""
    pos = np.zeros((N, 2))
    for l, idx in enumerate(LAYER_IDX):
        s = len(idx)
        ys = np.linspace(y_spread, -y_spread, s) if s > 1 else np.array([0.0])
        pos[idx, 0] = l * x_spacing
        pos[idx, 1] = ys
    return pos


# --- Visualization --------------------------------------------------------
def animate(params, z, out_path="network_run.gif", tf=1.0, dt=1e-3,
            beta=10.0, mu=1.0, stride=10, fps=20, seed=0):
    """Run a trajectory with the given params and save an animated GIF of the
    network: neuron nodes are colored AND labeled by their activation x_i(t),
    and every connection is drawn and labeled with its weight."""
    t, history = simulate_trajectory(params, z, tf=tf, dt=dt, beta=beta,
                                     mu=mu, seed=seed)
    x_spacing, y_spread = 3.0, 4.0
    pos = node_positions(x_spacing, y_spread)
    z_pos = np.array([-x_spacing, 0.0])

    # output y(t) = sum_i f_i x_i(t) over output neurons, vs target cos(2 pi z).
    y_trace = history[:, OUTPUT_IDX] @ params["f"]
    y_target = float(np.cos(2.0 * np.pi * z))
    err_trace = y_target - y_trace

    fig, (ax, ax_err) = plt.subplots(
        1, 2, figsize=(18, 11), gridspec_kw={"width_ratios": [4, 1]})
    ax.set_xlim(-x_spacing - 1.2, (len(LAYERS) - 1) * x_spacing + 1.2)
    ax.set_ylim(-y_spread - 1.2, y_spread + 1.2)
    ax.axis("off")

    # --- edges: one line per coupling, red/blue by sign, width by |w| --------
    segments, edge_colors, edge_widths = [], [], []
    for l, Wmat in enumerate(params["J"]):
        for a, i in enumerate(LAYER_IDX[l]):
            for c, j in enumerate(LAYER_IDX[l + 1]):
                w = Wmat[a, c]
                segments.append([pos[i], pos[j]])
                edge_colors.append((0.8, 0.1, 0.1) if w >= 0 else (0.1, 0.1, 0.8))
                edge_widths.append(abs(w))
    for a, i in enumerate(INPUT_IDX):                  # external z into layer 0
        w = params["W"][a]
        segments.append([z_pos, pos[i]])
        edge_colors.append((0.8, 0.1, 0.1) if w >= 0 else (0.1, 0.1, 0.8))
        edge_widths.append(abs(w))
    maxw = max(edge_widths) or 1.0
    edge_widths = [0.4 + 2.0 * (w / maxw) for w in edge_widths]
    ax.add_collection(LineCollection(segments, colors=edge_colors,
                                     linewidths=edge_widths, alpha=0.3, zorder=1))

    # --- nodes: scatter colored by activation, with a live value label -------
    vmax = max(2.0, np.percentile(np.abs(history), 99))
    nodes = ax.scatter(pos[:, 0], pos[:, 1], c=history[0], cmap="coolwarm",
                       vmin=-vmax, vmax=vmax, s=900, edgecolors="k",
                       linewidths=0.8, zorder=2)
    val_texts = [
        ax.text(pos[i, 0], pos[i, 1], f"{history[0, i]:.2f}", ha="center",
                va="center", fontsize=8, zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        for i in range(N)
    ]

    ax.scatter(*z_pos, s=800, marker="s", color="0.3", zorder=2)  # input node
    ax.text(z_pos[0], z_pos[1] - 0.55, f"z = {z:.2f}", ha="center", fontsize=11)
    for l in range(len(LAYERS)):
        ax.text(l * x_spacing, y_spread + 0.8, f"L{l}", ha="center",
                fontsize=12, color="0.4")
    title = ax.set_title("", fontsize=14)

    # --- error panel: output y(t) converging toward the target --------------
    ylim = max(1.2, 1.1 * np.abs(np.append(y_trace, y_target)).max())
    ax_err.set_xlim(0, t[-1])
    ax_err.set_ylim(-ylim, ylim)
    ax_err.set_xlabel("t")
    ax_err.set_title("output vs target", fontsize=12)
    ax_err.axhline(y_target, color="k", ls="--", lw=1.2, label=f"target {y_target:+.2f}")
    (y_line,) = ax_err.plot([], [], color="tab:green", lw=1.8, label="output y(t)")
    (err_bar,) = ax_err.plot([], [], color="tab:red", lw=2.0)        # current gap
    (y_dot,) = ax_err.plot([], [], "o", color="tab:green", ms=6)
    ax_err.legend(loc="upper right", fontsize=8)
    err_text = ax_err.text(0.04, 0.03, "", transform=ax_err.transAxes,
                           fontsize=11, va="bottom")

    frames = range(0, len(t), stride)

    def update(k):
        nodes.set_array(history[k])
        for i, txt in enumerate(val_texts):
            txt.set_text(f"{history[k, i]:.2f}")
        title.set_text(f"t = {t[k]:.3f}")
        y_line.set_data(t[:k + 1], y_trace[:k + 1])
        y_dot.set_data([t[k]], [y_trace[k]])
        err_bar.set_data([t[k], t[k]], [y_trace[k], y_target])
        err_text.set_text(f"error = {err_trace[k]:+.3f}")
        return [nodes, title, y_line, y_dot, err_bar, err_text, *val_texts]

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {out_path}")


# --- Trained weights ------------------------------------------------------
def params_from_population(pop, p):
    """Pull computer p's weights out of a GA population (batched torch tensors)
    and convert to the plain-numpy params dict the visualizer expects."""
    return {
        "W": pop["W"][p].detach().cpu().numpy(),
        "b": pop["b"][p].detach().cpu().numpy(),
        "J": [Wmat[p].detach().cpu().numpy() for Wmat in pop["J"]],
        "f": pop["f"][p].detach().cpu().numpy(),
    }


def save_params(params, path):
    """Save a numpy params dict to a .npz file."""
    arrs = {"W": params["W"], "b": params["b"], "f": params["f"]}
    for l, Jm in enumerate(params["J"]):
        arrs[f"J{l}"] = Jm
    np.savez(path, **arrs)
    print(f"saved weights -> {path}")


def load_params(path):
    """Load a numpy params dict saved by save_params."""
    d = np.load(path)
    n_J = sum(1 for k in d.files if k.startswith("J"))
    return {"W": d["W"], "b": d["b"], "f": d["f"],
            "J": [d[f"J{l}"] for l in range(n_J)]}


def train_and_visualize(z=0.5, out_path="network_trained.gif",
                        weights_path="trained_weights.npz",
                        generations=100, P=50, K=64, M=32, m_chunk=None,
                        var_weight=0.0, loss_mode="squared", device=None):
    """Train a population by GA, save the best computer's weights, then
    visualize how it runs. Uses the GPU automatically if a CUDA build of
    torch is installed. m_chunk caps how many samples are simulated at once
    (lower it if you hit GPU out-of-memory). var_weight (lambda ~ 1/M_inf) adds
    a per-sample output-variance penalty; loss_mode is "squared" (expected
    squared error) or "rms" (expected single-readout error magnitude)."""
    import torch
    from genetic_algorithm import train, population_loss, resolve_device

    device = resolve_device(device)
    pop, _ = train(generations=generations, P=P, K=K, M=M, m_chunk=m_chunk,
                   var_weight=var_weight, loss_mode=loss_mode, device=device)
    z_eval = torch.arange(K, dtype=torch.float32, device=device) / (K - 1)
    losses = population_loss(pop, z_eval, M=M, m_chunk=m_chunk,
                            var_weight=var_weight, loss_mode=loss_mode)
    best = int(losses.argmin())
    print(f"best loss after {generations} gens: {losses[best].item():.4f}")

    params = params_from_population(pop, best)
    save_params(params, weights_path)
    animate(params, z=z, out_path=out_path)


def visualize_saved(weights_path="trained_weights.npz", z=0.5,
                    out_path="network_trained.gif"):
    """Load saved weights and visualize a run (no training)."""
    animate(load_params(weights_path), z=z, out_path=out_path)


def main():
    # Random weights (fast). For a trained network use train_and_visualize().
    params = random_params(scale=0.5, seed=1)
    animate(params, z=0.25, out_path="network_run.gif")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
