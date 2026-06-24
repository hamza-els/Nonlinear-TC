import numpy as np
import matplotlib.pyplot as plt

from Graphs.neuron_dynamics import simulate, I_DEFAULT, J_QUARTIC


def main():
    # Fig. 1(d): quartic-potential neuron J = (0, 0, 1), beta = 100.
    t, x = simulate(J_QUARTIC, I_DEFAULT, tf=10.0, dt=1e-3, beta=100.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(I_DEFAULT)))
    for traj, c in zip(x, colors):
        ax.plot(t, traj, color=c, lw=0.6)

    ax.set_title("(0, 0, 1)")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig("quartic_graph.png", dpi=150)
    print("saved quartic_graph.png")


if __name__ == "__main__":
    main()
