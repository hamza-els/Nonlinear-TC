import numpy as np
import matplotlib.pyplot as plt

from neuron_dynamics import simulate, I_DEFAULT, J_QUADRATIC


def main():
    # Fig. 1(c): quadratic-potential neuron J = (1, 0, 0), beta = 100.
    t, x = simulate(J_QUADRATIC, I_DEFAULT, tf=10.0, dt=1e-3, beta=100.0)

    fig, ax = plt.subplots(figsize=(4, 4))
    # color by input: blue (low I) -> red (high I), as in the paper.
    colors = plt.cm.coolwarm(np.linspace(0.0, 1.0, len(I_DEFAULT)))
    for traj, c in zip(x, colors):
        ax.plot(t, traj, color=c, lw=0.6)

    ax.set_title("(1, 0, 0)")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    fig.tight_layout()
    fig.savefig("quadratic_graph.png", dpi=150)
    print("saved quadratic_graph.png")


if __name__ == "__main__":
    main()
