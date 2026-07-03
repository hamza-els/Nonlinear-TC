import numpy as np
import matplotlib.pyplot as plt

from neuron_dynamics import equilibrium_activation


def main():
    # Fig. 1(b): equilibrium activation function <x>_0(I) at beta = 1.
    I = np.linspace(-10.0, 10.0, 401)

    cases = [
        ((1.0, 0.0, 0.0), "(1, 0, 0)", "0.5"),      # quadratic: linear, gray
        ((0.0, 0.0, 1.0), "(0, 0, 1)", "tab:blue"),  # pure quartic: nonlinear
        ((1.0, 0.0, 1.0), "(1, 0, 1)", "tab:green"), # quadratic-quartic
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    for J, label, color in cases:
        m = equilibrium_activation(J, I, beta=1.0)
        ax.plot(I, m, color=color, lw=1.5, label=label)

    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("I")
    ax.set_ylabel(r"$\langle x \rangle_0$")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-2, 2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("activation_func_graph.png", dpi=150)
    print("saved activation_func_graph.png")


if __name__ == "__main__":
    main()
