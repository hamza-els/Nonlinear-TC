import numpy as np


def U(x, J, I):
    """Neuron potential U_J(x,I) = J2 x^2 + J3 x^3 + J4 x^4 - I x."""
    J2, J3, J4 = J
    return J2 * x ** 2 + J3 * x ** 3 + J4 * x ** 4 - I * x


def grad_U(x, J, I):
    """Gradient dU/dx of the neuron potential U_J(x,I) = J2 x^2 + J3 x^3 + J4 x^4 - I x."""
    J2, J3, J4 = J
    return 2.0 * J2 * x + 3.0 * J3 * x ** 2 + 4.0 * J4 * x ** 3 - I


def equilibrium_activation(J, I_values, beta=1.0, x_max=20.0, n_x=20001):
    """Equilibrium activation function <x>_0(I), Eq. (2):

        <x>_0 = int dx x exp(-beta U_J(x,I)) / int dx exp(-beta U_J(x,I))

    evaluated by numerical quadrature on a fixed grid in x. Returns an array
    aligned with I_values.
    """
    I_values = np.atleast_1d(np.asarray(I_values, dtype=float))
    x = np.linspace(-x_max, x_max, n_x)

    # Boltzmann weight per input; subtract the per-column min of beta*U for
    # numerical stability before exponentiating.
    bU = beta * U(x[:, None], J, I_values[None, :])
    w = np.exp(-(bU - bU.min(axis=0, keepdims=True)))

    Z = np.trapz(w, x, axis=0)
    mean = np.trapz(x[:, None] * w, x, axis=0) / Z
    return mean


def simulate(J, I_values, tf=10.0, dt=1e-3, beta=100.0, mu=1.0, x0=0.0):
    """Evolve a single thermodynamic neuron under overdamped Langevin dynamics (Eq. 3).

        x_dot = -mu * dU/dx + sqrt(2 mu kB T) eta(t)

    integrated with the Euler-Maruyama scheme. One independent trajectory is
    produced per value of I.

    Parameters
    ----------
    J : tuple (J2, J3, J4)   intrinsic couplings of the neuron.
    I_values : array-like     input signals; one trajectory each.
    tf, dt : float            total time and timestep (time in units of mu^-1).
    beta : float              inverse temperature 1/(kB T).
    mu : float                mobility (basic time constant).
    x0 : float                initial neuron activation.

    Returns
    -------
    t : (n_steps+1,) array of times.
    x : (len(I_values), n_steps+1) array of trajectories.
    """
    I_values = np.atleast_1d(np.asarray(I_values, dtype=float))
    kT = 1.0 / beta

    n_steps = int(round(tf / dt))
    t = np.arange(n_steps + 1) * dt

    x = np.empty((I_values.size, n_steps + 1))
    x[:, 0] = x0
    noise_amp = np.sqrt(2.0 * mu * kT * dt)

    xi = np.full(I_values.size, float(x0))
    for k in range(n_steps):
        drift = -mu * grad_U(xi, J, I_values)
        xi = xi + drift * dt + noise_amp * np.random.normal(size=I_values.size)
        x[:, k + 1] = xi

    return t, x


# Default settings used to reproduce Fig. 1(c,d): 11 evenly-spaced inputs in [-5, 5].
I_DEFAULT = np.linspace(-5.0, 5.0, 11)
J_QUADRATIC = (1.0, 0.0, 0.0)  # Fig. 1(c)
J_QUARTIC = (0.0, 0.0, 1.0)    # Fig. 1(d)
