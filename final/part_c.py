from toolbox import formulation
import matplotlib.pylab as plt
import numpy as np


def const_step_grid(x0: float, xf: float, h: float):
    x = x0
    xs = [x]
    while x < xf:
        x += h
        xs.append(x)
    return np.array(xs)


def make_contour_plt(ax, xi: np.ndarray, tau: np.ndarray, theta: np.ndarray):
    # get a grid of xy values
    X, Y = np.meshgrid(tau, xi)
    print(X.shape, Y.shape)
    # theta should have the shape (tau,xi,1)
    ax.contourf(X, Y, theta, levels=20, cmap="coolwarm")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(r"$\theta$")


params = {
    "A": 5.0,
    "xi0": 0.3,
    "sigma": 0.05,
    "beta": 4.0,
    "theta0": 0.0,
    "thetaf": 1.0,
}

for omega in (4, 400):
    n = 32
    xi = formulation.grid1d(0.0, 1.0, n)
    h = (xi[-1] - xi[0]) / n
    taus = const_step_grid(0.0, 0.25, 0.49 * h**2)
    fluid_fn = formulation.get_sin_plume_fn(
        params["A"],
        omega,
        params["xi0"],
        params["sigma"],
    )
    fluid_grid = np.array([fluid_fn(tau, xi) for tau in taus])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    make_contour_plt(ax, xi, taus, fluid_grid)
