import numpy as np
from typing import Union
from . import solvers
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def get_h(x0: float, xf: float, n: int) -> float:
    return (xf - x0) / n


def const_step_grid(x0: float, xf: float, h: float):
    x = x0
    xs = [x]
    while x < xf:
        x += h
        xs.append(x)
    return np.array(xs)


def solve_ss_analytic(
    x0: float,
    xf: float,
    fluid: float,
    beta: float,
    theta0: float,
    thetaf: float,
) -> callable:
    # get analytic solution for condition:
    # bc1 = x @ t=1
    # bc2 = x @ t=2

    # equation is of form
    # x(t) = c1 * exp( sqrt(beta) * t ) + c2 * exp( -sqrt(beta) * t ) + theta_f

    # solve c1, c2 for boundaries
    # x(0) = 0 = c1 * (1) + c2 * (1) + theta_f
    # x(1) = 1 = c1 * exp(sqrt(beta)) + c2 * exp(-sqrt(beta)) + theta_f

    # create matrix to find c1, c2
    if beta < 0.0:
        raise ValueError("Beta must be non-negative!")

    A = np.array(
        [
            [np.exp(np.sqrt(beta) * x0), np.exp(-np.sqrt(beta) * x0)],
            [np.exp(np.sqrt(beta) * xf), np.exp(-np.sqrt(beta) * xf)],
        ]
    )
    b = np.array([[theta0 - fluid], [thetaf - fluid]])
    c = np.linalg.solve(A, b)
    c0 = c[0, 0]
    c1 = c[1, 0]

    def soln(xi: np.ndarray) -> np.ndarray:
        c1term = c0 * np.exp(np.sqrt(beta) * xi)
        c2term = c1 * np.exp(-np.sqrt(beta) * xi)
        return c1term + c2term + fluid

    return soln


def grid1d(x0: float, xf: float, n: int = 8) -> np.ndarray:
    """get a 1-d grid of size `n` that spans the range `[x0, xf]`"""
    return np.linspace(x0, xf, n + 1)


def get_A(n, beta, h):
    A = np.zeros((n, n))
    # k=0 and k=n correspond to left, right boundary
    A[0, 0] = 1.0
    A[n - 1, n - 1] = 1.0
    # interior grid points
    for k in range(1, n - 1):
        A[k, k] = +2 / h**2 + beta
        A[k, k - 1] = -1.0 / h**2
        A[k, k + 1] = -1.0 / h**2
    return A


def get_b(n, beta, theta0, thetaf):
    b = beta * np.ones((n,))
    b[0], b[n - 1] = theta0, thetaf
    return b


def get_Ab(
    xi: np.ndarray,
    fluid: Union[np.ndarray, float],
    beta: float,
    theta0: float,
    thetaf: float,
) -> tuple:
    """Get a matrix A and vector b, which, when solved, produces the temperature profile."""

    n = xi.shape[0]
    # get tridiagonal matrix
    h = (xi[n - 1] - xi[0]) / (n - 1)
    A = get_A(n, beta, h)

    b = get_b(n, beta, theta0, thetaf)
    # fill interior points
    if isinstance(fluid, float):
        b[1 : n - 1] *= fluid
    # fill from array values
    elif isinstance(fluid, np.ndarray):
        b[1 : n - 1] *= fluid[1 : n - 1]
    else:
        # type
        raise TypeError(f"fluid must be either float or ndarray (type:{type(fluid)}")
    return A, b


def gaussian_plume(
    xi: np.ndarray,
    A: float,
    xi0: float,
    sigma: float,
) -> np.ndarray:
    """get an array of size `xi` that represents a temperature plume in theta_f centered at xi0, with amplitude A, and with width sigma"""
    return A * np.exp(-((xi - xi0) ** 2) / (2 * sigma**2))


def solve_time_rk2(
    xi: np.ndarray,
    taus: np.ndarray,
    beta: float,
    theta0: float,
    thetaf: float,
    fluid_eqn: callable,
    weight: float = 1.0,
) -> tuple:
    """solve the time-dependent equation dx/dtau = d2x / dxi^2 + beta * x + f(tau, xi)"""
    n = xi.shape[0]
    m = taus.shape[0]

    solution = np.empty(shape=(m, n))
    fluids = np.empty(shape=(m, n))
    # set initial fluid condition
    fluids[0, :] = fluid_eqn(taus[0], xi)

    # set IC
    # get steady state A,b
    ssA, ssb = get_Ab(xi, fluids[0], beta, theta0, thetaf)
    # solve SS system to get tau=0 I.C.
    ss_theta = solvers.tdma(ssA, ssb)
    # first member of solution array is the I.C.
    solution[0, :] = ss_theta

    # xi step
    h = (xi[n - 1] - xi[0]) / (n - 1)
    # tau step
    t = (taus[m - 1] - taus[0]) / (m - 1)

    c1 = 1.0 / (2.0 * weight)
    c2 = 1.0 - 1.0 / (2.0 * weight)

    A = get_A(n, beta, h) * -1.0

    for j, tau in enumerate(taus[1:]):
        fluid = fluid_eqn(tau, xi)

        # store into fluid array
        fluids[j + 1, :] = fluid

        # "b" vector = beta * theta_fluid
        b = beta * fluid
        # set left, right boundary
        b[0], b[n - 1] = theta0, thetaf
        # predictor
        k1 = A @ solution[j] + b
        y1 = solution[j] + weight * t * k1
        # corrector
        k2 = A @ y1 + b
        # weighted average
        solution[j + 1] = solution[j] + t * (c1 * k1 + c2 * k2)

    return fluids, solution


def solve_time_crank(
    xi: np.ndarray,
    taus: np.ndarray,
    beta: float,
    theta0: float,
    thetaf: float,
    fluid_eqn: callable,
) -> tuple:
    n = xi.shape[0]
    m = taus.shape[0]

    solution = np.empty(shape=(m, n))
    fluids = np.empty(shape=(m, n))
    # set initial fluid condition
    fluids[0, :] = fluid_eqn(taus[0], xi)

    # set IC
    # get steady state A,b
    ssA, ssb = get_Ab(xi, fluids[0], beta, theta0, thetaf)
    # solve SS system to get tau=0 I.C.
    ss_theta = solvers.tdma(ssA, ssb)
    # first member of solution array is the I.C.
    solution[0, :] = ss_theta

    # xi step
    h = (xi[n - 1] - xi[0]) / (n - 1)
    # tau step
    t = (taus[m - 1] - taus[0]) / (m - 1)

    A = get_A(n, beta, h)
    I = np.eye(n)

    for j, tau in enumerate(taus[1:]):
        fluid = fluid_eqn(tau, xi)
        # store into fluid array
        fluids[j + 1, :] = fluid

        b1 = get_b(n, beta, theta0, thetaf)
        b1[1 : n - 1] = fluids[j + 1, 1 : n - 1]
        b0 = get_b(n, beta, theta0, thetaf)
        b0[1 : n - 1] = fluids[j, 1 : n - 1]

        c = (I - 0.5 * t * A) @ solution[j] + 0.5 * t * (b0 + b1)
        B = I + 0.5 * t * A

        sol = solvers.tdma(B, c)

        solution[j + 1, :] = sol

    return fluids, solution


def get_sin_plume_fn(A: float, omega: float, xi0: float, sigma: float) -> callable:
    def f(tau: float, xi: np.ndarray):
        return A * np.sin(omega * tau) * np.exp(-((xi - xi0) ** 2) / (2 * sigma**2))

    return f


def get_step_plume_fn(A, t1, t2, xi0, sigma, xi):
    middle = (xi < (xi0 + sigma)) & (xi > (xi0 - sigma))

    def f(tau: float, xi: np.ndarray):
        fluid = np.empty_like(xi)
        if tau < t1:
            fluid = np.full_like(xi, 0.0)
        elif tau > t1 and tau < t2:
            fluid = np.full_like(xi, 0.0)
            fluid[middle] = A
        else:
            fluid = np.full_like(xi, 0.0)
        return fluid

    return f


def animate_results(xi, taus, fluids, solutions, skip=10, interval=1):
    """Make an animation of the temperature profile over time"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    (fluid,) = ax.plot([], [], "r-", label="Fluid")
    (solution,) = ax.plot([], [], "k-", label=r"$\theta$")
    text = ax.text(0.25, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="upper left")

    def init():
        xmax = xi.max()
        xmin = xi.min()
        ax.set_xlim(xmin, xmax)
        ymin = solutions.min() - 1.0
        ymax = solutions.max() + 1.0
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\theta$")

        return fluid, solution, text

    def update(i):
        fluid.set_data(xi, fluids[i * skip, :])
        solution.set_data(xi, solutions[i * skip, :])
        text.set_text(rf"$\tau$ = {taus[i * skip]:.2f}")
        return fluid, solution, text

    ani = FuncAnimation(
        fig,
        update,
        frames=list(range(taus.shape[0] // skip)),
        init_func=init,
        blit=True,
        interval=interval,
    )
    return ani


def make_contour_plt(
    ax,
    xi: np.ndarray,
    tau: np.ndarray,
    theta: np.ndarray,
    title: str,
):
    """Make a 2d contour plot of the temperature profile over time"""
    # get a grid of xy values
    X, Y = np.meshgrid(tau, xi)
    # theta should have the shape (tau,xi,1)
    ax.contourf(Y.T, X.T, theta, levels=40, cmap="coolwarm")
    ax.set_ylabel(r"$\tau$")
    ax.set_xlabel(r"$\xi$")
    ax.set_title(title)


if __name__ == "__main__":
    # test analytic solution against numerical solution
    import solvers
    import matplotlib.pyplot as plt

    beta = 4.0
    fluid_temp = 5.0
    theta0, thetaf = 4.0, 4.0
    n = 256
    xi = grid1d(0.0, 1.0, n)

    # true
    f = solve_ss_analytic(0.0, 1.0, fluid_temp, beta, theta0, thetaf)
    theta_actual = f(xi)
    A, b = get_Ab(xi, fluid_temp, beta, theta0, thetaf)
    theta = solvers.tdma(A, b)

    # plume
    fluid_temp = gaussian_plume(xi, 12.0, 0.3, 0.05)
    A_, b_ = get_Ab(xi, fluid_temp, beta, theta0, thetaf)
    theta_plume = solvers.tdma(A_, b_)

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(xi, theta_actual, label="True")
    ax[0].plot(xi, theta, label="Numerical")

    ax[1].plot(xi, theta_plume, label="Temperature")
    ax[1].plot(xi, fluid_temp, label="Fluid Temperature")
    for a in ax:
        a.legend()
    plt.show()
