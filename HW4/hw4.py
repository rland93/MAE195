import numpy as np
from cmath import sqrt
from enum import Enum
import math
import matplotlib.pyplot as plt

# set font sizes
plt.rcParams.update({"font.size": 9})


class damped(Enum):
    OVER = 0
    CRITICALLY = 1
    UNDER = 2


def get_lambda(alpha, omega):
    """Get eigenvalues for the problem

    Parameters
    ----------
    alpha : float
        alpha coefficient
    omega : float
        omega coefficient

    Returns
    -------
    tuple(complex, complex)
        the complex-valued eigenvalues. will always return complex numbers,
        even if eigenvalues have only real parts.
    """
    lambda1 = -complex(alpha / 2.0) + sqrt(complex(alpha**2 / 4.0 - omega**2))
    lambda2 = -complex(alpha / 2.0) - sqrt(complex(alpha**2 / 4.0 - omega**2))
    return lambda1, lambda2


def analytic_soln(alpha: float, omega: float, x0: float, dx0: float):
    """Get the analytic solution for the mass-spring system with
    normalized damping coefficient \alpha and normalized natural
    frequency \omega. The equation is of the form

    0 = m d2x/dt2 + c dx/dt + k x

    which is simplified

    0 = d2x/dt2 + alpha dx/dt + omega^2 x

    Parameters
    ----------
    alpha : float
        normalized damping coefficient c/m
    omega : float
        normalized natural frequency sqrt(k/m)
    x0 : float
        initial position of mass
    dx0 : float
        initial velocity of mass

    Returns
    -------
    callable
        The analytic solution x(t)
    """

    # convert to complex
    alpha = complex(alpha)
    omega = complex(omega)

    # get eigenvalues
    lambda1, lambda2 = get_lambda(alpha, omega)

    # evaluate critical/over/under damped
    if lambda1 == lambda2 and lambda1.imag == 0 and lambda2.imag == 0:
        case = damped.CRITICALLY
    elif lambda1.imag == 0 and lambda2.imag == 0:
        case = damped.OVER
    elif lambda1.imag != 0 and lambda2.imag != 0:
        case = damped.UNDER

    if case == damped.CRITICALLY:
        # The form of the analytic solution is:
        #
        # (c1 + t*c2) * exp(Re(eig) * t)
        #
        # therefore, we can write a linear system of the from
        # [A] {c} = {x_0}
        # to solve for c1, c2.
        A = np.array([[1, 0], [lambda1.real, 1]])
        x = np.array([x0, dx0])
        c = np.linalg.solve(A, x)
        c1, c2 = c[0], c[1]
        # return the functional form of the solution
        def solution(t):
            soln = (c1 + t * c2) * np.exp(lambda1.real * t)
            return np.real(soln)

    elif case == damped.OVER:
        # The form of the analytic solution is
        #
        # x = c1 exp(Re(lambda1) * t) + c2 exp(Re(lambda2) * t)
        #
        # Then, we can write a linear system of the form
        # [A] {c} = {x_0}
        # to solve for c1 and c2
        A = np.array([[1, 1], [lambda1.real, lambda2.real]])
        x = np.array([x0, dx0])
        c = np.linalg.solve(A, x)
        c1, c2 = c[0], c[1]
        # create function for analytic solution
        def solution(t):
            soln = c1 * np.exp(lambda1.real * t) + c2 * np.exp(lambda2.real * t)
            return np.real(soln)

    elif case == damped.UNDER:
        # The form of the analytic solution is
        #
        # x = exp(Re(lambda1) t) * (c1 cos(Im(lambda1) t) + c2 sin(Im(lambda1) t))
        #
        # Then, we can write a linear system of the form
        # [A] {c} = {x_0}
        # to solve for c1 and c2
        A = np.array([[1, 0], [lambda1.real, lambda1.imag]])
        x = np.array([x0, dx0])
        c = np.linalg.solve(A, x)
        c1, c2 = c[0], c[1]
        # create function for analytic solution
        def solution(t):
            exponent = np.exp(lambda1.real * t)
            sinusoid = c1 * np.cos(lambda1.imag * t) + c2 * np.sin(lambda1.imag * t)
            return np.real(exponent * sinusoid)

    # return the function and the case
    return solution, case


def condition(alpha: float, omega: float) -> float:
    """From alpha and omega parameters, calculate the spectral condition
    number (aka stiffness) of the problem.

    Print a warning if the spectral condition number is indeterminate.

    Parameters
    ----------
    alpha : float
        parameter alpha
    omega : float
        parameter omega

    Returns
    -------
    float
        spectral condition number.
    """

    lambda1, lambda2 = get_lambda(alpha, omega)
    print(f"λ1={lambda1.real} + {lambda1.imag}j")
    print(f"λ2={lambda2.real} + {lambda2.imag}j")
    if lambda2.real == 0 or lambda1.real == 0:
        print("WARNING: Eigenvalues of Zero!")
    try:
        num = max(abs(lambda1.real), abs(lambda2.real))
        denom = min(abs(lambda1.real), abs(lambda2.real))
        return num / denom
    except ZeroDivisionError:
        return float("inf")


def get_max_euler_stepsize(alpha: float, omega: float):
    """Given alpha, omega of the problem, calculate the maximum
    Euler step size.

    Parameters
    ----------
    alpha : float
        alpha parameter
    omega : float
        omega parameter

    Returns
    -------
    float
        Step Size
    """
    lambda1, lambda2 = get_lambda(alpha, omega)
    max_lambda = max(lambda1.real, lambda2.real)
    if max_lambda == 0:
        return float("inf")
    else:
        return 2.0 / max_lambda


def test_analytic_solns(
    plot=False, alphamax: int = 5, omegamax: int = 4, t_final: float = 10.0
):
    """Test and plot a bunch of analytic solutions for a bunch of
    alpha, omega values.

    will test all values of alpha, omega in the cartesian product
    of range(alphamax) X range(omegamax) ... This means alphamax * omegamax
    solutions are generated!

    Pass plot=True to see a time series of the analytic solution.

    Parameters
    ----------
    plot : bool, optional
        Whether to plot each solution, by default False
    alphamax : int, optional
        maximum alpha, by default 5
    omegamax : int, optional
        maximum omega, by default 4
    t_final : float, optional
        final time, by default 10.0
    """
    from itertools import product
    import matplotlib.pyplot as plt

    for alpha, omega in product(range(alphamax), range(omegamax)):
        print(f"\nalpha = {alpha}, omega = {omega}")
        print(condition(alpha, omega))
        ts = np.linspace(0, t_final, 400)
        soln, status = analytic_soln(alpha, omega, 4.0, 0)
        if plot:
            plt.plot(ts, soln(ts))
            plt.title(f"alpha={alpha}, omega={omega}, {status}")
            plt.show()


def get_ts_array(t_final: float, step: float) -> np.ndarray:
    """Get array of times in the interval [0, t_final], with step `step`

    Parameters
    ----------
    t_final : float
        stop time
    step : float
        step size h

    Returns
    -------
    np.ndarray
        the times array, size (t_final/h, )
    """
    t = 0.0
    ts = []
    # increase t by step till t_final reached
    while t <= t_final:
        ts.append(t)
        t += step
    return np.array(ts)


def euler(
    ts: float,
    alpha: float,
    omega: float,
    x0: float,
    dx0: float,
) -> np.ndarray:
    """Euler method a.k.a 1st order RK.

    Parameters
    ----------
    ts : float
        times
    alpha : float
        problem parameter
    omega : float
        problem parameter
    x0 : float
        initial condition
    dx0 : float
        initial condition

    Returns
    -------
    np.ndarray
        solution
    """
    # empty solution array
    ys = np.empty((ts.shape[0], 2))

    # equation
    A = np.array([[0, 1], [-(omega**2), -alpha]])

    # set initial condition
    ys[0, 0], ys[0, 1] = x0, dx0

    # solve forward
    for i, ti in enumerate(ts[:-1]):
        h = ts[i + 1] - ts[i]
        ys[i + 1] = ys[i] + h * A @ ys[i]
    # return solution
    return ys


def rk2(
    ts: float,
    alpha: float,
    omega: float,
    x0: float,
    dx0: float,
    p=1.0,
) -> np.ndarray:
    """Second order Runge-Kutta method.

    Parameters
    ----------
    ts : float
        times
    alpha : float
        problem parameter
    omega : float
        problem parameter
    x0 : float
        initial condition
    dx0 : float
        initial condition
    p : float, optional
        weight, by default 1.0
        must be 0 <= p <= 1

    Returns
    -------
    np.ndarray
        solution
    """
    # constants
    a1 = 1.0 / (2.0 * p)
    a2 = 1 - 1.0 / (2.0 * p)
    assert a1 + a2 - 1.0 <= 1e-15
    # empty solution array
    ys = np.empty((ts.shape[0], 2))

    # set iv
    ys[0, 0], ys[0, 1] = x0, dx0

    # equation
    A = np.array([[0, 1], [-(omega**2), -alpha]])

    # loop forward
    for i, ti in enumerate(ts[:-1]):
        h = ts[i + 1] - ts[i]

        k1 = A @ ys[i]
        y1 = ys[i] + p * h * k1

        k2 = A @ y1
        # weighted average
        ys[i + 1] = ys[i] + h * (a1 * k1 + a2 * k2)

    return ys


def adams_moulton(ts, h, A, y0):
    """adams moulton method"""
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (shape={A.shape})")

    # empty solution array
    ys = np.empty((ts.shape[0], A.shape[0]))

    # set i.v.
    ys[0, :] = y0

    # solve sys
    I = np.eye(N=A.shape[0], M=A.shape[1])
    LHS = I + A * h * 0.5
    RHS = I - A * h * 0.5

    for i, ti in enumerate(ts[:-1]):
        # solve system at each timestep
        ys[i + 1, :] = np.linalg.solve(RHS, LHS @ ys[i, :])

        # ys[i + 1, :] = RHS_INV @ (LHS @ ys[i, :])
    return ys


def make_xt_numeric_plot(
    ax,
    ts: np.ndarray,
    ys_analytic: np.ndarray,
    ys_numeric: np.ndarray,
    method: str,
):
    """Make a numeric plot on the 2-column axes `ax`. In the first column,
    this method puts the solutions to an ivp, given in arrays `y_analytic` and
    `y_numeric`.

    In the second column, it will put the difference between the analytic and
    numeric solution for each timestep, i.e. `y_analytic`-`y_numeric`.

    Alters `ax` in place.

    Parameters
    ----------
    ax : Axes
        plot axes
    ts : np.ndarray
        length (N, ) Discrete times for which a solution exists
    ys_analytic : np.ndarray
        length (N, ) Analytic ivp solution
    ys_numeric : np.ndarray
        length (N, ) Numerical ivp solution
    method : str
        Method, used for labeling purposes
    """
    # plot x(t) for actual and numeric
    ax[0].plot(ts, ys_numeric[:, 0], label=f"{method}")
    # style plot
    ax[0].set_title(rf"$x(t), h={h}$")
    ax[0].set_ylabel(r"$x$")
    ax[0].set_xlabel(r"$t$")

    # difference between analytic x(t) and numerical x(t) solution
    ax[1].plot(ts, ys_analytic - ys_numeric[:, 0], label=f"Diff ({method})")
    # style plot
    ax[1].set_title(r"$x_{analytic}(t) - x_{numerical}(t)$, " + rf"$h={h}$")
    ax[1].set_ylabel(r"$x$")
    ax[1].set_xlabel(r"$t$")


def get_fig_title(prob_no: str, condition):
    fig_title = r"$x(t)$ for "
    fig_title += f"P. {prob_no}, "
    fig_title += r"$\kappa=$"
    fig_title += f"{condition}, "
    return fig_title


def make_err_plot(ax, problem: dict, errors: list, method: str):
    """Make an error-vs-stepsize plot

    Parameters
    ----------
    ax : Axes
        matplotlib axes on which to make the plot
    problem : dict
        Problem definition
    errors : list
        list (or ndarray) of error values
    method : str
        method, used for writing labels
    """

    # log-log error plot
    ax.plot(problem["hs"], errors, label=f"{method}")
    ax.scatter(problem["hs"], errors, label=f"{method}")
    # O n^1
    Oh1 = [min(problem["hs"]) * 0.1, max(problem["hs"]) * 0.1]
    # O n^2
    Oh2 = [min(problem["hs"]) ** 2 * 0.1, max(problem["hs"]) ** 2 * 0.1]
    # add O n^1 and O n^2
    ax.plot(
        [min(problem["hs"]), max(problem["hs"])],
        Oh1,
        color="black",
        linestyle=":",
        label=r"$\mathcal{O}(h)$",
    )
    ax.plot(
        [min(problem["hs"]), max(problem["hs"])],
        Oh2,
        color="gray",
        linestyle="-.",
        label=r"$\mathcal{O}(h^2)$",
    )
    # labels, etc.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step-size")
    ax.set_ylabel("error")


if __name__ == "__main__":
    # import matplotlib

    # matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    #     color=["r", "g", "b", "y"]
    # )

    large_figure = (10, 7)
    small_figure = (9, 6)
    t_final = 4.0
    # 3.1 definition
    CASE1 = {
        "alpha": 1.0,
        "omega": 1.0,
        "x0": 1.0,
        "dx0": 0.0,
        "hs": [0.1, 0.01, 0.001, 0.005],
        "cond": condition(1.0, 1.0),
    }
    # 3.2 definition
    CASE2 = {
        "alpha": 1001.0,
        "omega": math.sqrt(1000.0),
        "x0": 1.0,
        "dx0": 0.0,
        "hs": [0.00201, 0.00200, 0.00199, 0.0015],
        "cond": condition(1001.0, math.sqrt(1000.0)),
    }
    ##################################################
    #####               3.1                     ######
    ##################################################
    err_31 = []
    fig31, axs31 = plt.subplots(
        nrows=len(CASE1["hs"]),
        ncols=2,
        sharex=True,
        tight_layout=True,
        figsize=large_figure,
    )
    fig31.suptitle(get_fig_title("3.1", CASE1["cond"]))
    for (h, ax) in zip(CASE1["hs"], axs31):
        ts = get_ts_array(t_final, h)
        # get analytic function
        an, _ = analytic_soln(
            CASE1["alpha"],
            CASE1["omega"],
            CASE1["x0"],
            CASE1["dx0"],
        )
        # get analytic solution
        ys_an = an(ts)
        # get euler solution
        ys_nm = euler(
            ts,
            CASE1["alpha"],
            CASE1["omega"],
            CASE1["x0"],
            CASE1["dx0"],
        )
        # get error
        err_31.append(abs(ys_an[-1] - ys_nm[-1, 0]))
        # plot numerical solution
        make_xt_numeric_plot(ax, ts, ys_an, ys_nm, "Euler")
        # plot analytic solution
        ax[0].plot(ts, ys_an, linestyle=":", label="Analytic")
        # write legend for both plots
        for a in ax:
            a.legend()
    fig31err, ax31err = plt.subplots(figsize=small_figure)
    make_err_plot(ax31err, CASE1, err_31, "Euler")
    ax31err.set_title("Error vs Step-size, p3.1")

    ##################################################
    #####               3.2                     ######
    ##################################################
    err_32 = []
    ###### ensemble plot
    fig32, axs32 = plt.subplots(
        nrows=len(CASE2["hs"]),
        ncols=2,
        sharex=True,
        tight_layout=True,
        figsize=large_figure,
    )
    fig32.suptitle(get_fig_title("3.2", CASE2["cond"]))
    for (h, ax) in zip(CASE2["hs"], axs32):
        ts = get_ts_array(t_final, h)
        an, _ = analytic_soln(
            CASE2["alpha"],
            CASE2["omega"],
            CASE2["x0"],
            CASE2["dx0"],
        )
        ys_an = an(ts)
        ys_nm = euler(
            ts,
            CASE2["alpha"],
            CASE2["omega"],
            CASE2["x0"],
            CASE2["dx0"],
        )
        # get error
        err_32.append(abs(ys_an[-1] - ys_nm[-1, 0]))
        # plot numerical solution
        make_xt_numeric_plot(ax, ts, ys_an, ys_nm, "Euler")
        # plot analytic solution
        ax[0].plot(ts, ys_an, linestyle=":", label="Analytic")
        # legend for both plots
        for a in ax:
            a.legend()
    ##### error
    fig32err, ax32err = plt.subplots(figsize=small_figure)
    make_err_plot(ax32err, CASE2, err_32, "Euler")
    ax32err.set_title("Error vs Step-size, p3.2")
    ax32err.legend()
    ##################################################
    #####               4.1                     ######
    ##################################################
    err_41 = []
    ###### ensemble plot
    fig41, axs41 = plt.subplots(
        nrows=len(CASE1["hs"]),
        ncols=2,
        sharex=True,
        tight_layout=True,
        figsize=large_figure,
    )
    fig41.suptitle(get_fig_title("4.1", CASE1["cond"]))
    for (h, ax) in zip(CASE1["hs"], axs41):
        ts = get_ts_array(t_final, h)
        # get analytic function
        an, _ = analytic_soln(
            CASE1["alpha"],
            CASE1["omega"],
            CASE1["x0"],
            CASE1["dx0"],
        )
        # get analytic solution
        ys_an = an(ts)
        # get euler solution
        ys_nm = rk2(
            ts,
            CASE1["alpha"],
            CASE1["omega"],
            CASE1["x0"],
            CASE1["dx0"],
        )
        # get error
        err_41.append(abs(ys_an[-1] - ys_nm[-1, 0]))
        # plot numerical solution
        make_xt_numeric_plot(ax, ts, ys_an, ys_nm, "RK2")
        # plot analytic solution
        ax[0].plot(ts, ys_an, linestyle=":", label="Analytic")
        # write legend for both plots
        for a in ax:
            a.legend()
    ##### Error plot
    # this time, we add the rk2 error to the already-made
    # euler error plot above.
    ax31err.plot(CASE1["hs"], err_41, label="RK2")
    ax31err.scatter(CASE1["hs"], err_41, label="RK2")
    ax31err.legend()
    ##################################################
    #####               5.1                     ######
    ##################################################
    err_51 = []
    ### ensemble plot
    fig51, axs51 = plt.subplots(
        nrows=len(CASE1["hs"]),
        ncols=2,
        sharex=True,
        tight_layout=True,
        figsize=large_figure,
    )
    fig51.suptitle(get_fig_title("5.1", CASE1["cond"]))

    for (h, ax) in zip(CASE1["hs"], axs51):
        ts = get_ts_array(t_final, h)
        # get analytic function
        an, _ = analytic_soln(
            CASE1["alpha"],
            CASE1["omega"],
            CASE1["x0"],
            CASE1["dx0"],
        )
        # get analytic solution
        ys_an = an(ts)
        # get AM solution
        A = np.array([[0, 1], [-(CASE1["omega"] ** 2), -CASE1["alpha"]]])
        ys_nm = adams_moulton(
            ts,
            h,
            A,
            np.array([CASE1["x0"], CASE1["dx0"]]),
        )
        # get error
        err_51.append(abs(ys_an[-1] - ys_nm[-1, 0]))
        # plot numerical solution
        make_xt_numeric_plot(ax, ts, ys_an, ys_nm, "Adams-Moulton")
        # plot analytic solution
        ax[0].plot(ts, ys_an, label="Analytic")
        # write legend for both plots
        for a in ax:
            a.legend()
    #### Error Plot
    fig51err, axs51err = plt.subplots(figsize=small_figure)
    make_err_plot(axs51err, CASE1, err_51, "Adams-Moulton")
    axs51err.set_title("Error vs Step-size, p5.1")
    axs51err.legend()

    ##################################################
    #####               5.2                     ######
    ##################################################
    err_52 = []
    ### Ensemble plot x(t)
    fig52, axs52 = plt.subplots(
        nrows=len(CASE2["hs"]),
        ncols=2,
        sharex=True,
        tight_layout=True,
        figsize=large_figure,
    )
    fig52.suptitle(get_fig_title("5.2", CASE2["cond"]))

    for (h, ax) in zip(CASE2["hs"], axs52):
        ts = get_ts_array(t_final, h)
        # get analytic function
        an, _ = analytic_soln(
            CASE2["alpha"],
            CASE2["omega"],
            CASE2["x0"],
            CASE2["dx0"],
        )
        # get analytic solution
        ys_an = an(ts)
        # get AM solution
        A = np.array([[0, 1], [-(CASE2["omega"] ** 2), -CASE2["alpha"]]])
        ys_nm = adams_moulton(
            ts,
            h,
            A,
            np.array([CASE2["x0"], CASE2["dx0"]]),
        )
        # get error
        err_52.append(abs(ys_an[-1] - ys_nm[-1, 0]))
        # plot numerical solution
        make_xt_numeric_plot(ax, ts, ys_an, ys_nm, "Adams-Moulton")
        # plot analytic solution
        ax[0].plot(ts, ys_an, label="Analytic")
        # write legend for both plots
        for a in ax:
            a.legend()
    # Error Plot
    fig52err, axs52err = plt.subplots(figsize=small_figure)
    make_err_plot(axs52err, CASE2, err_52, "Adams-Moulton")
    axs52err.set_title("Error vs Step-size, p5.2")
    axs52err.legend()

    SHOW = True
    SAVE = True
    if SHOW:
        plt.show()
    if SAVE:
        import os, pathlib

        figdir = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / "figures"
        os.makedirs(figdir, exist_ok=True)
        figs = (fig31, fig31err, fig41, fig51, fig51err, fig52, fig52err)
        names = ("p31", "p31err", "p41", "p51", "p51err", "p52", "p52err")
        for name, fig in zip(names, figs):
            fig.savefig(f"{figdir}/{name}.png", dpi=300)
