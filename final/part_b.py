from toolbox import solvers, formulation, differentiation
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

SAVEFIG = True


if __name__ == "__main__":
    all_figures = []
    savedir = Path("./report/figures/")
    savedir.mkdir(parents=True, exist_ok=True)

    # problem parameters
    params = {
        "A": 5.0,
        "xi0": 0.3,
        "sigma": 0.05,
        "beta": 40.0,
        "theta0": 0.0,
        "thetaf": 1.0,
    }
    # generate the figure
    fig1, axs = plt.subplots(
        ncols=2,
        nrows=2,
        num="ss_spatial_variation",
        figsize=(9, 5.5),
        tight_layout=True,
    )
    all_figures.append(fig1)
    ns = (16, 32)

    for n, ax in zip(ns, axs[0]):
        xi = formulation.grid1d(0.0, 1.0, n)
        h = 1.0 / n
        # get spatial varying fluid func
        fluid_timedep = formulation.gaussian_plume(
            xi, params["A"], params["xi0"], params["sigma"]
        )
        # set up system
        A, b = formulation.get_Ab(
            xi,
            fluid_timedep,
            params["beta"],
            params["theta0"],
            params["thetaf"],
        )

        # solve
        theta = solvers.tdma(A, b)
        # plot
        ax.plot(xi, theta, "k.-", label=r"$\theta$ (plume)")
        ax.set_title(rf"Fin Temperature ($\theta$) vs $\xi$, n={n}")
        # get smooth plume
        xi = formulation.grid1d(0, 1, 500)
        fluid_plume = formulation.gaussian_plume(xi, A=5.0, xi0=0.3, sigma=0.05)
        ax.plot(
            xi,
            fluid_plume / 5.0,
            color="r",
            label=r"$\theta_f(\xi)$",
        )

        # get analytic solution no plume
        xi = formulation.grid1d(0, 1, 500)
        theta_analytic = formulation.solve_ss_analytic(
            x0=0.0,
            xf=1.0,
            fluid=0.0,
            beta=params["beta"],
            theta0=params["theta0"],
            thetaf=params["thetaf"],
        )
        ax.plot(xi, theta_analytic(xi), "b--", label=r"$\theta$ (no plume)")

    for n, ax in zip((16, 32), axs[1, :]):
        xi = formulation.grid1d(0.0, 1.0, n)
        # get fluid temperature
        fluid_plume = formulation.gaussian_plume(xi, A=5.0, xi0=0.3, sigma=0.05)
        # get system
        A, b = formulation.get_Ab(
            xi, fluid_plume, params["beta"], params["theta0"], params["thetaf"]
        )
        # solve system
        theta = solvers.tdma(A, b)

        # solution without plume
        A, b = formulation.get_Ab(
            xi, 0.0, params["beta"], params["theta0"], params["thetaf"]
        )

        theta_no_plume = solvers.tdma(A, b)

        # get q
        q_plume = differentiation.derivative2(theta, xi[1] - xi[0])
        q_no_plume = differentiation.derivative2(theta_no_plume, xi[1] - xi[0])
        # plot q
        ax.plot(xi, q_plume, "r.-", label=r"$\phi$, plume")
        ax.plot(xi, q_no_plume, "kx-", label=r"$\phi$, no plume")
        ax.set_title(rf"Heat Flux ($\phi$) vs $\xi$, n={n}")

    if SAVEFIG:
        for fig in all_figures:
            fig.savefig(savedir / f"{fig.get_label()}.png", dpi=300)

    # label axes
    for i, j in np.ndindex(axs.shape):
        axs[i, j].set_xlabel(r"$\xi$")
        if i == 0:
            axs[i, j].set_ylabel(r"Temp $\theta$")
        elif i == 1:
            axs[i, j].set_ylabel(r"Heat Flux $\phi$")
        axs[i, j].legend()
    plt.show()
