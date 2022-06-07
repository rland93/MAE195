from toolbox import solvers, formulation, differentiation
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    SAVE = True
    all_figures = []
    savedir = Path("./report/figures/")
    savedir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(
        ncols=2, nrows=2, num="plume", figsize=(9, 5.5), tight_layout=True
    )
    all_figures.append(fig1)

    beta = 40.0
    theta0 = 0.0
    thetaf = 1.0
    fluid = 0.0

    for n, ax in zip((17, 33), ax1[0, :]):
        # get xi grid
        xi = formulation.grid1d(0, 1, n)
        # get fluid temperature
        fluid_plume = formulation.gaussian_plume(xi, A=5.0, xi0=0.3, sigma=0.05)
        # get system
        A, b = formulation.get_Ab(
            xi,
            fluid_plume,
            beta=beta,
            theta0=theta0,
            thetaf=thetaf,
        )
        # solve system
        theta = solvers.tdma(A, b)
        # plot
        ax.plot(xi, theta, ".-", label=r"$\theta$ (plume)")
        ax.set_title(rf"Fin Temperature ($\theta$) vs $\xi$, n={n-1}")

        # get smooth plume
        xi = formulation.grid1d(0, 1, 257)
        fluid_plume = formulation.gaussian_plume(xi, A=5.0, xi0=0.3, sigma=0.05)
        ax.fill_between(
            xi,
            fluid_plume / 5.0,
            color="blue",
            label=r"$\theta_f(\xi)$ (stream temperature)",
            alpha=0.2,
        )

        # get analytic solution no plume
        xi = formulation.grid1d(0, 1, 257)
        theta_analytic = formulation.solve_ss_analytic(
            x0=0.0,
            xf=1.0,
            fluid=0.0,
            beta=beta,
            theta0=theta0,
            thetaf=thetaf,
        )
        ax.plot(xi, theta_analytic(xi), label=r"$\theta$ (no plume)")

    for n, ax in zip((16, 32), ax1[1, :]):
        # get xi grid
        xi = formulation.grid1d(0, 1, n)
        # get fluid temperature
        fluid_plume = formulation.gaussian_plume(xi, A=5.0, xi0=0.3, sigma=0.05)
        # get system
        A, b = formulation.get_Ab(
            xi,
            fluid_plume,
            beta=beta,
            theta0=theta0,
            thetaf=thetaf,
        )
        # solve system
        theta = solvers.tdma(A, b)

        # solution without plume
        A, b = formulation.get_Ab(
            xi,
            fluid,
            beta=beta,
            theta0=theta0,
            thetaf=thetaf,
        )
        theta_no_plume = solvers.tdma(A, b)

        # get q
        q_plume = differentiation.derivative2(theta, xi[1] - xi[0])
        q_no_plume = differentiation.derivative2(theta_no_plume, xi[1] - xi[0])
        # plot q
        ax.plot(xi, q_plume, ".-", label=r"$\phi$, plume")
        ax.plot(xi, q_no_plume, "x-", label=r"$\phi$, no plume")
        ax.set_title(rf"Heat Flux ($\phi$) vs $\xi$, n={n-1}")

    for i, j in np.ndindex(ax1.shape):
        ax1[i, j].set_xlabel(r"$\xi$")
        if i == 0:
            ax1[i, j].set_ylabel(r"Temp $\theta$")
        elif i == 1:
            ax1[i, j].set_ylabel(r"Heat Flux $\phi$")
        ax1[i, j].legend()
    plt.show()

    if SAVE:
        for fig in all_figures:
            fig.savefig(savedir / f"{fig.get_label()}.png", dpi=300)
