from toolbox import formulation
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SAVEFIG = True


if __name__ == "__main__":
    params = {
        "A": 5.0,
        "xi0": 0.3,
        "sigma": 0.05,
        "beta": 4.0,
        "theta0": 0.0,
        "thetaf": 1.0,
    }
    all_figures, animations = [], []
    savedir = Path("./report/figures/")
    savedir.mkdir(parents=True, exist_ok=True)
    # make contour plot of the solution
    fig1, axs = plt.subplots(
        nrows=2,
        ncols=2,
        tight_layout=True,
        num=f"contour_time_rk2",
        figsize=(8, 7),
    )

    for i, omega in enumerate((4, 400)):
        n = 32
        xi = formulation.grid1d(0.0, 1.0, n)
        h = (xi[-1] - xi[0]) / n

        tau_final = 5 * 2 * np.pi / omega
        taus = formulation.const_step_grid(0.0, tau_final, 0.49 * h**2)
        print(f"Number of timesteps: {taus.shape[0]}")
        # get the fluid function
        fluid_fn = formulation.get_sin_plume_fn(
            params["A"],
            omega,
            params["xi0"],
            params["sigma"],
        )
        # solve time-dependent function
        fluids, solution = formulation.solve_time_rk2(
            xi,
            taus,
            params["beta"],
            theta0=params["theta0"],
            thetaf=params["thetaf"],
            fluid_eqn=fluid_fn,
        )

        # contour plot of fluid
        formulation.make_contour_plt(
            axs[i, 0],
            xi,
            taus,
            fluids,
            rf"Fluid Temperature for $\omega$={omega}",
        )
        # contour plot of fin temperature
        formulation.make_contour_plt(
            axs[i, 1],
            xi,
            taus,
            solution,
            rf"Fin Temperature for $\omega$={omega}",
        )
        # draw animation of the solution
        ani = formulation.animate_results(
            xi, taus, fluids, solution, skip=10, interval=4
        )
        animations.append(ani)

    all_figures.append(fig1)
    plt.show()

    # save figures
    if SAVEFIG:
        for fig in all_figures:
            fig.savefig(savedir / f"{fig.get_label()}.png", dpi=300)
