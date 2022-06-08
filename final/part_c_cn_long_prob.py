from toolbox import formulation
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SAVEFIG = True

if __name__ == "__main__":
    # set problem parameters
    params = {
        "A": 5.0,
        "xi0": 0.3,
        "sigma": 0.05,
        "beta": 5.0,
        "theta0": 0.0,
        "thetaf": 1.0,
    }
    all_figures, animations = [], []
    savedir = Path("./report/figures/")
    savedir.mkdir(parents=True, exist_ok=True)
    # make contour plot of the solution
    fig1, axs = plt.subplots(
        nrows=1,
        ncols=2,
        tight_layout=True,
        num=f"contour_time_crank_long",
        figsize=(8, 7),
    )

    n = 128
    xi = formulation.grid1d(0.0, 1.0, n)
    h = (xi[-1] - xi[0]) / n
    tau_final = 1.0
    taus = formulation.const_step_grid(0.0, tau_final, 0.001)

    print(f"Number of timesteps: {taus.shape[0]}")
    print(f"tau_final: {tau_final}")

    fluid_fn = formulation.get_step_plume_fn(
        params["A"],
        t1=0.25,
        t2=0.75,
        xi0=params["xi0"],
        sigma=0.2,
        xi=xi,
    )
    # solve time-dependent function
    fluids, solution = formulation.solve_time_crank(
        xi,
        taus,
        params["beta"],
        theta0=params["theta0"],
        thetaf=params["thetaf"],
        fluid_eqn=fluid_fn,
    )
    # contour plot of fluid
    formulation.make_contour_plt(
        axs[0],
        xi,
        taus,
        fluids,
        rf"Fluid Temperature",
    )
    # contour plot of fin temperature
    formulation.make_contour_plt(
        axs[1],
        xi,
        taus,
        solution,
        rf"Fin Temperature",
    )
    # draw animation of the solution
    ani = formulation.animate_results(
        xi, taus, fluids / 5.0, solution, interval=10, skip=1
    )
    animations.append(ani)

    all_figures.append(fig1)
    plt.show()

    # save figures
    if SAVEFIG:
        for fig in all_figures:
            fig.savefig(savedir / f"{fig.get_label()}.png", dpi=300)
