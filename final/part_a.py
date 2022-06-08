from toolbox import solvers, differentiation, formulation
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

SAVEFIG = True

if __name__ == "__main__":

    savedir = Path("./report/figures/")
    savedir.mkdir(parents=True, exist_ok=True)

    draw = {
        "3": True,
        "4": True,
        "5": True,
    }

    # problem 1
    # solution is in analytic.solve_analytic_1

    all_figures = []
    cases = [
        {
            "fluid": 2.0,
            "beta": 4.0,
            "theta0": 0.0,
            "thetaf": 1.0,
        },
        {
            "fluid": 2.0,
            "beta": 40.0,
            "theta0": 0.0,
            "thetaf": 1.0,
        },
        {
            "fluid": 0.0,
            "beta": 4.0,
            "theta0": 0.0,
            "thetaf": 1.0,
        },
        {
            "fluid": 0.0,
            "beta": 40.0,
            "theta0": 0.0,
            "thetaf": 1.0,
        },
    ]

    if draw["3"]:
        ################################################
        ######## PART A, Problem 3 #####################
        ################################################
        hs, rmses, Oh, Oh2 = [], [], [], []
        for n in (4, 8, 16, 32, 64, 128, 256):
            # get grid
            xi = formulation.grid1d(0.0, 1.0, n)
            # get analytic function
            f = formulation.solve_ss_analytic(
                0.0,
                1.0,
                cases[0]["fluid"],
                cases[0]["beta"],
                cases[0]["theta0"],
                cases[0]["thetaf"],
            )
            # get analytic solution
            theta_a = f(xi)

            # set up numerical solution
            A, b = formulation.get_Ab(
                xi,
                cases[0]["fluid"],
                cases[0]["beta"],
                cases[0]["theta0"],
                cases[0]["thetaf"],
            )
            # solve with TDMA
            theta = solvers.tdma(A, b)

            # get h for error analysis
            h = formulation.get_h(0.0, 1.0, n)
            print(
                "xi ",
                xi.shape,
                "n ",
                n,
                "theta ",
                theta.shape,
                "b ",
                b.shape,
                A.shape,
                "h ",
                h,
            )
            Oh += [h / 4.0]
            Oh2 += [h**2 / 4.0]
            rmse = np.sqrt(np.mean((theta - theta_a) ** 2))
            rmses += [rmse]
            hs += [h]

        # Error analysis plot
        fig3 = plt.figure(num="rms_error", figsize=(7, 4))
        ax3 = fig3.add_subplot(111)
        all_figures.append(fig3)
        ax3.plot(hs, rmses, "ko-", label="RMS Error")
        ax3.plot(hs, Oh, "b--", label=r"$\mathcal{O}(h)$")
        ax3.plot(hs, Oh2, "r--", label=r"$\mathcal{O}(h^2)$")
        ax3.set_xlabel(r"$h$")
        ax3.set_ylabel("RMSE")
        ax3.set_title(
            r"Error vs Step Size, SS Heat Transfer, Constant $\theta_{fluid}$"
        )
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.legend()
        plt.show()

    if draw["4"]:
        ################################################
        ######## PART A, Problem 3 #####################
        ################################################
        n = 16
        xi = formulation.grid1d(0.0, 1.0, n)

        # plot set up
        fig4, ax4 = plt.subplots(
            nrows=2,
            ncols=2,
            tight_layout=True,
            figsize=(9, 4.5),
            num="temp_dist_params",
        )
        markers = ("o", "s", "^", "v")
        all_figures.append(fig4)

        # plot each
        for case, (i, j), m in zip(cases, np.ndindex(ax4.shape), markers):
            # get analytic solution
            f = formulation.solve_ss_analytic(
                0.0,
                1.0,
                case["fluid"],
                case["beta"],
                case["theta0"],
                case["thetaf"],
            )
            theta_a = f(xi)
            # get numerical solution
            A, b = formulation.get_Ab(
                xi,
                case["fluid"],
                case["beta"],
                case["theta0"],
                case["thetaf"],
            )
            # solve with TDMA
            theta = solvers.tdma(A, b)

            # plot analytic solution
            ax4[i, j].plot(xi, theta_a, label="Exact Solution")

            # plot numerical solution
            ax4[i, j].scatter(
                xi, theta, marker="^", color="k", label="Numerical Solution"
            )

            # the temperature of the fin cannot exceed either the left boundary,
            # right boundary, or fluid temperature. So let's draw a horizontal line
            # to make that clear.
            physical_max = max((case["theta0"], case["thetaf"], case["fluid"]))
            ax4[i, j].axhline(
                physical_max,
                color="k",
                linestyle=":",
                label="Physical Maximum",
            )

            # break up fig.title because raw+fstrings don't play nicely
            title = rf"$\beta={case['beta']}$, $\theta_f={case['fluid']}$"

            # title and labels
            ax4[i, j].set_title(title)
            ax4[i, j].set_ylabel(r"$\theta$")
            ax4[i, j].legend()
            ax4[i, j].set_xlabel(r"$\xi$")

        plt.show()

    if draw["5"]:
        ################################################
        ######## PART A, Problem 3 #####################
        ################################################
        n = 16
        fig5, ax5 = plt.subplots(
            nrows=1,
            ncols=1,
            tight_layout=True,
            figsize=(6, 4),
            num="heat_flux_2",
        )
        all_figures.append(fig5)
        xi = formulation.grid1d(0.0, 1.0, n=n)

        betas, errs_romb, errs_romb6, errs_trap12, errs_trap, errs_trap2 = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        problems_random = []
        for b in range(1, 60):
            problem = {
                "fluid": 1.0,
                "beta": b * 0.5,
                "theta0": 0.0,
                "thetaf": 1.0,
            }
            problems_random.append(problem)

        for problem in problems_random:
            print("Problem:")
            for k, v in problem.items():
                print(f"\t{k}: {v}")
            f = formulation.solve_ss_analytic(
                0.0,
                1.0,
                problem["fluid"],
                problem["beta"],
                problem["theta0"],
                problem["thetaf"],
            )

            # convection
            romberg4 = differentiation.romberg_analytic(
                lambda x: f(x) - problem["fluid"],
                0.0,
                1.0,
                order=4,
            )
            romberg4 *= problem["beta"]
            romberg6 = differentiation.romberg_analytic(
                lambda x: f(x) - problem["fluid"],
                0.0,
                1.0,
                order=6,
            )
            romberg6 *= problem["beta"]
            trap12 = differentiation.trap_int(
                lambda x: f(x) - problem["fluid"],
                0.0,
                1.0,
                n=n // 2,
            )
            trap12 *= problem["beta"]
            trap = differentiation.trap_int(
                lambda x: f(x) - problem["fluid"],
                0.0,
                1.0,
                n=n,
            )
            trap *= problem["beta"]
            trap2 = differentiation.trap_int(
                lambda x: f(x) - problem["fluid"],
                0.0,
                1.0,
                n=2 * n,
            )
            trap2 *= problem["beta"]

            print(
                f"\tConvection:\n\t\tRomberg4 = {romberg4}\n\t\tTrap = {trap}\n\t\tTrap2 = {trap2}"
            )

            h = (xi[-1] - xi[0]) / n

            # conduction
            left = differentiation.derivative2(f(xi), h, nth=1, method="forward")[0]
            right = differentiation.derivative2(f(xi), h, nth=1, method="backward")[-1]
            print(f"\tConduction:\n\t\tLeft = {left}\n\t\tRight = {right}")
            print(f"\t\tTotal = {left - right}")

            # heat flux
            print(f"\tNet Heat Flux CV:")
            for conv, label in zip(
                (romberg4, trap, trap2), ("Romberg4", "Trap", "Trap2")
            ):
                cond = left - right
                print(f"\t\t{label}: 0 = {conv + cond}")

            betas.append([problem["beta"]])

            arrs = [errs_romb, errs_romb6, errs_trap12, errs_trap, errs_trap2]
            convs = [romberg4, romberg6, trap12, trap, trap2]
            for (arr, conv) in zip(arrs, convs):
                err = abs(left - right + conv)
                err /= problem["beta"]
                arr.append(err)

        ax5.scatter(betas, errs_romb, c="k", label=r"Romberg (order 4)", marker="+")
        ax5.scatter(betas, errs_romb6, c="b", label=r"Romberg (order 6)", marker=".")
        ax5.scatter(betas, errs_trap12, c="r", label=r"Trapezoid ($2 h$)", marker="x")
        ax5.scatter(betas, errs_trap, c="g", label=r"Trapezoid ($h$)", marker="3")
        ax5.scatter(
            betas, errs_trap2, c="purple", label=r"Trapezoid ($0.5 h$)", marker="4"
        )
        ax5.axhline(0.0, color="k", linestyle="--", label="Exact")
        ax5.set_xlabel(r"$\beta$")
        ax5.set_ylabel("Net CV Heat Flux")
        ax5.legend()
        plt.show()

    if SAVEFIG:
        for fig in all_figures:
            fig.savefig(savedir / f"{fig.get_label()}.png", dpi=300)
