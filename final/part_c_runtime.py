from toolbox import formulation
import matplotlib.pyplot as plt
import numpy as np
from time import time

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
    omega = 4
    ns, times = [], []
    On, On2 = [], []
    for n in range(1, 7):
        n = 2**n
        xi = formulation.grid1d(0.0, 1.0, n)
        h = (xi[-1] - xi[0]) / n
        tau_final = 3 * 2 * np.pi / omega
        taus = formulation.const_step_grid(0.0, tau_final, 0.49 * h**2)
        # get the fluid function
        fluid_fn = formulation.get_sin_plume_fn(
            params["A"],
            omega,
            params["xi0"],
            params["sigma"],
        )
        # solve time-dependent function
        t0 = time()
        fluids, solution = formulation.solve_time_rk2(
            xi,
            taus,
            params["beta"],
            theta0=params["theta0"],
            thetaf=params["thetaf"],
            fluid_eqn=fluid_fn,
        )
        t1 = time()
        ns.append(n)
        times.append(t1 - t0)
        print(f"{n}: {(t1 - t0):.2f}")
        On.append(1e-4 * n**1)
        On2.append(1e-4 * n**2)

    fig = plt.figure(num="explicit_runtime", figsize=(5, 3), tight_layout=True)
    ax = fig.add_subplot(111)
    ax.scatter(ns, times, marker="o", color="k", label="Explicit Method Runtime")
    ax.plot(ns, On, color="r", label="$\mathcal{O}(n)$")
    ax.plot(ns, On2, color="b", label="$\mathcal{O}(n^2)$")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("runtime (s)")
    ax.set_title(f"Explicit RK2 Method Run Time vs Grid Size")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    if SAVEFIG:
        fig.savefig("./report/figures/explicit_runtime.png", dpi=300)
    plt.show()
