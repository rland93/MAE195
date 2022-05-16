import numpy as np


def integrate_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    """
    Integrate fn(x) using the trapezoidal rule.
    """
    assert x.shape[0] > 1
    assert x.shape[0] == y.shape[0]
    # spacing is uniform
    h = x[1] - x[0]
    # end points
    sigma = y[0] + y[-1]
    for i in range(1, x.shape[0] - 1):
        # interior points
        sigma += 2.0 * y[i]
    return h * sigma / 2.0


def trap(n: int, a: float, b: float, f: callable):
    h = (b - a) / n
    x = a
    sigma = f(x)
    for _ in range(1, n - 1):
        x += h
        sigma += 2 * f(x)
    sigma += f(b)
    return (b - a) * sigma / (2 * n)


def romberg_integration(
    a: float,
    b: float,
    f: callable,
    max_iter=25,
    es=1e-10,
):
    P = np.zeros((max_iter, max_iter), dtype=float)
    n = 1
    P[0, 0] = trap(n, a, b, f)
    it, ea = 0, float("inf")
    while it < max_iter and ea > es:
        n = 2**it
        assert it < P.shape[0]
        P[it, 0] = trap(n, a, b, f)
        for k in range(1, it):
            j = it - k
            # print(f"Working on {j} {k}:")
            rho = 4**k * P[j, k - 1] - P[j - 1, k - 1]
            # print(f"\t4**(k-1)={4**k}")
            # print(f"\tP[j+1,k-1]={P[j, k-1]}")
            # print(f"\tP[j,k-1]={P[j-1, k-1]}")
            rho /= 4**k - 1
            # print(f"rho={rho}")
            P[j, k] = rho
        it += 1
        if it > 1:
            ea = abs(P[it - 1, 0] - P[it - 2, 0]) / P[it - 1, 0]
    return P[it - 1, 0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    hs = []
    trapz = []
    trapz_times = []
    trapz_numpy = []
    trapz_numpy_times = []
    rombs = []
    rombs_times = []

    analytic = -1.0 + np.e

    fig1 = plt.figure(figsize=(5, 4), tight_layout=True)
    # error plot
    ax1 = fig1.add_subplot(111)
    n = 1
    for iters, _ in enumerate(range(7)):
        print(f"Iteration {iters}", end="\r", flush=True)
        # n == number of splits.
        n *= 2
        # compare trapz and trapz numpy
        x = np.linspace(0, 1.0, num=n)
        y = np.exp(x)
        t0 = time.time()
        integral = integrate_trapezoidal(x, y)
        t1 = time.time()
        trapz_times.append(t1 - t0)
        t0 = time.time()
        integral_np = np.trapz(y, x)
        t1 = time.time()
        trapz_numpy_times.append(t1 - t0)
        # romberg integration

        romb = []
        romb_time = []
        for j in range(iters):
            t0 = time.time()
            romb_res = romberg_integration(0, 1, np.exp, max_iter=j + 1)
            romb.append(romb_res - analytic)
            t1 = time.time()
            romb_time.append(t1 - t0)
        rombs.append(romb)
        rombs_times.append(romb_time)

        # h
        hs.append(x[1] - x[0])

        # get absolute errors
        trapz.append(integral - analytic)
        trapz_numpy.append(integral_np - analytic)

    ax1.scatter(
        hs,
        trapz,
        marker="o",
        label="Trapezoid Rule",
    )
    ax1.scatter(
        hs,
        trapz_numpy,
        marker="+",
        label="Trapezoid Rule (Numpy)",
    )
    for i, r in enumerate(rombs):
        if len(r) != 0:
            print("\n")
            print(len(hs[:i]), len(r))
            ax1.scatter(
                hs[:i],
                r,
                marker="x",
                label=f"Romberg Integration, order={i}",
            )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("h (Log)")
    ax1.set_ylabel("Absolute Error (Log)")
    ax1.set_title("Absolute Error")
    ax1.legend()

    # Runtime plot
    fig2 = plt.figure(figsize=(5, 4), tight_layout=True)
    ax2 = fig2.add_subplot(111)
    ax2.scatter(
        hs,
        trapz_times,
        marker="o",
        label="Trapezoid Rule",
    )
    ax2.scatter(
        hs,
        trapz_numpy_times,
        marker="+",
        label="Trapezoid Rule (Numpy)",
    )
    for i, rt in enumerate(rombs_times):
        if len(rt) != 0:
            print(len(hs[:i]), len(rt))
            ax2.scatter(
                hs[:i],
                rt,
                marker="x",
                label=f"Romberg Integration, order={i}",
            )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime [s] (Log scale)")
    ax2.set_xlabel("h (Log)")
    ax2.set_title("Runtime")
    ax2.legend()

    fig1.savefig("./plots/integration.png", dpi=300)
    fig2.savefig("./plots/integration_runtime.png", dpi=300)

    plt.show()
