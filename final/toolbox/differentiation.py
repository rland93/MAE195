import numpy as np


def derivative2(x, h, nth=1, method="center"):
    res = np.full((x.shape[0],), fill_value=np.nan)
    if nth == 1:
        if method == "center":
            #              i+1       i-1
            res[1:] = (x[1:] - x[:-1]) / (2 * h)
        elif method == "forward":
            #               i+2         i+1           i
            res[:-2] = (-x[2:] + 4 * x[1:-1] - 3 * x[:-2]) / (2 * h)
        elif method == "backward":
            #               i        i-1       i-2
            res[2:] = (3 * x[2:] - 4 * x[1:-1] + x[:-2]) / (2 * h)
    elif nth == 2:
        if method == "center":
            #              i+1         i           i-1
            res[2:-2] = (x[2:] - 2 * x[1:-1] + x[:-2]) / h**2
        elif method == "forward":
            #             i+3          i+2           i+1           i
            res[:-3] = (-x[3:] + 4 * x[2:-1] - 5 * x[1:-2] + 2 * x[:-3]) / h**2
        elif method == "backward":
            #              i            i-1          i-2          i-3
            res[3:] = (2 * x[3:] - 5 * x[2:-1] + 4 * x[1:-2] - x[:-3]) / h**2
    return res


def trap_int(f, a, b, n):
    h = (b - a) / n
    x = a
    sigma = f(x)
    for _ in range(0, n - 1):
        x += h
        sigma += 2 * f(x)
    sigma += f(b)
    return sigma * h / 2


def romberg_analytic(f, a, b, order=4):
    R = np.zeros((order + 1, order + 1))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    it = 1
    for i in range(1, order + 1):
        h /= 2.0

        sigma = 0.0
        it = 2 * it
        for k in range(1, it, 2):
            sigma += f(a + k * h)

        R[i, 0] = 0.5 * R[i - 1, 0] + sigma * h

        coeff = 1
        for j in range(1, i + 1):
            coeff = 4 * coeff
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (coeff - 1)
    return R[order, order]


def romberg_discrete(xi, yi, order=2):
    assert np.log2(xi.shape[0]) % 1 == 0
    R = np.zeros((order + 1, order + 1))
    it = 1
    R[0, 0] = 0.5 * (yi[0] + yi[-1]) * (xi[1] - xi[0])
    for i in range(1, order + 1):
        it *= 2
        xis = xi[:: xi.shape[0] // it]
        yis = yi[:: yi.shape[0] // it]
        sigma = np.sum(yis)
        R[i, 0] = 0.5 * R[i - 1, 0] + sigma * (xis[1] - xis[0])

        coeff = 1
        for j in range(1, i + 1):
            coeff = 4 * coeff
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / (coeff - 1)
        print(R)
    return R[order, order]


if __name__ == "__main__":
    """
    import matplotlib.pyplot as plt

    np.set_printoptions(linewidth=400)

    coefs = np.random.uniform(0, 1, (10,))

    # some exact function
    def f(x):
        return np.polyval(coefs, x)

    def exactf(x):
        P = np.polyint(coefs, m=1)
        return np.polyval(P, x)

    def get_exact(a, b):
        return exactf(b) - exactf(a)

    x0 = 0
    x1 = 10
    ns, errs_r, errs_tz = [], [], []
    for n in range(1, 5):
        romberg_int = romberg_analytic(f, x0, x1, order=n)
        exact = get_exact(x0, x1)
        err_romberg = abs(romberg_int - exact)
        ns.append(n)
        errs_r.append(err_romberg)
        err_tz = abs(trap_int(f, x0, x1, n) - exact)
        errs_tz.append(err_tz)

    fig, axs = plt.subplots(ncols=2, figsize=(7, 3), tight_layout=True)
    axs[0].plot(ns, errs_r, "o-", label="Romberg")
    axs[1].plot(ns, errs_tz, "x--", label="Trapezoid")
    for ax in axs:
        ax.set_xlabel("n")
        ax.set_ylabel("Error")
        ax.set_title(r"Error vs. $n$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Romberg vs. Trapezoid Rule Error")
    plt.show()
    """
    xis = np.linspace(0, 1, 512)
    yis = np.sin(xis)
    yis_true = np.cos(xis)

    val1 = romberg_analytic(lambda x: np.cos(x), 0, 1, order=4)
    print("\n")
    val2 = romberg_discrete(xis, yis, order=4)

    print(val1, val2)
