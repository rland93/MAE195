from solver import gauss_elim
import numpy as np
from heateqn import A_dirichlet, b_dirichlet
import time
import matplotlib.pyplot as plt


def solvetime(n):
    top = np.full(shape=(n,), fill_value=400)
    bottom = np.full(shape=(n,), fill_value=300)
    left = np.full(shape=(n,), fill_value=100)
    right = np.full(shape=(n,), fill_value=300)
    A = A_dirichlet(n)
    b = b_dirichlet(top, bottom, left, right)

    t0 = time.time()
    x = np.linalg.solve(A, b)
    t_np = time.time() - t0

    t0 = time.time()
    x = gauss_elim(A, b)
    t_me = time.time() - t0

    return t_np, t_me


if __name__ == "__main__":
    k = 100
    nl, nh = 3, 17
    fig, ax = plt.subplots(1, 1)
    times_numpy, times_mine, ns = [], [], []
    print(f"Running {k} iterations of nxn matrix solve through n = {nh}")
    for n in range(nl, nh):
        print(n, end=" ", flush=True)
        ns.append(n)
        t_np, t_me = 0.0, 0.0
        for _ in range(k):
            numpy, custom = solvetime(n)
            t_np += numpy
            t_me += custom
        times_numpy.append(t_np / k)
        times_mine.append(t_me / k)

    ax.plot(ns, times_numpy, "b-", label="numpy")
    ax.plot(ns, times_mine, "r-", label="custom")

    ax.set_title(
        r"Runtime vs. $n \times n$ Matrix Size, Average of ${}$ Runs".format(k)
    )
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$t$ (s)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    plt.show()
