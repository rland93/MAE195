from solver import gauss_elim
from heateqn import A_dirichlet, b_dirichlet, plot_matr, unpack_solution
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import ticker

if __name__ == "__main__":
    times_np, times_custom, ns = [], [], []
    nupp = 31
    for i, n in enumerate(range(3, nupp)):
        print(n, end=" ", flush=True)
        top = np.full(shape=(n,), fill_value=100)
        bottom = np.full(shape=(n,), fill_value=0)
        left = np.full(shape=(n,), fill_value=50)
        right = np.full(shape=(n,), fill_value=75)

        A = A_dirichlet(n)
        b = b_dirichlet(top, bottom, left, right)

        # time & solve the system
        t0 = time.time()
        x_np = np.linalg.solve(A, b)
        t_np = time.time() - t0

        t0 = time.time()
        x_me = gauss_elim(A, b)
        t_me = time.time() - t0

        ns.append(n)
        times_np.append(t_np)
        times_custom.append(t_me)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ns, times_np, "b-", label="numpy.linalg.solve")
    ax.plot(ns, times_custom, "r-", label="Custom gauss elimination solver")
    ax.set_title("Runtime vs. nxn Matrix Size")
    ax.set_xlabel("n")
    nticks = [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 30]
    ax.set_ylabel("t (s)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    ax.set_xticks(nticks)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    fig.savefig("runtime_vs_n.png", dpi=200)
    plt.show()
