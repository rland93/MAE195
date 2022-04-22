from heateqn import A_dirichlet, b_dirichlet
from solver import gauss_elim
import numpy as np


def rel_errs_plot(low=2, upp=17):
    ns, rel_errs_np, rel_errs_ge = [], [], []
    conditions = []

    for n in range(low, upp):
        print(n, end=" ", flush=True)
        top = np.full(shape=(n,), fill_value=400)
        bottom = np.full(shape=(n,), fill_value=300)
        left = np.full(shape=(n,), fill_value=300)
        right = np.full(shape=(n,), fill_value=300)
        A = A_dirichlet(n)
        b = b_dirichlet(top, bottom, left, right)
        x_ge = gauss_elim(A, b)
        x_np = np.linalg.solve(A, b)

        # relative errors
        rel_err_np = np.linalg.norm(A @ x_np - b) / np.linalg.norm(b)
        rel_err_ge = np.linalg.norm(A @ x_ge - b) / np.linalg.norm(b)
        print(f"np: {rel_err_np:.4f} ge: {rel_err_ge:.4f}")
        rel_errs_np.append(rel_err_np)
        rel_errs_ge.append(rel_err_ge)

        # condition numbers
        cond = np.linalg.cond(A)
        print(f"cond: {cond:.4f}")
        conditions.append(cond)
        ns.append(n * n)

    return (
        np.array(ns),
        np.array(rel_errs_ge),
        np.array(rel_errs_np),
        np.array(conditions),
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ns, rel_err_ge, rel_err_np, conditions = rel_errs_plot(upp=31)
    fig1 = plt.figure(tight_layout=True, figsize=(8, 6))
    ax1 = fig1.add_subplot(211)
    ax1.set_title("Relative Error vs. Matrix Size")
    ax1.plot(ns, rel_err_ge, "b-", label="Custom GE Solver")
    ax1.plot(ns, rel_err_np, "r-", label="Numpy Solver")
    ax1.set_xlabel(r"Matrix Size: No. of elements")
    ax1.set_ylabel("Relative Error")
    ax1.legend()

    ax2 = fig1.add_subplot(212)
    ax2.plot(ns, conditions, "b-")
    ax2.set_title("Condition Number vs. Matrix Size")
    ax2.set_xlabel(r"Matrix Size: No. of elements")
    ax2.set_ylabel("Condition Number")

    fig1.savefig("rel_errs.png", dpi=200)
    plt.show()
