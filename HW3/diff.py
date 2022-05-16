import numpy as np


def first_deriv_order2(h: float, x: np.ndarray, i: int, f: callable):
    """calculate derivative of `f`, 2nd order, on station `i` of an array x"""
    assert i + 1 < x.shape[0]
    assert i - 1 >= 0
    dx = (f(x[i + 1]) - f(x[i - 1])) / (2 * h)
    return dx


def first_deriv_order4(h: float, x: np.ndarray, i: int, f: callable) -> float:
    """calculate derivative of `f`, 4th order, on station `i` of an array x"""
    assert i + 1 < x.shape[0]
    assert i - 1 >= 0
    dx = (-f(x[i + 2]) + 8 * f(x[i + 1]) - 8 * f(x[i - 1]) + f(x[i - 2])) / (12 * h)
    return dx


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hs, order2s, order4s = [], [], []
    func = np.exp
    for i in range(30):
        # use high precision
        dtype = np.longdouble
        # halve h each iteration
        h = 0.16 / (2**i)
        # 2nd order
        # get array with sufficient x points to calculate 2nd order derivative, with 0 in the center of the array
        idxs = 3
        x = np.empty((idxs,), dtype=dtype)
        for i in range(idxs):
            x[i] = -h * 1 + i * h

        # calculate 2nd order derivative
        order2_deriv = first_deriv_order2(h, x, idxs // 2, func)

        # 4th order
        # get array with sufficient x points to calculate 4th order derivative, with 0 in the center of the array
        idxs = 5
        x = np.empty((idxs,), dtype=dtype)
        for i in range(idxs):
            x[i] = -2 * h + i * h
        order4_deriv = first_deriv_order4(h, x, idxs // 2, func)

        hs.append(h)
        order2s.append(abs(order2_deriv - 1.0))
        order4s.append(abs(order4_deriv - 1.0))

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    ax.scatter(hs, order2s, marker="o", label="2nd order")
    ax.scatter(hs, order4s, marker="x", label="4th order")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("h (Log)")
    ax.set_ylabel("Absolute Error (Log)")
    ax.legend()
    fig.savefig("plots/derivs.png", dpi=300)
    plt.show()
