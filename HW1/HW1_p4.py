"""Write python code to evaluate each approximation of the taylor series of exp(x),
varying x over a range of values x=1.0, 0.1, 0.01, 0.001, to prove the order of
accuracy. Plot on a log-log plot."""

from math import factorial, exp
import matplotlib.pyplot as plt


def taylor_series(x, order=2):
    value = 1.0
    for n in range(order):
        value += x**n / factorial(n)
    return value


if __name__ == "__main__":
    xs = []
    # get some numbers within these orders of magnitude
    for p in (1.0, 0.1, 0.01, 0.001):
        xs.append(p)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)

    # evaluate and plot
    for order, marker in zip((2, 3, 4), (".", "+", "_")):
        # get y values
        ys = []
        es = []
        for x in xs:
            # get real value
            ys.append(taylor_series(x, order=order))
            # get error
            es.append(taylor_series(x, order=order) - x)
        # scatter plot and label
        ax.scatter(xs, es, label=f"order={order}", marker=marker)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.legend()

    plt.show()
