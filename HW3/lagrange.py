import numpy as np
from types import SimpleNamespace


def reorder_points_near(xx: float, x: np.array) -> np.array:
    """re-order points so that they are arranged by their
    distance to a point xx. This is useful for seeing the
    effect of order on Lagrange polynomials.

    e.g. in the array x = [-1, 2, 5, 10], if xx = 3, we could
    call x = x[reorder_points_near(3,x)], to get the array
    [2, 5, -1, 10]. If we then put this into our lagrange polynomial
    function, the points are in the correct order to produce
    lower-order polynomials.

    returns a sorted index"""
    return np.argsort(np.abs(x - xx))


def lagrange_polynomial(xx: float, x: np.array, y: np.array, n: int) -> float:
    assert x.shape[0] == y.shape[0]
    assert n >= 0 and n < x.shape[0]
    sigma = 0.0
    for i in range(n):
        pi = y[i]
        for j in range(n):
            if i != j:
                pi *= (xx - x[j]) / (x[i] - x[j])
        sigma += pi
    return sigma


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t0, tfinal = 0, 4.0

    # get a weird transcendental function
    def fn(x):
        res = np.arctan((x - 2) ** 2) / np.pi
        res += 1.0
        return res

    # get a random point to interpolate near
    xx = np.random.uniform(t0, tfinal)

    n = 7
    # Get a bunch of random x points in the interval [t0, tfinal]
    x = np.random.uniform(t0, tfinal, size=(n,))
    # Order them by how close they are to xx
    x = x[reorder_points_near(xx, x)]
    # apply the function to x and store in y
    y = fn(x)

    assert x.shape[0] == y.shape[0]

    # get "true" values of the function
    x_true = np.linspace(t0, tfinal, 100)
    y_true = fn(x_true)

    # create figures and scatter plot of data points
    fig, ax = plt.subplots(
        2,
        3,
        sharex=True,
        sharey=True,
        tight_layout=True,
        figsize=(9, 6),
    )
    for i, j in np.ndindex(ax.shape):
        # plot the data points
        ax[i, j].scatter(x, y, marker="o", label="measured")
        # plot the true function
        ax[i, j].plot(x_true, y_true, label="actual")
        # set limits to the true function so that large polynomials
        # don't dominate
        ax[i, j].set_ylim(y_true.min() - 0.3, y_true.max() + 0.3)

        ax[i, j].scatter(xx, fn(xx), marker="x", label="interp point")

    # dense, regular x-data for plotting interpolation
    x_dense = np.linspace(t0, tfinal, 500)

    # for each axes, create a kth order lagrange polynomial
    # around the point xx.
    for k, (i, j) in enumerate(np.ndindex(ax.shape)):
        # start with order 1, not 0!
        k += 1
        y_dense = np.empty_like(x_dense)
        # conduct the interopolation
        for dense_idx, dense_x in enumerate(x_dense):
            y_dense[dense_idx] = lagrange_polynomial(dense_x, x, y, k)
        ax[i, j].plot(x_dense, y_dense, linestyle="--", label=f"lagrange, k={k}")
        # add legend
        ax[i, j].legend()

    fig.savefig("plots/lagrange.png", dpi=300)
    plt.show()
