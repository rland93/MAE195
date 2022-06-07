import numpy as np


def tdma(A: np.ndarray, b: np.ndarray):
    assert A.shape[0] == A.shape[1]
    A, r = np.copy(A), np.copy(b)
    n = A.shape[0]

    # get diagonals
    e = A[np.eye(n, n, k=-1, dtype=bool)]
    f = A[np.eye(n, n, dtype=bool)]
    g = A[np.eye(n, n, k=1, dtype=bool)]

    for k in range(0, e.shape[0]):
        e[k] /= f[k]
        f[k + 1] -= e[k] * g[k]

    for k in range(1, r.shape[0]):
        r[k] -= e[k - 1] * r[k - 1]

    x = np.empty((n,))
    x[n - 1] = r[n - 1] / f[n - 1]
    for k in range(n - 2, -1, -1):
        x[k] = (r[k] - g[k] * x[k + 1]) / f[k]

    return x


def tri_mask(A):
    """a matrix with same size A, but Trues on the tridiagonal
    and False everywhere else."""
    mask = np.eye(*A.shape, dtype=bool)
    mask |= np.eye(*A.shape, dtype=bool, k=-1)
    mask |= np.eye(*A.shape, dtype=bool, k=1)
    return mask


if __name__ == "__main__":
    # example from the book
    # A = np.array(
    #     [
    #         [2.04, -1, 0, 0],
    #         [-1, 2.04, -1.0, 0],
    #         [0, -1, 2.04, -1],
    #         [0, 0, -1, 2.04],
    #     ]
    # )
    # b = np.array([40.8, 0.8, 0.8, 200.8])

    A = np.random.uniform(0, 1, (10, 10))
    A = np.where(tri_mask(A), A, 0)
    b = np.random.uniform(0, 1, (10,))

    x = tdma(A, b)
    xnp = np.linalg.solve(A, b)
    print("x: ", x)
    print("xnp: ", xnp)
    print("close: ", np.allclose(x, xnp))
