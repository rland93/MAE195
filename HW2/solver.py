import numpy as np
from typing import Tuple


def get_rand_Ab(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """obtain a random nxn A matrix and nx1 b vector"""
    A = np.random.uniform(low=0.0, high=1.0, size=(n, n))
    b = np.random.uniform(low=0.0, high=1.0, size=(n, 1))
    return A, b


def fe(G, h, n):
    """forward elimination. Alters G, h"""
    for k in range(0, n - 1):  # pivot row
        for i in range(k + 1, n):  # elimination row
            factor = G[i, k] / G[k, k]
            for j in range(k, n):  # elimination column
                G[i, j] = G[i, j] - factor * G[k, j]
            h[i] -= factor * h[k]
    return G, h


def backsub(G, h, n, x):
    """back substitution. Alters G, h, x"""
    # last
    x[n - 1] = h[n - 1] / G[n - 1, n - 1]
    # back substitution
    for i in range(n - 1, -1, -1):
        sigma = np.float128(np.squeeze(h[i]))
        for j in range(i + 1, n):
            sigma -= G[i, j] * x[j]
        x[i] = sigma / G[i, i]
    return G, h, x


def gauss_elim(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Perform Gauss Elimination for a system Ax = b"""
    n = A.shape[0]
    x = np.zeros((n,), dtype=np.float64)

    # make copies
    G = A.copy()
    h = np.squeeze(b.copy())

    # encapsulate for the profiler
    G, h = fe(G, h, n)
    # encapsulate for the profiler
    G, h, x = backsub(G, h, n, x)

    return np.atleast_2d(x).T


if __name__ == "__main__":
    tries = 100
    n = 4
    err = 0
    A, b = get_rand_Ab(n)

    x_mine = gauss_elim(A, b)
    x_theirs = np.linalg.solve(A, b)
    print("=============")
    print(f"x_mine: \n{x_mine}")
    print(f"x_theirs: \n{x_theirs}")
