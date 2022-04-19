import numpy as np
import sys, time
from solver import gauss_elim


def plot_matr(ax, T, text=False, cmap="RdYlBu_r"):
    ax.matshow(T, cmap=cmap, origin="lower")
    if text:  # show text
        for i, j in np.ndindex(T.shape):
            ax.text(j, i, round(T[i, j], 1), ha="center", va="center")
    ax.set_xlabel("x idx")
    ax.set_ylabel("y idx")
    ax.set_aspect("equal")
    return ax


def diag_indices2d(H, offset=0):
    """get 2d diagonal indices, with an offset applied
    e.g. for a 4x4 matrix H
    H  =[[ 0,  1,  0,  0],
         [-2,  0,  1,  0],
         [ 0, -2,  0,  1],
         [ 0,  0, -2,  0]]
    >>> diag_indices2d(H, offset=-1) ==
    [[0, 0, 0, 0],
     [1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0 ]]

    """
    mask = np.zeros_like(H, dtype=bool)
    n = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if offset >= 1:
        cols = np.arange(offset, n)
        rows = np.arange(0, n - offset)
    elif offset <= -1:
        cols = np.arange(0, n + offset)
        rows = np.arange(-1 * offset, n)
    elif offset == 0:
        cols = np.arange(n)
        rows = np.arange(n)
    mask[rows, cols] = 1
    return mask


def A_dirichlet(n, const=1):
    """Matrix for dirichlet boundary conditions for a 2d grid of size nxn
    the scale of this matrix is 1/h, where h is the grid step.
    """
    # go through block "rows"
    A = np.zeros((n * n, n * n))
    for i in range(n):
        # blockrow is an n x n*n "row" of the nxn A matrix.
        # thus, for an n*n x n*n matrix, there are n block rows.
        blockrow = A[i * n : (i + 1) * n, :]
        # go through block "columns"
        for j in range(n):
            block = blockrow[:, n * j : n * (j + 1)]  # this is nxn block
            if i == j:  # kth "diagonal" block
                block[diag_indices2d(block, 0)] = 4
                block[diag_indices2d(block, 1) | diag_indices2d(block, -1)] = -1
            elif i == j + 1:  # k-1 "lower" block
                block[diag_indices2d(block, 0)] = -1
            elif i + 1 == j:  # k+1 "upper" block
                block[diag_indices2d(block, 0)] = -1
    return A * const


def b_dirichlet(top, bottom, left, right):
    """dirichlet boundary condition"""
    assert top.shape == bottom.shape == left.shape == right.shape
    n = top.shape[0]
    b = np.zeros((n * n, 1))
    for i in range(n):  # rows
        for j in range(n):  # columns
            if j == 0:
                b[i + j * n] += bottom[j]
            elif j == n - 1:
                b[i + j * n] += top[j]
            if i == 0:
                b[i + j * n] += right[i]
            elif i == n - 1:
                b[i + j * n] += left[i]
    return b


def unpack_solution(x):
    """unpack solution vector x into a 2d array
    so that it can be visualized"""
    n = int(np.sqrt(x.shape[0]))
    return x.reshape((n, n))


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # formulate the problem from C & C ch 29.
    n = 3

    top = np.full(shape=(3,), fill_value=100)
    bottom = np.full(shape=(3,), fill_value=0)
    left = np.full(shape=(3,), fill_value=50)
    right = np.full(shape=(3,), fill_value=75)

    A = A_dirichlet(3)
    b = b_dirichlet(top, bottom, left, right)

    # solve
    # x = np.linalg.solve(A, b)
    x = gauss_elim(A, b)
    T = unpack_solution(x)

    # verify that the solution is identical to 29.1
    # fig 29.5, page 871 of C & C.
    fig1, ax1 = plt.subplots(1, 1)
    plot_matr(ax1, T)
    # problem 2
    for i, n in enumerate([3, 7, 15]):
        print(f"n = {n}")
        top = np.full(shape=(n,), fill_value=400)
        bottom = np.full(shape=(n,), fill_value=300)
        left = np.full(shape=(n,), fill_value=300)
        right = np.full(shape=(n,), fill_value=300)

        A = A_dirichlet(n)

        print(f"\tA matrix shape: {A.shape}")
        print(f"\tA matrix size: {sys.getsizeof(A) / 1024} KB")

        b = b_dirichlet(top, bottom, left, right)

        # time & solve the system
        t0 = time.time()
        x_np = np.linalg.solve(A, b)
        t_np = time.time() - t0

        t0 = time.time()
        x_me = gauss_elim(A, b)
        t_me = time.time() - t0

        fig, ax = plt.subplots(ncols=2)
        plot_matr(ax[0], unpack_solution(x_np))
        ax[0].set_title(f"Numpy Solve Routine: {round(t_np*1000, 0)} ms")
        plot_matr(ax[1], unpack_solution(x_me))
        ax[1].set_title(f"Custom Solve Routine: {round(t_me*1000, 0)} ms")
