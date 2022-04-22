import numpy as np
import sys, time
from solver import gauss_elim, gauss_seidel, diagonally_dominant


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
    ncc = 3
    topcc = np.full(shape=(ncc,), fill_value=100)
    bottomcc = np.full(shape=(ncc,), fill_value=0)
    leftcc = np.full(shape=(ncc,), fill_value=50)
    rightcc = np.full(shape=(ncc,), fill_value=75)
    Acc = A_dirichlet(ncc)
    if not diagonally_dominant(Acc):
        raise ValueError("A is not diagonally dominant")
    bcc = b_dirichlet(topcc, bottomcc, leftcc, rightcc)
    # solve
    x_cc = gauss_elim(Acc, bcc)
    T_cc = unpack_solution(x_cc)

    # formulate problem 2
    np2 = 19
    topp2 = np.full(shape=(np2,), fill_value=400)
    bottomp2 = np.full(shape=(np2,), fill_value=300)
    leftp2 = np.full(shape=(np2,), fill_value=300)
    rightp2 = np.full(shape=(np2,), fill_value=300)
    Ap2 = A_dirichlet(np2)
    if not diagonally_dominant(Ap2):
        raise ValueError("A is not diagonally dominant")
    bp2 = b_dirichlet(topp2, bottomp2, leftp2, rightp2)
    # solve
    x_p2 = gauss_elim(Ap2, bp2)
    T_p2 = unpack_solution(x_p2)

    fig1, ax1 = plt.subplots(1, 2, tight_layout=True)
    plot_matr(ax1[0], T_cc, text=True)
    ax1[0].set_title(r"Fig 29.5, page 871 of C & C")

    plot_matr(ax1[1], T_p2, text=False)
    ax1[1].set_title(r"Solution Problem 1, $n=19$")
    fig1.savefig(f"heateqn.png", dpi=200)
    plt.show()
