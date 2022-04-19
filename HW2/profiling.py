from solver import gauss_elim
from heateqn import A_dirichlet, b_dirichlet
import numpy as np
from pyinstrument import Profiler


if __name__ == "__main__":
    routine = "numpy"
    profiler = Profiler(interval=0.0000001)
    n = 63
    top = np.full(shape=(n,), fill_value=400)
    bottom = np.full(shape=(n,), fill_value=300)
    left = np.full(shape=(n,), fill_value=300)
    right = np.full(shape=(n,), fill_value=300)
    A = A_dirichlet(n)
    b = b_dirichlet(top, bottom, left, right)

    profiler.start()
    if routine == "mine":
        x = gauss_elim(A, b)
    elif routine == "numpy":
        x = np.linalg.solve(A, b)
    profiler.stop()
    profiler.print()
