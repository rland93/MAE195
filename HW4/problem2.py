import numpy as np
from cmath import sqrt
from enum import Enum
import warnings


class damped(Enum):
    OVER = 0
    CRITICALLY = 1
    UNDER = 2


def get_lambda(alpha, omega):
    lambda1 = -complex(alpha) + sqrt(complex(alpha**2 - 4 * omega**2))
    lambda2 = -complex(alpha) - sqrt(complex(alpha**2 - 4 * omega**2))
    return lambda1, lambda2


def analytic_soln(alpha: complex, omega: complex, x0: float, dx0: float):
    """Get callable solution, given alpha, omega and conditions at t=0

    Returns tuple of callable solution, and case (under, over, critically damped)

    """
    # eigenvalues
    lambda1, lambda2 = get_lambda(alpha, omega)
    # critically damped case
    if lambda1 == lambda2 and lambda1.imag == 0 and lambda2.imag == 0:
        # analytic solution for critically damped spring-mass-damper system
        # x = (c1 + c2t) exp(lambda1 t)
        # solve c1 and c2
        A = np.array([[1, 0], [lambda1.real, 1]])
        x = np.array([x0, dx0])
        c = np.linalg.solve(A, x)
        c1 = c[0]
        c2 = c[1]
        # return the functional form of the solution
        def solution(t):
            return (c1 + t * c2) * np.exp(lambda1.real * t)

        case = damped.CRITICALLY

    # overdamped case
    elif lambda1.imag == 0 and lambda2.imag == 0:
        # analytic solution for overdamped spring-mass-damper system
        # x = c1 exp(lambda1 t) + c2 exp(lambda2 t)
        # solve c1 and c2
        A = np.array([[1, 1], [lambda1.real, lambda2.real]])
        x = np.array([x0, dx0])
        c = np.linalg.solve(A, x)
        c1 = c[0]
        c2 = c[1]
        # return the functional form of the solution
        def solution(t):
            return c1 * np.exp(lambda1.real * t) + c2 * np.exp(lambda2.real * t)

        case = damped.OVER

    # underdamped case
    elif lambda1.imag != 0 and lambda2.imag != 0:
        assert lambda1.real == lambda2.real
        # analytic solution for underdamped spring-mass-damper system
        # x = exp(lambda1.real t) * (c1 cos(omega t) + c2 sin(omega t))
        # solve c1 and c2
        A = np.array([[1, 0], [lambda1.real, omega]])
        x = np.array([x0, dx0])
        c = np.linalg.solve(A, x)
        c1 = c[0]
        c2 = c[1]
        # return the functional form of the solution
        def solution(t):
            exponent = np.exp(lambda1.real * t)
            sinusoid = c1 * np.cos(omega * t) + c2 * np.sin(omega * t)
            return exponent * sinusoid

        case = damped.UNDER

    return solution, case


def stiffness(alpha, omega):
    lambda1, lambda2 = get_lambda(alpha, omega)
    if lambda2.real == 0 or lambda1.real == 0:
        warnings.warn(
            f"zero eigenvalue: λ1={lambda1.real} + {lambda1.imag}j, λ2={lambda2.real} + {lambda2.imag}j"
        )
    try:
        num = max(abs(lambda1.real), abs(lambda2.real))
        denom = min(abs(lambda1.real), abs(lambda2.real))
        return num / denom
    except ZeroDivisionError:
        return float("inf")


def test_analytic_solns(plot=False, alphamax=7, omegamax=3):
    from itertools import product
    import matplotlib.pyplot as plt

    for alpha, omega in product(range(alphamax), range(omegamax)):
        print(f"\nalpha = {alpha}, omega = {omega}")
        print(stiffness(alpha, omega))
        ts = np.linspace(0, 3, 200)
        soln, status = analytic_soln(alpha, omega, 4.0, 0)
        if plot:
            plt.plot(ts, soln(ts))
            plt.title(f"alpha={alpha}, omega={omega}, {status}")
            plt.show()


if __name__ == "__main__":
    # uncomment to do a test of the analytic solution generator.
    test_analytic_solns(plot=True)
    pass
