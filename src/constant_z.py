import numpy as np
from numpy.typing import NDArray
from sqrt_complex import sqrt_complex
from eq_coefs import solve_koefs_eq
from analytic_solution import get_analytic_solution

I = 1001
Ya = 0
Yb = 1
Pa = 10
Pb = 20


def get_solution(s: complex, C_func: callable, Pa: complex, Pb: complex):
    y_grid = np.linspace(Ya, Yb, num=I)
    omega = sqrt_complex(- C_func(s))
    C1, C2 = solve_koefs_eq(
        np.cos(omega * Ya), - np.sin(omega * Ya),
        np.cos(omega * Yb), - np.sin(omega * Yb),
        Pa / omega, Pb / omega
    )
    return np.array([
        C1 * np.sin(omega * y) + C2 * np.cos(omega * y)
        for y in y_grid
    ])

def get_alt(s: complex, C_func: callable, Pa: complex, Pb: complex):
    y_grid = np.linspace(Ya, Yb, num=I)
    omega = sqrt_complex(C_func(s))
    C1, C2 = solve_koefs_eq(
        np.exp(omega * Ya), - np.exp( - omega * Ya),
        np.exp(omega * Yb), - np.exp( - omega * Yb),
        Pa / omega, Pb / omega
    )
    return np.array([
        C1 * np.exp(omega * y) + C2 * np.exp( - omega * y)
        for y in y_grid
    ])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    p = get_solution(s=complex(1, 1), C_func=lambda t: 5 * t, Pa=Pa, Pb=Pb)
    plt.plot(p)
    plt.show()


