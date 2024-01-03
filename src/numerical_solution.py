from scipy import *
import numpy as np
from scipy import integrate




def f_real(t, y, s: complex, z_func: callable, C_func: callable, alpha:float):
    coef = C_func(s) * z_func(t) ** alpha
    return [
        y[1],
        coef * y[0]
    ]


def boundary_residual_real(ya, yb, s: complex, Pa: complex, Pb: complex):
    return np.array([
        ya[1] - Pa,
        yb[1] - Pb,
    ])


def get_numerical_solution(s_val: complex, alpha: float, z_func: callable, C_func: callable, Pa: complex, Pb: complex):
    f = lambda t, y: f_real(t, y, s=s_val, z_func=z_func, C_func=C_func, alpha=alpha)
    boundary_residual = lambda ya, yb: boundary_residual_real(ya, yb, s_val, Pa=Pa, Pb=Pb)

    a, b = 0, 1
    N = 1001
    x = np.linspace(a, b, N)
    y_guess = np.zeros((2, N), dtype=float)

    sol = integrate.solve_bvp(f, boundary_residual, x, y_guess)

    return sol.y[0]
