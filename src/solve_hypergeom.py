import numpy as np
import scipy.special
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy import *
from scipy import integrate

from src.eq_coefs import solve_koefs_eq




def get_numerical_solution(A, B, Pa, Pb, a_const, b_const):

    N = 1001
    x = np.linspace(A, B, N)
    y_guess = np.zeros((2, N), dtype=float)

    def f_real(t, y, C_func: callable):
        coef = C_func(t)
        return [
            y[1],
            coef * y[0]
        ]

    def boundary_residual_real(ya, yb):
        return np.array([
            ya[1] - Pa,
            yb[1] - Pb,
        ])

    f = lambda t, y: f_real(t, y, lambda z: (a_const + b_const * z) / z)
    b = lambda y1, y2: boundary_residual_real(y1, y2)

    sol = integrate.solve_bvp(
        f,b,
        x, y_guess
        )

    #sol = integrate.solve_ivp(f, [Pa, Pb], np.array([A, B]), rtol=1e-12)
    #
    # plt.plot(sol.y[0])
    # plt.show()
    # plt.plot(sol.y[0], sol.y[1])
    # plt.show()

    return sol.y[0], x

def solve(A, B, Pa, Pb, a_const, b_const):
    u = lambda a_u, b_u, z_u: scipy.special.hyperu(a_u, b_u, z_u)
    f1 = lambda a_f1, b_f1, z_f1: scipy.special.hyp1f1(a_f1, b_f1, z_f1)

    def f1_der(a, b, z):
        return (a / b) * f1(a + 1, b + 1, z)

    def u_der(a, b, z):
        return -a * u(a + 1, b + 1, z)

    C1, C2 = solve_koefs_eq(
        np.exp(- np.sqrt(b_const) * A) * (
                    2 * np.sqrt(b_const) * A * u_der(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * A) + u(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * A) * (1 - np.sqrt(b_const) * A)),
        np.exp(- np.sqrt(b_const) * A) * (
                    2 * np.sqrt(b_const) * A * f1_der(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * A) + f1(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * A) * (1 - np.sqrt(b_const) * A)),
        np.exp(- np.sqrt(b_const) * B) * (
                    2 * np.sqrt(b_const) * B * u_der(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * B) + u(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * B) * (1 - np.sqrt(b_const) * B)),
        np.exp(- np.sqrt(b_const) * B) * (
                    2 * np.sqrt(b_const) * B * f1_der(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * B) + f1(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * B) * (1 - np.sqrt(b_const) * B)),
        Pa,
        Pb,
    )

    # C1, C2 = solve_koefs_eq(
    #     A * np.exp(- np.sqrt(b_const) * A) * u(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * A),
    #     A * np.exp(- np.sqrt(b_const) * A) * f1(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * A),
    #     B * np.exp(- np.sqrt(b_const) * B) * u(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * B),
    #     B * np.exp(- np.sqrt(b_const) * B) * f1(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * B),
    #     Pa, Pb,
    # )

    g = 7



    def sol(z):
        return C1 * z * np.exp(- np.sqrt(b_const) * z) * u(a_const / (2 * np.sqrt(b_const)) + 1, 2,
                                                           2 * np.sqrt(b_const) * z) + C2 * z * np.exp(
            - np.sqrt(b_const) * z) * f1(a_const / (2 * np.sqrt(b_const)) + 1, 2, 2 * np.sqrt(b_const) * z)

    return sol

if __name__ == '__main__':
    A, B = 1, 2
    Pa, Pb = 5, 7
    a_const, b_const = 1, 1
    numerical, grid = get_numerical_solution(A, B, Pa, Pb, a_const, b_const)

    analytic = solve(A, B, Pa, Pb, a_const, b_const)
    analytic_sol = [analytic(z) for z in grid]

    tmp = (analytic_sol[1] - analytic_sol[0]) / (grid[1] - grid[0])

    plt.plot(numerical, label='n')
    plt.plot(analytic_sol, label='a')
    plt.legend()
    plt.show()
