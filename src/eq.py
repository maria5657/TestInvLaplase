import numpy as np
from sqrt_complex import sqrt_complex
from eq_coefs import get_coefs
from inverse_laplace_functions import get_inv_laplace
import matplotlib.pyplot as plt

a_coef = 5
a_coef_sqrt = np.sqrt(a_coef)
p_a = 10
p_b = 0
y_a = 0
y_b = 1

N = 1600
J = 40


def numeric_solution():
    tau = 1 / N
    h = 1 / J
    sol = np.zeros((N, J))
    sol[0, :] = 0
    sol[:, 0] = p_a
    sol[:, -1] = p_b
    tmp = tau / a_coef / h**2
    for n in range(0, N - 1):
        for j in range(1, J - 1):
            sol[n + 1, j] = sol[n, j] * (1 - 2 * tmp) + sol[n, j - 1] * tmp + sol[n, j+1] * tmp
    return sol[:, int(J / 2)], int(J / 2) * h


def f(s, y=0.1):
    a1, a2, a3, a4 = np.exp([a_coef_sqrt * sqrt_complex(s) * y_a,
                             -a_coef_sqrt * sqrt_complex(s) * y_a,
                             a_coef_sqrt * sqrt_complex(s) * y_b,
                             -a_coef_sqrt * sqrt_complex(s) * y_b,
                             ])
    b1, b2 = np.array([p_a, p_b]) / s

    c1, c2 = get_coefs(a1, a2, a3, a4, b1, b2)

    return c1 * np.exp(a_coef_sqrt * sqrt_complex(s) * y) + c2 * np.exp(- a_coef_sqrt * sqrt_complex(s) * y)


if __name__ == '__main__':
    numeric, y = numeric_solution()
    res = get_inv_laplace(np.linspace(1e-4, 1, 100), lambda s: f(s, y=y))
    plt.plot(np.linspace(1e-4, 1, 100), res, label='a')
    plt.plot(np.linspace(0, 1, N), numeric, label='n')
    plt.legend()
    plt.show()



