import numpy as np
from numpy.typing import NDArray
from inv_matrix import inv_matrix


def get_coefs(a1, a2, a3, a4, b1, b2):
    # | a1 a2 | | c1 | = | b1 |
    # | a3 a4 | | c2 | = | b2 |
    A_inv = inv_matrix(np.array([[a1, a2], [a3, a4]]))
    smth = np.linalg.inv(np.array([[a1, a2], [a3, a4]]))
    return A_inv @ np.array([b1, b2])


if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])

    C_analytic = get_coefs(A[0, 0], A[0, 1], A[1, 0], A[1, 1], b[0], b[1])
    C_numeric = np.linalg.solve(A, b)

    for it1, it2 in zip(C_numeric, C_analytic):
        assert(np.isclose(it1, it2))