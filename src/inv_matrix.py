import numpy as np
from numpy.typing import NDArray


def inv_matrix(A: NDArray):
    assert(all([it == 2 for it in A.shape]))

    a1, a2, a3, a4 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]

    determinant = a1 * a4 - a3 * a2

    assert(determinant != 0)

    return (1 / determinant) * np.array([[a4, -a2], [-a3, a1]])


if __name__ == '__main__':
    A = np.array([[10, 20], [30, 0.4]])
    A_inv_numeric = np.linalg.inv(A)
    A_inv_analytic = inv_matrix(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            assert(np.isclose(A_inv_analytic[i, j], A_inv_analytic[i, j]))