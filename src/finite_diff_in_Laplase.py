import numpy as np

Ya = 0
Yb = 1
I = 1001
h = (Yb - Ya) / (I - 1)



def get_analytic_solution(s_val: complex, alpha: float, z_func: callable, C_func: callable, Pa: complex, Pb: complex):
    def calc_z_i(i):
        return z_func(i * h) ** alpha

    def calc_k_i(i):
        return 1 / (h ** 2 * C_func(s_val) * calc_z_i(i) + 2)

    A = [[0 for i in range(I)] for j in range(I)]
    b = [0 for i in range(I)]
    for i in range(1, I - 1):
        A[i][i - 1] = - calc_k_i(i)
        A[i][i] = 1
        A[i][i + 1] = - calc_k_i(i)
    A[0][0] = 1
    A[0][1] = -1
    A[-1][-2] = -1
    A[-1][-1] = 1
    b[0] = - Pa * h
    b[-1] = Pb * h
    return np.linalg.solve(A, b)
