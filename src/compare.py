import numpy as np

from finite_diff_in_Laplase import get_analytic_solution
from constants import calc_z, calc_C, Constants
from analytic_solution import get_p_universal, calc_dZ_dz, calc_b, calc_dz_dx_D


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ALPHA = -1
    S = complex(1, 1)

    Pa = calc_b(
        x_D=0, vals=Constants()) / (S * calc_dZ_dz(x_D=0, vals=Constants(), s=S) * calc_dz_dx_D(vals=Constants())
                             )
    Pb = 0


    p_a = np.array([get_p_universal(x_D=_x, vals=Constants(), s=S) for _x in np.linspace(0, 1, 1001)])
    p_df = get_analytic_solution(s_val=S, alpha=ALPHA, z_func=calc_z, C_func=calc_C, Pa=Pa, Pb=Pb)

    def normal(arr):
        arr = abs(arr)
        arr = (arr - np.ones_like(arr) * min(arr)) / (max(arr) - min(arr))
        return arr

    # plt.plot(normal(p_a), label='analytic')
    # plt.plot(normal(p_df), label='df')

    plt.plot(p_a, label='analytic')
    plt.plot(-p_df, label='df')
    plt.yscale('log')
    plt.legend()
    plt.show()