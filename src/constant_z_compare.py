import numpy as np

from constant_z import get_solution, get_alt
from numerical_solution import get_numerical_solution
from finite_diff_in_Laplase import get_analytic_solution as get_fd_solution
from analytic_solution import get_analytic_solution
from Z_calculator import z_class


def C_func(s: complex):
    return 0.01 * s

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Pa = 10
    Pb = 20
    ALPHA = -1
    S = 10

    #p_a = get_solution(s=S, C_func=C_func, Pa=Pa, Pb=Pb)
    #p_a_a = get_alt(s=S, C_func=C_func, Pa=Pa, Pb=Pb)
    p_n = get_numerical_solution(s_val=S, alpha=ALPHA, z_func=z_class(), C_func=C_func, Pa=Pa, Pb=Pb)
    p_df = get_fd_solution(s_val=S, alpha=ALPHA, z_func=z_class(), C_func=C_func, Pa=Pa, Pb=Pb)
    p_our = [get_analytic_solution(s=S, C_func=C_func, alpha=ALPHA,  z=z_class(), x=_x, Pa=Pa, Pb=Pb) for _x in np.linspace(0, 1, 1000)]

    #plt.plot(p_a, label='analytic', lw=5)
    #plt.plot(p_a_a, label='analytic2', lw=4)
    plt.plot(p_n, label='numeric', lw=3)
    plt.plot(p_df, label='df', lw=2)
    plt.plot(np.array(p_our), label='our')
    plt.legend()
    plt.show()