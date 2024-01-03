from constant_z import get_solution
from numerical_solution import get_numerical_solution
from finite_diff_in_Laplase import get_analytic_solution


def C_func(s: complex):
    return - 10 * s


def z_func(x: float):
    return 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Pa = 10
    Pb = 20
    ALPHA = 1
    S = complex(1, 1)

    p_a = get_solution(s=S, C_func=C_func, Pa=Pa, Pb=Pb)
    p_df = get_analytic_solution(s_val=S, alpha=ALPHA, z_func=z_func, C_func=C_func, Pa=Pa, Pb=Pb)

    plt.plot(p_a, label='analytic')
    plt.plot(p_df, label='df')
    plt.legend()
    plt.show()