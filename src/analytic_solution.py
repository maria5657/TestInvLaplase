from typing import TypedDict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import *
from math import factorial
from inverse_laplace_functions import get_inv_laplace

def calc_alpha(vals: Constants):
    return - 1 / vals.val_n


def calc_q(vals: Constants):
    return (calc_alpha(vals) + 2) / 2


def calc_Z(x_D: float, vals: Constants, s: float):
    z = calc_z(x_D=x_D, vals=vals)
    C = calc_C(s=s, vals=vals)
    q = calc_q(vals=vals)
    return q ** (-1) * z ** q * np.sqrt(C)


def calc_dZ_dz(x_D: float, vals: Constants, s: float):
    z = calc_z(x_D=x_D, vals=vals)
    C = calc_C(s=s, vals=vals)
    q = calc_q(vals=vals)
    return z ** (q - 1) * np.sqrt(C)





def calc_a_nu(nu: int, vals: Constants, n_num: int):
    q = calc_q(vals=vals)
    assert q != 0
    if np.isclose((1 / q) % 2, 1):
        if nu == 0:
            return 1
        if nu > n_num:
            return 0
        res = 1
        for k in range(1, nu + 1):
            res *= ((2 * k - 1) * q - 1) / (k * q - 1)
        return res
    else:
        # чётное
        if nu < 2 * n_num:
            return 0
        if nu < 2 * n_num + 1:
            return 1
        res = 1
        for k in range(2 * n_num + 1, nu + 1):
            res *= ((2 * k - 1) * q - 1) / (k * q - 1)
        return res


def calc_b_nu(nu: int, vals: Constants, n_num: int):
    q = calc_q(vals=vals)
    assert q != 0
    if np.isclose((1 / q) % 2, 1):
        if nu == 0:
            return 1
        if nu > n_num:
            return 0
        res = 1
        for k in range(1, nu + 1):
            res *= ((2 * k - 1) * q + 1) / (k * q + 1)
        return res
    else:
        # чётное
        if nu < 2 * n_num:
            return 0
        if nu < 2 * n_num + 1:
            return 1
        res = 1
        res = 1
        for k in range(2 * n_num + 1, nu):
            res *= ((2 * k - 1) * q + 1) / (k * q + 1)
        return res


def calc_n_num(vals: Constants):
    if np.isclose((1 / calc_q(vals=vals)) % 2, 1):
        return int(abs((abs(calc_q(vals=vals)) - 1) // 2))
    else:
        # чётное
        return int((1 / calc_q(vals=vals)) / 2)


def calc_U_1(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(0, 15):
        a_nu = calc_a_nu(nu=nu, vals=vals, n_num=n_num)
        res += a_nu * Z ** nu * (-1) ** nu / factorial(nu)
    return res


def calc_U_2(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(0, 15):
        a_nu = calc_a_nu(nu=nu, vals=vals, n_num=n_num)
        res += a_nu * Z ** nu / factorial(nu)
    return res


def calc_V_1(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(0, 15):
        b_nu = calc_b_nu(nu=nu, vals=vals, n_num=n_num)
        res += b_nu * Z ** nu / factorial(nu)
    return res


def calc_V_2(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(0, 15):
        b_nu = calc_b_nu(nu=nu, vals=vals, n_num=n_num)
        res += b_nu * Z ** nu * (-1) ** nu / factorial(nu)
    return res


def calc_U_1_star(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    return np.exp(Z) * calc_U_1(x_D=x_D, vals=vals, s=s)


def calc_U_2_star(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    return np.exp(-Z) * calc_U_2(x_D=x_D, vals=vals, s=s)


def calc_V_1_star(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    return np.exp(-Z) * calc_V_1(x_D=x_D, vals=vals, s=s)


def calc_V_2_star(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    return np.exp(Z) * calc_V_2(x_D=x_D, vals=vals, s=s)


def calc_dU_1_dZ(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(1, 15):
        a_nu = calc_a_nu(nu=nu, vals=vals, n_num=n_num)
        res += a_nu * Z ** (nu - 1) * (-1) ** nu / factorial(nu - 1)
    return res


def calc_dU_2_dZ(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(1, 15):
        a_nu = calc_a_nu(nu=nu, vals=vals, n_num=n_num)
        res += a_nu * Z ** (nu - 1) / factorial(nu - 1)
    return res


def calc_dV_1_dZ(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(1, 15):
        b_nu = calc_b_nu(nu=nu, vals=vals, n_num=n_num)
        res += b_nu * Z ** (nu - 1) / factorial(nu - 1)
    return res


def calc_dV_2_dZ(x_D: float, vals: Constants, s: float):
    n_num = calc_n_num(vals=vals)
    res = 0
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    for nu in range(1, 15):
        b_nu = calc_b_nu(nu=nu, vals=vals, n_num=n_num)
        res += b_nu * Z ** (nu - 1) * (-1) ** nu / factorial(nu - 1)
    return res


def calc_dU_1_star_dZ(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    derivative = calc_dU_1_dZ(x_D=x_D, vals=vals, s=s)
    return np.exp(Z) * (calc_U_1(x_D=x_D, vals=vals, s=s) + derivative)


def calc_dU_2_star_dZ(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    derivative = calc_dU_2_dZ(x_D=x_D, vals=vals, s=s)
    return np.exp(-Z) * (-calc_U_2(x_D=x_D, vals=vals, s=s) + derivative)


def calc_dV_1_star_dZ(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    derivative = calc_dV_1_dZ(x_D=x_D, vals=vals, s=s)
    return np.exp(-Z) * (-calc_V_1(x_D=x_D, vals=vals, s=s) + derivative)


def calc_dV_2_star_dZ(x_D: float, vals: Constants, s: float):
    Z = calc_Z(x_D=x_D, vals=vals, s=s)
    derivative = calc_dV_2_dZ(x_D=x_D, vals=vals, s=s)
    return np.exp(Z) * (calc_V_2(x_D=x_D, vals=vals, s=s) + derivative)

def calc_eq_for_C1_C2(a: float, b: float, ks: float, d: float, e: float):
    # a * C1 + b * C2 = ks
    # d * C1 + e * C2 = 0
    C1 = - e * ks / (b * d - e * a)
    C2 = ks * d / (b * d - e * a)
    return C1, C2


def calc_C1_C2(x_D: float, vals: Constants, s: float):
    q = calc_q(vals=vals)
    ks = calc_b(
        x_D=0, vals=vals) / (s * calc_dZ_dz(x_D=0, vals=vals, s=s) * calc_dz_dx_D(vals=vals)
                             )
    if np.isclose((1 / q) % 2, 1):
        if q > 0:
            a = calc_dU_1_star_dZ(x_D=0, vals=vals, s=s)
            b = calc_dU_2_star_dZ(x_D=0, vals=vals, s=s)
            d = calc_dU_1_star_dZ(x_D=1, vals=vals, s=s)
            e = calc_dU_2_star_dZ(x_D=1, vals=vals, s=s)
        else:
            a = calc_dV_1_star_dZ(x_D=0, vals=vals, s=s)
            b = calc_dV_2_star_dZ(x_D=0, vals=vals, s=s)
            d = calc_dV_1_star_dZ(x_D=1, vals=vals, s=s)
            e = calc_dV_2_star_dZ(x_D=1, vals=vals, s=s)
        return calc_eq_for_C1_C2(a=a, b=b, d=d, e=e, ks=ks)
    else:
        # чётное
        a = calc_dU_1_star_dZ(x_D=0, vals=vals, s=s)
        b = calc_dV_2_star_dZ(x_D=0, vals=vals, s=s)
        d = calc_dU_1_star_dZ(x_D=1, vals=vals, s=s)
        e = calc_dV_2_star_dZ(x_D=1, vals=vals, s=s)
        return calc_eq_for_C1_C2(a=a, b=b, d=d, e=e, ks=ks)


def get_p_universal(x_D: float, vals: Constants, s: float):
    q = calc_q(vals=vals)

    if np.isclose((1 / q) % 2, 1):
        C1, C2 = calc_C1_C2(x_D=x_D, vals=vals, s=s)
        if q > 0:
            U_1_star = calc_U_1_star(x_D=x_D, vals=vals, s=s)
            U_2_star = calc_U_2_star(x_D=x_D, vals=vals, s=s)
            return C1 * U_1_star + C2 * U_2_star
        else:
            V_1_star = calc_V_1_star(x_D=x_D, vals=vals, s=s)
            V_2_star = calc_V_2_star(x_D=x_D, vals=vals, s=s)
            return C1 * V_1_star + C2 * V_2_star
    else:
        # чётное
        U_1_star = calc_U_1_star(x_D=x_D, vals=vals, s=s)
        V_2_star = calc_V_2_star(x_D=x_D, vals=vals, s=s)
        C1, C2 = calc_C1_C2(x_D=x_D, vals=vals, s=s)
        return C1 * U_1_star + C2 * V_2_star


if __name__ == "__main__":

    print('Обратное преобразование Лапласса')

    t_moments = np.exp(np.linspace(np.log(2), np.log(100), num=4))
    print(t_moments)

    p_f_sol = get_inv_laplace(t_moments, lambda q: get_p_universal(0, Constants(), q))

    plt.plot(t_moments, p_f_sol, '-o')
    plt.ylabel('Давление, бар')
    plt.xlabel('Время t, сек')
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    #plt.savefig(str(res_path / 'res1.png'), dpi=200)