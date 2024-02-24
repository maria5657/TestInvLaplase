import numpy as np
from Z_calculator import Z_class, z_class
from U_V_calculator import *
from calc_a_b import  calc_a_b
from eq_coefs import  solve_koefs_eq
from sqrt_complex import sqrt_complex
import special_functions as sf
import pyhypergeomatrix as hyper

def get_analytic_solution(s: complex, C_func: callable, alpha: float,  z: z_class, x: float, Pa: float, Pb: float):
    # решается следующее уравнение:
    # d^2 p / dx^2 = C * z(x) * p
    # dp / dx (0) = Pa
    # dp / dx (1) = Pb
    # Переходим к новой переменной z = z(x), dz / dx != 0, d^2 z / dx^2 = 0, тогда:
    # d^2 p / dz^2 = C * z(x) * p / (dz / dx (x))^2
    # dp / dx (0) = Pa / (dz / dx (0))
    # dp / dx (1) = Pb / (dz / dx (1))
    if np.isclose(alpha, 0):
        # тут скорее всего лажа может быть в делении С
        Z = Z_class(z_func=z, C=C_func(s) / ((z.derivative(0)) ** 2), q=(alpha + 2) / 2)
        z = z_class()
        Z_powers_left = Z.calc_powers(0)
        Z_powers_right = Z.calc_powers(1)
        a, b = calc_a_b(q=(alpha + 2) / 2)
        C1, C2 = solve_koefs_eq(
            np.exp(Z(0)) * (calc_U_1(a, Z_powers_left) + calc_U_1_derivative(a, Z_powers_left)),
            np.exp(-Z(0)) * (-calc_U_2(a, Z_powers_left) + calc_U_2_derivative(a, Z_powers_left)),
            np.exp(Z(1)) * (calc_U_1(a, Z_powers_right) + calc_U_1_derivative(a, Z_powers_right)),
            np.exp(-Z(1)) * (-calc_U_2(a, Z_powers_right) + calc_U_2_derivative(a, Z_powers_right)),
            Pa / (Z.derivative(0)),
            Pb / (Z.derivative(1)),
        )
        Z_powers_x = Z.calc_powers(x)
        return C1 * np.exp(Z(x)) * calc_U_1(a, Z_powers_x) + C2 * np.exp(- Z(x)) * calc_U_2(a, Z_powers_x)
    elif np.isclose(alpha, -1):
        Z_0 = sqrt_complex(C_func(s) / (z.derivative(0)) ** 2) * sqrt_complex(z(0))
        Z_1 = sqrt_complex(C_func(s) / (z.derivative(1)) ** 2) * sqrt_complex(z(1))
        dZ_dz_0 = sqrt_complex(C_func(s) / (z.derivative(0)) ** 2) / sqrt_complex(z(0)) / 2
        dZ_dz_1 = sqrt_complex(C_func(s) / (z.derivative(1)) ** 2) / sqrt_complex(z(1)) / 2
        C1, C2 = solve_koefs_eq(
            dZ_dz_0 * (sf.besseli(1, 2 * Z_0, 0) + 2 * Z_0 * sf.besseli(1, 2 * Z_0, 1)),
            dZ_dz_0 * (sf.besselk(1, 2 * Z_0, 0) + 2 * Z_0 * sf.besselk(1, 2 * Z_0, 1)),
            dZ_dz_1 * (sf.besseli(1, 2 * Z_1, 0) + 2 * Z_1 * sf.besseli(1, 2 * Z_1, 1)),
            dZ_dz_1 * (sf.besselk(1, 2 * Z_1, 0) + 2 * Z_1 * sf.besselk(1, 2 * Z_1, 1)),
            Pa / (z.derivative(0)),
            Pb / (z.derivative(1)),
        )

        Z_x = sqrt_complex(C_func(s)/ (z.derivative(x)) ** 2) * sqrt_complex(z(x))
        return C1 * sf.besseli(1, 2 * Z_x, 0) * Z_x + C2 * sf.besselk(1, 2 * Z_x, 0) * Z_x