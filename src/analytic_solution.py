import numpy as np
from Z_calculator import Z_class, z_class
from U_V_calculator import *
from calc_a_b import  calc_a_b
from eq_coefs import  solve_koefs_eq
from sqrt_complex import sqrt_complex
import special_functions as sf


def get_analytic_solution(s: complex, C_func: callable, alpha: float,  z: z_class, x: float, Pa: float, Pb: float):
    Z = Z_class(z_func=z, C=C_func(s), q=(alpha + 2) / 2)
    z = z_class()
    Z_powers_left = Z.calc_powers(0)
    Z_powers_right = Z.calc_powers(1)
    a, b = calc_a_b(q=(alpha + 2) / 2)
    if np.isclose(alpha, 0):
        C1, C2 = solve_koefs_eq(
            np.exp(Z(0)) * (calc_U_1(a, Z_powers_left) + calc_U_1_derivative(a, Z_powers_left)),
            np.exp(-Z(0)) * (-calc_U_2(a, Z_powers_left) + calc_U_2_derivative(a, Z_powers_left)),
            np.exp(Z(1)) * (calc_U_1(a, Z_powers_right) + calc_U_1_derivative(a, Z_powers_right)),
            np.exp(-Z(1)) * (-calc_U_2(a, Z_powers_right) + calc_U_2_derivative(a, Z_powers_right)),
            Pa / (Z.derivative(0) * z.derivative(0)),
            Pb / (Z.derivative(1) * z.derivative(1)),
        )
        Z_powers_x = Z.calc_powers(x)
        return C1 * np.exp(Z(x)) * calc_U_1(a, Z_powers_x) + C2 * np.exp(- Z(x)) * calc_U_2(a, Z_powers_x)
    elif np.isclose(alpha, -1):
        tmp0 = sqrt_complex(C_func(s)) / sqrt_complex(z(0)) / 2
        tmp1 = sqrt_complex(C_func(s)) / sqrt_complex(z(1)) / 2
        val0 = sqrt_complex(C_func(s)) * sqrt_complex(z(0))
        val1 = sqrt_complex(C_func(s)) * sqrt_complex(z(1))
        C1, C2 = solve_koefs_eq(
            tmp0 * sf.besseli(1, 2 * val0, 0) + sf.besseli(1, 2 * val0, 1) * C_func(s),
            tmp0 * sf.besselk(1, 2 * val0, 0) + sf.besselk(1, 2 * val0, 1) * C_func(s),
            tmp1 * sf.besseli(1, 2 * val1, 0) + sf.besseli(1, 2 * val1, 1) * C_func(s),
            tmp1 * sf.besselk(1, 2 * val1, 0) + sf.besselk(1, 2 * val1, 1) * C_func(s),
            Pa / z.derivative(0),
            Pb / z.derivative(1),
        )
        val = sqrt_complex(C_func(s)) * sqrt_complex(z(x))
        return C1 * sf.besseli(1, 2 * val, 0) * val + C2 * sf.besselk(1, 2 * val, 0) * val
