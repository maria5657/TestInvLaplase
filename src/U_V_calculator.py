import numpy as np
from numpy.typing import NDArray
from math import factorial

def calc_U_1(a: NDArray, Z_powers: NDArray):
    N = min(a.size, Z_powers.size)
    rez = 0
    for n in range(N):
        rez += ((-1) ** n) * a[n] * Z_powers[n] / factorial(n)
    return rez

def calc_U_2(a: NDArray, Z_powers: NDArray):
    N = min(a.size, Z_powers.size)
    rez = 0
    for n in range(N):
        rez += a[n] * Z_powers[n] / factorial(n)
    return rez

def calc_U_1_derivative(a: NDArray, Z_powers: NDArray):
    N = min(a.size, Z_powers.size)
    rez = 0
    for n in range(1, N):
        rez += ((-1) ** n) * a[n] * Z_powers[n] / factorial(n)
    return rez

def calc_U_2_derivative(a: NDArray, Z_powers: NDArray):
    N = min(a.size, Z_powers.size)
    rez = 0
    for n in range(1, N):
        rez += a[n] * Z_powers[n - 1] / factorial(n - 1)
    return rez

def calc_V_2(b: NDArray, Z_powers: NDArray):
    N = min(b.size, Z_powers.size)
    rez = 0
    for n in range(N):
        rez += ((-1) ** n) * b[n] * Z_powers[n] / factorial(n)
    return rez

def calc_V_2_derivative(b: NDArray, Z_powers: NDArray):
    N = min(b.size, Z_powers.size)
    rez = 0
    for n in range(1, N):
        rez += ((-1) ** n) * b[n] * Z_powers[n - 1] / factorial(n - 1)
    return rez