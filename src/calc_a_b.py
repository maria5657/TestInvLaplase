import numpy as np


def calc_a_b(q: float, N: int = 50):
    assert(q > 0)
    if np.isclose(1 / q % 2, 1):
        n = int((1 / q - 1) / 2)
        return calc_odd(q=q, n=n, N=N)
    elif np.isclose(1 / q % 2, 0):
        n = int((1 / q) / 2)
        return calc_even(q=q, n=n, N=N)


def calc_odd(q: float, n: int, N: int):
    a, b = np.zeros(N), np.zeros(N)
    for nu in range(N):
        if nu <= n:
            tmp_a = 1
            tmp_b = 1
            for k in range(1, nu + 1):
                tmp_a *= ((2 * k - 1) * q - 1) / (k * q - 1)
                tmp_b *= ((2 * k - 1) * q + 1) / (k * q + 1)
            a[nu], b[nu] = tmp_a, tmp_b
        else:
            a[nu], b[nu] = 0, 0
    return a, b


def calc_even(q: float, n: int, N: int):
    a, b = np.zeros(N), np.zeros(N)
    for nu in range(N):
        if nu < 2 * n:
            a[nu], b[nu] = 0, 0
        else:
            tmp_a = 1
            tmp_b = 1
            for k in range(2 * n + 1, nu + 1):
                tmp_a *= ((2 * k - 1) * q - 1) / (k * q - 1)
                tmp_b *= ((2 * k - 1) * q + 1) / (k * q + 1)
            a[nu], b[nu] = tmp_a, tmp_b
    return a, b

if __name__ == '__main__':
    a, b = calc_a_b(q=1/2)
    print(5)