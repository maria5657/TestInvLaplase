import numpy as np

from sqrt_complex import sqrt_complex


class Z_class:

    def __init__(self, z_func: callable, C: float, q: float):
        self.z_func = z_func
        self.C = C
        self.q = q

    def __call__(self, x: float):
        if np.isclose(self.q, 1):
            return (1 / self.q) * self.z_func(x) * sqrt_complex(self.C)
        elif np.isclose(self.q, 1 / 2):
            return (1 / self.q) * sqrt_complex(self.z_func(x)) * sqrt_complex(self.C)

    def derivative(self, x: float):
        if np.isclose(self.q, 1):
            return sqrt_complex(self.C)
        elif np.isclose(self.q, 1 / 2):
            return (1 / sqrt_complex(self.z_func(x))) * sqrt_complex(self.C)

    def calc_powers(self, x: float, N: int = 50):
        Z_arr = [0 for n in range(N)]
        Z_arr[0] = 1
        Z_val = self.__call__(x)
        for i in range(1, N):
            Z_arr[i] = Z_val * Z_arr[i - 1]
        return np.array(Z_arr)




class z_class:

    def __init__(self):
        pass

    def __call__(self, x: float):
        return 2 - x + 0.2

    def derivative(self, x: float):
        return -1