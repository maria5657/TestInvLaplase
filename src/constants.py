from scipy import *
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sqrt_complex import sqrt_complex


def darcy2si(val: float) -> float:
    return 1e-12 * val


def si2darcy(val: float) -> float:
    return 1e12 * val


@dataclass
class Constants:
    #val_a_top: float = -1
    val_x_fr: float = 100  # м
    val_k_f: float = darcy2si(10)
    val_k_r: float = darcy2si(1e-3)
    val_a_scl: float = 1
    val_n: float = 1
    val_w_0: float = 1


def calc_A(s: complex, vals: Constants = Constants()):
    tmp2 = vals.val_k_r * vals.val_x_fr / vals.val_k_f
    return tmp2 * sqrt_complex(s)


def calc_C(s: complex, vals: Constants = Constants()):
    return calc_A(s, vals) / (vals.val_x_fr ** 2)


def calc_w_f(x_D: float, vals: Constants):
    # return vals.val_w_0 * (1 - x_D) ** (1 / 2)
    return vals.val_w_0 * ((1 - x_D) + 0.01)


def calc_F_D(x_D: float, vals: Constants):
    tmp1 = vals.val_k_f * calc_w_f(x_D, vals)
    tmp2 = vals.val_k_r * vals.val_x_fr
    return tmp1 / tmp2


def calc_b(x_D: float, vals: Constants):
    return - np.pi / calc_F_D(x_D, vals)


def calc_ksi(C: float):
    return np.sqrt(abs(4 * C + 1)) / 2


def calc_z(x_D: float, vals: Constants = Constants()):
    return 0.5 * calc_w_f(x_D, vals)

def calc_dz_dx_D(vals: Constants):
    return - 0.5 * vals.val_w_0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    b = np.array([
        [0, 0.0],
        [1, -0.0],
        [2, 5.945920613703147e-07],
        [3, -2.1613365749556584e-10],
        [4, 8.249260120374002e-14],
        [5, -2.6987384957776748e-17],
        [6, 7.707770053897742e-21],
        [7, -1.951230726177325e-24],
        [8, 4.432940885706898e-28],
        [9, -9.131095877165193e-32],
        [10, 1.719920487495685e-35],
        [11, -2.9838589131532786e-39],
        [12, 4.797401572160501e-43],
        [13, -7.1861990888683225e-47],
    [14, 1.0075538465505852e-50],
    ])

    a = np.array([[ 0 ,  0.0 ],
[ 1 ,  -0.0 ],
[ 2 ,  5.945920613703147e-07 ],
[ 3 ,  -6.484009724866975e-10 ],
[ 4 ,  4.419246493057501e-13 ],
[ 5 ,  -2.2489487464813964e-16 ],
[ 6 ,  9.19677108703708e-20 ],
[ 7 ,  -3.151988096132603e-23 ],
[ 8 ,  9.309175859984489e-27 ],
[ 9 ,  -2.4170547910143167e-30 ],
[ 10 ,  5.60105685072608e-34 ],
[ 11 ,  -1.1722302873102168e-37 ],
[ 12 ,  2.2370492113661474e-41 ],
[ 13 ,  -3.923664702522105e-45 ],
[ 14 ,  6.3671805580627264e-49 ],])

    x = np.linspace(0, 1, num=100)
    plt.scatter(a[:, 0], a[:, 1], label='$U_1$')
    plt.scatter(b[:, 0], b[:, 1], label='$V_2$')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    # plt.xlabel('$x_{f}$', fontsize=10)
    # plt.ylabel('$w_{f}$', fontsize=10)
    plt.show()
