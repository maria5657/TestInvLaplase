from tqdm import tqdm
from numpy.typing import NDArray
from mpmath import *


mp.dps = 10
mp.pretty = True

def get_inv_laplace(t_moments: NDArray, func: callable):
    p_f_d_x_D_t = []
    for t_moment in tqdm(t_moments):
        p_f_d_x_D_t.append(invertlaplace(func, t_moment, method='talbot'))
    return p_f_d_x_D_t