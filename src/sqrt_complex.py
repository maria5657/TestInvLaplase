import numpy as np
import cmath


def sqrt_complex(s):
    s = complex(s)
    modulus = abs(s)
    phase = cmath.phase(s)
    return np.sqrt(modulus) * complex(np.cos(phase / 2), np.sin(phase / 2))


if __name__ == '__main__':
    assert(np.isclose(sqrt_complex(25), 5))