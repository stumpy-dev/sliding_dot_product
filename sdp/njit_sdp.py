import numpy as np
from numba import njit
import sdp


@njit(fastmath=True)
def _sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        result = 0.0
        for j in range(m):
            result += Q[j] * T[i + j]
        out[i] = result

    return out


def setup(Q, T):
    _sliding_dot_product(np.random.rand(50), np.random.rand(100))
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
