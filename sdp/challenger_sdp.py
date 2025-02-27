import numpy as np
from numba import njit


def setup(Q, T):
    sliding_dot_product(Q, T)
    return


@njit(fastmath=True)
def sliding_dot_product(Q, T):
    m = Q.shape[0]
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])

    return out
