import numpy as np
from numba import njit


def setup(Q, T):
    sliding_dot_product(Q, T)
    return


@njit(fastmath=True)
def sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)

    n_unroll = 4
    for i in range(0, l - l % n_unroll, n_unroll):
        result_0 = 0.0
        result_1 = 0.0
        result_2 = 0.0
        result_3 = 0.0
        for j in range(m):
            result_0 += Q[j] * T[i + j]
            result_1 += Q[j] * T[i + 1 + j]
            result_2 += Q[j] * T[i + 2 + j]
            result_3 += Q[j] * T[i + 3 + j]
        out[i] = result_0
        out[i + 1] = result_1
        out[i + 2] = result_2
        out[i + 3] = result_3

    # Handle any remaining iterations if `l` is not a multiple of `n_unroll`
    for i in range(l - l % n_unroll, l):
        result = 0.0
        for j in range(m):
            result += Q[j] * T[i + j]
        out[i] = result

    return out
