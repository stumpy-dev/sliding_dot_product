import numpy as np
from scipy.fft import next_fast_len
from numba import njit


def setup(Q, T):
    return sliding_dot_product(Q, T)


@njit
def sliding_dot_product(Q, T):
    n = len(T)
    m = len(Q)
    shape = next_fast_len(n)

    tmp = np.empty((2, shape))
    tmp[0, :m] = Q[::-1]
    tmp[0, m:] = 0.0
    tmp[1, :] = T
    fft_2d = np.fft.rfft(tmp, axis=-1)
    
    return np.fft.irfft(np.multiply(fft_2d[0], fft_2d[1]))[m - 1:n]
