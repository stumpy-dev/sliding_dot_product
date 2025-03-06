import numpy as np
from scipy.fft import next_fast_len


def setup(Q, T):
    return


def sliding_dot_product(Q, T, order="F"):
    n = len(T)
    m = len(Q)
    shape = next_fast_len(n, real=True)

    tmp = np.empty((2, shape), order=order)
    tmp[0, :m] = Q[::-1]
    tmp[0, m:] = 0.0
    tmp[1, :] = T
    fft_2d = np.fft.rfft(tmp, axis=-1)

    return np.fft.irfft(np.multiply(fft_2d[0], fft_2d[1]))[m - 1 : n]
