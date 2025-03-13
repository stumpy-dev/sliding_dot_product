import numpy as np
from scipy.fft import next_fast_len


def setup(Q, T):  # pragma: no cover
    return


def sliding_dot_product(Q, T, order="F"):
    n = len(T)
    m = len(Q)
    shape = next_fast_len(n)

    tmp = np.empty((2, shape), order=order)
    tmp[0, :m] = Q[::-1]
    tmp[0, m:] = 0.0
    tmp[1, :n] = T
    tmp[1, n:] = 0.0
    fft_2d = np.fft.rfft(tmp, axis=-1)

    return np.fft.irfft(np.multiply(fft_2d[0], fft_2d[1]), n=shape)[m - 1 : n]
