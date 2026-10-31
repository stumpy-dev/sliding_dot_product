import numpy as np
from scipy.fft import next_fast_len

# _duccfft replaced _pocketfft in scipy 1.18
try:
    from scipy.fft._duccfft.basic import c2r, r2c
except ModuleNotFoundError:  # pragma: no cover
    from scipy.fft._pocketfft.basic import c2r, r2c


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    n = len(T)
    m = len(Q)
    next_fast_n = next_fast_len(n, real=True)

    tmp = np.empty((2, next_fast_n))
    tmp[0, :m] = Q[::-1]
    tmp[0, m:] = 0.0
    tmp[1, :n] = T
    tmp[1, n:] = 0.0
    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[0], fft_2d[1]), n=next_fast_n)[m - 1 : n]
