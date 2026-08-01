import numpy as np
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import r2c, c2r


def _pocketfft_valid_convolve(Q, T):
    """
    Compute the valid convolution between ``Q`` and ``T``
    using circular convolution in the frequency domain
    """
    n = len(T)
    m = len(Q)
    next_fast_n = next_fast_len(n, real=True)

    tmp = np.empty((2, next_fast_n))
    tmp[0, :m] = Q
    tmp[0, m:] = 0.0
    tmp[1, :n] = T
    tmp[1, n:] = 0.0
    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[0], fft_2d[1]), n=next_fast_n)[
        len(Q) - 1 : len(T)
    ]


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    return _pocketfft_valid_convolve(Q[::-1], T)
