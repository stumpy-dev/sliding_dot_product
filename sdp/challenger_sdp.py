import math
import numpy as np
from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import r2c, c2r
from . import pocketfft_r2c_c2r_sdp


def _compute_block_size(m, n, block_size=None):
    """
    Return a block size for the overlap-add method.
    """
    if block_size is None:
        # Find optimal block_size based on m and n
        if m >= n / 2:
            block_size = n  # i.e. no blocking
        else:
            overlap = m - 1
            opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
            block_size = next_fast_len(math.ceil(opt_size), real=True)

    block_size = max(block_size, m)

    return min(block_size, n)


def _rfft_irfft_r2c2r_block(Q, T, block_size):
    m = Q.shape[0]
    n = T.shape[0]
    T_step = block_size - (m - 1)
    n_step = math.ceil(n / T_step)
    last_chunk_start = (n_step - 1) * T_step

    tmp = np.empty((n_step + 1, block_size), dtype=np.float64)

    # fill with T, block-wise
    tmp[: n_step - 1, :T_step] = T[:last_chunk_start].reshape(n_step - 1, T_step)
    tmp[: n_step - 1, T_step:] = 0.0
    tmp[n_step - 1, : n - last_chunk_start] = T[last_chunk_start:]
    tmp[n_step - 1, n - last_chunk_start :] = 0.0

    # fill with Q[::-1]
    tmp[n_step, :m] = Q[::-1]
    tmp[n_step, m:] = 0.0

    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[:-1], fft_2d[[-1]]), n=block_size)


def _sliding_dot_product(Q, T, block_size):
    m = Q.shape[0]
    n = T.shape[0]

    overlap = m - 1
    ret = _rfft_irfft_r2c2r_block(Q, T, block_size)
    out = ret[:, :-overlap]
    out[1:, :overlap] += ret[:-1, -overlap:]
    out = np.reshape(out, (-1,))

    return out[m - 1 : n]


def setup(Q, T):
    return


def sliding_dot_product(Q, T, block_size=None):
    m = Q.shape[0]
    n = T.shape[0]
    if m == n:
        return np.dot(Q, T)

    block_size = _compute_block_size(m, n, block_size=block_size)
    if block_size >= n:
        return pocketfft_r2c_c2r_sdp.sliding_dot_product(Q, T)
    else:
        return _sliding_dot_product(Q, T, block_size)
