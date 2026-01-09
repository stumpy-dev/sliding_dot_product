import math
import numpy as np
from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import r2c, c2r


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


def _sliding_dot_product_r2c2r(Q, T):
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


def _sliding_dot_product(Q, T):
    m = Q.shape[0]
    n = T.shape[0]

    # compute optimal block size
    overlap = m - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block_size = next_fast_len(math.ceil(opt_size), real=True)
    if block_size >= n:
        return _sliding_dot_product_r2c2r(Q, T)

    # perform fft-ifft operation on blocks
    ret = _rfft_irfft_r2c2r_block(Q, T, block_size)

    # overlap-add process
    out = ret[:, :-overlap]
    out[1:, :overlap] += ret[:-1, -overlap:]
    out = np.reshape(out, (-1,))

    return out[m - 1 : n]


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    if len(Q) == len(T):
        return np.dot(Q, T)
    return _sliding_dot_product(Q, T)
