import math

import numpy as np

from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy._lib._array_api import array_namespace
from scipy.signal._signaltools import _apply_conv_mode, _split
from scipy.fft._pocketfft.basic import r2c, c2r, r2cn, c2rn


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


def _rfft_irfft_r2c2rn(Q_2D, T_2D, shape):

    return c2rn(
        False,
        r2cn(True, T_2D, shape, axes=1) * r2cn(True, Q_2D, shape, axes=1),
        s=shape,
        axes=1,
    )


def _calc_oa_lens(n, m):
    fallback = (n + m - 1, None, n, m)
    # assuming n > m
    # otherwise, need to add the following:
    # if n == m or m >= n:
    #    return fallback

    overlap = m - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block_size = next_fast_len(math.ceil(opt_size), real=True)

    if block_size >= n:
        return fallback

    T_step = block_size - m + 1
    Q_step = m

    return (block_size, overlap, T_step, Q_step)


def _sliding_dot_product(T, Q):
    xp = array_namespace(T, Q)

    n = T.shape[0]
    m = Q.shape[0]

    shape_final = n + m - 1
    block_size, overlaps, T_step, Q_step = _calc_oa_lens(n, m)
    if T_step == n and Q_step == m:
        return _sliding_dot_product_r2c2r(Q, T)

    Q = Q[::-1]
    curnstep1 = math.ceil((n + 1) / T_step)
    if (block_size - overlaps) * curnstep1 < shape_final:
        curnstep1 += 1

    curpad1 = curnstep1 * T_step - n
    if curpad1 != 0:
        T = np.pad(T, [(0, curpad1)], mode="constant", constant_values=0)

    T = xp.reshape(T, (curnstep1, T_step))
    Q = xp.reshape(Q, (1, Q_step))
    ret = _rfft_irfft_r2c2rn(T, Q, block_size)

    # overlap-add
    overlap = overlaps
    if overlap is not None:
        ret, overpart = _split(ret, [-overlap], 1, xp=xp)
        overpart = _split(overpart, [-1], 0, xp=xp)[0]
        ret_overpart = _split(ret, [overlap], 1, xp=xp)[0]
        ret_overpart = _split(ret_overpart, [1], 0, xp)[1]
        ret_overpart += overpart

    # Reshape back to the correct dimensionality.
    shape_ret = [
        ret.shape[i] if i not in [1] else ret.shape[i] * ret.shape[i - 1]
        for i in range(ret.ndim)
        if i not in [0]
    ]
    ret = xp.reshape(ret, shape_ret)

    # Slice to the correct size.
    slice_final = tuple([slice(islice) for islice in [shape_final]])
    ret = ret[slice_final]

    return _apply_conv_mode(ret, (n,), (m,), "valid", [0], xp)


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    if len(Q) == len(T):
        return np.dot(Q, T)
    return _sliding_dot_product(T, Q)
