import math

import numpy as np
import scipy._lib.array_api_extra as xpx

from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy._lib._array_api import array_namespace
from scipy import fft as sp_fft
from scipy.signal._signaltools import _apply_conv_mode, _split  # , fftconvolve
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


##########
def _rfft_irfft_r2c2rn(Q_2D, T_2D, shape):

    return c2rn(
        False,
        r2cn(True, T_2D, shape, axis=1) * r2cn(True, Q_2D, shape, axis=1),
        n=shape,
        axis=1,
    )


##########


def _freq_domain_conv(T_2D, Q_2D, shape):
    sp1 = sp_fft.rfftn(T_2D, shape, axes=1)
    sp2 = sp_fft.rfftn(Q_2D, shape, axes=1)

    return sp_fft.irfftn(sp1 * sp2, shape, axes=1)


def _calc_oa_lens(n, m):
    fallback = (n + m - 1, None, n, m)
    if n == m or m >= n:
        return fallback

    overlap = m - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block_size = next_fast_len(math.ceil(opt_size), real=True)  # real=True?

    # Use conventional FFT convolve if there is only going to be one block.
    if block_size >= n:
        return fallback

    T_step = block_size - m + 1
    Q_step = m

    return (block_size, overlap, T_step, Q_step)


def _sliding_dot_product(T, Q):
    xp = array_namespace(T, Q)

    n = T.shape[0]
    m = Q.shape[0]

    # in1 = T
    # in2 = Q
    # axes = [0]
    s1 = (n,)
    s2 = (m,)

    shape_final = n + m - 1
    shape_final_lst = [shape_final]
    block_size, overlaps, T_step, Q_step = _calc_oa_lens(n, m)

    # block_size_tuple = (block_size,)
    # overlaps_tuple = (overlaps,)
    # T_step_tuple = (T_step,)
    # Q_step_tuple = (Q_step,)

    if T_step == n and Q_step == m:
        return _sliding_dot_product_r2c2r(Q, T)

    # flip Q:
    Q = Q[::-1]

    curnstep1 = math.ceil((n + 1) / T_step)
    if (block_size - overlaps) * curnstep1 < shape_final:
        curnstep1 += 1

    curpad1 = curnstep1 * T_step - n

    # nsteps1 = [curnstep1]
    # nsteps2 = [1]
    pad_size1 = [(0, curpad1)]
    # pad_size2 = [(0, 0)]

    if not all(curpad == (0, 0) for curpad in pad_size1):
        T = xpx.pad(T, pad_size1, mode="constant", constant_values=0, xp=xp)

    # Reshape the overlap-add parts to input block sizes.
    split_axes = [0]
    fft_axes = [1]

    # reshape_size1 = [curnstep1, T_step]  # reshape_size1.insert(0, curnstep1)
    # reshape_size2 = [1, Q_step]  # reshape_size2.insert(0, 1)

    T = xp.reshape(T, (curnstep1, T_step))
    Q = xp.reshape(Q, (1, Q_step))

    # fft_shape = [block_size]
    ret = _freq_domain_conv(T, Q, block_size)

    # Do the overlap-add.
    # ax = 0
    ax_fft = 1
    ax_split = 0

    overlap = overlaps
    if overlap is not None:
        ret, overpart = _split(ret, [-overlap], 1, xp=xp)
        overpart = _split(overpart, [-1], ax_split, xp=xp)[0]
        ret_overpart = _split(ret, [overlap], ax_fft, xp=xp)[0]
        ret_overpart = _split(ret_overpart, [1], ax_split, xp)[1]
        ret_overpart += overpart

    # Reshape back to the correct dimensionality.
    shape_ret = [
        ret.shape[i] if i not in fft_axes else ret.shape[i] * ret.shape[i - 1]
        for i in range(ret.ndim)
        if i not in split_axes
    ]
    ret = xp.reshape(ret, shape_ret)

    # Slice to the correct size.
    slice_final = tuple([slice(islice) for islice in shape_final_lst])
    ret = ret[slice_final]

    return _apply_conv_mode(ret, s1, s2, "valid", [0], xp)


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    if len(Q) == len(T):
        return np.dot(Q, T)
    return _sliding_dot_product(T, Q)
