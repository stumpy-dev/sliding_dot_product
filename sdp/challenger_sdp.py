import numpy as np

import math

import scipy

from scipy.special import lambertw
from scipy.fft import next_fast_len


def _pre_compute(n_Q, n_T):
    overlap = n_Q - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block_size = next_fast_len(math.ceil(opt_size))
    
    T_step = block_size - overlap
    Q_step = n_Q
    if (n_T == 2 * n_Q) or (block_size >= n_T) or (T_step == n_T):
        is_fallback = True
        out = [
            overlap,
            opt_size, 
            block_size, 
            T_step, 
            Q_step, 
            is_fallback,
            None,
            None,
            None,
            None,
        ]

        return out
    
    is_fallback = False

    shape_final = n_T + n_Q - 1
    if n_T > T_step: 
        T_curnstep = math.ceil((n_T + 1) / T_step) 
        if (block_size - overlap) * T_curnstep < shape_final:
            T_curnstep += 1
        T_curpad = T_curnstep * T_step - n_T
    else:
        T_curnstep = 1
        T_curpad = 0

    if n_Q > Q_step:
        Q_curnstep = math.ceil((n_Q+1)/Q_step)
        if (block_size - overlap) * Q_curnstep < shape_final:
            Q_curnstep += 1
        Q_curpad = Q_curnstep * Q_step - n_Q
    else:
        Q_curnstep = 1
        Q_curpad = 0
    
    out = [
        overlap,
        opt_size, 
        block_size, 
        T_step, 
        Q_step, 
        is_fallback,
        T_curpad,
        Q_curpad,
        T_curnstep,
        Q_curnstep,
    ]

    return out


def oaconvolve_1D(
    Q, 
    T, 
    overlap,
    opt_size,
    block_size,
    T_step,
    Q_step,
    T_curpad,
    Q_curpad,
    T_curnstep,
    Q_curnstep,
):
    n_Q = len(Q)
    n_T = len(T)
    
    nrow = n_T // T_step
    residue = n_T % T_step

    arr = np.empty((T_curnstep + 1, block_size), dtype=np.float64)  
    arr[:nrow, :T_step] = T[:nrow * T_step].reshape(nrow, T_step)
    arr[:nrow, T_step:] = 0.0
    arr[nrow, :residue] = T[-residue:]
    arr[nrow, residue:] = 0.0
    arr[-1, :n_Q] = Q
    arr[-1, n_Q:] = 0.0

    F = np.fft.rfft(arr, block_size, axis=1)
    ret = np.fft.irfft(
        np.multiply(F[:-1], F[-1].reshape(1,-1)), 
        block_size, 
        axis=1,
    )
   
    ret_split = ret[:, :-overlap]
    overpart = ret[:-1, -overlap:]
    ret_overpart = ret_split[1:, :overlap]
    ret_overpart += overpart

    return ret_split.reshape(-1,)[n_Q - 1:n_T]


def setup(Q, T):
    return sliding_dot_product(Q, T)


def sliding_dot_product(Q, T):
    n_T = len(T)
    n_Q = len(Q)

    if n_T == n_Q:
        return np.dot(Q, T)

    QT_INFO = _pre_compute(n_Q, n_T)
    (
        overlap,
        opt_size, 
        block_size, 
        T_step, 
        Q_step, 
        is_fallback,
        T_curpad,
        Q_curpad,
        T_curnstep,
        Q_curnstep,
    ) = QT_INFO

    if is_fallback:
        n = len(T)
        m = len(Q)
        shape = next_fast_len(n)

        tmp = np.empty((2, shape), order='F')
        tmp[0, :m] = Q[::-1]
        tmp[0, m:] = 0.0
        tmp[1, :] = T
        fft_2d = np.fft.rfft(tmp, axis=-1)

        return np.fft.irfft(np.multiply(fft_2d[0], fft_2d[1]))[m - 1 : n]
    else:
        return oaconvolve_1D(
            Q[::-1],
            T, 
            overlap,
            opt_size,
            block_size,
            T_step,
            Q_step,
            T_curpad,
            Q_curpad,
            T_curnstep,
            Q_curnstep,
        )