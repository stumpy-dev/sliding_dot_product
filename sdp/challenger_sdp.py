import math
import numpy as np

from numba import njit

@njit(fastmath=True)
def _rfft_preprocess_TQ(T, Q):
    n = len(T)
    m = len(Q)

    Q0 = np.empty(n, dtype=np.float64)
    Q0[:m] = Q[::-1]
    Q0[m:] = 0.0

    half_n = n // 2
    out = np.empty((2, half_n), dtype=np.complex128)
    
    m_stop = m // 2
    for i in range(m_stop):
        out[0, i] = T[2 * i] + 1j * T[2 * i + 1]
        out[1, i] = Q0[2 * i] + 1j * Q0[2 * i + 1]

    for i in range(m_stop, half_n):
        out[0, i] = T[2 * i] + 1j * T[2 * i + 1]
    out[1, m_stop:] = 0.0
        
    return out


@njit(fastmath=True)
def _rfft_postprocess_TQ(TQ):
    x_T = TQ[0]
    x_Q = TQ[1]

    n_x = len(x_T)
    half_n_x = n_x // 2

    F = np.empty(n_x + 1, dtype=np.complex128)

    # 0th element, half_n_x, and n_x
    F[0] = (x_T[0].real + x_T[0].imag) * (x_Q[0].real + x_Q[0].imag)
    F[half_n_x] = (x_T[half_n_x] * x_Q[half_n_x]).conjugate()
    F[n_x] = (x_T[0].real - x_T[0].imag) * (x_Q[0].real - x_Q[0].imag)

    theta0 = math.pi / n_x
    factor = math.cos(theta0) - 1j * math.sin(theta0)
    w = 0.5j
    for k in range(1, half_n_x):
        w = w * factor

        val_T = (x_T[k] - x_T[n_x - k].conjugate()) * (0.5 + w)
        val_Q = (x_Q[k] - x_Q[n_x - k].conjugate()) * (0.5 + w)

        F[k] = (x_T[k] - val_T) * (x_Q[k] - val_Q)
        F[n_x - k] = (x_T[n_x - k] + val_T.conjugate()) * (x_Q[n_x - k] + val_Q.conjugate())

    return F


def _rfft_TQ(T, Q):
    TQ = _rfft_preprocess_TQ(T, Q)
    np.fft.fft(TQ, axis=1, out=TQ)
    return _rfft_postprocess_TQ(TQ)


def sliding_dot_product(Q, T):
    n = len(T)
    m = len(Q)
    
    F = _rfft_TQ(T, Q)

    return np.fft.irfft(F)[m - 1 : n]


def setup(Q, T):
    return sliding_dot_product(Q, T)


