import numpy as np
from scipy.fft import dct


def _sliding_dot_product(Q, T):
    m = Q.shape[0]
    n = T.shape[0]

    # Padding Q and T
    p1 = (n - m + 1) // 2
    p2 = (m + 1) // 2

    # Note: T and Q will be padded with zeros as follows:
    # T_padded: 0_p1, 0_p2, T
    # Q_padded: 0_p1, Q, 0_p2, 0_{n - m}

    N = p1 + p2 + n
    Q_padded = np.zeros(N, dtype=np.float64)
    T_padded = np.zeros(N, dtype=np.float64)

    Q_padded[p1 : p1 + m] = Q
    T_padded[p1 + p2 :] = T

    QT_dct = np.empty(N + 1, dtype=np.float64)
    QT_dct[N] = 0
    # QT_dct[:N] will be filled with DCT compuation of Q & T

    QT_dct[:N] = dct(Q_padded, type=2, norm="ortho")
    np.multiply(QT_dct[:N], dct(T_padded, type=2, norm="ortho"), out=QT_dct[:N])
    QT_dct[0] *= np.sqrt(2)

    QT_dct[:] = dct(QT_dct, type=1, norm="ortho")
    QT_dct[0] *= 2

    return np.sqrt(2 * N) * QT_dct[p2 : p2 + (n - m + 1)]


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
