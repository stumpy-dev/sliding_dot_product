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

    Q_dct = dct(Q_padded, type=2, norm="ortho")
    T_dct = dct(T_padded, type=2, norm="ortho")

    QT_dct = Q_dct * T_dct
    QT_dct = np.append(QT_dct, 0)
    QT_dct[0] *= np.sqrt(2)

    QT = dct(QT_dct, type=1, norm="ortho")
    QT[0] *= 2

    return np.sqrt(2 * N) * QT[p2 : p2 + (n - m + 1)]


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
