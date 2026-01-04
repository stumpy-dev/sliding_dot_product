import numpy as np
from scipy.fft import dct


def _sliding_dot_product(Q, T):
    # MASS_V4 in https://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    m = Q.shape[0]
    n = T.shape[0]
    p1 = (n - m + 1) // 2
    p2 = (m + 1) // 2
    N = p1 + p2 + n  # The length of Q_padded and T_padded

    # Pad Q and T
    Q_padded = np.empty(N, dtype=np.float64)
    Q_padded[:p1] = 0
    Q_padded[p1 : p1 + m] = Q
    Q_padded[p1 + m :] = 0

    T_padded = np.empty(N, dtype=np.float64)
    T_padded[: p1 + p2] = 0
    T_padded[p1 + p2 :] = T

    # Use DCT to compute the sliding dot product
    QT_dct = np.empty(N + 1, dtype=np.float64)
    QT_dct[:N] = dct(Q_padded, type=2, norm="ortho")
    np.multiply(QT_dct[:N], dct(T_padded, type=2, norm="ortho"), out=QT_dct[:N])
    QT_dct[0] *= np.sqrt(2)
    QT_dct[N] = 0

    QT_dct[:] = dct(QT_dct, type=1, norm="ortho")

    return np.sqrt(2 * N) * QT_dct[p2 : p2 + (n - m + 1)]


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
