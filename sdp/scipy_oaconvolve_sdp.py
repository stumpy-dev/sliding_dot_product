from scipy.signal import oaconvolve
import numpy as np


def sliding_dot_product(Q, T):
    return oaconvolve(np.ascontiguousarray(Q[::-1]), T, mode="valid")
