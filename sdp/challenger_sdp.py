import numpy as np
from scipy.signal import convolve


def setup(Q, T):
    return


def sliding_dot_product(Q, T):
    return convolve(np.ascontiguousarray(Q[::-1]), T, method="direct", mode="valid")
    
