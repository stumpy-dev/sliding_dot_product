#!/usr/bin/env python

import numpy as np
import numpy.testing as npt
from sdp import (
    naive_sdp,
    njit_sdp,
    numpy_fft_sdp,
    scipy_oaconvolve_sdp,
    pyfftw_sdp
)
import time

if __name__ == "__main__":
    modules = [
        pyfftw_sdp,
        njit_sdp,
        numpy_fft_sdp,
        scipy_oaconvolve_sdp,
    ]

    n_iter = 3
    p_min = 6
    p_max = 27
    for mod in modules:
        for i in range(p_min, p_max):
            Q = np.random.rand(2**i)
            for j in range(i, p_max):
                T = np.random.rand(2**j)

                start = time.time()
                for _ in range(n_iter):
                    mod.sliding_dot_product(Q, T)
                elapsed_time = time.time() - start

                print(f"{mod.__name__},{len(Q)},{len(T)},{n_iter},{elapsed_time / n_iter}")
