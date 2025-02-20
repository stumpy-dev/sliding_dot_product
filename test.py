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

    n_iter = 4
    p_min = 6
    p_max = 28
    for mod in modules:
        for i in range(p_min, p_max):
            Q = np.random.rand(2**i)
            for j in range(i, p_max):
                T = np.random.rand(2**j)
                break_flag = False

                elapsed_times = []
                for _ in range(n_iter):
                    start = time.time()
                    mod.sliding_dot_product(Q, T)
                    diff = time.time() - start
                    if diff > 10.0:
                        break_flag = True
                        break
                    else:
                        elapsed_times.append(diff)

                if break_flag:
                    break

                elapsed_times.remove(min(elapsed_times))  # Remove smallest number from the list

                print(f"{mod.__name__},{len(Q)},{len(T)},{n_iter},{sum(elapsed_times) / len(elapsed_times)}", flush=True)
