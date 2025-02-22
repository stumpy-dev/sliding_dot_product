#!/usr/bin/env python

import numpy as np
import numpy.testing as npt
from sdp import naive_sdp, njit_sdp, numpy_fft_sdp, scipy_oaconvolve_sdp, pyfftw_sdp, challenger_sdp
import time
import warnings

if __name__ == "__main__":
    modules = [
        pyfftw_sdp,
        njit_sdp,
        numpy_fft_sdp,
        scipy_oaconvolve_sdp,
        challenger_sdp,
    ]

    n_iter = 4
    p_min = 6
    p_max = 28
    start_timing = time.time()
    for mod in modules:
        for i in range(p_min, p_max):
            Q = np.random.rand(2**i)
            break_Q = False
            for j in range(i, p_max):
                T = np.random.rand(2**j)
                break_T = False

                elapsed_times = []
                for _ in range(n_iter):
                    start = time.time()
                    mod.sliding_dot_product(Q, T)
                    diff = time.time() - start
                    if diff > 10.0:
                        break_T = True
                        warnings.warn(
                            f"SKIPPED: {mod.__name__},{len(Q)},{len(T)},{diff})"
                        )
                        break
                    else:
                        elapsed_times.append(diff)

                if break_T:
                    if j == 2 * i:
                        break_Q = True
                    break

                elapsed_times.remove(
                    min(elapsed_times)
                )  # Remove smallest number from the list

                print(
                    f"{mod.__name__},{len(Q)},{len(T)},{len(elapsed_times)},{sum(elapsed_times) / len(elapsed_times)}",
                    flush=True,
                )

            if break_Q:
                warnings.warn(f"SKIPPED: {mod.__name__},{len(Q)},>{len(T)},{diff})")
                break

    elapsed_timing = np.round((time.time() - start_timing) / 60.0, 2)
    warnings.warn(f"Test completed in {elapsed_timing} min")
