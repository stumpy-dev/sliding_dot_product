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

if __name__ == "__main__":
    Q = np.random.rand(50)
    T = np.random.rand(1000)

    ref = naive_sdp.sliding_dot_product(Q, T)
    modules = [
        njit_sdp,
        numpy_fft_sdp,
        scipy_oaconvolve_sdp,
        pyfftw_sdp
    ]

    for mod in modules:
        npt.assert_almost_equal(ref, mod.sliding_dot_product(Q, T))
        print(f"PASSED: {mod.__name__}")
