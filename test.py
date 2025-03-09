import numpy as np
import utils
import warnings

from numpy import testing as npt
from scipy.fft import next_fast_len


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


def test_sdp_case0():
    # This tests cases where `T` is complete power of 2.
    pmin = 3
    pmax = 13

    modules = utils.import_sdp_mods()
    for mod in modules:
        try:
            for q in range(pmin, pmax + 1):
                n_Q = 2**q
                for p in range(q, pmax + 1):
                    n_T = 2**p
                    Q = np.random.rand(n_Q)
                    T = np.random.rand(n_T)

                    ref = naive_sliding_dot_product(Q, T)
                    comp = mod.sliding_dot_product(Q, T)
                    npt.assert_allclose(comp, ref)

        except Exception as e:
            msg = f"Error in {mod.__name__}, with q={q} and p={p}"
            print(msg)
            raise e

    return


def test_sdp_case1():
    # This tests cases where the length of `T` is even
    # and its next_fast_len is the same as the len(T). 
    # To this end, we choose 2, 3 and 5

    n_T = 2 * (3**2) * (5**3)
    shape = next_fast_len(n_T)
    if shape != n_T:
        warnings.warn(f"next_fast_len({n_T}) = {shape}")

    modules = utils.import_sdp_mods()
    for mod in modules:
        for n_Q in range(2, n_T + 1):
            try:
                Q = np.random.rand(n_Q)
                T = np.random.rand(n_T)

                ref = naive_sliding_dot_product(Q, T)
                comp = mod.sliding_dot_product(Q, T)
                npt.assert_allclose(comp, ref)

            except Exception as e:
                msg = f"Error in {mod.__name__}, with n_Q={n_Q} and n_T={n_T}"
                print(msg)
                raise e

    return


def test_sdp_case2():
    # This tests cases where the length of `T` is odd
    # and its next_fast_len is the same as the len(T). 
    # To this end, we choose 3 and 5

    n_T = (3**2) * (5**3) 
    shape = next_fast_len(n_T)
    if shape != n_T:
        warnings.warn(f"next_fast_len({n_T}) != {shape}")

    modules = utils.import_sdp_mods()
    for mod in modules:
        for n_Q in range(2, n_T + 1):
            try:
                Q = np.random.rand(n_Q)
                T = np.random.rand(n_T)

                ref = naive_sliding_dot_product(Q, T)
                comp = mod.sliding_dot_product(Q, T)
                npt.assert_allclose(comp, ref)

            except Exception as e:
                msg = f"Error in {mod.__name__}, with n_Q={n_Q} and n_T={n_T}"
                print(msg)
                raise e

    return


def test_sdp_case3():
    # This tests cases where the length of `T` is even
    # and its next_fast_len is not the same as
    # the len(T) but longer. To this end, we choose
    # factors 2, 7, 11, 13.

    n_T = 2 * 7 * 11 * 13
    shape = next_fast_len(n_T)
    if shape == n_T:
        warnings.warn(f"next_fast_len({n_T}) == {shape}")

    modules = utils.import_sdp_mods()
    for mod in modules:
        for n_Q in range(2, n_T + 1):
            try:
                Q = np.random.rand(n_Q)
                T = np.random.rand(n_T)

                ref = naive_sliding_dot_product(Q, T)
                comp = mod.sliding_dot_product(Q, T)
                npt.assert_allclose(comp, ref)

            except Exception as e:
                msg = f"Error in {mod.__name__}, with n_Q={n_Q} and n_T={n_T}"
                print(msg)
                raise e

    return


def test_sdp_case4():
    # This tests cases where the length of `T` is odd
    # and its next_fast_len is not the same as
    # the len(T) but longer. To this end, we choose
    # factors 7, 11, 13.

    n_T = 7 * 11 * 13
    shape = next_fast_len(n_T)
    if shape == n_T:
        warnings.warn(f"next_fast_len({n_T}) == {shape}")

    modules = utils.import_sdp_mods()
    for mod in modules:
        for n_Q in range(2, n_T + 1):
            try:
                Q = np.random.rand(n_Q)
                T = np.random.rand(n_T)

                ref = naive_sliding_dot_product(Q, T)
                comp = mod.sliding_dot_product(Q, T)
                npt.assert_allclose(comp, ref)

            except Exception as e:
                msg = f"Error in {mod.__name__}, with n_Q={n_Q} and n_T={n_T}"
                print(msg)
                raise e

    return
