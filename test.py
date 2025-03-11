import numpy as np
import pytest
import utils

from numpy import testing as npt
from operator import eq, lt
from scipy.fft import next_fast_len

# README
# Real FFT algorithm performs more efficiently when the length
# of the input array `arr` is composed of small prime factors.
# The next_fast_len(arr, real=True) function from Scipy returns
# the same length if len(arr) is composed of a subset of
# prime numbers 2, 3, 5. Therefore, these radices are
# considered as the most efficient for the real FFT algorithm.

# To ensure that the tests cover different cases, the following cases
# are considered:
# 1. len(T) is even, and len(T) == next_fast_len(len(T))
# 2. len(T) is odd, and len(T) == next_fast_len(len(T))
# 3. len(T) is even, and len(T) < next_fast_len(len(T))
# 4. len(T) is odd, and len(T) < next_fast_len(len(T))
# And 5. a special case of 1, where len(T) is power of 2.

# Therefore:
# 1. len(T) is composed of 2 and a subset of {3, 5}
# 2. len(T) is composed of a subset of {3, 5}
# 3. len(T) is composed of a subset of {7, 11, 13, ...} and 2
# 4. len(T) is composed of a subset of {7, 11, 13, ...}
# 5. len(T) is power of 2

# In some cases, the prime factors are powered to a certain degree
# to increase the length of array to be around 1000-2000. This
# can just let us test sliding_dot_product for wider range of
# query lengths.

test_inputs = [
    (
        2 * (3**2) * (5**3),
        0,
        eq,
    ),  # = 2250, Even `len(T)`, and `len(T) == next_fast_len(len(T))`
    (
        (3**2) * (5**3),
        1,
        eq,
    ),  # = 1125, Odd `len(T)`, and `len(T) == next_fast_len(len(T))`.
    (
        2 * 7 * 11 * 13,
        0,
        lt,
    ),  # = 2002, Even `len(T)`, and `len(T) < next_fast_len(len(T))`
    (7 * 11 * 13, 1, lt),  # = 1001, Odd `len(T)`, and `len(T) < next_fast_len(len(T))`
]


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_remainder(n_T, remainder):
    assert n_T % 2 == remainder


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_comparator(n_T, comparator):
    shape = next_fast_len(n_T)
    assert comparator(n_T, shape)


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_sdp():
    # test_sdp for cases 1-4
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


def test_sdp_power2():
    # test for case 5. len(T) is power of 2
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
