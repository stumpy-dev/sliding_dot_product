import importlib
import numpy as np
import pkgutil
import pytest
import sdp
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
# 1. len(T) is even, and len(T) == next_fast_len(len(T), real=True)
# 2. len(T) is odd, and len(T) == next_fast_len(len(T), real=True)
# 3. len(T) is even, and len(T) < next_fast_len(len(T), real=True)
# 4. len(T) is odd, and len(T) < next_fast_len(len(T), real=True)
# And 5. a special case of 1, where len(T) is power of 2.

# Therefore:
# 1. len(T) is composed of 2 and a subset of {3, 5}
# 2. len(T) is composed of a subset of {3, 5}
# 3. len(T) is composed of a subset of {7, 11, 13, ...} and 2
# 4. len(T) is composed of a subset of {7, 11, 13, ...}
# 5. len(T) is power of 2

# In some cases, the prime factors are raised to a power of
# certain degree to increase the length of array to be around
# 1000-2000. This allows us to test sliding_dot_product for
# wider range of query lengths.

test_inputs = [
    # Input format:
    # (
    #     len(T),
    #     remainder,  #  from `len(T) % 2`  
    #     comparator,  # for len(T) comparator next_fast_len(len(T), real=True)
    # )
    (
        2 * (3**2) * (5**3),
        0,
        eq,
    ),  # = 2250, Even `len(T)`, and `len(T) == next_fast_len(len(T), real=True)`
    (
        (3**2) * (5**3),
        1,
        eq,
    ),  # = 1125, Odd `len(T)`, and `len(T) == next_fast_len(len(T), real=True)`.
    (
        2 * 7 * 11 * 13,
        0,
        lt,
    ),  # = 2002, Even `len(T)`, and `len(T) < next_fast_len(len(T), real=True)`
    (
        7 * 11 * 13,
        1,
        lt,
    ),  # = 1001, Odd `len(T)`, and `len(T) < next_fast_len(len(T), real=True)`
]


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_remainder(n_T, remainder, comparator):
    assert n_T % 2 == remainder


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_comparator(n_T, remainder, comparator):
    shape = next_fast_len(n_T, real=True)
    assert comparator(n_T, shape)


@pytest.mark.parametrize("n_T, remainder, comparator", test_inputs)
def test_sdp(n_T, remainder, comparator):
    # test_sdp for cases 1-4

    n_Q_prime = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
    n_Q_power2 = [2, 4, 8, 16, 32, 64]
    n_Q_values = n_Q_prime + n_Q_power2 + [n_T]
    n_Q_values = sorted(n_Q for n_Q in set(n_Q_values) if n_Q <= n_T)

    modules = utils.import_sdp_mods()
    for n_Q in n_Q_values:
        Q = np.random.rand(n_Q)
        T = np.random.rand(n_T)
        ref = naive_sliding_dot_product(Q, T)
        for mod in modules:
            try:
                comp = mod.sliding_dot_product(Q, T)
                npt.assert_allclose(comp, ref)
            except Exception as e:  # pragma: no cover
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

        except Exception as e:  # pragma: no cover
            msg = f"Error in {mod.__name__}, with q={q} and p={p}"
            print(msg)
            raise e

    return


def test_setup():
    Q = np.random.rand(3)
    T = np.random.rand(10)

    for m in sorted(list(pkgutil.iter_modules(sdp.__path__))):
        if m[1].endswith("_sdp"):
            # test if the module has the setup function
            mod_path = f"sdp/{m[1]}.py"
            try:
                assert utils.func_exists(mod_path, "setup")
            except AssertionError as e:  # pragma: no cover
                msg = f"Error in {mod_path}"
                print(msg)
                raise e

            # test if setup function returns None
            mod_name = f"sdp.{m[1]}"
            mod = importlib.import_module(mod_name)
            try:
                assert mod.setup(Q, T) is None
            except AssertionError as e:  # pragma: no cover
                msg = f"Error in {mod_name}"
                print(msg)
                raise e

    return
