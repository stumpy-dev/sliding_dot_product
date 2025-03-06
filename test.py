import numpy as np
import utils

from numpy import testing as npt


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


def test_sdp_power2():
    pmin = 3
    pmax = 13

    modules = utils.import_sdp_mods()
    for mod in modules:
        try:
            for q in range(pmin, pmax + 1):
                for p in range(q, pmax + 1):
                    Q = np.random.rand(2**q)
                    T = np.random.rand(2**p)

                    ref = naive_sliding_dot_product(Q, T)
                    comp = mod.sliding_dot_product(Q, T)
                    npt.assert_allclose(comp, ref)

        except Exception as e:
            msg = f"Error in {mod.__name__}, with q={q} and p={p}"
            print(msg)
            raise e

    return


def test_sdp_power2_plus1():
    pmin = 3
    pmax = 13

    modules = utils.import_sdp_mods()
    for mod in modules:
        try:
            for q in range(pmin, pmax + 1):
                for p in range(q, pmax + 1):
                    Q = np.random.rand(2**q + 1)
                    T = np.random.rand(2**p + 1)

                    ref = naive_sliding_dot_product(Q, T)
                    comp = mod.sliding_dot_product(Q, T)
                    npt.assert_allclose(comp, ref)

        except Exception as e:
            msg = f"Error in {mod.__name__}, with q={q} and p={p}"
            print(msg)
            raise e

    return


def test_sdp_power2_minus1():
    pmin = 3
    pmax = 13

    modules = utils.import_sdp_mods()
    for mod in modules:
        try:
            for q in range(pmin, pmax + 1):
                for p in range(q, pmax + 1):
                    Q = np.random.rand(2**q - 1)
                    T = np.random.rand(2**p - 1)

                    ref = naive_sliding_dot_product(Q, T)
                    comp = mod.sliding_dot_product(Q, T)
                    npt.assert_allclose(comp, ref)

        except Exception as e:
            msg = f"Error in {mod.__name__}, with q={q} and p={p}"
            print(msg)
            raise e

    return
