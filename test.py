import numpy as np
import pytest

from numpy import testing as npt
from sdp import challenger_sdp

from utils import import_sdp_mods


def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


def test_modules():
    pmin = 3
    pmax = 13

    modules = import_sdp_mods()
    for mod in modules:
        try:
            for q in range(pmin, pmax + 1):
                for p in range(q, pmax + 1):
                    Q = np.random.rand(2 ** q)
                    T = np.random.rand(2 ** p)
                    ref = naive_sliding_dot_product(Q, T)
                    comp = mod.sliding_dot_product(Q, T)

                    np.testing.assert_allclose(comp, ref)
        except Exception as e:
            print(f"Error in {mod.__name__}: {str(e)}")
            raise e

    return