import numpy as np
import pytest

from sdp import challenger_sdp

def naive_sliding_dot_product(Q, T):
    m = len(Q)
    l = T.shape[0] - m + 1
    out = np.empty(l)
    for i in range(l):
        out[i] = np.dot(Q, T[i : i + m])
    return out


def test_challenger():
    pmin = 2
    pmax = 10
    for q in range(pmin, pmax):
        for p in range(q, pmax):
            Q = np.random.rand(2 ** q)
            T = np.random.rand(2 ** p)
            ref = naive_sliding_dot_product(Q, T)
            comp = challenger_sdp.sliding_dot_product(Q, T)
            
            np.testing.assert_allclose(comp, ref)

    return