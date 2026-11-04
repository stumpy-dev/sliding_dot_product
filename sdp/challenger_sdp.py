import math

import numpy as np

from sdp import pocketfft_r2c_c2r_sdp


def _valid_oaconvolve(Q, T, chunk_size):
    """
    Compute the valid convolution between Q and T using the overlap-add method.

    `T` is split into non-overlapping chunks of `chunk_size` elements
    (the last chunk may be shorter). Each chunk is fully convolved with `Q`,
    and the chunk results are summed at their proper offsets (overlap-add),
    which gives the full linear convolution between `Q` and `T`. Finally,
    only the "valid" part is kept.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence, of length `m`.

    T : numpy.ndarray
        Time series or sequence, of length `n`.

    chunk_size : int
        Number of elements of `T` in each chunk. Must be at least 1.

    Returns
    -------
    out : numpy.ndarray
        The valid convolution between Q and T, of length `n - m + 1`.

    Notes
    -----
    The full linear convolution between a chunk (length `chunk_size`) and
    `Q` (length `m`) has length `chunk_size + m - 1`. Consecutive chunk
    results therefore overlap by `m - 1` samples, which are summed.
    """
    m = Q.shape[0]
    n = T.shape[0]

    # Full linear convolution of Q and T has n + m - 1 elements
    out = np.zeros(n + m - 1, dtype=np.float64)

    # TODO:
    # 1. Vectorize the for-loop
    # 2. Use FFT-based circular convolution
    for i in range(math.ceil(n / chunk_size)):
        start = i * chunk_size
        stop = start + chunk_size
        stop = min(stop, n)  # the last chunk may be shorter
        chunk_conv = np.convolve(Q, T[start:stop], mode="full")

        # Overlap-add: chunk_conv has length len(chunk) + m - 1, so it
        # overlaps the next chunk's result by m - 1 samples
        out[start : start + len(chunk_conv)] += chunk_conv

    # Valid part: indices m - 1 to n - 1 (where Q fully overlaps T),
    # i.e. n - m + 1 samples
    return out[m - 1 : n]


def _valid_convolve(Q, T, chunk_size=None):
    """
    Compute the valid convolution between Q and T.

    The overlap-add method is used unless `chunk_size` is smaller than 1
    or covers all of `T`, in which case no chunking is done.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence.

    T : numpy.ndarray
        Time series or sequence.

    chunk_size : int, default None
        Number of elements of `T` in each chunk of the overlap-add method.
        If None, `len(T) // 2` is used.

    Returns
    -------
    out : numpy.ndarray
        The valid convolution between Q and T, of length
        `len(T) - len(Q) + 1`.

    Notes
    -----
    The valid convolution between ``Q`` and ``T`` is equivalent to
    the sliding dot product between Q[::-1] and T.
    """
    n = len(T)
    if chunk_size is None:
        # TODO: Improve the dummy default
        chunk_size = n // 2

    if chunk_size < 1 or chunk_size >= n:
        # TODO: Improve the fallback condition
        # No chunking: convolve Q with the whole of T at once
        return pocketfft_r2c_c2r_sdp._valid_convolve(Q, T)

    return _valid_oaconvolve(Q, T, chunk_size)


def setup(Q, T):
    return


def sliding_dot_product(Q, T, chunk_size=None):
    """
    Compute the sliding dot product between Q and T.

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence.

    T : numpy.ndarray
        Time series or sequence.

    chunk_size : int, default None
        Number of elements of `T` per chunk in the overlap-add convolution.
        See `_valid_convolve`.

    Returns
    -------
    out : numpy.ndarray
        The sliding dot product between Q and T, of length
        `len(T) - len(Q) + 1`.
    """
    if len(Q) == len(T):
        # Only one window, so the result is a single dot product
        return np.array([np.dot(Q, T)])

    # Convolution flips its kernel, so flipping Q first turns the
    # convolution into a sliding dot product.
    return _valid_convolve(Q[::-1], T, chunk_size=chunk_size)
