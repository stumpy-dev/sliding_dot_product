import math
import numpy as np
from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import r2c, c2r
from . import pocketfft_r2c_c2r_sdp


def _compute_block_size(m, n, conv_block_size=None):
    """
    Return a block size for the overlap-add method.

    Parameters
    ----------
    m : int
        Length of the query array Q.

    n : int
        Length of the time series T.

    conv_block_size : int, default None
        Block size for the convolution. When `conv_block_size` is None,
        it will be automatically set to an optimal value, internally
        computed based on the lengths of Q and T.

    Returns
    -------
    conv_block_size : int
        Block size for the convolution. Will be at least `m` and at most `n`.
    """
    if conv_block_size is None:
        if m >= n / 2:
            conv_block_size = n
        else:
            # To minimize Eq. 3 in
            # https://en.wikipedia.org/wiki/Overlap–add_method
            overlap = m - 1
            opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
            conv_block_size = next_fast_len(math.ceil(opt_size), real=True)

    # Ensure that conv_block_size is at least m, so that
    # it can cover at least one element of `T` in each block
    conv_block_size = max(conv_block_size, m)

    return min(conv_block_size, n)


def _pocketfft_circular_convolve_block(Q, T, conv_block_size):
    m = Q.shape[0]
    n = T.shape[0]

    # Each block in overlap-add method needs to be padded
    # with `m-1` zeros. Therefore, the effective block size
    # for T is `conv_block_size - (m-1)`.
    T_block_size = conv_block_size - (m - 1)
    n_blocks = math.ceil(n / T_block_size)
    last_block_start = (n_blocks - 1) * T_block_size

    # To compute the circular convolution between the zero-padded Q
    # and each zero-padded block of T, the data can be loaded into
    # a 2D array with `n_blocks + 1` rows, where the first `n_blocks`
    # rows correspond to the blocks of T, and the last row is the
    # zero-padded Q.
    tmp = np.empty((n_blocks + 1, conv_block_size), dtype=np.float64)
    tmp[: n_blocks - 1, :T_block_size] = T[:last_block_start].reshape(
        n_blocks - 1, T_block_size
    )
    tmp[: n_blocks - 1, T_block_size:] = 0.0
    tmp[n_blocks - 1, : n - last_block_start] = T[last_block_start:]
    tmp[n_blocks - 1, n - last_block_start :] = 0.0

    tmp[n_blocks, :m] = Q
    tmp[n_blocks, m:] = 0.0

    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[:-1], fft_2d[[-1]]), n=conv_block_size)


def _pocketfft_valid_oaconvolve(Q, T, conv_block_size):
    """
    Compute the valid convolution between Q and T using the overlap-add method.
    This method performs several circular convolutions between Q and blocks of T,
    and then combines the results to obtain the valid convolution between Q and T

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence.

    T : numpy.ndarray
        Time series or sequence.

    conv_block_size : int
        Block size for the overlap-add method.
        The value cannot be less than len(Q).

    Returns
    -------
    out : numpy.ndarray
        The valid convolution between Q and T.

    Notes
    -----
    Each block of the convolution contains part of `T`, padded with `len(Q)-1`
    zeros. Therefore, `conv_block_size` must be at least `len(Q)` so that it
    can cover at least one element of `T` in each block.
    """
    # performs several circular convolutions between
    # zero-padded Q and zero-padded blocks of T
    # and returns a 2D array of the results,
    # where each row is associated with a block of T
    QT_conv_blocks = _pocketfft_circular_convolve_block(Q, T, conv_block_size)

    # The subsequences at the boundaries of the blocks
    # are shared between adjacent blocks.
    # The following logic is needed to reconstruct
    # the valid convolution between Q and T
    overlap = len(Q) - 1
    out = QT_conv_blocks[:, :-overlap]
    out[1:, :overlap] += QT_conv_blocks[:-1, -overlap:]

    return np.reshape(out, (-1,))[len(Q) - 1 : len(T)]


def _valid_convolve(Q, T, conv_block_size=None):
    """
    Compute the valid convolution between Q and T

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence.

    T : numpy.ndarray
        Time series or sequence.

    conv_block_size : int, default None
        Block size for the overlap-add method. When `conv_block_size`
        is None, it will automatically be set to an optimal value,
        internally computed based on the lengths of Q and T.

    Returns
    -------
    out : numpy.ndarray
        The valid convolution between Q and T.

    Notes
    -----
    The valid convolution between `Q` and `T` is computed by sliding Q[::-1]
    over T and computing the dot product at each position when there is a
    a full overlap between Q and a subsequence of T.
    """
    m = len(Q)
    n = len(T)
    conv_block_size = _compute_block_size(m, n, conv_block_size=conv_block_size)
    if conv_block_size >= n:
        out = pocketfft_r2c_c2r_sdp._pocketfft_valid_convolve(Q, T)
    else:
        out = _pocketfft_valid_oaconvolve(Q, T, conv_block_size)

    return out


def setup(Q, T):
    return


def sliding_dot_product(Q, T, conv_block_size=None):
    """
    Compute the sliding dot product between Q and T

    Parameters
    ----------
    Q : numpy.ndarray
        Query array or subsequence.

    T : numpy.ndarray
        Time series or sequence.

    conv_block_size : int, default None
        Block size for the overlap-add method. When `conv_block_size`
        is None, it will automatically be set to an optimal value,
        internally computed based on the lengths of Q and T.

    Returns
    -------
    out : numpy.ndarray
        The sliding dot product between Q and T.
    """
    if len(Q) == len(T):
        return np.dot(Q, T)
    else:
        return _valid_convolve(Q[::-1], T, conv_block_size=conv_block_size)
