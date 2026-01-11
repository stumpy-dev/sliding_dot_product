import math
import numpy as np
from scipy.special import lambertw
from scipy.fft import next_fast_len
from scipy.fft._pocketfft.basic import r2c, c2r
from . import pocketfft_r2c_c2r_sdp


def _compute_block_size(m, n, conv_block_size=None):
    """
    Return a block size for the overlap-add method.
    """
    if conv_block_size is None:
        # Find optimal block_size based on m and n
        if m >= n / 2:
            conv_block_size = n  # i.e. no blocking
        else:
            overlap = m - 1
            opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
            conv_block_size = next_fast_len(math.ceil(opt_size), real=True)

    conv_block_size = max(conv_block_size, m)

    return min(conv_block_size, n)


def _pocketfft_oaconvolve_block(Q, T, conv_block_size):
    m = Q.shape[0]
    n = T.shape[0]

    T_chunk_size = conv_block_size - (m - 1)
    n_chunks = math.ceil(n / T_chunk_size)
    last_chunk_start = (n_chunks - 1) * T_chunk_size

    tmp = np.empty((n_chunks + 1, conv_block_size), dtype=np.float64)

    # fill with T, block-wise
    tmp[: n_chunks - 1, :T_chunk_size] = T[:last_chunk_start].reshape(
        n_chunks - 1, T_chunk_size
    )
    tmp[: n_chunks - 1, T_chunk_size:] = 0.0
    tmp[n_chunks - 1, : n - last_chunk_start] = T[last_chunk_start:]
    tmp[n_chunks - 1, n - last_chunk_start :] = 0.0

    # fill with Q
    tmp[n_chunks, :m] = Q
    tmp[n_chunks, m:] = 0.0

    fft_2d = r2c(True, tmp, axis=-1)

    return c2r(False, np.multiply(fft_2d[:-1], fft_2d[[-1]]), n=conv_block_size)


def _pocketfft_oaconvolve(Q, T, conv_block_size):
    # Linear convolution between two 1D arrays X and Y
    # (of same length N) is an array with length N,
    # and its n-th element is defined as:
    # out[n] = sum_{i=0}^{N-1} X[i] * Y[(n - i)]
    # and, for any out-of-range index, the value is 0.

    # To compute this linear convolution:
    # Approach I: Regular Method
    # X: 1 2 3 4 5 6 --conv-- Y: b a 0 0 0 0

    # X_conv_Y:
    # [
    # 1b,
    # 1a + 2b,
    # 2a + 3b,
    # 3a + 4b,
    # 4a + 5b,
    # 5a + 6b
    # ]

    # Approach II: Overlap-Add Method
    # Step1: Chunk T and compute convolution block-wise
    # Choose a block size -->  3
    # Find chunk size -->  block_size - (M - 1) = 2
    # Chunk T according to chunk size, and Pad
    # each chunk with M-1 zeros at the end

    # X1: 1 2 0  --conv-- Y: b a 0
    # X2: 3 4 0  --conv-- Y: b a 0
    # X3: 5 6 0  --conv-- Y: b a 0

    # X1_conv_Y: [1b, 1a + 2b, 2a]
    # X2_conv_Y: [3b, 3a + 4b, 4a]
    # X3_conv_Y: [5b, 5a + 6b, 6a]

    # Step 2: Add the overlapping parts
    # Slice the blocks based on `chunk_size`, so:
    # [1b, 1a + 2b]
    # [3b, 3a + 4b]
    # [5b, 5a + 6b]

    # The last `M-1` element(s) of k-th block
    # will be added to first `M-1` element(s) of (k+1)-th block.
    # Final output:
    # [
    # 1b
    # 1a + 2b,
    # 3b (+ 2a),
    # 3a + 4b,
    # 5b (+ 4a),
    # 5a + 6b
    # ]
    # Now this is equivalent to X_conv_Y.
    overlap = len(Q) - 1
    QT_conv_blocks = _pocketfft_oaconvolve_block(Q, T, conv_block_size)
    out = QT_conv_blocks[:, :-overlap]
    out[1:, :overlap] += QT_conv_blocks[:-1, -overlap:]
    return np.reshape(out, (-1,))


def _sliding_dot_product(Q, T, conv_block_size):
    return _pocketfft_oaconvolve(Q[::-1], T, conv_block_size)[len(Q) - 1 : len(T)]


def setup(Q, T):
    return


def sliding_dot_product(Q, T, conv_block_size=None):
    m = Q.shape[0]
    n = T.shape[0]
    if m == n:
        return np.dot(Q, T)

    conv_block_size = _compute_block_size(m, n, conv_block_size=conv_block_size)
    if conv_block_size >= n:
        return pocketfft_r2c_c2r_sdp.sliding_dot_product(Q, T)
    else:
        return _sliding_dot_product(Q, T, conv_block_size)
