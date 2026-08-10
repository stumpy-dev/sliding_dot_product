import pyfftw
import numpy as np


class SLIDING_DOT_PRODUCT:
    # https://stackoverflow.com/a/30615425/2955541
    def __init__(self, max_n=2**20):
        """
        Parameters
        ----------
        max_n : int
            Maximum length to preallocate arrays for. This will be the size of the
            the real-valued array. A complex-valued array of size `1 + (max_n // 2)`
            will also be preallocated.
        """
        # Preallocate arrays
        self.real_arr_A = pyfftw.empty_aligned(max_n, dtype="float64")
        self.real_arr_B = pyfftw.empty_aligned(max_n, dtype="float64")
        self.toggle_one = (-1) ** np.arange(max_n)

        # Store FFTW objects, keyed by (next_fast_n, n_threads, planning_flag)
        self.dct_typeII_objects = {}
        self.dct_typeI_objects = {}

    def __call__(self, Q, T, n_threads=1, planning_flag="FFTW_ESTIMATE"):
        """
        Compute the sliding dot product between `Q` and `T` using FFTW via pyfftw.

        Parameters
        ----------
        Q : numpy.ndarray
            Query array or subsequence.

        T : numpy.ndarray
            Time series or sequence.

        n_threads : int, default=1
            Number of threads to use for FFTW computations.

        planning_flag : str, default="FFTW_MEASURE"
            The planning flag that will be used in FFTW for planning.
            See pyfftw documentation for details. Current options include:
            "FFTW_ESTIMATE", "FFTW_MEASURE", "FFTW_PATIENT", and "FFTW_EXHAUSTIVE".

        Returns
        -------
        out : numpy.ndarray
            Sliding dot product between `Q` and `T`.
        """
        m = Q.shape[0]
        n = T.shape[0]
        p1 = (n - m + 1) // 2
        p2 = (m + 1) // 2
        N = p1 + p2 + n  # The length of Q_padded and T_padded

        # Update preallocated arrays if needed
        N_plus_1 = N + 1
        if N_plus_1 > len(self.real_arr_A):
            self.real_arr_A = pyfftw.empty_aligned(N_plus_1, dtype="float64")
            self.real_arr_B = pyfftw.empty_aligned(N_plus_1, dtype="float64")
            self.toggle_one = (-1) ** np.arange(N_plus_1)

        real_arr_A = self.real_arr_A[:N_plus_1]
        real_arr_B = self.real_arr_B[:N_plus_1]

        # Get or create FFTW objects
        key = (N, n_threads, planning_flag)

        dct_typeII_object = self.dct_typeII_objects.get(key, None)
        if dct_typeII_object is None:
            dct_typeII_object = pyfftw.FFTW(
                input_array=real_arr_A[:N],
                output_array=real_arr_B[:N],
                direction="FFTW_REDFT10",
                flags=(planning_flag,),
                threads=n_threads,
            )
            self.dct_typeII_objects[key] = dct_typeII_object
        else:
            dct_typeII_object.update_arrays(real_arr_A[:N], real_arr_B[:N])

        dct_typeI_object = self.dct_typeI_objects.get(key, None)
        if dct_typeI_object is None:
            dct_typeI_object = pyfftw.FFTW(
                input_array=real_arr_B,
                output_array=real_arr_A,
                direction="FFTW_REDFT00",
                flags=(planning_flag, "FFTW_DESTROY_INPUT"),
                threads=n_threads,
            )
            self.dct_typeI_objects[key] = dct_typeI_object
        else:
            dct_typeI_object.update_arrays(real_arr_B, real_arr_A)

        # Pad Q
        real_arr_A[:p1] = 0
        real_arr_A[p1 : p1 + m] = Q
        real_arr_A[p1 + m : N] = 0

        dct_typeII_object.execute()  # output is in real_arr_B
        dct_Q = real_arr_B[:N].copy()

        # Pad T
        real_arr_A[: p1 + p2] = 0
        real_arr_A[p1 + p2 : N] = T
        dct_typeII_object.execute()

        # Multiply Result and some modifications
        np.multiply(real_arr_B[:N], dct_Q, out=real_arr_B[:N])
        real_arr_B[0] *= np.sqrt(2) / (4 * N)
        real_arr_B[1:N] *= 1 / (2 * N)
        real_arr_B[N] = 0
        dct_typeI_object.execute()

        # real_arr_B --> real_arr_A
        # Need to correct output real_arr_A since scipy's dct type 1...
        # with norm 'ortho' is used
        n_arr = len(real_arr_A)
        real_arr_A += (np.sqrt(2) - 1) * (
            real_arr_B[0] + self.toggle_one[:n_arr] * real_arr_B[-1]
        )
        real_arr_A[:] = real_arr_A / np.sqrt(2 * (n_arr - 1))
        real_arr_A[0] /= np.sqrt(2)
        real_arr_A[-1] /= np.sqrt(2)

        return np.sqrt(2 * N) * real_arr_A[p2 : p2 + (n - m + 1)]


_sliding_dot_product = SLIDING_DOT_PRODUCT()


def setup(Q, T, n_threads=1, planning_flag="FFTW_ESTIMATE"):
    _sliding_dot_product(Q, T, n_threads=n_threads, planning_flag=planning_flag)
    return


def sliding_dot_product(Q, T, n_threads=1, planning_flag="FFTW_ESTIMATE"):
    return _sliding_dot_product(Q, T, n_threads=n_threads, planning_flag=planning_flag)
