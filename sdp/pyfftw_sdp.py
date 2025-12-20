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
        self.n = 0
        self.next_fast_n = 0

        # Preallocate arrays
        self.real_arr = pyfftw.empty_aligned(max_n, dtype="float64")
        self.complex_arr = pyfftw.empty_aligned(1 + (max_n // 2), dtype="complex128")

        # Store FFTW objects, keyed by (next_fast_n, n_threads, planning_flag)
        self.rfft_objects = {}
        self.irfft_objects = {}

    def __call__(self, Q, T, n_threads=1, planning_flag="FFTW_MEASURE"):
        """
        Parameters
        ----------
        Q : numpy.ndarray
            Query array or subsequence

        T : numpy.ndarray
            Time series or sequence

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
        if self.n != T.shape[0]:
            self.n = T.shape[0]
            self.next_fast_n = pyfftw.next_fast_len(self.n)

        # Update preallocated arrays if needed
        if self.next_fast_n > len(self.real_arr):
            self.real_arr = pyfftw.empty_aligned(self.next_fast_n, dtype="float64")
            self.complex_arr = pyfftw.empty_aligned(
                1 + (self.next_fast_n // 2), dtype="complex128"
            )

        real_arr = self.real_arr[: self.next_fast_n]
        complex_arr = self.complex_arr[: 1 + (self.next_fast_n // 2)]

        # Get or create FFTW objects
        key = (self.next_fast_n, n_threads, planning_flag)

        rfft_obj = self.rfft_objects.get(key, None)
        if rfft_obj is None:
            rfft_obj = pyfftw.FFTW(
                input_array=real_arr,
                output_array=complex_arr,
                direction="FFTW_FORWARD",
                flags=(planning_flag,),
                threads=n_threads,
            )
            self.rfft_objects[key] = rfft_obj
        else:
            rfft_obj.update_arrays(real_arr, complex_arr)

        irfft_obj = self.irfft_objects.get(key, None)
        if irfft_obj is None:
            irfft_obj = pyfftw.FFTW(
                input_array=complex_arr,
                output_array=real_arr,
                direction="FFTW_BACKWARD",
                flags=(planning_flag, "FFTW_DESTROY_INPUT"),
                threads=n_threads,
            )
            self.irfft_objects[key] = irfft_obj
        else:
            irfft_obj.update_arrays(complex_arr, real_arr)

        # RFFT(T)
        real_arr[: self.n] = T
        real_arr[self.n :] = 0.0
        rfft_obj.execute()  # output is in complex_arr
        complex_arr_T = complex_arr.copy()

        # RFFT(Q)
        # Scale by 1/next_fast_n to account for
        # FFTW's unnormalized inverse FFT via execute()
        real_arr[:m] = Q[::-1] / self.next_fast_n
        real_arr[m:] = 0.0
        rfft_obj.execute()  # output is in complex_arr

        # RFFT(T) * RFFT(Q)
        np.multiply(complex_arr, complex_arr_T, out=complex_arr)

        # IRFFT
        # input is in complex_arr
        irfft_obj.execute()  # output is in real_arr

        return real_arr[m - 1 : self.n]


_sliding_dot_product = SLIDING_DOT_PRODUCT()


def setup(Q, T, n_threads=1, planning_flag="FFTW_MEASURE"):
    _sliding_dot_product(Q, T, n_threads=n_threads, planning_flag=planning_flag)
    return


def sliding_dot_product(Q, T, n_threads=1, planning_flag="FFTW_MEASURE"):
    return _sliding_dot_product(Q, T, n_threads=n_threads, planning_flag=planning_flag)
