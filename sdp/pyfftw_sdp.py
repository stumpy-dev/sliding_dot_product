import pyfftw
import numpy as np


class SLIDING_DOT_PRODUCT:
    # https://stackoverflow.com/a/30615425/2955541
    def __init__(self, max_shape=2**10):
        """
        Parameters
        ----------
        max_shape : int
            Maximum shape to preallocate arrays for. This will be the size of the
            the real-valued array. A complex-valued array of size 1 + max_shape // 2
            will also be preallocated.
        """
        self.n = 0
        self.shape = 0
        self.real_arr = pyfftw.empty_aligned(max_shape, dtype="float64")
        self.complex_arr = pyfftw.empty_aligned(1 + max_shape // 2, dtype="complex128")
        self.rfft_irfft_objects = {}

    def __call__(self, Q, T, threads=1, planning_flag="FFTW_MEASURE"):
        m = Q.shape[0]
        if self.n != T.shape[0]:
            self.n = T.shape[0]
            self.shape = pyfftw.next_fast_len(self.n)

        if self.shape > len(self.real_arr):
            self.real_arr = pyfftw.empty_aligned(self.shape, dtype="float64")
            self.complex_arr = pyfftw.empty_aligned(
                1 + self.shape // 2, dtype="complex128"
            )
        real_arr = self.real_arr[: self.shape]
        complex_arr = self.complex_arr[: 1 + self.shape // 2]

        rfft_irfft_obj = self.rfft_irfft_objects.get(self.shape, None)
        if rfft_irfft_obj is None:
            rfft_obj = pyfftw.FFTW(
                input_array=real_arr,
                output_array=complex_arr,
                direction="FFTW_FORWARD",
                flags=(planning_flag,),
                threads=threads,
            )
            irfft_obj = pyfftw.FFTW(
                input_array=complex_arr,
                output_array=real_arr,
                direction="FFTW_BACKWARD",
                flags=(planning_flag, "FFTW_DESTROY_INPUT"),
                threads=threads,
            )
            self.rfft_irfft_objects[self.shape] = (rfft_obj, irfft_obj)
        else:
            rfft_obj, irfft_obj = rfft_irfft_obj
            rfft_obj.update_arrays(real_arr, complex_arr)
            irfft_obj.update_arrays(complex_arr, real_arr)

        # RFFT(T)
        real_arr[: self.n] = T
        real_arr[self.n :] = 0.0
        rfft_obj.execute()  # output is in self.complex_arr
        complex_arr_T = complex_arr.copy()

        # RFFT(Q)
        # Scale by 1/shape to account for
        # FFTW's unnormalized inverse FFT via execute()
        real_arr[:m] = Q[::-1] / self.shape
        real_arr[m:] = 0.0
        rfft_obj.execute()  # output is in self.complex_arr

        # RFFT(T) * RFFT(Q)
        np.multiply(complex_arr, complex_arr_T, out=complex_arr)

        # IRFFT
        # input is in self.complex_arr
        # output is in self.real_arr
        irfft_obj.execute()

        return real_arr[m - 1 : self.n]


_sliding_dot_product = SLIDING_DOT_PRODUCT(max_shape=2**10)


def setup(Q, T, threads=1, planning_flag="FFTW_MEASURE"):
    _sliding_dot_product(Q, T, threads=threads, planning_flag=planning_flag)
    return


def sliding_dot_product(Q, T, threads=1, planning_flag="FFTW_MEASURE"):
    return _sliding_dot_product(Q, T, threads=threads, planning_flag=planning_flag)
