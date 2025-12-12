import pyfftw
import numpy as np


class SLIDING_DOT_PRODUCT:
    # https://stackoverflow.com/a/30615425/2955541
    def __init__(self):
        self.m = 0
        self.n = 0
        self.shape = 0
        self.threads = 1

    def __call__(self, Q, T):
        self.m = Q.shape[0]

        need_new_obj = False
        if self.n != T.shape[0]:
            # need to re-compute shape
            self.n = T.shape[0]
            shape = pyfftw.next_fast_len(self.n)
            if self.shape != shape:
                self.shape = shape
                need_new_obj = True

        if need_new_obj:
            # create input and output arrays and FFTW objects
            self.real_arr = pyfftw.empty_aligned(self.shape, dtype="float64")
            self.complex_arr = pyfftw.empty_aligned(
                1 + self.shape // 2, dtype="complex128"
            )

            self.rfft_obj = pyfftw.FFTW(
                input_array=self.real_arr,
                output_array=self.complex_arr,
                flags=("FFTW_MEASURE",),
                direction="FFTW_FORWARD",
                threads=self.threads,
            )

            self.irfft_obj = pyfftw.FFTW(
                input_array=self.complex_arr,
                output_array=self.real_arr,
                flags=("FFTW_MEASURE", "FFTW_DESTROY_INPUT"),
                direction="FFTW_BACKWARD",
                threads=self.threads,
            )

        # RFFT(T)
        self.real_arr[: self.n] = T
        self.real_arr[self.n : self.shape] = 0.0
        self.rfft_obj.execute()  # output is in self.complex_arr
        complex_arr_T = self.complex_arr.copy()

        # RFFT(Q)
        self.real_arr[: self.m] = Q[::-1] / self.shape  # reversed Q and scale
        self.real_arr[self.m : self.shape] = 0.0
        self.rfft_obj.execute()  # output is in self.complex_arr

        # RFFT(T) * RFFT(Q)
        np.multiply(self.complex_arr, complex_arr_T, out=self.complex_arr)

        # IRFFT
        # input is in self.complex_arr
        # output is in self.real_arr
        self.irfft_obj.execute()

        return self.real_arr[self.m - 1 : self.n]


_sliding_dot_product = SLIDING_DOT_PRODUCT()


def setup(Q, T):
    _sliding_dot_product(Q, T)
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
