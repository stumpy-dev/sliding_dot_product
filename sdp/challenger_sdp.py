import numpy as np
import pyfftw


class SLIDING_DOT_PRODUCT:
    # https://stackoverflow.com/a/30615425/2955541
    def __init__(self):
        self.m = 0
        self.n = 0
        self.threads = 1
        self.rfft_Q_obj = None
        self.rfft_T_obj = None
        self.irfft_obj = None

    def __call__(self, Q, T):
        if Q.shape[0] != self.m or T.shape[0] != self.n:
            self.m = Q.shape[0]
            self.n = T.shape[0]
            self.shape = pyfftw.next_fast_len(self.n)

            self.rfft_input = pyfftw.empty_aligned(self.shape, dtype=np.float64)
            self.rfft_obj = pyfftw.builders.rfft(
                self.rfft_input,
                overwrite_input=True,
                avoid_copy=True,
                n=self.shape,
                threads=self.threads,
            )

            self.irfft_input = pyfftw.empty_aligned(
                1 + self.shape // 2, dtype=np.complex128
            )
            self.irfft_obj = pyfftw.builders.irfft(
                self.irfft_input,
                overwrite_input=True,
                avoid_copy=True,
                n=self.shape,
                threads=self.threads,
            )

        self.rfft_input[: self.m] = Q[::-1] / self.shape  # Reverse/flip Q and scale
        self.rfft_input[self.m :] = 0.0
        self.rfft_obj.execute()
        self.irfft_input[:] = self.rfft_obj.output_array

        self.rfft_input[: self.n] = T
        self.rfft_input[self.n :] = 0.0
        self.rfft_obj.execute()
        self.irfft_input *= self.rfft_obj.output_array

        # irfft_input is ready
        self.irfft_obj.execute()

        return self.irfft_obj.output_array[self.m - 1 : self.n]


_sliding_dot_product = SLIDING_DOT_PRODUCT()


def setup(Q, T):
    _sliding_dot_product(Q, T)
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
