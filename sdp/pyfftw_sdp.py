import pyfftw
import numpy as np


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
            shape = pyfftw.next_fast_len(self.n)
            self.rfft_Q_obj = pyfftw.builders.rfft(
                np.empty(self.m), overwrite_input=True, n=shape, threads=self.threads
            )
            self.rfft_T_obj = pyfftw.builders.rfft(
                np.empty(self.n), overwrite_input=True, n=shape, threads=self.threads
            )
            self.irfft_obj = pyfftw.builders.irfft(
                self.rfft_Q_obj.output_array,
                overwrite_input=True,
                n=shape,
                threads=self.threads,
            )

        Qr = Q[::-1]  # Reverse/flip Q
        rfft_padded_Q = self.rfft_Q_obj(Qr)
        rfft_padded_T = self.rfft_T_obj(T)

        return self.irfft_obj(np.multiply(rfft_padded_Q, rfft_padded_T)).real[
            self.m - 1 : self.n
        ]


_sliding_dot_product = SLIDING_DOT_PRODUCT()


def setup(Q, T):  # pragma: no cover
    _sliding_dot_product(Q, T)
    return


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)
