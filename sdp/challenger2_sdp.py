import numpy as np
import pyfftw


class SLIDING_DOT_PRODUCT:
    # https://stackoverflow.com/a/30615425/2955541
    def __init__(self):
        self.m = 0
        self.n = 0
        self.shape = 0
        self.threads = 1

        self.rfft_obj = None
        self.irfft_obj = None

        self.store_obj = False
        self.historical_obj = {}

        self.real_arr = pyfftw.empty_aligned(2**20, dtype=np.float64)
        self.complex_arr = pyfftw.empty_aligned(2**20, dtype=np.complex128)

    def __call__(self, Q, T):
        if T.shape[0] == self.n:
            shape = self.shape
        else:
            shape = pyfftw.next_fast_len(T.shape[0])

        self.n = T.shape[0]
        self.m = Q.shape[0]

        if self.shape != shape:
            # need to set rfft/irfft objects
            self.shape = shape
            rfft_irfft_objects = self.historical_obj.get(self.shape, None)

            if rfft_irfft_objects is not None:
                self.rfft_obj = rfft_irfft_objects[0]
                self.irfft_obj = rfft_irfft_objects[1]
            else:
                self.rfft_obj = pyfftw.builders.rfft(
                    pyfftw.empty_aligned(self.shape, dtype=np.float64),
                    n=self.shape,
                    overwrite_input=True,
                    avoid_copy=True,
                    threads=self.threads,
                )

                self.irfft_obj = pyfftw.builders.irfft(
                    pyfftw.empty_aligned(1 + self.shape // 2, dtype=np.complex128),
                    overwrite_input=True,
                    avoid_copy=True,
                    n=self.shape,
                    threads=self.threads,
                )

        if self.store_obj:  # pragma: no cover
            self.historical_obj[self.shape] = (self.rfft_obj, self.irfft_obj)

        rfft_shape = 1 + self.shape // 2
        if self.shape <= self.real_arr.shape[0]:
            real_arr = self.real_arr[: self.shape]
            complex_arr = self.complex_arr[:rfft_shape]
        else:  # pragma: no cover
            real_arr = pyfftw.empty_aligned(self.shape, dtype=np.float64)
            complex_arr = pyfftw.empty_aligned(rfft_shape, dtype=np.complex128)

        real_arr[: self.n] = T
        real_arr[self.n :] = 0
        self.rfft_obj(real_arr)
        complex_arr[:] = self.rfft_obj.output_array

        real_arr[: self.m] = Q[::-1]
        real_arr[self.m :] = 0
        self.rfft_obj(real_arr)
        complex_arr *= self.rfft_obj.output_array

        self.irfft_obj(complex_arr)

        out = self.irfft_obj.output_array[self.m - 1 : self.n]

        return out

    def create_reusable_objects(self, pmin=2, pmax=20):
        for p in range(pmin, pmax + 1):
            n = 2**p
            rfft_obj = pyfftw.builders.rfft(
                pyfftw.empty_aligned(n, dtype=np.float64),
                overwrite_input=True,
                avoid_copy=True,
                n=n,
                threads=self.threads,
            )
            irfft_obj = pyfftw.builders.irfft(
                pyfftw.empty_aligned(1 + n // 2, dtype=np.complex128),
                overwrite_input=True,
                avoid_copy=True,
                n=n,
                threads=self.threads,
            )
            self.historical_obj[n] = (rfft_obj, irfft_obj)

        return


_sliding_dot_product = SLIDING_DOT_PRODUCT()


def setup(Q, T):
    _sliding_dot_product.create_reusable_objects()
    _sliding_dot_product(Q, T)
    return


def sliding_dot_product(Q, T):
    if len(Q) == len(T):
        out = np.dot(Q, T)
    else:
        out = _sliding_dot_product(Q, T)

    return out
