import math
import time

import numba
import numpy as np
import scipy
from scipy.io import loadmat
from numba import njit
import numpy.testing as npt


# rfft
@njit(fastmath=True)  
def _fft0(n, s, eo, x, y, c): 
    """
    A recursive function that is used as part of fft algorithm
    
    n : int
    s : int
    eo: bool
    x : numpy.array 1D
    y : numpy.array 1D
    """
    if n == 2:
        if eo:
            z = y
        else:
            z = x
        
        for i in range(s):
            j = i + s
            a = x[i]
            b = x[j]
            z[i] = a + b
            z[j] = a - b
            
    elif n >= 4:
        m = n // 2
        sm = s * m
    
        twiddle_factor = 1.0
        for p in range(m):
            sp = s * p
            two_sp = 2 * sp
            for q in range(s):
                i = sp + q
                j = i + sm
                
                k = two_sp + q
                y[k] = x[i] + x[j]
                y[k + s] = (x[i] - x[j]) * twiddle_factor
        
            twiddle_factor = twiddle_factor * c
        
        _fft0(m, 2*s, not eo, y, x, c * c)
        
    else:
        pass
    

@njit(fastmath=True)
def _sixstep_fft(x, y):
    N = len(x)
    n = int(np.sqrt(N))
    
    for k in range(n):
        for p in range(k + 1, n):
            i = k + p * n
            j = p + k * n
            x[i], x[j] = x[j], x[i]
    
    theta = 2 * math.pi / n
    c_thata = math.cos(theta) - 1j * math.sin(theta)
    for p in range(n):
        start = p * n
        _fft0(n, 1, False, x[start:], y[start:], c_thata)
    
    theta_init = 2 * math.pi / N
    n_plus_1 = n + 1
    for p in range(n):
        theta0 = theta_init * p
        ppn = p * n_plus_1
        
        c = math.cos(theta0) - 1j * math.sin(theta0)
        w = math.cos(theta0 * p) - 1j * math.sin(theta0 * p)
        for alpha in range(0, n - p):
            i = ppn + alpha
    
            if alpha == 0:
                x[i] = x[i] * w
            else:
                j = ppn + alpha * n
                x[j], x[i] = x[i] * w, x[j] * w
                
            w = w * c
        
    for k in range(n):
        start = k * n
        _fft0(n, 1, False, x[start:], y[start:], c_thata)
        
    for k in range(n):
        kn = k * n
        for p in range(k + 1, n):
            i = k + p * n
            j = p + kn
            x[i], x[j] = x[j], x[i]
            

@njit(fastmath=True)
def _eightstep_fft(x, y):
    n = len(x)
    m = n // 2
    
    theta0 = math.pi / m    
    wp = 1.0
    factor = math.cos(theta0) - 1j * math.sin(theta0)
    for i in range(m):
        j = i + m
        y[i] = x[i] + x[j]
        y[j] = (x[i] - x[j]) * wp
        wp = wp * factor

    _sixstep_fft(y[:m], x[:m])
    _sixstep_fft(y[m:], x[m:])

    for p in range(m):
        x[2 * p] = y[p]
        x[2 * p + 1] = y[p + m]
        
    return
        

@njit(fastmath=True)  
def _compute_fft(x, y):
    n = len(x)
    logtwo_n = int(np.log2(n))
    
    if logtwo_n == 1: 
        theta = 2 * math.pi / n
        c_thata = math.cos(theta) - 1j * math.sin(theta)
        _fft0(n, 1, False, x, y, c_thata)
    elif logtwo_n % 2 == 0:
        _sixstep_fft(x, y)
    else:
        _eightstep_fft(x, y)
        
    return

@njit(fastmath=True)  
def _rfft(T):
    n = len(T)
    half_n = n // 2
    y = np.empty(half_n + 1, dtype=np.complex128)
    
    x = np.empty(half_n, dtype=np.complex128)
    for i in range(half_n):
        x[i] = T[2 * i] + 1j * T[2 * i + 1]
    _compute_fft(x, y[:half_n])
    
    y[0] = x[0].real + x[0].imag
    y[n // 4] = x[n // 4].conjugate()
    y[half_n] = x[0].real - x[0].imag
    
    theta0 = math.pi / half_n
    factor = math.cos(theta0) - 1j * math.sin(theta0)
    w = 0.5j
    for k in range(1, n // 4):
        w = w * factor
        val = (x[k] - x[half_n - k].conjugate()) * (0.5 + w)
        y[k] = x[k] - val
        y[half_n - k] = x[half_n - k] + val.conjugate()
        
    return y

################################
# irfft (IIRC, j in rfft is changed to -j here)
@njit(fastmath=True)  
def _fft0_with_conjugate(n, s, eo, x, y, c): 
    """
    A recursive function that is used as part of fft algorithm
    
    n : int
    s : int
    eo: bool
    x : numpy.array 1D
    y : numpy.array 1D
    """
    if n == 2:
        if eo:
            z = y
        else:
            z = x
        
        for i in range(s):
            j = i + s
            a = x[i]
            b = x[j]
            z[i] = a + b
            z[j] = a - b
            
    elif n >= 4:
        m = n // 2
        sm = s * m
     
        twiddle_factor = 1.0
        for p in range(m):
            sp = s * p
            two_sp = 2 * sp
            for q in range(s):
                i = sp + q
                j = i + sm
                
                k = two_sp + q
                y[k] = x[i] + x[j]
                y[k + s] = (x[i] - x[j]) * twiddle_factor
        
            twiddle_factor = twiddle_factor * c
        
        _fft0_with_conjugate(m, 2*s, not eo, y, x, c * c)
        
    else:
        pass
    

@njit(fastmath=True)
def _sixstep_fft_with_conjugate(x, y):
    N = len(x)
    n = int(np.sqrt(N))
     
    for k in range(n):
        kn = k * n
        for p in range(k + 1, n):
            i = k + p * n
            j = p + kn
            x[i], x[j] = x[j], x[i]
    
    theta = 2 * math.pi / n
    c_thata = math.cos(theta) + 1j * math.sin(theta)
    for p in range(n):
        start = p * n
        _fft0_with_conjugate(n, 1, False, x[start:], y[start:], c_thata)
    
    theta_init = 2 * math.pi / N
    n_plus_1 = n + 1
    for p in range(n):
        theta0 = theta_init * p
        ppn = p * n_plus_1
        
        c = math.cos(theta0) + 1j * math.sin(theta0)
        w = math.cos(theta0 * p) + 1j * math.sin(theta0 * p)
        for alpha in range(0, n - p):
            i = ppn + alpha
    
            if alpha == 0:
                x[i] = x[i] * w
            else:
                j = ppn + alpha * n
                x[j], x[i] = x[i] * w, x[j] * w
                
            w = w * c
    
    for k in range(n):
        start = k * n
        _fft0_with_conjugate(n, 1, False, x[start:], y[start:], c_thata)    
    
    for k in range(n):
        kn = k * n
        for p in range(k + 1, n):
            i = k + p * n
            j = p + kn
            x[i], x[j] = x[j], x[i]


@njit(fastmath=True)
def _eightstep_fft_with_conjugate(x, y):
    n = len(x)
    m = n // 2
    
    theta0 = math.pi / m
    factor = math.cos(theta0) + 1j * math.sin(theta0)
    wp = 1.0
    for i in range(m):
        j = i + m
        y[i] = x[i] + x[j]
        y[j] = (x[i] - x[j]) * wp

        wp = wp * factor

    _sixstep_fft_with_conjugate(y[:m], x[:m])
    _sixstep_fft_with_conjugate(y[m:], x[m:])

    for p in range(m):
        x[2 * p] = y[p]
        x[2 * p + 1] = y[p + m]
        
    return


@njit(fastmath=True)
def _ifft(x):        
    n = len(x)
    y = np.empty(n, dtype=np.complex128)
    
    logtwo_n = int(np.log2(n))
    if logtwo_n == 1: 
        theta = 2 * math.pi / n
        c_thata = math.cos(theta) + 1j * math.sin(theta)
        _fft0_with_conjugate(n, 1, False, x, y, c_thata)
    elif logtwo_n % 2 == 0:
        _sixstep_fft_with_conjugate(x, y)
    else:
        _eightstep_fft_with_conjugate(x, y)


@njit(fastmath=True)
def _irfft(y, out):
    m = len(y)
    if m == 1:
        out = np.real(y)
        return out
    v = y

    m_minus_1 = m - 1
    m_minus_1_inverse = 1.0 / m_minus_1
    
    v[m_minus_1] = v[m_minus_1].real  # force nyquist element real

    factor = math.cos(math.pi * m_minus_1_inverse) - 1j * math.sin(math.pi * m_minus_1_inverse)
    val = 0.5j
    for i in range(math.ceil(m / 2)):
        j = m_minus_1 - i
        
        v_i_conj = v[i].conjugate()
        cs = 0.5 + val
        v[i] += cs.conjugate() * (v[j].conjugate() - v[i])
        v[j] += cs * (v_i_conj - v[j])

        val = val * factor
        
    _ifft(v[:-1])
    
    for i in range(m_minus_1):
        out[2 * i] = v[i].real * m_minus_1_inverse
        out[2 * i + 1] = v[i].imag * m_minus_1_inverse


################################
# sliding dot product
njit(fastmath=True)
def _sliding_dot_product(Q, T):
    m = len(Q)
    Qr = np.empty(len(T), dtype=np.float64)
    Qr[:m] = Q[::-1]
    Qr[m:] = 0.0

    _irfft(np.multiply(_rfft(T), _rfft(Qr)), out=Qr)

    return Qr[m - 1:]


Q = np.random.rand(50)
T = np.random.rand(100)
_sliding_dot_product(Q, T)


def sliding_dot_product(Q, T):
    return _sliding_dot_product(Q, T)

