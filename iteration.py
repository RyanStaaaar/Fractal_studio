import numpy as np
from numba import njit, prange
class Poly :
    def __init__(self, a, b, c) :
        self.a= a
        self.b=b
        self.c=c
    def evaluate(self,z) :
        return self.a*z*z + self.b*z + self.c


# smooth=True : coloration lissée (escape time logarithmique, dégradés continus)
# smooth=False : version classique (compte d'itérations avant échappement)
@njit(parallel=True)
def escape_speed(Z, a, b, c, n=100, B=2, smooth=True):
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            z = Z[y, x]
            for i in range(n):
                z = a*z*z + b*z + c
                if z.real*z.real + z.imag*z.imag > B2:
                    if smooth:
                        log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                        smooth_i = i + 1 - np.log2(log_zn / np.log(B))
                        V[y, x] = max(0.0, min(1.0, 1.0 - smooth_i / n))
                    else:
                        V[y, x] = (n - i) / n
                    break
    return V

@njit(parallel=True)
def mandelbrot(C, seed, n=100, B=2.0, smooth=True):
    H, W = C.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            c = C[y, x]
            z = seed
            for i in range(n):
                z = z*z + c
                if z.real*z.real + z.imag*z.imag > B2:
                    if smooth:
                        log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                        smooth_i = i + 1 - np.log2(log_zn / np.log(B))
                        V[y, x] = max(0.0, min(1.0, 1.0 - smooth_i / n))
                    else:
                        V[y, x] = (n - i) / n
                    break
    return V
