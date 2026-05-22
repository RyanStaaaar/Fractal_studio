import numpy as np
from numba import njit, prange
class Poly :
    def __init__(self, a, b, c) :
        self.a= a
        self.b=b
        self.c=c
    def evaluate(self,z) :
        return self.a*z*z + self.b*z + self.c
    
"""
    def escape_speed(self, Z, n=100, B=2) :
        V= np.zeros(Z.shape)
        Z_copy= Z.copy()
        for i in range(n) :
            vivant = V==0
            Z_copy[vivant]=self.evaluate(Z_copy[vivant])
            escaped = np.abs(Z_copy)>B
            V[escaped & vivant ]=(n-i)/n
        return V
"""

@njit(parallel=True)
def escape_speed(Z, a, b, c, n=100, B=2.0):
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            z = Z[y, x]
            for i in range(n):
                z = a*z*z + b*z + c
                if z.real*z.real + z.imag*z.imag > B2:
                    V[y, x] = (n - i) / n
                    break
    return V
   
#if __name__ == "__main__" :
