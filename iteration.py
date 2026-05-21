import numpy as np
class Poly :
    def __init__(self, a, b, c) :
        self.a= a
        self.b=b
        self.c=c
    def evaluate(self,z) :
        return self.a*z*z + self.b*z + self.c
    
    def escape_speed(self, Z, n=100, B=2) :
        V= np.zeros(Z.shape)
        Z_copy= Z.copy()
        for i in range(n) :
            vivant = V==0
            Z_copy[vivant]=self.evaluate(Z_copy[vivant])
            escaped = np.abs(Z_copy)>B
            V[escaped & vivant ]=(n-i)/n
        return V
    
#if __name__ == "__main__" :
