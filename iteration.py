import numpy as np
class Poly :
    def __init__(self, a, b, c) :
        self.a= a
        self.b=b
        self.c=c
    def evaluate(self,z) :
        return self.a*z*z + self.b*z + self.c
    def iter(self, z, n) :
        if n==1 :
            return self.evaluate(z)
        else : 
            return self.iter(self.evaluate(z), n-1)
    def orbit_bounded(self, z, n, B) :
        f_z = self.evaluate(z)
        if n==1 :
            return (abs(f_z)< B)
        else :
            if abs(f_z) > B :
                return False
            else :
                return self.orbit_bounded(f_z, n-1, B) 
    
    def escape_speed(self, Z, n=100, B=2) :
        V= np.zeros(Z.shape)
        Z_copy= Z.copy()
        for i in range(n) :
            vivant = V==0
            Z_copy[vivant]=self.evaluate(Z_copy[vivant])
            escaped = abs(Z_copy)>B
            V[escaped & vivant ]=(n-i)/n
        return V
    
if __name__ == "__main__" :
    f = Poly(1,1,0.1)
    z= complex(0,-0.5)
    y= complex(2,1)
    nul=complex(0,0)

    n=10
    B= 1000
    #print(f.orbit_bounded(z,n,B))
    #print(f.orbit_bounded(y,n,B))
    print(f.orbit_bounded(nul,n,B))
    print(f.escape_speed(z, 100))