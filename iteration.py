from complex import Complex
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
        mod_f_z = f_z.modulus()
        if n==1 :
            return (mod_f_z< B)
        else :
            if mod_f_z > B :
                return False
            else :
                return self.orbit_bounded(f_z, n-1, B) 
    def orbit_behaviour(self, z, n, n_tot=100, B=2):
        if n==0 :
            return 0
        else : 
            f_z = self.evaluate(z)
            mod_f_z = f_z.modulus()
            if mod_f_z >B :
                return int(255*n/n_tot)
            else :
                return self.orbit_behaviour(f_z, n-1, n_tot, B)

        mod_f_z = f_z.modulus()
        if n==1 :
            return (mod_f_z< B)


if __name__ == "__main__" :
    f = Poly(1,1,0.1)
    z= Complex(0,1)
    y= Complex(2,1)
    nul=Complex(0,0)

    n=10
    B= 1000
    #print(f.orbit_bounded(z,n,B))
    #print(f.orbit_bounded(y,n,B))
    print(f.orbit_bounded(nul,n,B))