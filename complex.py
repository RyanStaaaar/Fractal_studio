from math import sqrt
class Complex :
    # Dunder methods
    def __init__(self, real=0,im=0) :
        self.real= real
        self.im= im
    def __add__(self, autre ) :
        if isinstance(autre, (int,float)) :
            return Complex(self.real + autre, self.im)
        if isinstance(autre, Complex) :
            return Complex(self.real + autre.real, self.im + autre.im)
        return NotImplemented
    def __radd__(self, autre) :
        return self.__add__(autre)
    def __sub__(self, autre : Complex ) :
        if isinstance(autre, (int,float)) :
            return Complex(self.real-autre, self.im)
        if isinstance(autre, Complex) :
            return Complex(self.real - autre.real, self.im - autre.im)
        return NotImplemented
    def __rsub__(self,autre) :
        if isinstance(autre, (int,float)) :
            return Complex(autre - self.real, -self.im)
    def __mul__(self, autre : Complex ) :
        if isinstance(autre, (int,float)) :
            return Complex(self.real*autre,self.im*autre)
        if isinstance(autre, Complex) :
            return Complex(self.real * autre.real - self.im * autre.im, self.real * autre.im + autre.real * self.im )
        return NotImplemented
    def __rmul__(self,autre) :
        return self.__mul__(autre)
    def __str__(self) :
        if self.im == 1 :
            return f"{self.real} + i"
        elif self.im == -1 :
            return f"{self.real} - i"
        elif self.im < 0 :
            return f"{self.real} - {abs(self.im)}i"
        else : 
            return f"{self.real} + {self.im}i"
    
    # get methods
    def get_coordinate(self):
        return (self.real, self.im)
    def get_real(self) :
        return(self.real)
    def get_im(self) :
        return(self.im)
    # autres
    def conjugate(self) :
        return Complex(self.real,-self.im)
    def norm(self) :
        return (self*(self.conjugate())).get_real()
    def modulus(self) :
        return sqrt(self.norm())

if __name__=="__main__" :
    z= Complex(3,4)
    y= Complex(0,-1)
    y2= Complex(0,1)
    v= Complex(2,-8)
    print(z)
    print(y)
    print(y2)
    print(v)

    print(z*z.conjugate())
    print(z.norm())
    print(z.modulus())

   