import iteration
import numpy as np

"""
def julia(f, size, borne=2, n=100):
    xs = np.linspace(-borne, borne, size)
    ys = np.linspace(-borne, borne, size)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    return iteration.escape_speed(Z, f.a, f.b, f.c, n=100, B=2.0)
"""

def julia(f, height, width, borne=2, n=100):
    borne_y = borne * height / width
    xs = np.linspace(-borne, borne, width)
    ys = np.linspace(-borne_y, borne_y, height)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    return iteration.escape_speed(Z, f.a, f.b, f.c, n=100, B=2.0)


def mosaique_mandelbrot(size=100, n_sub=11, borne_julia=2, borne_c=2, n=100):
    meta_size = n_sub * size
    V = np.zeros((meta_size, meta_size))

    cs = np.linspace(-borne_c, borne_c, n_sub)
    for i, cy in enumerate(cs):
        for j, cx in enumerate(cs):
            c = complex(cx, cy)
            f = iteration.Poly(1, 0, c)
            V[i*size:(i+1)*size, j*size:(j+1)*size] = julia(f, size, size, borne_julia, n)
    return V
#if __name__=="__main__" :



