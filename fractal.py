import iteration
import render
from PIL import ImageDraw, ImageFont
from math import pi
import numpy as np

def julia(f, size, borne=2, n=100):
    xs = np.linspace(-borne, borne, size)
    ys = np.linspace(-borne, borne, size)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    return f.escape_speed(Z, n)
def julia_picture(scale ,f: iteration.Poly, size = 10, n_iter = 10, borne = 2, ) :
    cons=2
    step = pi/size
    # initialisation du plan
    plan = render.Plan(size)
    for x in range(-size//cons, size//cons) :
        for y in range(-size//cons, size//cons) :
            z= complex(step*x,step*y)
            escape_speed = f.escape_speed(z, n_iter, B=borne)
            plan.draw_point((x,y),color=(scale.level(escape_speed)))
    return plan
def mosaique_mandelbrot(size=100, n_sub=11, borne_julia=2, borne_c=2, n=100):
    meta_size = n_sub * size
    V = np.zeros((meta_size, meta_size))

    cs = np.linspace(-borne_c, borne_c, n_sub)
    for i, cy in enumerate(cs):
        for j, cx in enumerate(cs):
            c = complex(cx, cy)
            print(c)
            f = iteration.Poly(1, 0, c)
            V[i*size:(i+1)*size, j*size:(j+1)*size] = julia(f, size, borne_julia, n)
    return V
if __name__=="__main__" :
    a=1
    b=0
    c=complex(-0.4, -0.601)
    f=iteration.Poly(a,b,c)
    color_0=render.Color.white
    color_1= render.Color.black
    scale= render.Color_scale(color_0, color_1)

    j=julia_picture(scale, f,size=5000, n_iter = 100)
    j.display()


