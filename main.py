import fractal
import render
from PIL import Image
import iteration
scale = render.Color_scale(render.Color.black, render.Color.white)
f = iteration.Poly(1, 0, complex(-0.4, -0.601))

V =fractal.julia(f,size=10000)  
#V =fractal.mosaique_mandelbrot(size=100, n_sub=100, n=100)  
C = render.coloriser(V, scale)                       
im = Image.fromarray(C)
im.show()