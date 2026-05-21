import fractal
import render
from PIL import Image

scale = render.Color_scale(render.Color.black, render.Color.white)

V =fractal.mosaique_mandelbrot(size=1000, n_sub=100, n=10)  
C = render.coloriser(V, scale)                       
im = Image.fromarray(C)
im.show()