import fractal
import render
from PIL import Image
import iteration
import numpy as np

borne = 1
c = complex(np.random.uniform(-borne, borne), np.random.uniform(-borne, borne))
print(c)
scale = render.Color_scale(render.Color.black, render.Color.white)
f = iteration.Poly(1, 0, c)


V =fractal.julia(f,height=1964, width=3024, n=100)  
#V =fractal.mosaique_mandelbrot(size=1000, n_sub=40, n=100)  
C = render.coloriser(V, scale)                       
im = Image.fromarray(C)
im.show()