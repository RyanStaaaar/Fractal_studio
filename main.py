import fractal
import render
from PIL import Image
import iteration
import numpy as np

borne = 1
c = complex(np.random.uniform(-borne, borne), np.random.uniform(-borne, borne))
scale = render.ColorScale(render.Color.black, render.Color.white)
f = iteration.Poly(1, 0, c)


#V =fractal.mandelbrot(height=1964, width=3024, seed = 0+0j, n=100)  
V =fractal.mosaique_mandelbrot(size=1000, n_sub=40, n=100)  
C = render.coloriser(V, scale)                       
im = Image.fromarray(C)
im.save("/Users/ryanmounir/Desktop/INTERFAAAACE/Mandelbrot_project/Wallpapers/wallpaper_0.png")