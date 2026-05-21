from PIL import Image
import numpy as np
import iteration
import render
import fractal

f = iteration.Poly(1, 0, complex(-0.4, -0.601))
scale = render.Color_scale(render.Color.red, render.Color.black)

V = fractal.julia(f, size=800)
C = render.coloriser(V, scale)
im = Image.fromarray(C)
im.show()
