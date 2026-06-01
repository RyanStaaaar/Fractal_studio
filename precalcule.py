import fractal
import numpy as np

V = fractal.mandelbrot(height=1000, width=1000, borne=2, n=100)
np.save("mandelbrot_map.npy", V)