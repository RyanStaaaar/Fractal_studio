import fractal
import numpy as np

V = fractal.FractalGenerator(height=1000, width=1000, n_iter=100).generate_mandelbrot(borne=2)
np.save("mandelbrot_map.npy", V)
