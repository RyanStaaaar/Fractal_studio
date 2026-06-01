import fractal
import render
from PIL import Image

gen = fractal.FractalGenerator(height=1000, width=1000, n_iter=100)
V = gen.generate_mosaic(n_sub=11, tile_size=100)
palette = [[0.0, 0.5, 1.0], [[0, 0, 0], [128, 0, 200], [255, 255, 255]]]
Image.fromarray(render.FractalRenderer(palette).render(V)).save("wallpaper_0.png")
