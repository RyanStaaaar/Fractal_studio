from PIL import Image, ImageDraw, ImageFont
import numpy as np
import iteration
import render
import fractal
import subprocess
import datetime

borne = 1
c = complex(np.random.uniform(-borne, borne), np.random.uniform(-borne, borne))
scale = render.Color_scale(render.Color.black, render.Color.white)
f = iteration.Poly(1, 0, c)


V =fractal.julia(f,height=1964, width=3024, n=100)  
#V =fractal.mosaique_mandelbrot(size=1000, n_sub=40, n=100)  
C = render.coloriser(V, scale)                       
im = Image.fromarray(C)

font=ImageFont.truetype("/System/Library/Fonts/Supplemental/AmericanTypewriter.ttc", size = 100)
ajd= datetime.date.today()
draw = ImageDraw.Draw(im)
draw.text((10,10), ajd.strftime("%-d"),font= font, anchor = "lt", fill = "black")

im.save("/Users/ryanmounir/Desktop/INTERFAAAACE/Mandelbrot_project/Wallpapers/wallpaper_.png")

subprocess.run(["desktoppr", "/Users/ryanmounir/Desktop/INTERFAAAACE/Mandelbrot_project/Wallpapers/wallpaper_.png"])
