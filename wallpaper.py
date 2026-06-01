from PIL import Image, ImageDraw, ImageFont
import numpy as np
import iteration
import render
import fractal
import subprocess
import random
from datetime import datetime
import glob, os # pour trier et éviter l'accumulation de wallpaper

dossier = "/Users/ryanmounir/Desktop/INTERFAAAACE/Mandelbrot_project/Wallpapers"
fichiers = sorted(glob.glob(f"{dossier}/*.png"), key=os.path.getmtime)

# supprime tout sauf les 7 plus récents
# for old in fichiers[:-7]:
#     os.remove(old)
position = [0, random.uniform(0,0.8),random.uniform(0,0.8), random.uniform(0,0.8), 1]
position.sort()

nuancier= [render.Color(0,0,0), render.Color(255,255,255), render.Color(116, 0, 184),render.Color(105, 48, 195),render.Color(94, 96, 206),render.Color(83, 144, 217),render.Color(78, 168, 222),
           render.Color(72, 191, 227),render.Color(86, 207, 225),render.Color(100, 223, 223),render.Color(114, 239, 221),render.Color(128, 255, 219)]
color_choices=random.sample(nuancier, k=5)
couleurs = [ color_choices[0].get_rgb(), color_choices[1].get_rgb(),color_choices[2].get_rgb(), color_choices[3].get_rgb(),color_choices[4].get_rgb()]
palette=[position, couleurs]
print(palette)


c =fractal.choisir_c()
#c =complex(0.37, 0.1)
f = iteration.Poly(1, 0, c)


V=fractal.julia(f,height=1964, width=3024, n=80, borne = 2)  
#V=fractal.mandelbrot(height=1964, width=3024)
#V =fractal.mosaique_mandelbrot(size=1000, n_sub=40, n=100) 


C = render.coloriser(V, palette)                       
im = Image.fromarray(C)
ajd = datetime.today().strftime("%d_%m_%Y")
#rd = random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13,14])

chemin = f"/Users/ryanmounir/Desktop/INTERFAAAACE/Mandelbrot_project/Wallpapers/wallpaper_{ajd}.png"

im.save(chemin)
print(f"image sauvegardée : {chemin}")
result = subprocess.run(["/usr/local/bin/desktoppr", chemin],  capture_output=True, text=True)
print(f"desktoppr retour : {result.returncode} | {result.stdout} | {result.stderr}")
subprocess.run(["killall", "Dock"])
print("dock relancé")

