"""Compare SSAA 1→4 sur 3 fractales avec les réglages de wallpaper.py.

Pour chaque fractale, 4 images identiques sauf le niveau de supersampling.
Sortie : Exports/ssaa_compare/<slug>_ssaa<k>.png

Lance :  myenv/bin/python ssaa_compare.py
"""
from pathlib import Path
import random
import numpy as np

import iteration
import fractal
import render
from transform import parse_transform

# ------------------------------------------------------------------ #
#  Réglages repris de WallpaperApp
# ------------------------------------------------------------------ #
MODE       = "oklab"
SMOOTH     = True
REPEAT     = random.randint(2, 4)
EQUALIZE   = True
CLIP_LIMIT = 5.5
N_ITER     = 100
TRANSFORM  = "z"
OUT_W      = 1200
OUT_H      = 779          # ratio 3024:1964

SSAA_LEVELS = [1, 2, 3, 4]

FRACTALES = [
    (complex(-0.4,    0.6),     200, "f1_arbre"),
    (complex(-0.7269, 0.1889),   55, "f2_spirale"),
    (complex(0.285,   0.013),   130, "f3_nautile"),
]
# ------------------------------------------------------------------ #


def main():
    out = Path(__file__).parent / "Exports" / "ssaa_compare"
    out.mkdir(parents=True, exist_ok=True)

    palettes  = render.load_sanzo_palettes()
    transform = parse_transform(TRANSFORM)
    total     = len(FRACTALES) * len(SSAA_LEVELS)
    done      = 0

    print(f"REPEAT={REPEAT}  →  {total} images dans {out}/\n")

    for c, pal_idx, slug in FRACTALES:
        palette = palettes[pal_idx % 348]
        renderer = render.FractalRenderer(palette, mode=MODE, n_iter=N_ITER,
                                          repeat=REPEAT, equalize=EQUALIZE,
                                          clip_limit=CLIP_LIMIT)
        for k in SSAA_LEVELS:
            gen = fractal.FractalGenerator(OUT_H * k, OUT_W * k, N_ITER,
                                           smooth=SMOOTH, transform=transform)
            V   = gen.generate_julia(iteration.Poly(1, 0, c))
            img = render.downscale(renderer.render(V), OUT_W, OUT_H)
            fname = f"{slug}_ssaa{k}.png"
            from PIL import Image
            Image.fromarray(img).save(out / fname)
            done += 1
            print(f"  [{done:2d}/{total}]  {fname}")

    print(f"\nTerminé — {done} images enregistrées.")


if __name__ == "__main__":
    main()
