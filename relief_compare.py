"""Planche comparative des réglages de relief (ombrage pictural).

Génère une image où l'on voit, côte à côte, la fractale SANS relief (référence)
et une matrice de rendus AVEC relief : lignes = depth (force du relief),
colonnes = warmth (décalage de teinte = distorsion de la palette Wada).
But : choisir le réglage le plus agréable (assez de relief, peu de distorsion).

Édite les RÉGLAGES ci-dessous puis lance :  myenv/bin/python relief_compare.py
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import iteration
import fractal
import render

# ------------------------------------------------------------------ #
#  RÉGLAGES
# ------------------------------------------------------------------ #
C = complex(-0.4, 0.6)        # paramètre c de la Julia (structure à comparer)
PALETTE_INDEX = 200           # combinaison Sanzo Wada (0..347)
MODE = "oklab"                # interpolation des couleurs
N_ITER = 200
SHADOW_FLOOR = 0.5            # fixe ici ; baisse-le pour des ombres plus profondes
AZIMUTH, ELEVATION = 135, 45  # direction de la lumière
DEPTHS = [1, 2, 4, 6]         # lignes : relief de + en + prononcé
WARMTHS = [0.0, 0.3, 0.6, 1.0]  # colonnes : distorsion de teinte croissante
TILE_W, TILE_H = 380, 250
PAD, CAP = 12, 26             # marge et hauteur de légende
# ------------------------------------------------------------------ #


def _load_font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _load_font(15)
FONT_HDR = _load_font(17)


def _tile(arr: np.ndarray, caption: str) -> Image.Image:
    canvas = Image.new("RGB", (TILE_W, TILE_H + CAP), (25, 25, 25))
    canvas.paste(Image.fromarray(arr), (0, 0))
    ImageDraw.Draw(canvas).text((6, TILE_H + 5), caption, fill=(235, 235, 235), font=FONT)
    return canvas


def main():
    gen = fractal.FractalGenerator(TILE_H, TILE_W, N_ITER, smooth=True)
    V = gen.generate_julia(iteration.Poly(1, 0, C))        # champ calculé une seule fois
    palette = render.load_sanzo_palettes()[PALETTE_INDEX % 348]

    ref = render.FractalRenderer(palette, MODE).render(V)   # référence, sans relief
    ref_tile = _tile(ref, "SANS RELIEF (référence)")

    grid = []
    for d in DEPTHS:
        row = []
        for w in WARMTHS:
            arr = render.FractalRenderer(palette, MODE, light=True, azimuth=AZIMUTH,
                                         elevation=ELEVATION, depth=d, warmth=w,
                                         shadow_floor=SHADOW_FLOOR).render(V)
            row.append(_tile(arr, f"depth={d}   warmth={w}"))
        grid.append(row)

    cols = len(WARMTHS)
    tile_full_h = TILE_H + CAP
    header_h = 46
    width = PAD + cols * (TILE_W + PAD)
    height = header_h + (len(DEPTHS) + 1) * (tile_full_h + PAD) + PAD
    sheet = Image.new("RGB", (width, height), (15, 15, 15))

    hdr = (f"Relief — c={C.real:+.3f}{C.imag:+.3f}i, palette #{PALETTE_INDEX}, "
           f"shadow_floor={SHADOW_FLOOR}, lumière az={AZIMUTH}° el={ELEVATION}°   |   "
           f"colonnes = warmth (distorsion couleur), lignes = depth (relief)")
    ImageDraw.Draw(sheet).text((PAD, 12), hdr, fill=(255, 235, 150), font=FONT_HDR)

    sheet.paste(ref_tile, (PAD, header_h))                  # référence en haut à gauche
    y = header_h + tile_full_h + PAD
    for row in grid:
        x = PAD
        for tile in row:
            sheet.paste(tile, (x, y))
            x += TILE_W + PAD
        y += tile_full_h + PAD

    out = Path(__file__).parent / "Exports"
    out.mkdir(exist_ok=True)
    path = out / "relief_compare.png"
    sheet.save(path)
    print(f"planche enregistrée : {path}  ({width}x{height})")


if __name__ == "__main__":
    main()
