"""Planche comparant 5 façons d'ajouter du relief sans (trop) abîmer la palette.

Colonnes = méthodes, lignes = intensité croissante, + référence sans relief :
  - Actuel (lum.) : ombrage actuel, luminosité seule (warmth=0), shadow_floor décroissant
  - 1 palette     : on décale la COORDONNÉE dans le dégradé -> reste 100% couleurs Wada
  - 2 emboss      : seul le détail haute-fréquence (arêtes) est éclairé, aplats intacts
  - 3 soft-light  : couche d'ombrage fusionnée en « lumière douce » à opacité croissante
  - 4 contour     : lignes d'encre sombres sur les arêtes (aplats Wada intacts)

Lance :  myenv/bin/python relief_methods.py   -> Exports/relief_methods.png
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import iteration
import fractal
import render

# ------------------------------------------------------------------ #
#  RÉGLAGES
# ------------------------------------------------------------------ #
C = complex(-0.4, 0.6)
PALETTE_INDEX = 200
MODE = "oklab"
N_ITER = 200
AZIMUTH, ELEVATION, DEPTH = 135, 45, 3.0   # géométrie de lumière fixe
INTENSITIES = [0.25, 0.5, 0.75, 1.0]       # lignes
TILE_W, TILE_H = 340, 226
PAD, CAP, HDR_H = 12, 26, 70
# ------------------------------------------------------------------ #


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT, FONT_HDR = _font(14), _font(17)


def _lambert(V):
    a, e = np.radians(AZIMUTH), np.radians(ELEVATION)
    lx, ly, lz = np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)
    gy, gx = np.gradient(V.astype(np.float64))
    nx, ny, nz = -DEPTH * gx, -DEPTH * gy, np.ones_like(gx)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.clip((nx * lx + ny * ly + nz * lz) * inv, 0.0, 1.0)


# ---- les 5 méthodes : (V, base, palette, d, intensité) -> image RGB uint8 ---- #
def m_actuel(V, base, palette, d, I):
    sf = 1.0 - 0.6 * I                       # plus I est grand, plus les ombres sont basses
    return render.shade(V, base, AZIMUTH, ELEVATION, DEPTH, warmth=0.0, shadow_floor=sf)


def m_palette(V, base, palette, d, I):
    v_lit = np.clip(V + (0.6 * I) * (d - 0.5), 0.0, 1.0)   # décale la coordonnée couleur
    return render.FractalRenderer(palette, MODE).render(v_lit)


def m_emboss(V, base, palette, d, I):
    blur = np.asarray(Image.fromarray((d * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(6)), np.float64) / 255.0
    detail = d - blur                                       # composante locale (arêtes)
    factor = np.clip(1.0 + (5.0 * I) * detail, 0.0, 2.0)
    return np.clip(base.astype(np.float64) * factor[..., None], 0, 255).astype(np.uint8)


def _soft_light(b, s):
    dd = np.where(b <= 0.25, ((16 * b - 12) * b + 4) * b, np.sqrt(b))
    return np.where(s <= 0.5, b - (1 - 2 * s) * b * (1 - b), b + (2 * s - 1) * (dd - b))


def m_softlight(V, base, palette, d, I):
    b = base.astype(np.float64) / 255.0
    sl = _soft_light(b, d[..., None])
    out = (1 - I) * b + I * sl
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def m_contour(V, base, palette, d, I):
    gy, gx = np.gradient(V.astype(np.float64))
    g = np.sqrt(gx * gx + gy * gy)
    g = g / (np.percentile(g, 97) + 1e-9)                  # normalise (évite qu'une arête écrase)
    t = np.clip(g, 0.0, 1.0)
    factor = 1.0 - (0.9 * I) * t                           # encre sombre sur les arêtes
    return np.clip(base.astype(np.float64) * factor[..., None], 0, 255).astype(np.uint8)


METHODS = [("Actuel (lum.)", m_actuel), ("1 palette", m_palette), ("2 emboss", m_emboss),
           ("3 soft-light", m_softlight), ("4 contour", m_contour)]


def _tile(arr, caption):
    canvas = Image.new("RGB", (TILE_W, TILE_H + CAP), (25, 25, 25))
    canvas.paste(Image.fromarray(arr), (0, 0))
    ImageDraw.Draw(canvas).text((6, TILE_H + 5), caption, fill=(235, 235, 235), font=FONT)
    return canvas


def main():
    gen = fractal.FractalGenerator(TILE_H, TILE_W, N_ITER, smooth=True)
    V = gen.generate_julia(iteration.Poly(1, 0, C))
    palette = render.load_sanzo_palettes()[PALETTE_INDEX % 348]
    base = render.FractalRenderer(palette, MODE).render(V)
    d = _lambert(V)

    cols = len(METHODS)
    tfh = TILE_H + CAP
    width = PAD + cols * (TILE_W + PAD)
    height = HDR_H + (len(INTENSITIES) + 1) * (tfh + PAD) + PAD
    sheet = Image.new("RGB", (width, height), (15, 15, 15))
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 10), f"Relief — méthodes comparées   c={C.real:+.3f}{C.imag:+.3f}i, "
              f"palette #{PALETTE_INDEX}, lumière az={AZIMUTH}° el={ELEVATION}° depth={DEPTH}",
              fill=(255, 235, 150), font=FONT_HDR)
    draw.text((PAD, 36), "colonnes = méthode | lignes = intensité croissante | "
              "1 et 4 préservent exactement les couleurs Wada",
              fill=(200, 200, 200), font=FONT)

    # référence (sans relief) en haut à gauche
    sheet.paste(_tile(base, "RÉFÉRENCE (sans relief)"), (PAD, HDR_H))
    # bandeau noms de méthodes au-dessus de la grille
    y0 = HDR_H + tfh + PAD
    for j, (name, _) in enumerate(METHODS):
        draw.text((PAD + j * (TILE_W + PAD), y0 - 14), name, fill=(150, 220, 255), font=FONT)

    for i, I in enumerate(INTENSITIES):
        y = y0 + i * (tfh + PAD)
        for j, (name, fn) in enumerate(METHODS):
            arr = fn(V, base, palette, d, I)
            x = PAD + j * (TILE_W + PAD)
            sheet.paste(_tile(arr, f"{name}  ·  I={I}"), (x, y))

    out = Path(__file__).parent / "Exports"
    out.mkdir(exist_ok=True)
    path = out / "relief_methods.png"
    sheet.save(path)
    print(f"planche enregistrée : {path}  ({width}x{height})")


if __name__ == "__main__":
    main()
