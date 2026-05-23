#!/usr/bin/env python3
"""
daily_wallpaper.py

À placer DANS le dossier Mandelbrot_obsession, à côté de main.py.
Réutilise tes modules (fractal, iteration, render) sans les modifier, et ajoute
seulement ce qu'il faut pour en faire un fond d'écran quotidien :
  - un c déterministe selon la date (même jour -> même fractale)
  - une présélection : on teste plusieurs c et on garde le plus riche
    visuellement, pour ne pas tomber sur de la poussière déconnectée
  - l'enregistrement de l'image (au lieu de im.show())
  - la pose en fond d'écran macOS (desktoppr si présent, sinon AppleScript)

Lancer à la main pour tester :  python3 daily_wallpaper.py
"""

import datetime as dt
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import fractal      # tes modules
import iteration
import render

# ----------------------------------------------------------------- réglages
WIDTH, HEIGHT = 3024, 1964          # ta résolution (14" MBP) ; ajuste si besoin
N_ITER        = 100                 # profondeur d'itération (comme ton main.py)
N_CANDIDATES  = 12                  # nombre de c testés chaque jour
PROBE         = 200                 # taille basse résolution pour scorer les candidats
BORNE_C       = 1.0                 # même domaine de tirage de c que ton main.py
SAVE_DIR      = Path.home() / "Pictures" / "fractal-wallpapers"

# on pioche deux couleurs de TA palette, variables selon le jour
PALETTE = [render.Color.black, render.Color.white, render.Color.red,
           render.Color.blue, render.Color.violet, render.Color.orange,
           render.Color.cyan, render.Color.magenta, render.Color.green2]


def date_rng(date):
    """Générateur aléatoire ensemencé par la date (même jour -> même image)."""
    seed = int(hashlib.sha256(date.isoformat().encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def interest(V):
    """Richesse visuelle à partir de TON tableau V (0 = dans l'ensemble)."""
    inside = float((V == 0).mean())          # proportion dans l'ensemble
    esc = V[V > 0]
    if esc.size == 0:
        return 0.0
    return float(esc.std()) * 4 * inside * (1 - inside)   # pic près du bord


def choose_c(rng):
    """Teste plusieurs c en basse résolution, renvoie le plus intéressant."""
    best_c, best = None, -1.0
    for _ in range(N_CANDIDATES):
        c = complex(rng.uniform(-BORNE_C, BORNE_C), rng.uniform(-BORNE_C, BORNE_C))
        f = iteration.Poly(1, 0, c)
        V = fractal.julia(f, height=PROBE, width=PROBE, n=N_ITER)
        s = interest(V)
        if s > best:
            best, best_c = s, c
    return best_c


def set_wallpaper(path):
    """Pose le fond d'écran. desktoppr de préférence, sinon AppleScript."""
    if shutil.which("desktoppr"):
        subprocess.run(["desktoppr", str(path)], check=True)
    else:
        script = ('tell application "System Events" to tell every desktop '
                  f'to set picture to "{path}"')
        subprocess.run(["osascript", "-e", script], check=True)


def main():
    today = dt.date.today()
    rng = date_rng(today)

    c = choose_c(rng)
    i0, i1 = rng.choice(len(PALETTE), size=2, replace=False)
    scale = render.Color_scale(PALETTE[int(i0)], PALETTE[int(i1)])

    f = iteration.Poly(1, 0, c)
    V = fractal.julia(f, height=HEIGHT, width=WIDTH, n=N_ITER)
    C = render.coloriser(V, scale)
    im = Image.fromarray(C)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out = SAVE_DIR / f"{today.isoformat()}.png"
    im.save(out)
    set_wallpaper(out)
    print(f"{today}: c = {c.real:.4f}{c.imag:+.4f}i  ->  {out}")


if __name__ == "__main__":
    main()
