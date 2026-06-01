from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import glob, os
import subprocess
import random

import iteration
import render
import fractal


# ------------------------------------------------------------------ #
#  Conversions RGB <-> HSV vectorisées (numpy, h/s/v dans [0, 1])
# ------------------------------------------------------------------ #
def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc == 0, 0.0, delta / np.where(maxc == 0, 1.0, maxc))

    deltac = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac
    h = np.where(maxc == r, bc - gc, 0.0)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = np.where(delta == 0, 0.0, (h / 6.0) % 1.0)
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    w = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, w, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [w, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, w, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    gray = np.stack([v, v, v], axis=-1)
    return np.where(s[..., None] == 0, gray, rgb)


def coloriser_hsv(V, palette):
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)

    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    colors = colors[order]

    stops_hsv = rgb_to_hsv(colors / 255.0)        # conversion des stops uniquement

    V_clamped = np.clip(V, positions[0], positions[-1])
    idx_right = np.clip(np.searchsorted(positions, V_clamped, side="right"),
                        1, len(positions) - 1)
    idx_left = idx_right - 1

    span = positions[idx_right] - positions[idx_left]
    span[span == 0] = 1
    t = (V_clamped - positions[idx_left]) / span   # (H, W)

    lhsv = stops_hsv[idx_left]
    rhsv = stops_hsv[idx_right]

    # teinte : interpolation circulaire (plus court chemin)
    lh, rh = lhsv[..., 0], rhsv[..., 0]
    dh = (rh - lh + 0.5) % 1.0 - 0.5
    h = (lh + t * dh) % 1.0

    # saturation et valeur : interpolation linéaire
    s = lhsv[..., 1] * (1 - t) + rhsv[..., 1] * t
    v = lhsv[..., 2] * (1 - t) + rhsv[..., 2] * t

    rgb = hsv_to_rgb(np.stack([h, s, v], axis=-1)) * 255.0
    return rgb.astype(np.uint8)


def coloriser_rgb(V, palette):
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)

    # tri par position, en conservant le lien couleur <-> position
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    colors = colors[order]

    V_clamped = np.clip(V, positions[0], positions[-1])

    idx_right = np.searchsorted(positions, V_clamped, side="right")
    idx_right = np.clip(idx_right, 1, len(positions) - 1)
    idx_left = idx_right - 1

    span = positions[idx_right] - positions[idx_left]
    span[span == 0] = 1
    t = (V_clamped - positions[idx_left]) / span

    lcolor = colors[idx_left]
    rcolor = colors[idx_right]

    t = t[:, :, None]
    return (lcolor * (1 - t) + rcolor * t).astype(np.uint8)

def coloriser(V, palette):
    if mode_var.get() == "hsv":
        return coloriser_hsv(V, palette)
    return coloriser_rgb(V, palette)
# ------------------------------------------------------------------ #
#  Constantes
# ------------------------------------------------------------------ #
FULL_W, FULL_H = 3024, 1964                      # résolution d'export (wallpaper)
PREVIEW_W = 450                                  # résolution de l'aperçu
PREVIEW_H = round(PREVIEW_W * FULL_H / FULL_W)   # même ratio que le HD
N_ITER = 80
BORNE = 2
GRAD_H = 28                                       # hauteur de la barre de palette

dossier = "/Users/ryanmounir/Desktop/INTERFAAAACE/Mandelbrot_project/Wallpapers"


# ------------------------------------------------------------------ #
#  Palette de départ (tirée au hasard, comme dans ton script)
# ------------------------------------------------------------------ #
position = [0, random.uniform(0, 0.8), random.uniform(0, 0.8),
            random.uniform(0, 0.8), 1]
position.sort()

nuancier = [render.Color(0, 0, 0), render.Color(255, 255, 255),
            render.Color(116, 0, 184), render.Color(105, 48, 195),
            render.Color(94, 96, 206), render.Color(83, 144, 217),
            render.Color(78, 168, 222), render.Color(72, 191, 227),
            render.Color(86, 207, 225), render.Color(100, 223, 223),
            render.Color(114, 239, 221), render.Color(128, 255, 219)]
color_choices = random.sample(nuancier, k=5)

# couleurs en LISTES (mutables) pour pouvoir éditer un canal à la fois
couleurs = [list(c.get_rgb()) for c in color_choices]
palette = [position, couleurs]

# barre de dégradé servant d'aperçu de la palette
barre = np.linspace(np.zeros(GRAD_H), np.ones(GRAD_H), PREVIEW_W).T


# ------------------------------------------------------------------ #
#  Fenêtre
# ------------------------------------------------------------------ #
root = tk.Tk()
root.title("Fractale de Julia + palette")

# valeur de c, pilotée par deux DoubleVar
c0 = fractal.choisir_c()
c_re = tk.DoubleVar(value=round(c0.real, 4))
c_im = tk.DoubleVar(value=round(c0.imag, 4))

# mode d'interpolation des couleurs : "rgb" ou "hsv"
mode_var = tk.StringVar(value="rgb")

# aperçu de la fractale
fractal_canvas = tk.Canvas(root, width=PREVIEW_W, height=PREVIEW_H)
fractal_canvas.pack(padx=10, pady=(10, 4))
fractal_item = fractal_canvas.create_image(0, 0, anchor="nw")

# barre de palette
grad_canvas = tk.Canvas(root, width=PREVIEW_W, height=GRAD_H)
grad_canvas.pack(padx=10, pady=(0, 10))
grad_item = grad_canvas.create_image(0, 0, anchor="nw")

V_preview = None   # cache du tableau d'itérations (recalculé seulement si c change)


# ------------------------------------------------------------------ #
#  Cœur : calcul vs colorisation
# ------------------------------------------------------------------ #
def current_c():
    return complex(c_re.get(), c_im.get())


def compute_V(h, w):
    f = iteration.Poly(1, 0, current_c())
    return fractal.julia(f, height=h, width=w, n=N_ITER, borne=BORNE)


def redraw_fractal():
    # ne recolorise que le V en cache -> opération bon marché, temps réel
    if V_preview is None:
        return
    C = coloriser(V_preview, palette)
    photo = ImageTk.PhotoImage(Image.fromarray(C))
    fractal_canvas.itemconfig(fractal_item, image=photo)
    fractal_canvas.image = photo   # référence gardée (sinon le GC efface l'image)


def update_gradient():
    C = coloriser(barre, palette)
    photo = ImageTk.PhotoImage(Image.fromarray(C))
    grad_canvas.itemconfig(grad_item, image=photo)
    grad_canvas.image = photo


def apply_palette():
    # appelé quand une couleur ou une position change : pas de recalcul de fractale
    update_gradient()
    redraw_fractal()


def recompute_fractal(*_):
    # appelé quand c change : recalcul complet de l'aperçu
    global V_preview
    V_preview = compute_V(PREVIEW_H, PREVIEW_W)
    redraw_fractal()


# ------------------------------------------------------------------ #
#  Contrôles de palette (une ligne par couleur)
# ------------------------------------------------------------------ #
def hexcolor(rgb):
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


swatches = []


def refresh_swatch(i):
    swatches[i].config(bg=hexcolor(palette[1][i]))


def make_pos_cb(i):
    def cb(val):
        palette[0][i] = float(val)
        apply_palette()
    return cb


def make_rgb_cb(i, channel, var):
    def cb(*_):
        try:
            v = int(float(var.get()))
        except (ValueError, tk.TclError):
            return
        palette[1][i][channel] = max(0, min(255, v))
        refresh_swatch(i)
        apply_palette()
    return cb


mode_frame = tk.Frame(root)
mode_frame.pack(padx=10, pady=(4, 0), anchor="w")
tk.Label(mode_frame, text="Interpolation :").pack(side="left")
tk.Radiobutton(mode_frame, text="RGB", variable=mode_var, value="rgb",
               command=apply_palette).pack(side="left")
tk.Radiobutton(mode_frame, text="HSV", variable=mode_var, value="hsv",
               command=apply_palette).pack(side="left")


pal_frame = tk.LabelFrame(root, text="Palette")
pal_frame.pack(padx=10, pady=4, fill="x")

for i in range(len(palette[0])):
    row = tk.Frame(pal_frame)
    row.pack(fill="x", pady=3, padx=6)

    sw = tk.Label(row, width=3, relief="solid", bd=1, bg=hexcolor(palette[1][i]))
    sw.pack(side="left", padx=(0, 8))
    swatches.append(sw)

    for channel, name in enumerate(("R", "G", "B")):
        tk.Label(row, text=name).pack(side="left")
        var = tk.StringVar(value=str(palette[1][i][channel]))
        var.trace_add("write", make_rgb_cb(i, channel, var))
        ttk.Spinbox(row, from_=0, to=255, width=4,
                    textvariable=var).pack(side="left", padx=(0, 6))

    tk.Label(row, text="pos").pack(side="left", padx=(8, 4))
    sc = tk.Scale(row, from_=0, to=1, resolution=0.01,
                  orient="horizontal", length=150)
    sc.set(palette[0][i])
    sc.config(command=make_pos_cb(i))
    sc.pack(side="left")


# ------------------------------------------------------------------ #
#  Contrôles de c (Julia)
# ------------------------------------------------------------------ #
c_frame = tk.LabelFrame(root, text="Paramètre c")
c_frame.pack(padx=10, pady=4, fill="x")

c_label_var = tk.StringVar()


def update_c_label(*_):
    signe = "+" if c_im.get() >= 0 else "-"
    c_label_var.set(f"c = {c_re.get():.3f} {signe} {abs(c_im.get()):.3f} i")


c_re.trace_add("write", update_c_label)
c_im.trace_add("write", update_c_label)
update_c_label()

re_row = tk.Frame(c_frame)
re_row.pack(fill="x", padx=6, pady=2)
tk.Label(re_row, text="Re(c)", width=6).pack(side="left")
sc_re = tk.Scale(re_row, from_=-1.5, to=1.5, resolution=0.001,
                 orient="horizontal", length=320, variable=c_re, showvalue=False)
sc_re.pack(side="left", fill="x", expand=True)

im_row = tk.Frame(c_frame)
im_row.pack(fill="x", padx=6, pady=2)
tk.Label(im_row, text="Im(c)", width=6).pack(side="left")
sc_im = tk.Scale(im_row, from_=-1.5, to=1.5, resolution=0.001,
                 orient="horizontal", length=320, variable=c_im, showvalue=False)
sc_im.pack(side="left", fill="x", expand=True)

# recalcul seulement quand on relâche le curseur (la fractale est chère à calculer)
sc_re.bind("<ButtonRelease-1>", recompute_fractal)
sc_im.bind("<ButtonRelease-1>", recompute_fractal)

tk.Label(c_frame, textvariable=c_label_var).pack(pady=(2, 4))


def c_aleatoire():
    nc = fractal.choisir_c()
    c_re.set(round(nc.real, 4))
    c_im.set(round(nc.imag, 4))
    recompute_fractal()


# ------------------------------------------------------------------ #
#  Export HD
# ------------------------------------------------------------------ #
def exporter_hd():
    V_full = compute_V(FULL_H, FULL_W)
    C = coloriser(V_full, palette)
    im = Image.fromarray(C)

    ajd = datetime.today().strftime("%d_%m_%Y")
    chemin = f"{dossier}/wallpaper_{ajd}.png"
    im.save(chemin)
    print(f"image sauvegardée : {chemin}")

    # ménage : ne garder que les 7 plus récents
    # fichiers = sorted(glob.glob(f"{dossier}/*.png"), key=os.path.getmtime)
    # for old in fichiers[:-7]:
    #     os.remove(old)

    try:
        result = subprocess.run(["/usr/local/bin/desktoppr", chemin],
                                capture_output=True, text=True)
        print(f"desktoppr : {result.returncode} | {result.stdout} | {result.stderr}")
        subprocess.run(["killall", "Dock"])
    except Exception as e:
        print(f"pose du wallpaper ignorée : {e}")


# ------------------------------------------------------------------ #
#  Boutons
# ------------------------------------------------------------------ #
btns = tk.Frame(root)
btns.pack(pady=8)
tk.Button(btns, text="Régénérer", command=recompute_fractal).pack(side="left", padx=4)
tk.Button(btns, text="c aléatoire", command=c_aleatoire).pack(side="left", padx=4)
tk.Button(btns, text="Exporter en HD", command=exporter_hd).pack(side="left", padx=4)


# ------------------------------------------------------------------ #
#  Premier rendu
# ------------------------------------------------------------------ #
recompute_fractal()   # calcule V_preview et l'affiche
update_gradient()

root.mainloop()