import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk


def coloriser(V, palette):
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)

    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    colors = colors[order]

    V_clamped = np.clip(V, positions[0], positions[-1])

    idx_right = np.searchsorted(positions, V_clamped, side="right")
    idx_right = np.clip(idx_right, 1, len(positions) - 1)
    idx_left = idx_right - 1

    lpos = positions[idx_left]
    rpos = positions[idx_right]

    span = rpos - lpos
    span[span == 0] = 1
    t = (V_clamped - lpos) / span

    lcolor = colors[idx_left]
    rcolor = colors[idx_right]

    t = t[:, :, None]
    return (lcolor * (1 - t) + rcolor * t).astype(np.uint8)


# --- palette : positions d'un côté, couleurs (en listes mutables) de l'autre ---
palette = [
    [0.0, 1.0, 0.5, 1.0],
    [[255, 255, 255], [230, 100, 35], [77, 99, 180], [0, 255, 0]],
]

height = 50
width = 500
barre = np.linspace(np.zeros(height), np.ones(height), width).T


# --- fenêtre ---
root = tk.Tk()
root.title("Éditeur de palette")

canvas = tk.Canvas(root, width=width, height=height)
canvas.pack(padx=10, pady=10)
item_id = canvas.create_image(0, 0, anchor="nw")


def update_gradient():
    C = coloriser(barre, palette)
    img = Image.fromarray(C)
    photo = ImageTk.PhotoImage(img)
    canvas.itemconfig(item_id, image=photo)
    canvas.image = photo  # référence gardée, sinon le ramasse-miettes efface l'image


def hexcolor(rgb):
    r, g, b = (int(c) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


swatches = []


def refresh_swatch(i):
    swatches[i].config(bg=hexcolor(palette[1][i]))


def make_pos_cb(i):
    def cb(val):
        palette[0][i] = float(val)
        update_gradient()
    return cb


def make_rgb_cb(i, channel, var):
    def cb(*_):
        try:
            v = int(float(var.get()))
        except (ValueError, tk.TclError):
            return
        palette[1][i][channel] = max(0, min(255, v))
        refresh_swatch(i)
        update_gradient()
    return cb


controls = tk.Frame(root)
controls.pack(padx=10, pady=(0, 10), fill="x")

for i in range(len(palette[0])):
    row = tk.Frame(controls)
    row.pack(fill="x", pady=4)

    # pastille de la couleur i
    sw = tk.Label(row, width=3, relief="solid", bd=1, bg=hexcolor(palette[1][i]))
    sw.pack(side="left", padx=(0, 8))
    swatches.append(sw)

    # trois spinbox R / G / B
    for channel, name in enumerate(("R", "G", "B")):
        tk.Label(row, text=name).pack(side="left")
        var = tk.StringVar(value=str(palette[1][i][channel]))
        var.trace_add("write", make_rgb_cb(i, channel, var))
        ttk.Spinbox(row, from_=0, to=255, width=4,
                    textvariable=var).pack(side="left", padx=(0, 6))

    # scale de position
    tk.Label(row, text="pos").pack(side="left", padx=(8, 4))
    sc = tk.Scale(row, from_=0, to=1, resolution=0.01,
                  orient="horizontal", length=160)
    sc.set(palette[0][i])           # valeur initiale AVANT de brancher le command
    sc.config(command=make_pos_cb(i))
    sc.pack(side="left")


def sauvegarder():
    with open("couleur.txt", "w") as f:
        f.write(str(palette))


tk.Button(root, text="Sauvegarder", command=sauvegarder).pack(pady=(0, 10))

update_gradient()   # premier rendu
root.mainloop()