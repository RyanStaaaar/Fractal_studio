#!/usr/bin/env python3
"""
palette_calibrator.py — Interface de calibration des 348 palettes Sanzo Wada.

Pour chaque combinaison, teste toutes les permutations d'ordre des couleurs
et choisit le nombre de miroir (1–3). Les choix sont enregistrés dans
data/palette_calibration.xlsx.

Lancement :  myenv/bin/python palette_calibrator.py
"""

import itertools
import json
import tkinter as tk
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk
import openpyxl

import render
import fractal
import iteration

# ── Réglages du rendu de prévisualisation ─────────────────────────────────────
PREVIEW_W     = 580
PREVIEW_H     = int(PREVIEW_W * 1964 / 3024)   # ≈ 377, ratio 3024:1964
N_ITER        = 100
MODE          = "oklab"
SMOOTH        = True
EQUALIZE      = True
CLIP_LIMIT    = 5.5
TRAP_CX       = 0.0
TRAP_CY       = 0.0
TRAP_NORM_MAX = 0.5
JULIA_C       = complex(-0.7, 0.27)             # Julia canonique pour comparaisons cohérentes

EXCEL_PATH = Path(__file__).parent / "data" / "palette_calibration.xlsx"
HEADERS    = ["palette_index", "name", "perm_index", "color_order", "mirror", "validated_at"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_perms(colors: list) -> list[list]:
    return [list(p) for p in itertools.permutations(colors)]


def _hexcolor(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _make_palette(colors: list) -> list:
    n = len(colors)
    pos = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return [pos, [list(c) for c in colors]]


# ── Excel I/O ─────────────────────────────────────────────────────────────────

def _load_validated() -> dict[int, dict]:
    """Charge le fichier Excel → dict[palette_index → {perm_index, mirror}]."""
    if not EXCEL_PATH.exists():
        return {}
    wb = openpyxl.load_workbook(EXCEL_PATH)
    ws = wb.active
    result = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            result[int(row[0])] = {"perm_index": int(row[2]), "mirror": int(row[4])}
    return result


def _save_entry(idx: int, name: str, perm_index: int,
                color_order: list, mirror: int) -> None:
    EXCEL_PATH.parent.mkdir(exist_ok=True)
    if EXCEL_PATH.exists():
        wb = openpyxl.load_workbook(EXCEL_PATH)
        ws = wb.active
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(HEADERS)

    for row in ws.iter_rows(min_row=2):
        if row[0].value == idx:
            row[1].value = name
            row[2].value = perm_index
            row[3].value = json.dumps(color_order)
            row[4].value = mirror
            row[5].value = datetime.now().isoformat(timespec="seconds")
            wb.save(EXCEL_PATH)
            return

    ws.append([idx, name, perm_index, json.dumps(color_order),
               mirror, datetime.now().isoformat(timespec="seconds")])
    wb.save(EXCEL_PATH)


# ── Widget : frame scrollable ─────────────────────────────────────────────────

class _ScrollFrame(tk.Frame):
    """Frame avec scrollbar verticale interne."""
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        bg = kw.get("bg", "#1e1e1e")
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        sb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._win_id, width=e.width)

    def _on_mousewheel(self, e):
        self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")


# ── Application principale ────────────────────────────────────────────────────

class PaletteCalibrator:
    def __init__(self):
        self.palettes  = render.load_sanzo_palettes()
        self.names     = render.load_sanzo_names()
        self.validated = _load_validated()
        self.idx       = 0
        self._cache_V  = None   # champ scalaire mis en cache par palette

        # Trouver la première palette non validée
        for i in range(len(self.palettes)):
            if i not in self.validated:
                self.idx = i
                break

        self.root = tk.Tk()
        self.root.title("Calibration Palettes Sanzo Wada")
        self.root.configure(bg="#1e1e1e")
        self._build_ui()
        self._load_palette(self.idx)
        self.root.bind("<Return>", lambda _: self._validate())
        self.root.bind("<Right>",  lambda _: self._skip())
        self.root.bind("<Left>",   lambda _: self._prev())

    # ── Construction de l'UI ──────────────────────────────────────────────────

    def _build_ui(self):
        # ── Barre de navigation ──────────────────────────────────────────────
        nav = tk.Frame(self.root, bg="#1e1e1e")
        nav.pack(fill="x", padx=10, pady=(8, 0))

        tk.Button(nav, text="← Préc",  command=self._prev,
                  bg="#333", fg="white", relief="flat", padx=6).pack(side="left", padx=2)
        tk.Button(nav, text="Suiv →",  command=self._skip,
                  bg="#333", fg="white", relief="flat", padx=6).pack(side="left", padx=2)

        self._nav_label = tk.Label(nav, text="", bg="#1e1e1e", fg="#bbb",
                                    font=("Helvetica", 11))
        self._nav_label.pack(side="left", padx=14)

        self._status_label = tk.Label(nav, text="", bg="#1e1e1e", fg="#5a5",
                                       font=("Helvetica", 10))
        self._status_label.pack(side="right", padx=10)

        tk.Button(nav, text="✓  Valider & Suivant", command=self._validate,
                  bg="#2a7a2a", fg="white", font=("Helvetica", 11, "bold"),
                  relief="flat", padx=10).pack(side="right", padx=6)

        # ── Barre de progression ─────────────────────────────────────────────
        prog_bar_frame = tk.Frame(self.root, bg="#1e1e1e")
        prog_bar_frame.pack(fill="x", padx=10, pady=(4, 0))
        self._prog_label = tk.Label(prog_bar_frame, text="", bg="#1e1e1e",
                                     fg="#666", font=("Helvetica", 9))
        self._prog_label.pack(side="left")

        # ── Corps principal ──────────────────────────────────────────────────
        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Prévisualisation (gauche)
        preview_frame = tk.Frame(body, bg="#000", bd=1, relief="solid")
        preview_frame.pack(side="left", padx=(0, 10))
        self._canvas_lbl = tk.Label(preview_frame, bg="#000",
                                     width=PREVIEW_W, height=PREVIEW_H)
        self._canvas_lbl.pack()

        # Panneau de droite
        right = tk.Frame(body, bg="#1e1e1e")
        right.pack(side="left", fill="y")

        # Ordre des couleurs (scrollable)
        perm_outer = tk.LabelFrame(right, text="Ordre des couleurs",
                                    bg="#1e1e1e", fg="#ccc", bd=1,
                                    font=("Helvetica", 10, "bold"))
        perm_outer.pack(fill="both", expand=True, pady=(0, 8))
        self._scroll_frame = _ScrollFrame(perm_outer, bg="#1e1e1e",
                                           height=260, width=240)
        self._scroll_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self._perm_var = tk.IntVar(value=0)

        # Miroir
        mirror_frame = tk.LabelFrame(right, text="Miroir (repeat)",
                                      bg="#1e1e1e", fg="#ccc", bd=1,
                                      font=("Helvetica", 10, "bold"))
        mirror_frame.pack(fill="x")
        self._mirror_var = tk.IntVar(value=2)
        for v in (1, 2, 3):
            tk.Radiobutton(mirror_frame, text=str(v), variable=self._mirror_var,
                           value=v, bg="#1e1e1e", fg="white",
                           activebackground="#1e1e1e", activeforeground="white",
                           selectcolor="#333",
                           command=self._on_setting_change).pack(
                               side="left", padx=14, pady=6)

    # ── Chargement d'une palette ──────────────────────────────────────────────

    def _load_palette(self, idx: int):
        self.idx = idx % len(self.palettes)
        base_colors  = self.palettes[self.idx][1]
        self._perms  = _all_perms(base_colors)
        self._cache_V = None   # forcer recalcul pour la nouvelle palette

        saved = self.validated.get(self.idx)
        self._perm_var.set(saved["perm_index"] if saved else 0)
        self._mirror_var.set(saved["mirror"]      if saved else 2)

        self._rebuild_perm_radios()
        self._update_labels()
        self._refresh_preview()

    def _rebuild_perm_radios(self):
        for w in self._scroll_frame.inner.winfo_children():
            w.destroy()

        for pi, perm in enumerate(self._perms):
            row = tk.Frame(self._scroll_frame.inner, bg="#1e1e1e")
            row.pack(fill="x", padx=2, pady=1)
            tk.Radiobutton(row, variable=self._perm_var, value=pi,
                           bg="#1e1e1e", activebackground="#1e1e1e",
                           selectcolor="#333",
                           command=self._on_setting_change).pack(side="left")
            for rgb in perm:
                tk.Label(row, width=3, bg=_hexcolor(rgb),
                         relief="flat").pack(side="left", padx=1)
            # petit indicateur hex
            hex_str = "  " + "  ".join(_hexcolor(c) for c in perm)
            tk.Label(row, text=hex_str, bg="#1e1e1e", fg="#555",
                     font=("Courier", 8)).pack(side="left", padx=4)

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def _refresh_preview(self):
        pi     = self._perm_var.get()
        mirror = self._mirror_var.get()
        colors = self._perms[pi]

        if self._cache_V is None:
            gen = fractal.FractalGenerator(PREVIEW_H, PREVIEW_W, N_ITER, smooth=SMOOTH)
            poly = iteration.Poly(1, 0, JULIA_C)
            trap_params = np.array([TRAP_CX, TRAP_CY, 0.0])
            self._cache_V = gen.generate_julia_trap(
                poly, trap_type=0, trap_params=trap_params, norm_max=TRAP_NORM_MAX)

        palette  = _make_palette(colors)
        renderer = render.FractalRenderer(palette, mode=MODE, n_iter=N_ITER,
                                           repeat=mirror, equalize=EQUALIZE,
                                           clip_limit=CLIP_LIMIT)
        arr = renderer.render(self._cache_V)
        pil = Image.fromarray(arr)
        self._tk_img = ImageTk.PhotoImage(pil)
        self._canvas_lbl.config(image=self._tk_img)

    # ── Mise à jour des labels ────────────────────────────────────────────────

    def _update_labels(self):
        n    = len(self.palettes)
        done = len(self.validated)
        self._nav_label.config(
            text=f"{self.idx + 1} / {n}  ·  {self.names[self.idx]}")
        self._prog_label.config(
            text=f"{done} / {n} validées  ({100*done//n}%)")
        saved = self.validated.get(self.idx)
        if saved:
            self._status_label.config(
                text=f"✓  perm {saved['perm_index']}  ·  miroir {saved['mirror']}",
                fg="#5a5")
        else:
            self._status_label.config(text="non validé", fg="#666")

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_setting_change(self):
        self._refresh_preview()

    def _validate(self):
        pi     = self._perm_var.get()
        mirror = self._mirror_var.get()
        colors = self._perms[pi]
        _save_entry(self.idx, self.names[self.idx], pi, colors, mirror)
        self.validated[self.idx] = {"perm_index": pi, "mirror": mirror}
        self._update_labels()
        # Avancer à la prochaine palette non validée
        start = self.idx + 1
        for offset in range(len(self.palettes)):
            candidate = (start + offset) % len(self.palettes)
            if candidate not in self.validated:
                self._load_palette(candidate)
                return
        # Toutes validées
        self._load_palette(self.idx + 1)

    def _skip(self):
        self._load_palette(self.idx + 1)

    def _prev(self):
        self._load_palette(self.idx - 1)

    # ── Lancement ─────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    PaletteCalibrator().run()
