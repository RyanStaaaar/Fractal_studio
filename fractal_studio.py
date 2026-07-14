#!/usr/bin/env python3
"""
fractal_studio.py  —  Studio interactif de vidéos fractales (Julia / Mandelbrot).

Atelier modulaire : on assemble dans la zone du bas les modules qui animent le
rendu (chemin de c, orbit traps, biomorphe, palette, zoom, rotation), on règle
les courbes de vélocité, et on exporte en MP4.
"""

from __future__ import annotations

import cmath
import itertools
import math
import random
import re
import threading
import tkinter as tk
import tkinter.colorchooser as tkcc
import tkinter.filedialog as tkfd
import tkinter.ttk as ttk
from pathlib import Path

import numpy as np
from numba import njit, prange
from PIL import Image, ImageTk

import iteration
import orbit_trap
import render

# ── Constantes ────────────────────────────────────────────────────────────────
N_ITER  = 80
PREV_W  = 520
PREV_H  = int(PREV_W * 1964 / 3024)   # ~338

MANDEL_W    = 200
MANDEL_H    = 200
MANDEL_NPY  = Path(__file__).parent / "mandelbrot_map.npy"
MANDEL_BORN = 2.0

OTRAP_W    = 200
OTRAP_H    = 200
OTRAP_BORN = 2.0

ZOOM_W    = 200
ZOOM_H    = 200
ZOOM_BORN = 2.0    # demi-largeur du plan affiché dans la carte zoom

NORM_W          = 132    # largeur de la colonne norme
NORM_H          = 200    # hauteur = même que la carte OTRAP
NORM_MIN        = 0.05
NORM_MAX        = 2.0
NORM_SLIDER_TOP = 16     # y haut de la piste du slider
NORM_SLIDER_BOT = 122    # y bas de la piste du slider
KNOB_CX         = NORM_W // 2   # centre x du knob
KNOB_CY         = 160    # centre y du knob
KNOB_R          = 22     # rayon du knob

RAD_MIN = 0.05   # rayon min du trap cercle
RAD_MAX = 2.0    # rayon max du trap cercle


CLOSE_THRESH = 18
SAMPLE_DIST  =  4
CTRL_MASK    = 0x4

FPS_DEFAULT = 24
DUR_DEFAULT = 10.0

# Vitesse uniforme : plafond du coût DEM (1/distance à ∂M). Sans plafond, un
# chemin qui touche ou traverse M sature à 1/eps = 1e4 et monopolise la quasi
# totalité des frames — le reste du chemin est parcouru en 1-2 frames.
REPARAM_RHO_CAP     = 25.0
REPARAM_IMG_SAMPLES = 400   # previews rendues pour la métrique image
REPARAM_IMG_SIZE    = 128   # côté (px) des previews de la métrique image

# Éditeur de vélocités (lanes d'automation)
LANE_W = 560   # largeur du canvas d'une lane (px)
LANE_H = 80    # hauteur du canvas d'une lane (px)


def _lane_eval(pts: list, bends: list, t: float) -> float:
    """Évalue une courbe de vélocité en t ∈ [0,1].

    pts : points (t, v) triés, bornes t=0 et t=1 incluses.
    bends : courbure exponentielle par segment, k ∈ [-1, 1] (0 = linéaire)."""
    if t <= pts[0][0]:
        return pts[0][1]
    for i in range(len(pts) - 1):
        t0, v0 = pts[i]
        t1, v1 = pts[i + 1]
        if t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            k = bends[i] if i < len(bends) else 0.0
            if abs(k) > 1e-6:
                a = 4.0 * k
                u = (math.exp(a * u) - 1.0) / (math.exp(a) - 1.0)
            return v0 + u * (v1 - v0)
    return pts[-1][1]

BG     = "#1a1a1a"
BG2    = "#242424"
BG3    = "#2e2e2e"
FG     = "#dddddd"
FG2    = "#888888"
ACCENT = "#4a8abf"
GREEN  = "#2e6e2e"
RED    = "#6e2e2e"

PALETTES      = render.load_sanzo_palettes()
PALETTE_NAMES = render.load_sanzo_names()

# Sérialise les rendus numba entre threads (preview, export, métrique image) :
# la couche workqueue de numba (parallel=True) n'est pas thread-safe — deux
# régions parallèles concurrentes font avorter le process.
_RENDER_LOCK = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_palette(colors: list) -> list:
    n   = len(colors)
    pos = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return [pos, [list(c) for c in colors]]


def _hexcolor(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _load_texture(path: str, svg_side: int = 1024) -> np.ndarray:
    """Charge une texture RGBA uint8 pour les traps image / géométrique.

    PNG (et tout format PIL) via Pillow ; SVG rasterisé depuis sa géométrie
    vectorielle (svg_trap), plus grand côté = svg_side px."""
    if str(path).lower().endswith(".svg"):
        import svg_trap
        return svg_trap.load_svg(path, raster_side=svg_side)[1]
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


# ── Menu déroulant palette ────────────────────────────────────────────────────

class _PaletteDropdown:
    """Bouton-trigger + popup Canvas scrollable : index + swatches par ligne."""

    ROW_H         = 22
    SW_W          = 12      # largeur d'un swatch
    SW_H          = 10      # hauteur d'un swatch
    SW_PAD        = 2       # espacement entre swatches
    IDX_W         = 32      # colonne index (pixels)
    TRIGGER_SW_W  = 112     # largeur fixe de la zone swatches dans le trigger
    POPUP_W       = 196     # largeur totale popup (hors scrollbar)
    SB_W          = 14      # largeur scrollbar
    VIS_ROWS      = 14      # lignes visibles

    _BG_ROW   = "#242424"
    _BG_HOVER = "#2a4060"
    _BG_SEL   = "#1a3050"
    _FG_IDX   = "#666"
    _BORDER   = "#3a3a3a"

    def __init__(self, parent: tk.Widget, root: tk.Tk,
                 palettes: list, initial_idx: int = 0,
                 on_select=None):
        self._palettes  = palettes
        self._root      = root
        self._on_select = on_select
        self._sel       = initial_idx
        self._popup: tk.Toplevel | None = None
        self._cv:    tk.Canvas    | None = None
        self._hovered = -1
        self._root_bind: str | None = None

        # ── Trigger ──────────────────────────────────────────────────────────
        self._frame = tk.Frame(parent, bg=BG2, cursor="hand2",
                               highlightthickness=1,
                               highlightbackground=self._BORDER)
        inner = tk.Frame(self._frame, bg=BG2)
        inner.pack(padx=6, pady=4)

        self._idx_lbl = tk.Label(inner, text="", bg=BG2, fg=FG2,
                                  font=("Courier", 9), width=4, anchor="e")
        self._idx_lbl.pack(side="left")

        self._sw_cv = tk.Canvas(inner, bg=BG2,
                                 width=self.TRIGGER_SW_W,
                                 height=self.SW_H + 2,
                                 highlightthickness=0)
        self._sw_cv.pack(side="left", padx=(4, 2))

        self._arrow = tk.Label(inner, text="▾", bg=BG2, fg=FG2,
                                font=("Helvetica", 9))
        self._arrow.pack(side="left")

        # Bind click sur tous les enfants
        for w in (self._frame, inner, self._idx_lbl,
                  self._sw_cv, self._arrow):
            w.bind("<Button-1>", self._toggle)

        self._draw_trigger()

    # ── Public ───────────────────────────────────────────────────────────────

    def pack(self, **kw):
        self._frame.pack(**kw)

    @property
    def selected_index(self) -> int:
        return self._sel

    def set_index(self, idx: int):
        self._sel = idx % len(self._palettes)
        self._draw_trigger()

    # ── Trigger ──────────────────────────────────────────────────────────────

    def _draw_trigger(self):
        idx    = self._sel
        colors = self._palettes[idx][1]
        n      = len(colors)

        self._idx_lbl.config(text=f"{idx + 1}")

        # Largeur fixe — les swatches s'affichent de gauche a droite
        self._sw_cv.delete("all")
        for i, rgb in enumerate(colors):
            x0 = i * (self.SW_W + self.SW_PAD)
            self._sw_cv.create_rectangle(
                x0, 1, x0 + self.SW_W, 1 + self.SW_H,
                fill=_hexcolor(rgb), outline="")

    # ── Popup ─────────────────────────────────────────────────────────────────

    def _toggle(self, _event=None):
        if self._popup and self._popup.winfo_exists():
            self._close()
        else:
            self._open()

    def _open(self):
        self._frame.update_idletasks()
        rx = self._frame.winfo_rootx()
        ry = self._frame.winfo_rooty() + self._frame.winfo_height() + 1

        popup_h = self.ROW_H * self.VIS_ROWS
        total_w = self.POPUP_W + self.SB_W

        # Eviter le debordement hors ecran (winfo_width peut etre 1 avant rendu)
        screen_w = self._root.winfo_screenwidth()
        rx = min(rx, screen_w - total_w)
        rx = max(0, rx)

        self._popup = tk.Toplevel()
        self._popup.withdraw()
        self._popup.overrideredirect(True)
        self._popup.configure(bg=self._BORDER)
        self._popup.geometry(f"{total_w}x{popup_h}+{rx}+{ry}")

        cv = tk.Canvas(self._popup, bg=self._BG_ROW,
                        width=self.POPUP_W, highlightthickness=0)
        sb = tk.Scrollbar(self._popup, orient="vertical",
                           command=cv.yview, width=self.SB_W)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        self._cv = cv

        # Dessiner toutes les lignes
        total_h = len(self._palettes) * self.ROW_H
        cv.configure(scrollregion=(0, 0, self.POPUP_W, total_h))

        for i, (_, colors) in enumerate(self._palettes):
            y   = i * self.ROW_H
            bg  = self._BG_SEL if i == self._sel else self._BG_ROW
            # Rectangle de fond (tag individuel)
            cv.create_rectangle(0, y, self.POPUP_W, y + self.ROW_H,
                                  fill=bg, outline="", tags=f"bg{i}")
            # Index
            cv.create_text(self.IDX_W - 4, y + self.ROW_H // 2,
                            text=str(i + 1), anchor="e",
                            fill=self._FG_IDX, font=("Courier", 8),
                            tags=f"lbl{i}")
            # Swatches
            sx = self.IDX_W + 4
            for rgb in colors:
                cv.create_rectangle(
                    sx, y + (self.ROW_H - self.SW_H) // 2,
                    sx + self.SW_W,
                    y + (self.ROW_H - self.SW_H) // 2 + self.SW_H,
                    fill=_hexcolor(rgb), outline="",
                    tags=f"sw{i}")
                sx += self.SW_W + self.SW_PAD

        # Scroll vers la selection
        frac = max(0.0, (self._sel / len(self._palettes)) - 0.15)
        cv.yview_moveto(frac)

        cv.bind("<Motion>",   self._on_hover)
        cv.bind("<Leave>",    self._on_leave)
        cv.bind("<Button-1>", self._on_click)

        # Scroll trackpad : bind sur le canvas ET la scrollbar + le popup
        # (le trackpad envoie de nombreux petits delta, on utilise juste le signe)
        def _scroll(e):
            cv.yview_scroll(-1 if e.delta > 0 else 1, "units")

        cv.bind("<MouseWheel>",           _scroll)
        sb.bind("<MouseWheel>",           _scroll)
        self._popup.bind("<MouseWheel>",  _scroll)

        # Fermer sur clic exterieur
        self._root_bind = self._root.bind(
            "<ButtonPress>", self._check_outside, "+")

        self._popup.deiconify()
        self._popup.lift()
        self._arrow.config(text="▴")

    def _close(self):
        if self._root_bind:
            try:
                self._root.unbind("<ButtonPress>", self._root_bind)
            except Exception:
                pass
            self._root_bind = None
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        self._popup   = None
        self._cv      = None
        self._hovered = -1
        self._arrow.config(text="▾")

    def _check_outside(self, event):
        if not (self._popup and self._popup.winfo_exists()):
            return
        px = self._popup.winfo_rootx()
        py = self._popup.winfo_rooty()
        pw = self._popup.winfo_width()
        ph = self._popup.winfo_height()
        if not (px <= event.x_root < px + pw and
                py <= event.y_root < py + ph):
            self._close()

    # ── Interactions popup ────────────────────────────────────────────────────

    def _row_at(self, event_y: int) -> int:
        if self._cv is None:
            return -1
        return int(self._cv.canvasy(event_y)) // self.ROW_H

    def _on_hover(self, event):
        row = self._row_at(event.y)
        if row == self._hovered or self._cv is None:
            return
        if 0 <= self._hovered < len(self._palettes):
            prev_bg = self._BG_SEL if self._hovered == self._sel else self._BG_ROW
            self._cv.itemconfig(f"bg{self._hovered}", fill=prev_bg)
        self._hovered = row
        if 0 <= row < len(self._palettes):
            self._cv.itemconfig(f"bg{row}", fill=self._BG_HOVER)

    def _on_leave(self, _event):
        if self._cv and 0 <= self._hovered < len(self._palettes):
            bg = self._BG_SEL if self._hovered == self._sel else self._BG_ROW
            self._cv.itemconfig(f"bg{self._hovered}", fill=bg)
        self._hovered = -1

    def _on_click(self, event):
        row = self._row_at(event.y)
        if 0 <= row < len(self._palettes):
            self._sel = row
            self._draw_trigger()
            self._close()
            if self._on_select:
                self._on_select(row)


# ── Formules Julia libres ─────────────────────────────────────────────────────

_FORMULA_GLOBALS: dict = {
    'np'     : np,
    'sin'    : np.sin,   'cos'    : np.cos,   'tan'    : np.tan,
    'sinh'   : np.sinh,  'cosh'   : np.cosh,  'tanh'   : np.tanh,
    'asin'   : np.arcsin,'acos'   : np.arccos,'atan'   : np.arctan,
    'arcsin' : np.arcsin,'arccos' : np.arccos,'arctan' : np.arctan,
    'asinh'  : np.arcsinh,'acosh' : np.arccosh,'atanh' : np.arctanh,
    'arcsinh': np.arcsinh,'arccosh':np.arccosh,'arctanh':np.arctanh,
    'exp'    : np.exp,   'log'    : np.log,   'log10'  : np.log10,
    'log2'   : np.log2,  'sqrt'   : np.sqrt,  'abs'    : np.abs,
    'angle'  : np.angle, 'conj'   : np.conj,
    'real'   : np.real,  'imag'   : np.imag,
    'pi'     : np.pi,    'e'      : np.e,     'i'      : 1j,
}
_JULIA_ITER_CACHE: dict = {}

_DEFAULT_FORMULA_NORMALIZED = 'z**2+c'

def _normalize_formula(formula: str) -> str:
    return formula.strip().replace('^', '**')

def _compile_julia_iter(formula: str):
    """Compile une formule libre en fonction vectorisée f(z_arr, c) -> arr."""
    key = _normalize_formula(formula).replace(' ', '')
    if key in _JULIA_ITER_CACHE:
        return _JULIA_ITER_CACHE[key]
    code = f"def _f(z, c):\n    return {_normalize_formula(formula)}\n"
    globs = dict(_FORMULA_GLOBALS)
    exec(code, globs)
    fn = globs['_f']
    _JULIA_ITER_CACHE[key] = fn
    return fn


# ── Compilation numba des formules libres ─────────────────────────────────────
# Le moteur numpy vectorisé plafonne à 2-9 fps sur la preview ; les kernels
# générés ci-dessous (expression inlinée, parallel=True) retrouvent la vitesse
# du chemin z²+c. En cas d'échec de compilation, repli silencieux sur numpy.

def _w(f):
    """Wrapper njit : promeut l'argument en complexe puis applique f (cmath)."""
    return njit(lambda z, _f=f: _f(z + 0j))

_NUMBA_FORMULA_GLOBALS: dict = {
    'np': np, 'math': math, 'prange': prange,
    'sin'    : _w(cmath.sin),   'cos'    : _w(cmath.cos),
    'tan'    : _w(cmath.tan),
    'sinh'   : _w(cmath.sinh),  'cosh'   : _w(cmath.cosh),
    'tanh'   : _w(cmath.tanh),
    'asin'   : _w(cmath.asin),  'acos'   : _w(cmath.acos),
    'atan'   : _w(cmath.atan),
    'arcsin' : _w(cmath.asin),  'arccos' : _w(cmath.acos),
    'arctan' : _w(cmath.atan),
    'asinh'  : _w(cmath.asinh), 'acosh'  : _w(cmath.acosh),
    'atanh'  : _w(cmath.atanh),
    'arcsinh': _w(cmath.asinh), 'arccosh': _w(cmath.acosh),
    'arctanh': _w(cmath.atanh),
    'exp'    : _w(cmath.exp),   'log'    : _w(cmath.log),
    'log10'  : _w(cmath.log10),
    'log2'   : njit(lambda z: cmath.log(z + 0j) / 0.6931471805599453),
    'sqrt'   : _w(cmath.sqrt),  'abs': abs,
    'angle'  : njit(lambda z: math.atan2((z + 0j).imag, (z + 0j).real)),
    'conj'   : njit(lambda z: complex((z + 0j).real, -(z + 0j).imag)),
    'real'   : njit(lambda z: (z + 0j).real),
    'imag'   : njit(lambda z: (z + 0j).imag),
    'pi'     : math.pi, 'e': math.e, 'i': 1j,
    '_dist_point' : orbit_trap._dist_point,
    '_dist_line'  : orbit_trap._dist_line,
    '_dist_cross' : orbit_trap._dist_cross,
    '_dist_circle': orbit_trap._dist_circle,
    '_dist_square': orbit_trap._dist_square,
    '_dist_ring'  : orbit_trap._dist_ring,
    '_dist_sinus' : orbit_trap._dist_sinus,
}

_NUMBA_KERNEL_SRC = '''
def _esc_kernel(Z, c, n, B, smooth, mandel):
    H, W = Z.shape
    V = np.zeros((H, W))
    logB = math.log(B)
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            v = 0.0
            for it in range(n):
                z = (EXPR) + 0j
                zr = z.real
                zi = z.imag
                if zr != zr or zi != zi:
                    break                      # NaN : orbite morte
                m2 = zr * zr + zi * zi
                if m2 > B2:
                    if smooth:
                        lz = 0.5 * math.log(m2)
                        if lz / logB > 1e-12:
                            si = it + 1 - math.log2(lz / logB)
                        else:
                            si = it + 1.0
                        v = min(max(1.0 - si / n, 0.0), 1.0)
                    else:
                        v = (n - it) / n
                    break
            V[y, x] = v
    return V


def _svg_kernel(Z, c, edges, offs, bboxes, colors, view, rect,
                n, B, min_iter, angle, mandel):
    H, W = Z.shape
    re_c = (rect[0] + rect[1]) * 0.5
    im_c = (rect[2] + rect[3]) * 0.5
    re_w = rect[1] - rect[0]
    im_h = rect[3] - rect[2]
    half_w = re_w * 0.5
    half_h = im_h * 0.5
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    out = np.zeros((H, W, 4), dtype=np.uint8)
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            for it in range(n):
                z = (EXPR) + 0j
                zr = z.real
                zi = z.imag
                if zr != zr or zi != zi:
                    break
                if it >= min_iter:
                    dre = zr - re_c
                    dim = zi - im_c
                    local_re = dre * cos_a + dim * sin_a
                    local_im = -dre * sin_a + dim * cos_a
                    if (-half_w <= local_re <= half_w
                            and -half_h <= local_im <= half_h):
                        u = (local_re + half_w) / re_w
                        v = (half_h - local_im) / im_h
                        sx = view[0] + u * view[2]
                        sy = view[1] + v * view[3]
                        s = _svg_shape_at(edges, offs, bboxes, sx, sy)
                        if s >= 0:
                            out[y, x, 0] = colors[s, 0]
                            out[y, x, 1] = colors[s, 1]
                            out[y, x, 2] = colors[s, 2]
                            out[y, x, 3] = 255
                            break
                if zr * zr + zi * zi > B2:
                    break
    return out


def _svg_geom_kernel(Z, c, edges, offs, bboxes, colors, view,
                     N, r, cx, cy, base_size, angle_step, n, B, mandel):
    H, W = Z.shape
    aspect = view[3] / view[2]
    scales = np.empty(N, dtype=np.float64)
    cx_ks  = np.empty(N, dtype=np.float64)
    cy_ks  = np.empty(N, dtype=np.float64)
    sc = 1.0
    for k in range(N):
        scales[k] = sc
        ang = float(k + 1) * angle_step
        cx_ks[k] = cx + base_size * sc * math.sin(ang)
        cy_ks[k] = cy + base_size * sc * (1.0 - math.cos(ang))
        sc *= r
    out = np.zeros((H, W, 4), dtype=np.uint8)
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            hit = False
            for it in range(n):
                z = (EXPR) + 0j
                zr = z.real
                zi = z.imag
                if zr != zr or zi != zi:
                    break
                for k in range(N):
                    hw = base_size * scales[k] * 0.5
                    hh = hw * aspect
                    dre = zr - cx_ks[k]
                    dim = zi - cy_ks[k]
                    if -hw <= dre <= hw and -hh <= dim <= hh:
                        u = (dre + hw) / (2.0 * hw)
                        v = (hh - dim) / (2.0 * hh)
                        sx = view[0] + u * view[2]
                        sy = view[1] + v * view[3]
                        s = _svg_shape_at(edges, offs, bboxes, sx, sy)
                        if s >= 0:
                            out[y, x, 0] = colors[s, 0]
                            out[y, x, 1] = colors[s, 1]
                            out[y, x, 2] = colors[s, 2]
                            out[y, x, 3] = 255
                            hit = True
                            break
                if hit:
                    break
                if zr * zr + zi * zi > B2:
                    break
    return out


def _trap_kernel(Z, c, trap_type, trap_params, n, B, norm_max, mandel):
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            min_d = 1e18
            for it in range(n):
                z = (EXPR) + 0j
                zr = z.real
                zi = z.imag
                if zr != zr or zi != zi:
                    break
                if trap_type == 0:
                    d = _dist_point(zr, zi, trap_params[0], trap_params[1])
                elif trap_type == 1:
                    d = _dist_line(zr - trap_params[0], zi - trap_params[1],
                                   trap_params[2])
                elif trap_type == 2:
                    d = _dist_cross(zr - trap_params[0], zi - trap_params[1])
                elif trap_type == 3:
                    d = _dist_circle(zr, zi, trap_params[0], trap_params[1],
                                     trap_params[2])
                elif trap_type == 4:
                    d = _dist_square(zr, zi, trap_params[0], trap_params[1],
                                     trap_params[2])
                elif trap_type == 6:
                    d = _dist_ring(zr, zi, trap_params[0], trap_params[1],
                                   trap_params[2], trap_params[3],
                                   trap_params[4])
                else:
                    d = _dist_sinus(zr, zi, trap_params[0], trap_params[1],
                                    trap_params[2], trap_params[3])
                if d < min_d:
                    min_d = d
                if zr * zr + zi * zi > B2:
                    break
            V[y, x] = 1.0 - math.exp(-min_d / norm_max)
    return V


def _bio_kernel(Z, c, n, mod_bail, L, color_by_iter, mandel):
    # Biomorphe de Pickover appliqué à la formule (EXPR). Classification OU
    # composante sur le z final ; V = log(1+min(|re|,|im|)) ou escape lissé.
    H, W = Z.shape
    V = np.zeros((H, W))
    mb2 = mod_bail * mod_bail
    logB = math.log(mod_bail)
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            escaped = False
            it_esc = n
            for it in range(n):
                z = (EXPR) + 0j
                zr = z.real
                zi = z.imag
                if zr != zr or zi != zi:
                    break
                if zr * zr + zi * zi > mb2:
                    escaped = True
                    it_esc = it
                    break
            are = abs(z.real)
            aim = abs(z.imag)
            q = are if are < aim else aim
            if color_by_iter:
                if escaped:
                    m2 = z.real * z.real + z.imag * z.imag
                    if m2 > 1.0:
                        lz = 0.5 * math.log(m2)
                        if lz / logB > 1e-12:
                            si = it_esc + 1 - math.log2(lz / logB)
                        else:
                            si = it_esc + 1.0
                    else:
                        si = it_esc + 1.0
                    V[y, x] = min(max(1.0 - si / n, 0.0), 1.0)
                else:
                    V[y, x] = 0.0
            else:
                V[y, x] = math.log(1.0 + q)
    return V
'''

_JULIA_NUMBA_CACHE: dict = {}

def _compile_julia_numba(formula: str):
    """Compile la formule en kernels numba (escape, trap scalaire, trap SVG,
    trap SVG géométrique) — lève si numba ne peut pas typer l'expression ;
    l'appelant replie alors sur numpy."""
    key = _normalize_formula(formula).replace(' ', '')
    if key in _JULIA_NUMBA_CACHE:
        return _JULIA_NUMBA_CACHE[key]
    import svg_trap
    globs = dict(_NUMBA_FORMULA_GLOBALS)
    globs['_svg_shape_at'] = svg_trap._svg_shape_at
    exec(_NUMBA_KERNEL_SRC.replace('EXPR', _normalize_formula(formula)), globs)
    esc      = njit(parallel=True, error_model='numpy')(globs['_esc_kernel'])
    trap     = njit(parallel=True, error_model='numpy')(globs['_trap_kernel'])
    svg_img  = njit(parallel=True, error_model='numpy')(globs['_svg_kernel'])
    svg_geom = njit(parallel=True, error_model='numpy')(globs['_svg_geom_kernel'])
    bio      = njit(parallel=True, error_model='numpy')(globs['_bio_kernel'])
    # Typage immédiat sur une grille minuscule : échec précoce si la formule
    # n'est pas supportée par numba. Sous verrou : ces kernels sont parallèles.
    Zs = np.zeros((4, 4), dtype=np.complex128)
    ed = np.zeros((1, 4)); of = np.array([0, 1], dtype=np.int64)
    bb = np.zeros((1, 4)); co = np.zeros((1, 3), dtype=np.uint8)
    vw = np.array([0.0, 0.0, 1.0, 1.0]); rc = np.array([0.0, 1.0, 0.0, 1.0])
    with _RENDER_LOCK:
        esc(Zs, 0j, 2, 256.0, True, False)
        trap(Zs, 0j, 0, np.zeros(5), 2, 256.0, 1.0, False)
        svg_img(Zs, 0j, ed, of, bb, co, vw, rc, 2, 256.0, 1, 0.0, False)
        svg_geom(Zs, 0j, ed, of, bb, co, vw, 2, 0.5, 0.0, 0.0, 1.0, 0.0,
                 2, 256.0, False)
        bio(Zs, 0j, 2, 100.0, 10.0, False, False)
    _JULIA_NUMBA_CACHE[key] = (esc, trap, svg_img, svg_geom, bio)
    return _JULIA_NUMBA_CACHE[key]

def _trap_julia_numpy(Z: np.ndarray, iter_fn, c: complex,
                      trap_params: np.ndarray,
                      n: int = 80, B: float = 256.0,
                      norm_max: float = 1.0,
                      trap_type: int = 0, mandel: bool = False) -> np.ndarray:
    """Orbit trap avec formule arbitraire — numpy vectorisé, supporte toute expression.
    trap_type 0 = point, 1 = droite (angle [2]), 2 = croix, 3 = cercle
    (rayon [2]), 5 = sinus (amplitude [2], fréquence [3]),
    6 = anneau 3D (rayon [2], inclinaison [3], axe [4], radians).
    mandel : plan des c (z₀=0, c=grille) au lieu du plan des z."""
    if mandel:
        z = np.zeros_like(Z); c = Z
    else:
        z = Z.copy()
    trap   = complex(trap_params[0], trap_params[1])
    radius = float(trap_params[2])
    if trap_type == 1:
        sin_a, cos_a = math.sin(radius), math.cos(radius)   # [2] = angle
    elif trap_type == 5:
        amp, freq = radius, float(trap_params[3])
    if trap_type == 6:
        tilt = float(trap_params[3])
        phi  = float(trap_params[4])
        ea   = radius                                     # demi-grand axe
        eb   = max(radius * abs(math.cos(tilt)), 1e-4 * radius)
        rot  = complex(math.cos(phi + math.pi / 2), -math.sin(phi + math.pi / 2))
    min_d = np.full(Z.shape, 1e18, dtype=np.float64)
    # Orbites suivies tant qu'elles ne sont ni échappées ni NaN — même
    # convention que le kernel canonique orbit_trap.trap_julia.
    alive = np.ones(Z.shape, dtype=bool)
    with np.errstate(all='ignore'):
        for _ in range(n):
            zn = np.asarray(iter_fn(z, c), dtype=np.complex128)
            z  = np.where(alive, zn, z)
            alive &= np.isfinite(z.real) & np.isfinite(z.imag)
            np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            if trap_type == 6:
                p  = (z - trap) * rot                     # repère de l'ellipse
                px = np.abs(p.real)
                py = np.abs(p.imag)
                k0 = np.hypot(px / ea, py / eb)
                k1 = np.hypot(px / (ea * ea), py / (eb * eb))
                d  = np.where(k1 < 1e-12, eb,
                              np.abs(k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)))
            elif trap_type == 1:
                d = np.abs((z.real - trap.real) * sin_a
                           - (z.imag - trap.imag) * cos_a)
            elif trap_type == 2:
                d = np.minimum(np.abs(z.real - trap.real),
                               np.abs(z.imag - trap.imag))
            elif trap_type == 5:
                d = np.abs(z.imag - trap.imag
                           - amp * np.sin(freq * (z.real - trap.real)))
            else:
                d = np.abs(z - trap)
                if trap_type == 3:
                    d = np.abs(d - radius)
            np.minimum(min_d, d, out=min_d, where=alive)
            alive &= np.abs(z) <= B
            if not alive.any():
                break
    return 1.0 - np.exp(-min_d / norm_max)


def _trap_image_numpy(Z: np.ndarray, iter_fn, c: complex,
                      tex: np.ndarray, rect: np.ndarray,
                      n: int = 80, B: float = 256.0,
                      min_iter: int = 2, angle: float = 0.0,
                      mandel: bool = False) -> np.ndarray:
    """Trap image avec formule arbitraire — équivalent numpy de
    orbit_trap.trap_image_julia : RGBA, alpha=0 si l'orbite ne touche pas."""
    H, W = Z.shape
    TH, TW = tex.shape[0], tex.shape[1]
    out  = np.zeros((H, W, 4), dtype=np.uint8)
    if mandel:
        z = np.zeros_like(Z); c = Z
    else:
        z = Z.copy()
    re_c   = (rect[0] + rect[1]) * 0.5
    im_c   = (rect[2] + rect[3]) * 0.5
    re_w   = rect[1] - rect[0]
    im_h   = rect[3] - rect[2]
    half_w = re_w * 0.5
    half_h = im_h * 0.5
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    alive = np.ones((H, W), dtype=bool)   # ni piégé ni échappé
    with np.errstate(all='ignore'):
        for it in range(n):
            zn = np.asarray(iter_fn(z, c), dtype=np.complex128)
            z  = np.where(alive, zn, z)
            np.nan_to_num(z, copy=False, nan=1e6, posinf=1e6, neginf=-1e6)
            if it >= min_iter:
                dre = z.real - re_c
                dim = z.imag - im_c
                lre = dre * cos_a + dim * sin_a    # repère du rectangle
                lim = -dre * sin_a + dim * cos_a
                inside = (alive & (np.abs(lre) <= half_w)
                          & (np.abs(lim) <= half_h))
                if inside.any():
                    tj = np.clip(np.round((lre[inside] + half_w)
                                          / re_w * (TW - 1)).astype(np.int64),
                                 0, TW - 1)
                    ti = np.clip(np.round((half_h - lim[inside])
                                          / im_h * (TH - 1)).astype(np.int64),
                                 0, TH - 1)
                    texel  = tex[ti, tj]
                    opaque = texel[:, 3] != 0
                    iy, ix = np.nonzero(inside)
                    iy, ix = iy[opaque], ix[opaque]
                    out[iy, ix, :3] = texel[opaque, :3]
                    out[iy, ix, 3]  = 255
                    alive[iy, ix]   = False
            alive &= np.abs(z) <= B
            if not alive.any():
                break
    return out


def _trap_geom_numpy(Z: np.ndarray, iter_fn, c: complex,
                     tex: np.ndarray, N: int, r: float,
                     cx: float, cy: float,
                     base_size: float, angle_step: float,
                     n: int = 80, B: float = 256.0,
                     mandel: bool = False) -> np.ndarray:
    """Trap série géométrique avec formule arbitraire — équivalent numpy de
    orbit_trap.trap_image_geom_series_julia (N copies en spirale, premier
    contact gagne, dès l'itération 0)."""
    H, W = Z.shape
    TH, TW = tex.shape[0], tex.shape[1]
    aspect = float(TH) / float(TW)
    out   = np.zeros((H, W, 4), dtype=np.uint8)
    if mandel:
        z = np.zeros_like(Z); c = Z
    else:
        z = Z.copy()
    alive = np.ones((H, W), dtype=bool)
    # Échelles et centres des copies (cf. kernel : sin / 1-cos)
    scales, cxs, cys = [], [], []
    sc = 1.0
    for k in range(N):
        ang = float(k + 1) * angle_step
        scales.append(sc)
        cxs.append(cx + base_size * sc * math.sin(ang))
        cys.append(cy + base_size * sc * (1.0 - math.cos(ang)))
        sc *= r
    with np.errstate(all='ignore'):
        for _ in range(n):
            zn = np.asarray(iter_fn(z, c), dtype=np.complex128)
            z  = np.where(alive, zn, z)
            np.nan_to_num(z, copy=False, nan=1e6, posinf=1e6, neginf=-1e6)
            for k in range(N):
                hw = base_size * scales[k] * 0.5
                hh = hw * aspect
                dre = z.real - cxs[k]
                dim = z.imag - cys[k]
                inside = (alive & (np.abs(dre) <= hw) & (np.abs(dim) <= hh))
                if not inside.any():
                    continue
                tj = np.clip(np.round((dre[inside] + hw) / (2 * hw)
                                      * (TW - 1)).astype(np.int64), 0, TW - 1)
                ti = np.clip(np.round((hh - dim[inside]) / (2 * hh)
                                      * (TH - 1)).astype(np.int64), 0, TH - 1)
                texel  = tex[ti, tj]
                opaque = texel[:, 3] != 0
                iy, ix = np.nonzero(inside)
                iy, ix = iy[opaque], ix[opaque]
                out[iy, ix, :3] = texel[opaque, :3]
                out[iy, ix, 3]  = 255
                alive[iy, ix]   = False
            alive &= np.abs(z) <= B
            if not alive.any():
                break
    return out


def _escape_julia_numpy(Z: np.ndarray, iter_fn, c: complex,
                        n: int = 80, B: float = 256.0,
                        smooth: bool = True,
                        mandel: bool = False) -> np.ndarray:
    """Escape time avec formule arbitraire — numpy vectorisé.
    Même convention que iteration.escape_speed : V dans [0,1], 0 = non échappé."""
    if mandel:
        z = np.zeros_like(Z, dtype=np.complex128); c = Z
    else:
        z = Z.astype(np.complex128)
    V     = np.zeros(Z.shape, dtype=np.float64)
    alive = np.ones(Z.shape, dtype=bool)
    logB  = math.log(B)
    with np.errstate(all='ignore'):
        for i in range(n):
            zn = np.asarray(iter_fn(z, c), dtype=np.complex128)
            z  = np.where(alive, zn, z)
            np.nan_to_num(z, copy=False, nan=1e6, posinf=1e6, neginf=-1e6)
            mag = np.abs(z)
            esc = alive & (mag > B)
            if esc.any():
                if smooth:
                    log_zn   = np.log(np.maximum(mag[esc], 1e-12))
                    smooth_i = i + 1 - np.log2(np.maximum(log_zn / logB, 1e-12))
                    V[esc]   = np.clip(1.0 - smooth_i / n, 0.0, 1.0)
                else:
                    V[esc] = (n - i) / n
                alive &= ~esc
            if not alive.any():
                break
    return V


def _biomorph_formula_numpy(Z: np.ndarray, iter_fn, c: complex,
                            n: int = 50, mod_bail: float = 100.0,
                            L: float = 10.0, mandel: bool = False,
                            color_by_iter: bool = False) -> np.ndarray:
    """Biomorphe de Pickover avec formule arbitraire — repli numpy vectorisé.
    Classification OU sur le z final (cf. iteration.biomorph)."""
    if mandel:
        z = np.zeros_like(Z, dtype=np.complex128); c = Z
    else:
        z = Z.astype(np.complex128)
    alive  = np.ones(Z.shape, dtype=bool)
    it_esc = np.full(Z.shape, n, dtype=np.float64)
    escd   = np.zeros(Z.shape, dtype=bool)
    logB   = math.log(mod_bail)
    with np.errstate(all='ignore'):
        for i in range(n):
            zn = np.asarray(iter_fn(z, c), dtype=np.complex128)
            z  = np.where(alive, zn, z)
            bad = ~(np.isfinite(z.real) & np.isfinite(z.imag))
            alive &= ~bad                       # NaN/inf : orbite morte
            np.nan_to_num(z, copy=False, nan=1e6, posinf=1e6, neginf=-1e6)
            esc = alive & (np.abs(z) > mod_bail)
            if esc.any():
                it_esc[esc] = i
                escd[esc]   = True
                alive &= ~esc
            if not alive.any():
                break
    are = np.abs(z.real)
    aim = np.abs(z.imag)
    q   = np.minimum(are, aim)
    if color_by_iter:
        V = np.zeros(Z.shape, dtype=np.float64)
        mag = np.abs(z)
        log_zn   = np.log(np.maximum(mag[escd], 1e-12))
        smooth_i = it_esc[escd] + 1 - np.log2(np.maximum(log_zn / logB, 1e-12))
        V[escd]  = np.clip(1.0 - smooth_i / n, 0.0, 1.0)
        return V
    return np.log1p(q)


# ── Rendu d'une frame ─────────────────────────────────────────────────────────

def render_frame(t: float, duration: float,
                 width: int, height: int,
                 c: complex,
                 colors: list,
                 mirror: int = 3,
                 equalize: bool = True,
                 clip_limit: float = 3.0,
                 trap_x: float = 0.0,
                 trap_y: float = 0.0,
                 trap_norm: float = 0.5,
                 color_mode: str = "oklab",
                 view_rot_deg: float = 0.0,
                 julia_iter_fn=None,
                 use_trap: bool = True,
                 smooth: bool = True,
                 trap_type: int = 0,
                 trap_radius: float = 0.0,
                 trap_tilt: float = 0.0,
                 trap_axis: float = 0.0,
                 trap_tex: np.ndarray | None = None,
                 julia_numba: tuple | None = None,
                 trap_count: float = 0.0,
                 trap_svg: tuple | None = None,
                 aa: int = 1,
                 mandelbrot: bool = False,
                 zoom: float = 1.0,
                 center_re: float = 0.0,
                 center_im: float = 0.0,
                 biomorph: bool = False,
                 bio_L: float = 10.0,
                 bio_modbail: float = 100.0,
                 bio_iter: int = 50,
                 bio_color_iter: bool = False) -> np.ndarray:
    # Anti-crénelage : rendu à aa× puis moyennage (chaque pixel final reçoit
    # aa² orbites au lieu d'une décision binaire piégé/pas piégé).
    out_w, out_h = width, height
    if aa > 1:
        width, height = width * aa, height * aa
    # Module zoom : demi-largeur de la vue = 2 / facteur, recentrée sur (cx, cy)
    half_x  = 2.0 / max(zoom, 1e-12)
    half_y  = half_x * height / width
    xs = np.linspace(-half_x, half_x, width)
    ys = np.linspace(-half_y, half_y, height)
    Z  = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    if view_rot_deg:
        theta = math.radians(view_rot_deg)
        Z = Z * complex(math.cos(theta), -math.sin(theta))
    Z = Z + complex(center_re, center_im)

    # Mode biomorphe de Pickover : classification OU sur le z final de la
    # FORMULE Julia active (mêmes 3 moteurs que le rendu normal), colorisée
    # par la même palette OKLab. `c` est celui du module c (animé) le cas
    # échéant. Court-circuite escape/trap sans les casser (OFF = inchangé).
    if biomorph:
        if julia_iter_fn is None:
            V, _mask = iteration.biomorph(
                Z, c, bio_iter, bio_modbail, bio_L, mandelbrot, bio_color_iter)
        elif julia_numba is not None:
            V = julia_numba[4](np.ascontiguousarray(Z), c, bio_iter,
                               bio_modbail, bio_L, bio_color_iter, mandelbrot)
        else:
            V = _biomorph_formula_numpy(Z, julia_iter_fn, c, bio_iter,
                                        bio_modbail, bio_L, mandelbrot,
                                        bio_color_iter)
        palette  = _make_palette(colors)
        renderer = render.FractalRenderer(
            palette, mode=color_mode, n_iter=N_ITER, repeat=mirror,
            equalize=equalize, clip_limit=clip_limit,
            # color_by_iter borne déjà V dans [0,1] ; le champ structure est
            # à plage variable → égalisation automatique.
            eq_range=(0.0, 1.0) if bio_color_iter else None)
        out = renderer.render(V)
        if aa > 1:
            out = render.downscale(out, out_w, out_h)
        return out

    # Les traps image (type 7) et géométrique (type 8) se compositent sur
    # un fond escape time
    scalar_trap = use_trap and trap_type not in (7, 8)
    if scalar_trap:
        trap_params = np.array([trap_x, trap_y, trap_radius, trap_tilt, trap_axis])
        if julia_iter_fn is None and mandelbrot:
            V = orbit_trap.trap_mandelbrot(Z, trap_type, trap_params,
                                           N_ITER, 256.0, trap_norm)
        elif julia_iter_fn is None:
            V = orbit_trap.trap_julia(Z, 1 + 0j, 0 + 0j, c,
                                      trap_type, trap_params,
                                      N_ITER, 256.0, trap_norm)
        elif julia_numba is not None:
            # Formule libre compilée en kernel numba (rapide)
            V = julia_numba[1](np.ascontiguousarray(Z), c, trap_type,
                               trap_params, N_ITER, 256.0, trap_norm,
                               mandelbrot)
        else:
            V = _trap_julia_numpy(Z, julia_iter_fn, c, trap_params,
                                  N_ITER, 256.0, trap_norm, trap_type,
                                  mandelbrot)
    else:
        # Mode escape time classique (pas de module orbit trap, ou fond du
        # trap image)
        if julia_iter_fn is None and mandelbrot:
            V = iteration.mandelbrot(Z, 0 + 0j, N_ITER, 256.0, smooth)
        elif julia_iter_fn is None:
            V = iteration.escape_speed(Z, 1 + 0j, 0 + 0j, c,
                                       N_ITER, 256.0, smooth)
        elif julia_numba is not None:
            V = julia_numba[0](np.ascontiguousarray(Z), c,
                               N_ITER, 256.0, smooth, mandelbrot)
        else:
            V = _escape_julia_numpy(Z, julia_iter_fn, c,
                                    N_ITER, 256.0, smooth, mandelbrot)
    palette  = _make_palette(colors)
    renderer = render.FractalRenderer(
        palette, mode=color_mode, n_iter=N_ITER,
        repeat=mirror, equalize=equalize, clip_limit=clip_limit,
        # Escape time : plage fixe pour que l'égalisation ne saute pas
        # d'une frame à l'autre quand les extrêmes de V fluctuent.
        eq_range=None if scalar_trap else (0.0, 1.0),
    )
    out = renderer.render(V)

    if use_trap and trap_type == 7 and (trap_tex is not None
                                        or trap_svg is not None):
        # Incruste l'image là où l'orbite touche son rectangle.
        # Slots : trap_radius = largeur du rectangle, trap_tilt = angle (rad).
        # SVG : test vectoriel exact ; raster en repli (formules libres).
        if trap_svg is not None:
            aspect = trap_svg[4][3] / trap_svg[4][2]
        else:
            aspect = trap_tex.shape[0] / trap_tex.shape[1]
        w = max(trap_radius, 1e-6)
        h = w * aspect
        rect = np.array([trap_x - w / 2, trap_x + w / 2,
                         trap_y - h / 2, trap_y + h / 2])
        if julia_iter_fn is None and trap_svg is not None:
            import svg_trap
            rgba = svg_trap.trap_svg_julia(Z, 1 + 0j, 0 + 0j, c,
                                           *trap_svg, rect,
                                           N_ITER, 256.0, 2, trap_tilt,
                                           mandelbrot)
        elif trap_svg is not None and julia_numba is not None:
            # Formule libre + SVG : kernel vectoriel généré pour la formule
            rgba = julia_numba[2](np.ascontiguousarray(Z), c, *trap_svg,
                                  rect, N_ITER, 256.0, 2, trap_tilt,
                                  mandelbrot)
        elif julia_iter_fn is None and mandelbrot:
            rgba = orbit_trap.trap_image_mandelbrot(Z, trap_tex, rect,
                                                    N_ITER, 256.0, 2, trap_tilt)
        elif julia_iter_fn is None:
            rgba = orbit_trap.trap_image_julia(Z, 1 + 0j, 0 + 0j, c,
                                               trap_tex, rect,
                                               N_ITER, 256.0, 2, trap_tilt)
        else:
            rgba = _trap_image_numpy(Z, julia_iter_fn, c, trap_tex, rect,
                                     N_ITER, 256.0, 2, trap_tilt, mandelbrot)
        hit = rgba[..., 3] > 0
        out[hit] = rgba[hit, :3]
    elif use_trap and trap_type == 8 and (trap_tex is not None
                                          or trap_svg is not None):
        # Série géométrique : N copies de l'image, copie k réduite par
        # ratio^k et déplacée en spirale. Slots : trap_radius = taille de
        # base, trap_tilt = pas d'angle (rad), trap_axis = ratio,
        # trap_count = nombre de copies.
        N_copies = max(1, int(round(trap_count)))
        ratio    = max(trap_axis, 1e-3)
        base     = max(trap_radius, 1e-6)
        if julia_iter_fn is None and trap_svg is not None:
            import svg_trap
            rgba = svg_trap.trap_svg_geom_julia(
                Z, 1 + 0j, 0 + 0j, c, *trap_svg,
                N_copies, ratio, trap_x, trap_y, base, trap_tilt,
                N_ITER, 256.0, mandelbrot)
        elif trap_svg is not None and julia_numba is not None:
            # Formule libre + SVG : kernel vectoriel généré pour la formule
            rgba = julia_numba[3](np.ascontiguousarray(Z), c, *trap_svg,
                                  N_copies, ratio, trap_x, trap_y,
                                  base, trap_tilt, N_ITER, 256.0, mandelbrot)
        elif julia_iter_fn is None:
            rgba = orbit_trap.trap_image_geom_series_julia(
                Z, 1 + 0j, 0 + 0j, c, trap_tex,
                N_copies, ratio, trap_x, trap_y, base, trap_tilt,
                N_ITER, 256.0, mandelbrot)
        else:
            rgba = _trap_geom_numpy(Z, julia_iter_fn, c, trap_tex,
                                    N_copies, ratio, trap_x, trap_y,
                                    base, trap_tilt, N_ITER, 256.0, mandelbrot)
        hit = rgba[..., 3] > 0
        out[hit] = rgba[hit, :3]
    if aa > 1:
        out = render.downscale(out, out_w, out_h)
    return out


# ── Application ───────────────────────────────────────────────────────────────

class FractalStudio:

    def __init__(self):
        self._t          = 0.0
        self._fps        = FPS_DEFAULT
        self._duration   = DUR_DEFAULT
        self._playing        = False
        self._play_after     = None
        self._loop_infinite  = False
        self._meta_n         = 1      # nombre de passes de la méta-boucle
        self._current_pass   = 0      # passe en cours (0-indexé)

        self._c_re = -0.7
        self._c_im =  0.27

        # Boucle c
        self._loop_pts:      list[tuple[float, float]] = []
        self._loop_closed:   bool = False
        self._loop_drawing:  bool = False
        self._loop_ctrl:     bool = False
        self._loop_start_px: tuple[int, int] = (0, 0)
        self._loop_last_px:  tuple[int, int] = (0, 0)
        self._loop_drag_px:  tuple[int, int] = (0, 0)
        self._loop_n:        int  = 1
        self._uniform_speed: bool = False   # reparamétrisation du chemin c
        self._reparam_method: str = "dem"   # 'dem' | 'image' | 'hybrid'
        self._reparam_overlay_px: list[tuple[int, int]] = []
        self._img_cost_cache: tuple | None = None   # (clé, rho image)
        self._imgcost_thread: threading.Thread | None = None
        self._loop_bounce:   bool = False

        # Boucle orbit trap
        self._otrap_x    = 0.0
        self._otrap_y    = 0.0
        self._otrap_pts:      list[tuple[float, float]] = []
        self._otrap_closed:   bool = False
        self._otrap_drawing:  bool = False
        self._otrap_ctrl:     bool = False
        self._otrap_start_px: tuple[int, int] = (0, 0)
        self._otrap_last_px:  tuple[int, int] = (0, 0)
        self._otrap_drag_px:  tuple[int, int] = (0, 0)
        self._otrap_n:        int  = 1
        self._otrap_bounce:   bool = False
        # Centre de la vue de la carte position (le tracé peut déborder du
        # cadre : la fenêtre glisse pour suivre le crayon, échelle fixe).
        self._otrap_view_cx:  float = 0.0
        self._otrap_view_cy:  float = 0.0

        # Boucle norme — knob + slider
        self._norm_range_lo: float = 0.5    # borne basse
        self._norm_range_hi: float = 1.5    # borne haute
        self._knob_angle:    float = 0.0    # balayage en degrés (0–360°)
        self._norm_n:        int   = 1
        self._norm_bounce:   bool  = False
        self._norm_drag_what: str | None   = None

        # Module orbit trap cercle — rayon animable
        self._rad_lo:        float = 0.3
        self._rad_hi:        float = 0.8
        self._rad_n:         int   = 1
        self._rad_bounce:    bool  = False
        self._rad_drag_what: str | None = None

        # Module orbit trap anneau 3D — orientation animable (degrés)
        self._tilt_lo:    float = 0.0     # inclinaison départ
        self._tilt_hi:    float = 360.0   # inclinaison arrivée (bascule complète)
        self._axis_lo:    float = 0.0     # axe départ
        self._axis_hi:    float = 0.0     # axe arrivée (fixe par défaut)
        self._ori_n:      int   = 1
        self._ori_bounce: bool  = False

        # Sous-modules génériques à paires départ→arrivée animables
        # (droite : angle ; sinus : amplitude + fréquence ; image : taille + angle)
        self._pp: dict[str, dict] = {
            "lin": {"n": 1, "bounce": False,
                    "rows": {"angle": [0.0, 180.0]}},
            "sin": {"n": 1, "bounce": False,
                    "rows": {"amp": [0.5, 0.5], "freq": [3.0, 3.0]}},
            "img": {"n": 1, "bounce": False,
                    "rows": {"taille": [1.0, 1.0], "angle": [0.0, 0.0]}},
            "geo": {"n": 1, "bounce": False,
                    "rows": {"taille": [1.0, 1.0], "ratio": [0.5, 0.5],
                             "angle": [0.0, 0.0], "copies": [4.0, 4.0]}},
        }

        # Éditeur de vélocités : courbes d'automation par lane
        self._lanes: dict[str, dict] = {}
        self._vel_win = None
        self._vel_canvases: dict[str, tk.Canvas] = {}
        self._lane_drag_ref: tuple | None = None

        # Module orbit trap image — texture RGBA (PNG détouré) ; pour les
        # SVG, géométrie vectorielle empaquetée (test exact point-dans-forme)
        self._img_tex:  np.ndarray | None = None
        self._svg_pack: tuple | None = None
        self._img_name: str = "(aucune image)"
        demo = Path(__file__).parent / "trap_demo.png"
        if demo.exists():
            try:
                self._img_tex  = _load_texture(str(demo))
                self._img_name = demo.name
            except Exception:
                pass

        # Module rotation
        self._rot_start:    float = 0.0     # angle de départ (degrés)
        self._rot_end:      float = 360.0   # angle d'arrivée (degrés)
        self._rot_n:        int   = 1
        self._rot_bounce:   bool  = False
        self._knob_drag_ref:  tuple | None = None   # (start_mouse_angle, start_knob)

        # Module zoom — facteur d'agrandissement animable (interpolation log)
        # autour du centre (cx, cy) du plan complexe.
        self._zoom_start:   float = 1.0
        self._zoom_end:     float = 1.0
        self._zoom_cx:      float = 0.0
        self._zoom_cy:      float = 0.0
        self._zoom_n:       int   = 1
        self._zoom_bounce:  bool  = False

        # Source des couleurs : module palette sanzo OU couleur aléatoire
        # (mutuellement exclusifs). _pal_idx persiste même module retiré.
        self._pal_idx     = 0    # index de la palette Sanzo sélectionnée
        self._perm_idx    = 10   # permutation 11
        self._crand_n     = 5    # nombre d'arrêts du dégradé aléatoire
        self._crand_colors: list[tuple[int, int, int]] = [
            tuple(random.randint(0, 255) for _ in range(3))
            for _ in range(self._crand_n)]
        self._mirror      = 1
        self._equalize    = True
        self._clip_limit  = 3.0
        self._color_mode  = "oklab"
        self._smooth      = True    # lissage escape time
        self._mandelbrot  = False   # plan des c (z₀=0) au lieu du plan des z
        # Module biomorphe (présence = actif) : classification OU de Pickover
        # appliquée à la formule Julia active ; c vient du module c.
        self._bio_L          = 10.0    # biomorph bailout (test OU)
        self._bio_modbail    = 100.0   # bailout modulus
        self._bio_iter       = 50      # max_iter biomorphe
        self._bio_color_iter = False   # False = structure, True = escape time
        self._julia_formula_str      = "z^2 + c"
        self._julia_formula_compiled = _DEFAULT_FORMULA_NORMALIZED
        self._julia_iter_fn          = None   # None = z^2+c fast numba path
        self._julia_numba            = None   # kernels numba de la formule libre

        self._prev_thread: threading.Thread | None = None
        self._prev_dirty  = False
        self._refresh_id  = None
        self._mandel_img: ImageTk.PhotoImage | None = None

        # Registre des modules dynamiques (zone du bas)
        self._mod_frames:   dict[str, tk.Frame] = {}
        self._module_order: list[str] = []
        self._dragging_mod: str | None = None

        self.root = tk.Tk()
        self.root.title("Fractal Studio")
        self.root.configure(bg=BG)

        self._build_ui()
        self.root.update_idletasks()
        # Fige la hauteur de la zone modules sur la configuration la plus
        # haute (tous les modules construits) pour que l'ajout/retrait de
        # modules ne redimensionne jamais la fenêtre.
        self._bottom.config(height=self._bottom.winfo_reqheight())
        self._bottom.pack_propagate(False)
        # Lancement par défaut : modules c et palette sanzo actifs
        for mid in (*self._TRAP_MODS, "rotation", "zoom", "crandom", "biomorphe"):
            self._remove_module(mid, refresh=False)
        # Fige la taille de la fenêtre : les labels animés ne peuvent plus
        # redimensionner la fenêtre pendant la lecture.
        self.root.geometry(f"{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}")
        self.root.resizable(False, False)
        self._randomize_launch()
        self._load_mandelbrot()
        self._bind_keys()
        self._request_preview()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Barre haute ──────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG2)
        top.pack(fill="x")

        tk.Label(top, text="DISNEYLAND", bg=BG2, fg=ACCENT,
                 font=("Helvetica", 12, "bold"),
                 padx=14, pady=6).pack(side="left")

        # Bascule plan : Julia (z) ↔ Mandelbrot (c)
        self._mandel_btn = self._btn(
            top, "Julia", self._toggle_mandelbrot,
            bg=BG3, fg=FG, font=("Helvetica", 9, "bold"), padx=10, pady=3)
        self._mandel_btn.pack(side="left", padx=(4, 0))

        tk.Label(top, text="FPS", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(20, 2))
        self._fps_var = tk.IntVar(value=self._fps)
        fps_spin = tk.Spinbox(top, textvariable=self._fps_var,
                              from_=1, to=60, width=3,
                              bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                              command=self._on_fps_change)
        # Le command d'un Spinbox ne se déclenche que sur les flèches : il faut
        # aussi propager les valeurs tapées au clavier.
        fps_spin.bind("<Return>",   lambda _e: self._on_fps_change())
        fps_spin.bind("<FocusOut>", lambda _e: self._on_fps_change())
        fps_spin.pack(side="left")

        tk.Label(top, text="passe", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(10, 2))
        self._dur_var = tk.DoubleVar(value=self._duration)
        dur_spin = tk.Spinbox(top, textvariable=self._dur_var,
                              from_=1, to=600, increment=1, width=4,
                              bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                              command=self._on_dur_change)
        dur_spin.bind("<Return>",   lambda _e: self._on_dur_change())
        dur_spin.bind("<FocusOut>", lambda _e: self._on_dur_change())
        dur_spin.pack(side="left")
        tk.Label(top, text="s", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(2, 2))

        self._btn(top, "  ▶  RENDER  ", self._open_render_dialog,
                  bg="#1e3d1e", fg="#88ee88",
                  font=("Helvetica", 10, "bold"), padx=10, pady=4
                  ).pack(side="right", padx=12, pady=3)
        self._btn(top, " ∿ VÉLOCITÉS ", self._open_velocity_editor,
                  bg=BG3, fg=FG, font=("Helvetica", 9), padx=8, pady=4
                  ).pack(side="right", padx=2, pady=3)

        # ── Corps : preview (gauche) + parametres statiques (droite) ─────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="x")

        left = tk.Frame(body, bg="#000")
        left.pack(side="left")
        self._canvas = tk.Label(left, bg="#000", width=PREV_W, height=PREV_H)
        self._canvas.pack()

        right = tk.Frame(body, bg=BG, padx=12, pady=8)
        right.pack(side="left", fill="y")
        self._build_static_params(right)

        # ── Module lecteur ───────────────────────────────────────────────────
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        player_mod = tk.Frame(self.root, bg=MOD_BG,
                              highlightthickness=1, highlightbackground=MOD_BOR)
        player_mod.pack(fill="x", padx=10, pady=(8, 4))

        # En-tête
        phdr = tk.Frame(player_mod, bg=HDR_BG)
        phdr.pack(fill="x")

        tk.Label(phdr, text="lecture", bg=HDR_BG, fg=ACCENT,
                 font=("Helvetica", 10, "bold"),
                 padx=8, pady=5).pack(side="left")

        loop_ctrl = tk.Frame(phdr, bg=HDR_BG)
        loop_ctrl.pack(side="right", padx=8)

        tk.Label(loop_ctrl, text="boucle", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left")
        self._meta_n_var = tk.IntVar(value=self._meta_n)
        tk.Spinbox(loop_ctrl, textvariable=self._meta_n_var,
                   from_=1, to=999, width=3,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_meta_n_change).pack(side="left", padx=(4, 2))
        tk.Label(loop_ctrl, text="×", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 8))

        self._inf_btn = self._btn(
            loop_ctrl, "∞", self._toggle_infinite,
            bg=HDR_BG, fg=FG2, font=("Helvetica", 13), padx=6, pady=2)
        self._inf_btn.pack(side="left")

        tk.Frame(player_mod, bg=MOD_BOR, height=1).pack(fill="x")

        # Corps
        body = tk.Frame(player_mod, bg=MOD_BG)
        body.pack(fill="x", padx=8, pady=6)

        self._time_lbl = tk.Label(
            body, text=self._fmt_time(), bg=MOD_BG, fg=FG2,
            font=("Courier", 10), width=24, anchor="w")
        self._time_lbl.pack(side="left")

        self._btn(body, "<<", self._rewind,
                  bg=BG2).pack(side="left", padx=2)
        self._btn(body, "<",  lambda: self._step(-1),
                  bg=BG2).pack(side="left", padx=2)
        self._play_btn = self._btn(
            body, "  PLAY  ", self._toggle_play,
            bg=GREEN, fg="#aaffaa", font=("Helvetica", 11, "bold"))
        self._play_btn.pack(side="left", padx=6)
        self._btn(body, ">",  lambda: self._step(1),
                  bg=BG2).pack(side="left", padx=2)
        self._btn(body, ">>", lambda: self._seek(self._duration),
                  bg=BG2).pack(side="left", padx=2)

        # Scrubber
        self._scrub_var = tk.DoubleVar(value=0.0)
        self._scrub = tk.Scale(
            player_mod, variable=self._scrub_var,
            from_=0.0, to=self._duration,
            resolution=1 / max(1, FPS_DEFAULT),
            orient="horizontal", showvalue=False,
            bg=MOD_BG, troughcolor=BG2, highlightthickness=0,
            command=self._on_scrub,
        )
        self._scrub.pack(fill="x", padx=8, pady=(0, 4))

        tk.Frame(self.root, bg=BG3, height=1).pack(fill="x")

        # ── Bas : zone de modules dynamiques ─────────────────────────────────
        self._bottom = tk.Frame(self.root, bg=BG, pady=4)
        self._bottom.pack(fill="x", padx=10)
        for seq in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            self._bottom.bind(seq, self._show_module_menu)

        self._mod_placeholder = tk.Label(
            self._bottom, text="clic droit : ajouter un module",
            bg=BG, fg="#555555", font=("Helvetica", 9, "italic"))
        for seq in ("<Button-2>", "<Button-3>", "<Control-Button-1>"):
            self._mod_placeholder.bind(seq, self._show_module_menu)

        # Construit tous les modules (la hauteur max est mesurée dans __init__,
        # puis tous sauf c sont retirés pour le lancement par défaut).
        for mid in ("c", "psanzo", "crandom", "otrap", "ocircle", "oring",
                    "odroite", "ocroix", "osinus", "oimage", "ogeom",
                    "biomorphe", "rotation", "zoom"):
            self._build_module(mid)
            self._module_order.append(mid)
        self._repack_modules()

    # ── Gestion des modules dynamiques ────────────────────────────────────────

    _MOD_LABELS = {"c": "c (Mandelbrot)", "otrap": "orbit trap",
                   "ocircle": "orbit trap cercle",
                   "oring": "orbit trap anneau 3D",
                   "odroite": "orbit trap droite",
                   "ocroix": "orbit trap croix",
                   "osinus": "orbit trap sinus",
                   "oimage": "orbit trap image",
                   "ogeom": "orbit trap géométrique",
                   "rotation": "rotation",
                   "zoom": "zoom",
                   "psanzo": "palette sanzo",
                   "crandom": "couleur aléatoire",
                   "biomorphe": "biomorphe"}
    # Un seul type de trap dans le rendu : ces modules sont mutuellement exclusifs
    _TRAP_MODS = ("otrap", "ocircle", "oring", "odroite", "ocroix",
                  "osinus", "oimage", "ogeom")
    # Source du champ de rendu : un trap OU le biomorphe (alternatives)
    _FIELD_MODS = (*_TRAP_MODS, "biomorphe")
    # Une seule source de couleurs à la fois (palette sanzo ou couleur aléatoire)
    _COLOR_MODS = ("psanzo", "crandom")

    def _build_module(self, mod_id: str):
        if mod_id == "c":
            self._build_mandelbrot_section(self._bottom)
        elif mod_id == "otrap":
            self._build_otrap_section(self._bottom)
        elif mod_id == "ocircle":
            self._build_ocircle_section(self._bottom)
        elif mod_id == "oring":
            self._build_oring_section(self._bottom)
        elif mod_id == "odroite":
            self._build_odroite_section(self._bottom)
        elif mod_id == "ocroix":
            self._build_ocroix_section(self._bottom)
        elif mod_id == "osinus":
            self._build_osinus_section(self._bottom)
        elif mod_id == "oimage":
            self._build_oimage_section(self._bottom)
        elif mod_id == "ogeom":
            self._build_ogeom_section(self._bottom)
        elif mod_id == "rotation":
            self._build_rotation_section(self._bottom)
        elif mod_id == "zoom":
            self._build_zoom_section(self._bottom)
        elif mod_id == "psanzo":
            self._build_psanzo_section(self._bottom)
        elif mod_id == "crandom":
            self._build_crandom_section(self._bottom)
        elif mod_id == "biomorphe":
            self._build_biomorphe_section(self._bottom)

    def _repack_modules(self):
        self._mod_placeholder.pack_forget()
        for mid in self._module_order:
            self._mod_frames[mid].pack_forget()
        if not self._module_order:
            self._mod_placeholder.pack(expand=True, pady=20)
            return
        for mid in self._module_order:
            self._mod_frames[mid].pack(side="left", padx=(0, 8),
                                        anchor="n", pady=(2, 0))

    def _add_module(self, mod_id: str):
        if mod_id in self._mod_frames:
            return
        for group in (self._FIELD_MODS, self._COLOR_MODS):
            if mod_id in group:
                for other in group:
                    if other != mod_id:
                        self._remove_module(other, refresh=False)
        self._build_module(mod_id)
        self._module_order.append(mod_id)
        self._repack_modules()
        if mod_id == "c":
            self._draw_mandelbrot()
        self._refresh_velocity_window()
        self._update_labels()
        self._request_preview()

    def _remove_module(self, mod_id: str, refresh: bool = True):
        frame = self._mod_frames.pop(mod_id, None)
        if frame is None:
            return
        if mod_id in self._module_order:
            self._module_order.remove(mod_id)
        frame.destroy()
        self._repack_modules()
        if refresh:
            self._refresh_velocity_window()
            self._request_preview()

    def _show_module_menu(self, event):
        avail = [m for m in ("c", *self._COLOR_MODS, *self._TRAP_MODS,
                             "biomorphe", "rotation", "zoom")
                 if m not in self._mod_frames]
        menu = tk.Menu(self.root, tearoff=0,
                       bg=BG2, fg=FG,
                       activebackground=BG3, activeforeground=FG)
        if avail:
            for m in avail:
                menu.add_command(label=f"+ {self._MOD_LABELS[m]}",
                                 command=lambda m=m: self._add_module(m))
        else:
            menu.add_command(label="tous les modules sont actifs",
                             state="disabled")
        menu.tk_popup(event.x_root, event.y_root)

    # ── Drag & drop des modules (par leur header) ─────────────────────────────

    def _bind_module_drag(self, mod_id: str, *widgets):
        for w in widgets:
            w.config(cursor="fleur")
            w.bind("<ButtonPress-1>",
                   lambda e, m=mod_id: self._mod_drag_start(e, m))
            w.bind("<B1-Motion>",       self._mod_drag_motion)
            w.bind("<ButtonRelease-1>", self._mod_drag_end)

    def _mod_drag_start(self, _event, mod_id: str):
        self._dragging_mod = mod_id

    def _mod_drag_motion(self, event):
        mid = self._dragging_mod
        if not mid or mid not in self._mod_frames:
            return
        order = self._module_order
        i = order.index(mid)
        x = event.x_root
        for j, other in enumerate(order):
            if other == mid:
                continue
            f  = self._mod_frames[other]
            cx = f.winfo_rootx() + f.winfo_width() // 2
            if (j < i and x < cx) or (j > i and x > cx):
                order.pop(i)
                order.insert(j, mid)
                self._repack_modules()
                break

    def _mod_drag_end(self, _event):
        self._dragging_mod = None

    def _build_static_params(self, parent):
        # La palette vit désormais dans le module « palette sanzo » (zone du bas).
        # Repeat
        row = tk.Frame(parent, bg=BG)
        row.pack(anchor="w", pady=3)
        tk.Label(row, text="Repeat", bg=BG, fg=FG2,
                 font=("Helvetica", 9), width=10, anchor="w").pack(side="left")
        self._mirror_var = tk.IntVar(value=self._mirror)
        tk.Spinbox(row, textvariable=self._mirror_var,
                   from_=1, to=8, width=3,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_mirror_change).pack(side="left")

        # Égalisation + Clip sur une seule ligne
        row2 = tk.Frame(parent, bg=BG)
        row2.pack(anchor="w", pady=3)
        tk.Label(row2, text="Égalisation", bg=BG, fg=FG2,
                 font=("Helvetica", 9), width=10, anchor="w").pack(side="left")
        self._eq_btn = self._btn(
            row2, "  ON  ", self._toggle_equalize,
            bg=GREEN, fg="#aaffaa", font=("Helvetica", 9), padx=6, pady=2)
        self._eq_btn.pack(side="left")
        self._clip_var = tk.DoubleVar(value=self._clip_limit)
        tk.Spinbox(row2, textvariable=self._clip_var,
                   from_=0.5, to=20.0, increment=0.5, width=4,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_clip_change).pack(side="left", padx=(6, 0))

        # Lissage (escape time)
        rowS = tk.Frame(parent, bg=BG)
        rowS.pack(anchor="w", pady=3)
        tk.Label(rowS, text="Lissage", bg=BG, fg=FG2,
                 font=("Helvetica", 9), width=10, anchor="w").pack(side="left")
        self._smooth_btn = self._btn(
            rowS, "  ON  ", self._toggle_smooth,
            bg=GREEN, fg="#aaffaa", font=("Helvetica", 9), padx=6, pady=2)
        self._smooth_btn.pack(side="left")

        # Color space
        row3 = tk.Frame(parent, bg=BG)
        row3.pack(anchor="w", pady=3)
        tk.Label(row3, text="Color space", bg=BG, fg=FG2,
                 font=("Helvetica", 9), width=10, anchor="w").pack(side="left")
        self._cmode_var = tk.StringVar(value="OkLab")
        om = tk.OptionMenu(row3, self._cmode_var,
                           "OkLab", "RGB", "HSV", "Cyclic",
                           command=self._on_color_mode_change)
        om.config(bg=BG2, fg=FG, activebackground=BG3, activeforeground=FG,
                  highlightthickness=0, relief="flat", font=("Helvetica", 9))
        om["menu"].config(bg=BG2, fg=FG,
                          activebackground=BG3, activeforeground=FG)
        om.pack(side="left")

        # Julia form
        row4 = tk.Frame(parent, bg=BG)
        row4.pack(anchor="w", pady=3)
        tk.Label(row4, text="Julia form", bg=BG, fg=FG2,
                 font=("Helvetica", 9), width=10, anchor="w").pack(side="left")
        self._julia_entry_var = tk.StringVar(value=self._julia_formula_str)
        self._julia_entry = tk.Entry(row4, textvariable=self._julia_entry_var,
                                     bg=BG2, fg=FG, insertbackground=FG,
                                     relief="flat", font=("Courier", 9), width=15)
        self._julia_entry.bind("<Return>",   lambda _e: self._on_julia_formula_change())
        self._julia_entry.bind("<FocusOut>", lambda _e: self._on_julia_formula_change())
        self._julia_entry.pack(side="left")
        # Dé : génère une formule Julia aléatoire (et donc un biomorphe aléatoire
        # quand le module biomorphe est actif).
        self._btn(row4, "🎲", self._random_julia_formula,
                  bg=BG2, fg=FG2, font=("Helvetica", 11),
                  padx=4, pady=1).pack(side="left", padx=(4, 0))

    def _bio_kwargs(self) -> dict:
        """Paramètres biomorphe pour render_frame (présence du module = actif)."""
        return dict(biomorph="biomorphe" in self._mod_frames,
                    bio_L=self._bio_L, bio_modbail=self._bio_modbail,
                    bio_iter=self._bio_iter,
                    bio_color_iter=self._bio_color_iter)

    # ── Dé : formule Julia aléatoire ──────────────────────────────────────────

    _RAND_FUNCS = ("sin", "cos", "sinh", "tanh", "exp")

    def _random_julia_formula(self):
        """Génère une formule Julia valide aléatoire et l'applique."""
        for _ in range(8):
            f = random.choice(self._RAND_FUNCS)
            a = round(random.uniform(0.2, 0.8), 2)
            k = random.randint(2, 6)
            j = random.randint(2, 4)
            formula = random.choice([
                f"z^{k} + c",
                f"{f}(z) + c",
                f"z^{k} + {a}*{f}(z) + c",
                f"z^{k} + {a}*conj(z) + c",
                f"{f}(z)*{a} + c",
                f"z^{k} + {a}*z^{j} + c",
            ])
            try:
                _compile_julia_iter(formula)
            except Exception:
                continue
            self._julia_entry_var.set(formula)
            self._on_julia_formula_change()
            return

    # ── Sous-modules partagés des modules trap ────────────────────────────────

    _SUB_BOR = "#454545"

    def _trap_mod_id(self) -> str | None:
        """Id du module trap actif (un de _TRAP_MODS), ou None."""
        for mid in self._TRAP_MODS:
            if mid in self._mod_frames:
                return mid
        return None

    def _trap_render_args(self, trap_mod: str, progress: float,
                          ) -> tuple[int, float, float, float, float]:
        """Type numba du trap actif et ses slots de paramètres [2], [3], [4]
        plus le compteur (copies du trap géométrique).

        Slots par type : cercle/anneau → rayon (+ inclinaison, axe) ;
        droite → angle (rad) ; sinus → amplitude, fréquence ; image →
        taille, angle (rad) ; géométrique → taille, angle (rad), ratio."""
        trap_type = {"otrap": 0, "odroite": 1, "ocroix": 2, "ocircle": 3,
                     "osinus": 5, "oring": 6, "oimage": 7,
                     "ogeom": 8}.get(trap_mod, 0)
        p2 = p3 = p4 = p5 = 0.0
        if trap_mod in ("ocircle", "oring"):
            p2 = self._rad_at(progress)
            if trap_mod == "oring":
                tilt, axis = self._ori_at(progress)
                p3, p4 = math.radians(tilt), math.radians(axis)
        elif trap_mod == "odroite":
            p2 = math.radians(self._pp_at("lin", progress)["angle"])
        elif trap_mod == "osinus":
            v = self._pp_at("sin", progress)
            p2, p3 = v["amp"], v["freq"]
        elif trap_mod == "oimage":
            v = self._pp_at("img", progress)
            p2, p3 = v["taille"], math.radians(v["angle"])
        elif trap_mod == "ogeom":
            v = self._pp_at("geo", progress)
            p2, p3 = v["taille"], math.radians(v["angle"])
            p4, p5 = v["ratio"], v["copies"]
        return trap_type, p2, p3, p4, p5

    def _build_pos_submodule(self, body):
        """Sous-module position : carte 2D du centre du trap."""
        MOD_BG, SUB_BOR, SUB_HDR = "#202020", self._SUB_BOR, self._SUB_HDR

        pos_mod = tk.Frame(body, bg=MOD_BG,
                           highlightthickness=1, highlightbackground=SUB_BOR)
        pos_mod.pack(side="left", fill="y", padx=(0, 6))

        pos_hdr = tk.Frame(pos_mod, bg=SUB_HDR)
        pos_hdr.pack(fill="x")
        tk.Label(pos_hdr, text="position", bg=SUB_HDR, fg=FG,
                 font=("Helvetica", 9, "bold"),
                 padx=6, pady=3).pack(side="left")

        loop_ctrl = tk.Frame(pos_hdr, bg=SUB_HDR)
        loop_ctrl.pack(side="right", padx=6)
        tk.Label(loop_ctrl, text="boucle", bg=SUB_HDR, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left")
        self._otrap_n_var = tk.IntVar(value=self._otrap_n)
        tk.Spinbox(loop_ctrl, textvariable=self._otrap_n_var,
                   from_=1, to=999, width=3,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_otrap_n_change).pack(side="left", padx=(4, 2))
        tk.Label(loop_ctrl, text="×", bg=SUB_HDR, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 6))
        self._otrap_bounce_btn = self._btn(
            loop_ctrl, "⇄", self._toggle_otrap_bounce,
            bg=SUB_HDR, fg=FG2, font=("Helvetica", 11), padx=4, pady=1)
        self._otrap_bounce_btn.pack(side="left")

        tk.Frame(pos_mod, bg=SUB_BOR, height=1).pack(fill="x")

        self._otrap_cv = tk.Canvas(pos_mod, width=OTRAP_W, height=OTRAP_H,
                                    bg="#0c0c0c", highlightthickness=0,
                                    cursor="crosshair")
        self._otrap_cv.pack()
        self._otrap_cv.bind("<ButtonPress-1>",  self._otrap_press)
        self._otrap_cv.bind("<B1-Motion>",       self._otrap_drag)
        self._otrap_cv.bind("<ButtonRelease-1>", self._otrap_release)
        # Clic droit : recentre la vue sur le chemin (le tracé peut déborder)
        self._otrap_cv.bind("<Button-2>", self._otrap_recenter)
        self._otrap_cv.bind("<Button-3>", self._otrap_recenter)

        tk.Frame(pos_mod, bg=SUB_BOR, height=1).pack(fill="x")

        footer = tk.Frame(pos_mod, bg=MOD_BG)
        footer.pack(fill="x", padx=8, pady=5)
        self._otrap_lbl = tk.Label(footer, text=self._fmt_otrap(),
                                    bg=MOD_BG, fg=FG2, font=("Courier", 9))
        self._otrap_lbl.pack(side="left")
        self._btn(footer, "Lisser", self._smooth_otrap,
                  bg=BG2, fg=FG2, font=("Helvetica", 8),
                  padx=6, pady=2).pack(side="right")
        self._btn(footer, "⌖", self._otrap_recenter,
                  bg=BG2, fg=FG2, font=("Helvetica", 10),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

    def _build_norm_submodule(self, body):
        """Sous-module norm : plage animable de normalisation."""
        MOD_BG, SUB_BOR, SUB_HDR = "#202020", self._SUB_BOR, self._SUB_HDR

        norm_mod = tk.Frame(body, bg=MOD_BG,
                            highlightthickness=1, highlightbackground=SUB_BOR)
        norm_mod.pack(side="left", fill="y")

        norm_hdr = tk.Frame(norm_mod, bg=SUB_HDR)
        norm_hdr.pack(fill="x")
        tk.Label(norm_hdr, text="norm", bg=SUB_HDR, fg=FG,
                 font=("Helvetica", 9, "bold"),
                 padx=6, pady=3).pack(side="left")
        nc = tk.Frame(norm_hdr, bg=SUB_HDR)
        nc.pack(side="right", padx=6)
        self._norm_n_var = tk.IntVar(value=self._norm_n)
        tk.Spinbox(nc, textvariable=self._norm_n_var,
                   from_=1, to=999, width=2,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_norm_n_change).pack(side="left")
        tk.Label(nc, text="×", bg=SUB_HDR, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(2, 2))
        self._norm_bounce_btn = self._btn(
            nc, "⇄", self._toggle_norm_bounce,
            bg=SUB_HDR, fg=FG2, font=("Helvetica", 10), padx=3, pady=1)
        self._norm_bounce_btn.pack(side="left")

        tk.Frame(norm_mod, bg=SUB_BOR, height=1).pack(fill="x")

        self._norm_cv = tk.Canvas(norm_mod, width=NORM_W, height=NORM_H,
                                   bg="#0c0c0c", highlightthickness=0,
                                   cursor="hand2")
        self._norm_cv.pack()
        self._norm_cv.bind("<ButtonPress-1>",  self._norm_press)
        self._norm_cv.bind("<B1-Motion>",       self._norm_drag)
        self._norm_cv.bind("<ButtonRelease-1>", self._norm_release)

        tk.Frame(norm_mod, bg=SUB_BOR, height=1).pack(fill="x")

        norm_footer = tk.Frame(norm_mod, bg=MOD_BG)
        norm_footer.pack(fill="x", pady=4)
        self._norm_lbl = tk.Label(norm_footer, text=self._fmt_norm(),
                                   bg=MOD_BG, fg=FG2, font=("Courier", 7),
                                   justify="center")
        self._norm_lbl.pack()

    def _build_rad_submodule(self, body):
        """Sous-module rayon : plage animable du rayon du trap cercle."""
        MOD_BG, SUB_BOR, SUB_HDR = "#202020", self._SUB_BOR, self._SUB_HDR

        rad_mod = tk.Frame(body, bg=MOD_BG,
                           highlightthickness=1, highlightbackground=SUB_BOR)
        rad_mod.pack(side="left", fill="y", padx=(0, 6))

        rad_hdr = tk.Frame(rad_mod, bg=SUB_HDR)
        rad_hdr.pack(fill="x")
        tk.Label(rad_hdr, text="rayon", bg=SUB_HDR, fg=FG,
                 font=("Helvetica", 9, "bold"),
                 padx=6, pady=3).pack(side="left")
        rc = tk.Frame(rad_hdr, bg=SUB_HDR)
        rc.pack(side="right", padx=6)
        self._rad_n_var = tk.IntVar(value=self._rad_n)
        tk.Spinbox(rc, textvariable=self._rad_n_var,
                   from_=1, to=999, width=2,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_rad_n_change).pack(side="left")
        tk.Label(rc, text="×", bg=SUB_HDR, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(2, 2))
        self._rad_bounce_btn = self._btn(
            rc, "⇄", self._toggle_rad_bounce,
            bg=SUB_HDR, fg=FG2, font=("Helvetica", 10), padx=3, pady=1)
        self._rad_bounce_btn.pack(side="left")

        tk.Frame(rad_mod, bg=SUB_BOR, height=1).pack(fill="x")

        self._rad_cv = tk.Canvas(rad_mod, width=NORM_W, height=NORM_H,
                                  bg="#0c0c0c", highlightthickness=0,
                                  cursor="hand2")
        self._rad_cv.pack()
        self._rad_cv.bind("<ButtonPress-1>",  self._rad_press)
        self._rad_cv.bind("<B1-Motion>",       self._rad_drag)
        self._rad_cv.bind("<ButtonRelease-1>", self._rad_release)

        tk.Frame(rad_mod, bg=SUB_BOR, height=1).pack(fill="x")

        rad_footer = tk.Frame(rad_mod, bg=MOD_BG)
        rad_footer.pack(fill="x", pady=4)
        self._rad_lbl = tk.Label(rad_footer, text=self._fmt_rad(),
                                  bg=MOD_BG, fg=FG2, font=("Courier", 7),
                                  justify="center")
        self._rad_lbl.pack()

    def _build_orient_submodule(self, body):
        """Sous-module orientation : inclinaison et axe de l'anneau 3D (degrés)."""
        MOD_BG, SUB_BOR, SUB_HDR = "#202020", self._SUB_BOR, self._SUB_HDR

        ori_mod = tk.Frame(body, bg=MOD_BG,
                           highlightthickness=1, highlightbackground=SUB_BOR)
        ori_mod.pack(side="left", fill="y", padx=(0, 6))

        ori_hdr = tk.Frame(ori_mod, bg=SUB_HDR)
        ori_hdr.pack(fill="x")
        tk.Label(ori_hdr, text="orientation", bg=SUB_HDR, fg=FG,
                 font=("Helvetica", 9, "bold"),
                 padx=6, pady=3).pack(side="left")
        oc = tk.Frame(ori_hdr, bg=SUB_HDR)
        oc.pack(side="right", padx=6)
        self._ori_n_var = tk.IntVar(value=self._ori_n)
        tk.Spinbox(oc, textvariable=self._ori_n_var,
                   from_=1, to=999, width=2,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_ori_n_change).pack(side="left")
        tk.Label(oc, text="×", bg=SUB_HDR, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(2, 2))
        self._ori_bounce_btn = self._btn(
            oc, "⇄", self._toggle_ori_bounce,
            bg=SUB_HDR, fg=FG2, font=("Helvetica", 10), padx=3, pady=1)
        self._ori_bounce_btn.pack(side="left")

        tk.Frame(ori_mod, bg=SUB_BOR, height=1).pack(fill="x")

        # Champs inclinaison / axe : départ → arrivée (degrés)
        fields = tk.Frame(ori_mod, bg=MOD_BG)
        fields.pack(padx=6, pady=4)
        self._ori_vars = {}
        for key, label, lo, hi in (
                ("tilt", "incl.", self._tilt_lo, self._tilt_hi),
                ("axis", "axe",   self._axis_lo, self._axis_hi)):
            row = tk.Frame(fields, bg=MOD_BG)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=label, bg=MOD_BG, fg=FG2,
                     font=("Helvetica", 8), width=5, anchor="w").pack(side="left")
            v_lo = tk.StringVar(value=f"{lo:g}")
            v_hi = tk.StringVar(value=f"{hi:g}")
            self._ori_vars[key] = (v_lo, v_hi)
            for var in (v_lo, v_hi):
                e = tk.Entry(row, textvariable=var, width=4,
                             bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                             font=("Courier", 9), justify="right")
                e.bind("<Return>",   lambda _e: self._on_ori_field_change())
                e.bind("<FocusOut>", lambda _e: self._on_ori_field_change())
                e.pack(side="left")
                if var is v_lo:
                    tk.Label(row, text="→", bg=MOD_BG, fg=FG2,
                             font=("Helvetica", 8)).pack(side="left", padx=1)
            tk.Label(row, text="°", bg=MOD_BG, fg=FG2,
                     font=("Helvetica", 8)).pack(side="left", padx=(2, 0))

        tk.Frame(ori_mod, bg=SUB_BOR, height=1).pack(fill="x")

        # Aperçu de la projection (ellipse) de l'anneau
        self._ori_cv = tk.Canvas(ori_mod, width=NORM_W, height=104,
                                  bg="#0c0c0c", highlightthickness=0)
        self._ori_cv.pack()

        tk.Frame(ori_mod, bg=SUB_BOR, height=1).pack(fill="x")

        ori_footer = tk.Frame(ori_mod, bg=MOD_BG)
        ori_footer.pack(fill="x", pady=4)
        self._ori_lbl = tk.Label(ori_footer, text=self._fmt_ori(),
                                  bg=MOD_BG, fg=FG2, font=("Courier", 7),
                                  justify="center")
        self._ori_lbl.pack()

    def _build_pp_submodule(self, body, key: str, title: str,
                            rows: list[tuple[str, str]]):
        """Sous-module générique : paires départ→arrivée animables.

        rows : [(row_key, unité)] — les bornes initiales vivent dans
        self._pp[key]["rows"]. Boucle N× et ⇄ partagés par les lignes."""
        MOD_BG, SUB_BOR, SUB_HDR = "#202020", self._SUB_BOR, self._SUB_HDR
        st = self._pp[key]

        mod = tk.Frame(body, bg=MOD_BG,
                       highlightthickness=1, highlightbackground=SUB_BOR)
        mod.pack(side="left", fill="y", padx=(0, 6))

        hdr = tk.Frame(mod, bg=SUB_HDR)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, bg=SUB_HDR, fg=FG,
                 font=("Helvetica", 9, "bold"),
                 padx=6, pady=3).pack(side="left")
        ctl = tk.Frame(hdr, bg=SUB_HDR)
        ctl.pack(side="right", padx=6)
        st["n_var"] = tk.IntVar(value=st["n"])
        tk.Spinbox(ctl, textvariable=st["n_var"], from_=1, to=999, width=2,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=lambda k=key: self._on_pp_n_change(k)).pack(side="left")
        tk.Label(ctl, text="×", bg=SUB_HDR, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(2, 2))
        st["btn"] = self._btn(
            ctl, "⇄", lambda k=key: self._toggle_pp_bounce(k),
            bg=SUB_HDR, fg=FG2, font=("Helvetica", 10), padx=3, pady=1)
        st["btn"].pack(side="left")

        tk.Frame(mod, bg=SUB_BOR, height=1).pack(fill="x")

        fields = tk.Frame(mod, bg=MOD_BG)
        fields.pack(padx=6, pady=4)
        st["vars"] = {}
        for row_key, unit in rows:
            lo, hi = st["rows"][row_key]
            row = tk.Frame(fields, bg=MOD_BG)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=row_key, bg=MOD_BG, fg=FG2,
                     font=("Helvetica", 8), width=6, anchor="w").pack(side="left")
            v_lo = tk.StringVar(value=f"{lo:g}")
            v_hi = tk.StringVar(value=f"{hi:g}")
            st["vars"][row_key] = (v_lo, v_hi)
            for var in (v_lo, v_hi):
                e = tk.Entry(row, textvariable=var, width=5,
                             bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                             font=("Courier", 9), justify="right")
                e.bind("<Return>",   lambda _e, k=key: self._on_pp_field_change(k))
                e.bind("<FocusOut>", lambda _e, k=key: self._on_pp_field_change(k))
                e.pack(side="left")
                if var is v_lo:
                    tk.Label(row, text="→", bg=MOD_BG, fg=FG2,
                             font=("Helvetica", 8)).pack(side="left", padx=1)
            tk.Label(row, text=unit, bg=MOD_BG, fg=FG2,
                     font=("Helvetica", 8), width=1).pack(side="left", padx=(2, 0))

        tk.Frame(mod, bg=SUB_BOR, height=1).pack(fill="x")

        footer = tk.Frame(mod, bg=MOD_BG)
        footer.pack(fill="x", pady=4)
        st["lbl"] = tk.Label(footer, text=self._fmt_pp(key),
                             bg=MOD_BG, fg=FG2, font=("Courier", 7),
                             justify="left")
        st["lbl"].pack()
        return mod

    def _pp_at(self, key: str, progress: float) -> dict[str, float]:
        """Valeurs courantes des paires de key pour progress ∈ [0, 1]."""
        st = self._pp[key]
        p  = self._anim_p(key, progress, st["n"], st["bounce"])
        return {rk: lo + p * (hi - lo) for rk, (lo, hi) in st["rows"].items()}

    def _fmt_pp(self, key: str) -> str:
        progress = self._t / max(self._duration, 1e-9)
        cur = self._pp_at(key, progress)
        return "\n".join(f"{rk} {v:8.2f}" for rk, v in cur.items())

    def _on_pp_field_change(self, key: str):
        st = self._pp[key]
        try:
            new = {rk: [float(pair[0].get().replace(",", ".")),
                        float(pair[1].get().replace(",", "."))]
                   for rk, pair in st["vars"].items()}
        except ValueError:
            for rk, pair in st["vars"].items():
                pair[0].set(f"{st['rows'][rk][0]:g}")
                pair[1].set(f"{st['rows'][rk][1]:g}")
            return
        st["rows"] = new
        st["lbl"].config(text=self._fmt_pp(key))
        self._request_preview()

    def _on_pp_n_change(self, key: str):
        st = self._pp[key]
        try:
            st["n"] = max(1, int(st["n_var"].get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _toggle_pp_bounce(self, key: str):
        st = self._pp[key]
        st["bounce"] = not st["bounce"]
        if st["bounce"]:
            st["btn"].config(bg=ACCENT, fg=FG, font=("Helvetica", 10, "bold"))
        else:
            st["btn"].config(bg=self._SUB_HDR, fg=FG2, font=("Helvetica", 10))
        self._request_preview()

    # ── Vélocités : lanes d'automation (façon Logic Pro) ──────────────────────

    _LANE_LABELS = {"c": "c", "position": "position", "norm": "norm",
                    "rayon": "rayon", "orientation": "orient.",
                    "lin": "angle", "sin": "forme", "img": "image",
                    "geo": "série", "rotation": "rotation", "zoom": "zoom"}

    def _lane(self, lane: str) -> dict:
        if lane not in self._lanes:
            self._lanes[lane] = {"pts": [(0.0, 0.0), (1.0, 1.0)],
                                 "bend": [0.0], "custom": False}
        return self._lanes[lane]

    def _anim_p(self, lane: str, progress: float, n: int, bounce: bool) -> float:
        """Phase p ∈ [0,1] d'une grandeur animée : courbe de vélocité si la
        lane est personnalisée, sinon boucle N× / ⇄ (comportement existant)."""
        d = self._lanes.get(lane)
        if d is not None and d["custom"]:
            return min(1.0, max(0.0, _lane_eval(d["pts"], d["bend"],
                                                min(1.0, max(0.0, progress)))))
        raw = progress * max(1, n)
        p   = raw % 1.0
        if bounce and int(raw) % 2 == 1:
            p = 1.0 - p
        return p

    def _lane_loop_params(self, lane: str) -> tuple[int, bool]:
        table = {"c":          lambda: (self._loop_n,  self._loop_bounce),
                 "position":   lambda: (self._otrap_n, self._otrap_bounce),
                 "norm":       lambda: (self._norm_n,  self._norm_bounce),
                 "rayon":      lambda: (self._rad_n,   self._rad_bounce),
                 "orientation":lambda: (self._ori_n,   self._ori_bounce),
                 "rotation":   lambda: (self._rot_n,   self._rot_bounce),
                 "zoom":       lambda: (self._zoom_n,  self._zoom_bounce)}
        if lane in table:
            return table[lane]()
        st = self._pp[lane]
        return st["n"], st["bounce"]

    def _active_lanes(self) -> list[str]:
        lanes = []
        if "c" in self._mod_frames:
            lanes.append("c")
        tm = self._trap_mod_id()
        if tm:
            lanes.append("position")
        if tm in ("ocircle", "oring"):
            lanes.append("rayon")
        if tm == "oring":
            lanes.append("orientation")
        pp = {"odroite": "lin", "osinus": "sin",
              "oimage": "img", "ogeom": "geo"}.get(tm)
        if pp:
            lanes.append(pp)
        if tm and tm not in ("oimage", "ogeom"):
            lanes.append("norm")
        if "rotation" in self._mod_frames:
            lanes.append("rotation")
        if "zoom" in self._mod_frames:
            lanes.append("zoom")
        return lanes

    # — Fenêtre —

    def _open_velocity_editor(self):
        if self._vel_win and self._vel_win.winfo_exists():
            self._vel_win.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Vélocités")
        win.configure(bg=BG)
        win.resizable(False, False)
        self._vel_win = win
        self._rebuild_velocity_lanes()

    def _refresh_velocity_window(self):
        if self._vel_win and self._vel_win.winfo_exists():
            self._rebuild_velocity_lanes()

    def _rebuild_velocity_lanes(self):
        win = self._vel_win
        for child in win.winfo_children():
            child.destroy()
        self._vel_canvases = {}

        # Règle temporelle (alignée sur les canvases : colonne gauche 96 px)
        ruler_row = tk.Frame(win, bg=BG)
        ruler_row.pack(padx=10, pady=(8, 0), anchor="w")
        tk.Frame(ruler_row, bg=BG, width=96).pack(side="left")
        ruler = tk.Canvas(ruler_row, width=LANE_W, height=16,
                          bg=BG, highlightthickness=0)
        ruler.pack(side="left")
        dur = max(self._duration, 1e-9)
        step = max(1, int(dur / 10))
        s = 0.0
        while s <= dur + 1e-9:
            x = min(LANE_W - 1, s / dur * LANE_W)
            ruler.create_line(x, 10, x, 16, fill="#555555")
            ruler.create_text(x + 2, 8, text=f"{s:g}s", anchor="w",
                              fill="#555555", font=("Courier", 7))
            s += step

        lanes = self._active_lanes()
        if not lanes:
            tk.Label(win, text="aucun module animable actif",
                     bg=BG, fg="#555555",
                     font=("Helvetica", 9, "italic")).pack(padx=20, pady=20)
            return
        for lane in lanes:
            self._build_lane_row(win, lane)
        tk.Label(win, text="double-clic : ajouter un point — clic droit : "
                           "supprimer — tirer un segment : le courber",
                 bg=BG, fg="#555555",
                 font=("Helvetica", 8)).pack(pady=(2, 8))

    def _build_lane_row(self, parent, lane: str):
        row = tk.Frame(parent, bg=BG)
        row.pack(padx=10, pady=3, anchor="w")
        left = tk.Frame(row, bg=BG, width=96)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        tk.Label(left, text=self._LANE_LABELS.get(lane, lane),
                 bg=BG, fg=ACCENT, font=("Helvetica", 9, "bold"),
                 anchor="w").pack(anchor="w")
        presets = ["défaut (boucle)", "linéaire", "ease", "expo", "boucle N×"]
        if lane == "c":
            presets.append("vitesse uniforme")
        var = tk.StringVar(value="préréglage")
        om = tk.OptionMenu(left, var, *presets,
                           command=lambda p, l=lane: self._lane_preset(l, p))
        om.config(bg=BG2, fg=FG2, activebackground=BG3, activeforeground=FG,
                  highlightthickness=0, relief="flat", font=("Helvetica", 8))
        om["menu"].config(bg=BG2, fg=FG,
                          activebackground=BG3, activeforeground=FG)
        om.pack(anchor="w", pady=(4, 0))

        cv = tk.Canvas(row, width=LANE_W, height=LANE_H, bg="#101010",
                       highlightthickness=1, highlightbackground=BG3)
        cv.pack(side="left", padx=(0, 0))
        self._vel_canvases[lane] = cv
        cv.bind("<ButtonPress-1>",   lambda e, l=lane: self._lane_press(l, e))
        cv.bind("<B1-Motion>",       lambda e, l=lane: self._lane_drag(l, e))
        cv.bind("<ButtonRelease-1>", lambda e, l=lane: self._lane_release(l, e))
        cv.bind("<Double-Button-1>", lambda e, l=lane: self._lane_addpoint(l, e))
        cv.bind("<Button-2>",        lambda e, l=lane: self._lane_delpoint(l, e))
        cv.bind("<Button-3>",        lambda e, l=lane: self._lane_delpoint(l, e))
        self._redraw_lane(lane)

    # — Dessin —

    def _redraw_lane(self, lane: str):
        cv = self._vel_canvases.get(lane)
        if cv is None or not cv.winfo_exists():
            return
        cv.delete("all")
        # Grille aux quarts
        for q in (0.25, 0.5, 0.75):
            cv.create_line(q * LANE_W, 0, q * LANE_W, LANE_H, fill="#1c1c1c")
            cv.create_line(0, q * LANE_H, LANE_W, q * LANE_H, fill="#1c1c1c")
        d = self._lane(lane)
        if d["custom"]:
            pts_px = [(t * LANE_W, (1.0 - _lane_eval(d["pts"], d["bend"], t))
                       * LANE_H)
                      for t in (k / 200 for k in range(201))]
            flat = [c for p in pts_px for c in p]
            cv.create_line(*flat, fill=ACCENT, width=2)
            for (t, v) in d["pts"]:
                x, y = t * LANE_W, (1.0 - v) * LANE_H
                cv.create_oval(x - 4, y - 4, x + 4, y + 4,
                               fill="#0c0c0c", outline=ACCENT, width=2)
        else:
            # Courbe effective boucle N×/⇄ (grisée : cliquer pour éditer)
            n, bounce = self._lane_loop_params(lane)
            pts_px = []
            for k in range(201):
                t = k / 200
                p = self._anim_p("__none__", t, n, bounce)
                pts_px.extend((t * LANE_W, (1.0 - p) * LANE_H))
            cv.create_line(*pts_px, fill="#666666", width=1)
            cv.create_text(LANE_W - 6, 8, anchor="e",
                           text="via boucle N×/⇄ — cliquer pour éditer",
                           fill="#444444", font=("Helvetica", 7))
        # Playhead
        progress = self._t / max(self._duration, 1e-9)
        cv._playhead = cv.create_line(progress * LANE_W, 0,
                                      progress * LANE_W, LANE_H,
                                      fill="#ff4444", width=1)

    def _update_velocity_playheads(self):
        if not (self._vel_win and self._vel_win.winfo_exists()):
            return
        progress = self._t / max(self._duration, 1e-9)
        x = progress * LANE_W
        for cv in self._vel_canvases.values():
            if cv.winfo_exists() and hasattr(cv, "_playhead"):
                cv.coords(cv._playhead, x, 0, x, LANE_H)

    # — Interactions —

    def _lane_event_tv(self, e) -> tuple[float, float]:
        t = min(1.0, max(0.0, e.x / LANE_W))
        v = min(1.0, max(0.0, 1.0 - e.y / LANE_H))
        return t, v

    def _lane_make_custom(self, lane: str):
        """Première édition : fige la courbe effective (boucle N×/⇄) en
        points éditables, pour partir de ce que la lane affiche."""
        d = self._lane(lane)
        if d["custom"]:
            return d
        n, bounce = self._lane_loop_params(lane)
        n = max(1, n)
        if n == 1 and not bounce:
            pts = [(0.0, 0.0), (1.0, 1.0)]
        elif bounce:
            # Triangle : sommets alternés 0/1
            pts = [(j / n, float(j % 2)) for j in range(n + 1)]
        else:
            # Dents de scie : retombée quasi verticale entre les cycles
            pts = [(0.0, 0.0)]
            for j in range(n):
                end = (j + 1) / n
                pts.append((end - 1e-4, 1.0))
                if j < n - 1:
                    pts.append((end, 0.0))
            pts.append((1.0, 1.0 if n == 1 else 0.0))
            if n > 1:
                pts[-1] = (1.0, 0.0)
        d["pts"]    = pts
        d["bend"]   = [0.0] * (len(pts) - 1)
        d["custom"] = True
        return d

    def _lane_press(self, lane: str, e):
        d = self._lane_make_custom(lane)
        # Point sous le curseur ?
        for i, (t, v) in enumerate(d["pts"]):
            if (abs(t * LANE_W - e.x) <= 6
                    and abs((1.0 - v) * LANE_H - e.y) <= 6):
                self._lane_drag_ref = (lane, "pt", i, 0, 0.0)
                self._redraw_lane(lane)
                return
        # Sinon : courbure du segment sous x
        t = e.x / LANE_W
        for i in range(len(d["pts"]) - 1):
            if d["pts"][i][0] <= t <= d["pts"][i + 1][0]:
                self._lane_drag_ref = (lane, "seg", i, e.y, d["bend"][i])
                self._redraw_lane(lane)
                return
        self._lane_drag_ref = None

    def _lane_drag(self, lane: str, e):
        ref = self._lane_drag_ref
        if not ref or ref[0] != lane:
            return
        d = self._lane(lane)
        _, kind, i, y0, b0 = ref
        if kind == "pt":
            t, v = self._lane_event_tv(e)
            if i == 0:
                t = 0.0
            elif i == len(d["pts"]) - 1:
                t = 1.0
            else:
                t = min(max(t, d["pts"][i - 1][0] + 1e-3),
                        d["pts"][i + 1][0] - 1e-3)
            d["pts"][i] = (t, v)
        else:
            d["bend"][i] = min(1.0, max(-1.0, b0 + (y0 - e.y) / 40.0))
        self._redraw_lane(lane)
        self._schedule_preview()

    def _lane_release(self, lane: str, _e):
        if self._lane_drag_ref and self._lane_drag_ref[0] == lane:
            self._lane_drag_ref = None
            self._request_preview()

    def _lane_addpoint(self, lane: str, e):
        d = self._lane_make_custom(lane)
        t, v = self._lane_event_tv(e)
        for i in range(len(d["pts"]) - 1):
            if d["pts"][i][0] < t < d["pts"][i + 1][0]:
                d["pts"].insert(i + 1, (t, v))
                d["bend"].insert(i, 0.0)
                d["bend"][i + 1] = 0.0
                break
        self._redraw_lane(lane)
        self._request_preview()

    def _lane_delpoint(self, lane: str, e):
        d = self._lane(lane)
        if not d["custom"]:
            return
        for i in range(1, len(d["pts"]) - 1):   # bornes non supprimables
            t, v = d["pts"][i]
            if (abs(t * LANE_W - e.x) <= 6
                    and abs((1.0 - v) * LANE_H - e.y) <= 6):
                d["pts"].pop(i)
                d["bend"].pop(i - 1)
                break
        self._redraw_lane(lane)
        self._request_preview()

    def _lane_preset(self, lane: str, preset: str):
        d = self._lane(lane)
        if preset == "défaut (boucle)":
            d["custom"] = False
        elif preset == "linéaire":
            d.update(pts=[(0.0, 0.0), (1.0, 1.0)], bend=[0.0], custom=True)
        elif preset == "ease":
            d.update(pts=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
                     bend=[0.9, -0.9], custom=True)
        elif preset == "expo":
            d.update(pts=[(0.0, 0.0), (1.0, 1.0)], bend=[1.0], custom=True)
        elif preset == "boucle N×":
            d["custom"] = False
            self._lane_make_custom(lane)
        elif preset == "vitesse uniforme" and len(self._loop_pts) >= 2:
            # Cuit le warp DEM dans la lane : courbe éditable ensuite
            import path_reparam
            S, tg = path_reparam.reparam_warp(self._loop_c_at,
                                              n_samples=2000,
                                              rho_cap=REPARAM_RHO_CAP)
            ss = [k / 16 for k in range(17)]
            ts = np.interp(ss, S, tg)
            d.update(pts=[(float(s), float(tv)) for s, tv in zip(ss, ts)],
                     bend=[0.0] * 16, custom=True)
        self._redraw_lane(lane)
        self._request_preview()

    # ── Sur-modules trap ──────────────────────────────────────────────────────

    def _build_trap_supermodule(self, parent, mod_id: str, title: str):
        """Cadre commun des sur-modules trap : header + ✕ + drag. Retourne body."""
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames[mod_id] = module

        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text=title, bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"),
                      padx=8, pady=3)
        t1.pack(side="left")
        self._bind_module_drag(mod_id, hdr, t1)

        self._btn(hdr, "✕", lambda: self._remove_module(mod_id),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=4, pady=4)
        return body

    def _build_otrap_section(self, parent):
        """Sur-module orbit trap (point) : position + norm."""
        body = self._build_trap_supermodule(parent, "otrap", "orbit trap")
        self._build_pos_submodule(body)
        self._build_norm_submodule(body)
        self._draw_otrap()
        self._draw_norm()

    def _build_ocircle_section(self, parent):
        """Sur-module orbit trap cercle : position + rayon + norm."""
        body = self._build_trap_supermodule(parent, "ocircle", "orbit trap cercle")
        self._build_pos_submodule(body)
        self._build_rad_submodule(body)
        self._build_norm_submodule(body)
        self._draw_otrap()
        self._draw_rad()
        self._draw_norm()

    def _build_oring_section(self, parent):
        """Sur-module orbit trap anneau 3D : position + rayon + orientation + norm."""
        body = self._build_trap_supermodule(parent, "oring", "orbit trap anneau 3D")
        self._build_pos_submodule(body)
        self._build_rad_submodule(body)
        self._build_orient_submodule(body)
        self._build_norm_submodule(body)
        self._draw_otrap()
        self._draw_rad()
        self._draw_ori()
        self._draw_norm()

    def _build_odroite_section(self, parent):
        """Sur-module orbit trap droite : position + angle + norm."""
        body = self._build_trap_supermodule(parent, "odroite", "orbit trap droite")
        self._build_pos_submodule(body)
        self._build_pp_submodule(body, "lin", "angle", [("angle", "°")])
        self._build_norm_submodule(body)
        self._draw_otrap()
        self._draw_norm()

    def _build_ocroix_section(self, parent):
        """Sur-module orbit trap croix : position + norm."""
        body = self._build_trap_supermodule(parent, "ocroix", "orbit trap croix")
        self._build_pos_submodule(body)
        self._build_norm_submodule(body)
        self._draw_otrap()
        self._draw_norm()

    def _build_osinus_section(self, parent):
        """Sur-module orbit trap sinus : position + forme (amp, freq) + norm."""
        body = self._build_trap_supermodule(parent, "osinus", "orbit trap sinus")
        self._build_pos_submodule(body)
        self._build_pp_submodule(body, "sin", "forme",
                                 [("amp", ""), ("freq", "")])
        self._build_norm_submodule(body)
        self._draw_otrap()
        self._draw_norm()

    def _build_oimage_section(self, parent):
        """Sur-module orbit trap image : position + taille/angle + fichier."""
        body = self._build_trap_supermodule(parent, "oimage", "orbit trap image")
        self._build_pos_submodule(body)
        self._build_pp_submodule(body, "img", "image",
                                 [("taille", ""), ("angle", "°")])
        self._build_imgfile_submodule(body)
        self._draw_otrap()

    def _build_ogeom_section(self, parent):
        """Sur-module orbit trap géométrique : position + série + fichier.

        Texture partagée avec le module orbit trap image (self._img_tex)."""
        body = self._build_trap_supermodule(parent, "ogeom",
                                            "orbit trap géométrique")
        self._build_pos_submodule(body)
        self._build_pp_submodule(body, "geo", "série",
                                 [("taille", ""), ("ratio", ""),
                                  ("angle", "°"), ("copies", "")])
        self._build_imgfile_submodule(body)
        self._draw_otrap()

    def _build_imgfile_submodule(self, body):
        """Sous-module fichier : chargement du PNG détouré du trap image."""
        MOD_BG, SUB_BOR, SUB_HDR = "#202020", self._SUB_BOR, self._SUB_HDR
        mod = tk.Frame(body, bg=MOD_BG,
                       highlightthickness=1, highlightbackground=SUB_BOR)
        mod.pack(side="left", fill="y", padx=(0, 6))
        hdr = tk.Frame(mod, bg=SUB_HDR)
        hdr.pack(fill="x")
        tk.Label(hdr, text="fichier", bg=SUB_HDR, fg=FG,
                 font=("Helvetica", 9, "bold"),
                 padx=6, pady=3).pack(side="left")
        tk.Frame(mod, bg=SUB_BOR, height=1).pack(fill="x")
        inner = tk.Frame(mod, bg=MOD_BG)
        inner.pack(padx=8, pady=8)
        self._btn(inner, "Charger PNG…", self._load_oimage,
                  bg=BG2, fg=FG2, font=("Helvetica", 8),
                  padx=6, pady=2).pack(anchor="w")
        self._img_name_lbl = tk.Label(inner, text=self._img_name,
                                      bg=MOD_BG, fg=FG2,
                                      font=("Courier", 7),
                                      wraplength=110, justify="left")
        self._img_name_lbl.pack(anchor="w", pady=(6, 4))
        # Vignette de l'image chargée (raster d'affichage, y compris SVG)
        self._img_thumb_cv = tk.Canvas(inner, width=110, height=70,
                                       bg="#0c0c0c", highlightthickness=1,
                                       highlightbackground=SUB_BOR)
        self._img_thumb_cv.pack(anchor="w")
        self._update_img_thumb()

    def _load_oimage(self):
        path = tkfd.askopenfilename(
            title="Charger une image avec transparence",
            filetypes=[("Images", "*.png *.svg"), ("PNG", "*.png"),
                       ("SVG", "*.svg"), ("Tous", "*.*")])
        if not path:
            return
        svg_pack = None
        try:
            if path.lower().endswith(".svg"):
                # Géométrie vectorielle (trap exact à toute échelle) + raster
                # dérivé pour la vignette et le repli des formules libres
                import svg_trap
                svg_pack, tex = svg_trap.load_svg(path)
            else:
                tex = _load_texture(path)
        except Exception:
            return
        self._img_tex  = tex
        self._svg_pack = svg_pack
        self._img_name = Path(path).name
        self._img_name_lbl.config(text=self._img_name)
        self._update_img_thumb()
        self._request_preview()

    def _update_img_thumb(self):
        """Vignette du sous-module fichier (texture raster d'affichage)."""
        cv = getattr(self, "_img_thumb_cv", None)
        if cv is None or not cv.winfo_exists():
            return
        cv.delete("all")
        if self._img_tex is None:
            return
        pil = Image.fromarray(self._img_tex)
        pil.thumbnail((108, 68), Image.BILINEAR)
        self._img_thumb_ref = ImageTk.PhotoImage(pil)
        cv.create_image(56, 36, image=self._img_thumb_ref, anchor="center")

    def _build_mandelbrot_section(self, parent):
        """Widget boucle autonome pour le parametre c."""
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        # ── Conteneur module ─────────────────────────────────────────────────
        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["c"] = module

        # ── En-tete : "c  Mandelbrot" à gauche | boucle à droite ─────────────
        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")

        t1 = tk.Label(hdr, text="c", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"),
                      padx=8, pady=5)
        t1.pack(side="left")
        t2 = tk.Label(hdr, text="Mandelbrot", bg=HDR_BG, fg=FG2,
                      font=("Helvetica", 8))
        t2.pack(side="left")
        self._bind_module_drag("c", hdr, t1, t2)

        # Croix de fermeture — tout à droite
        self._btn(hdr, "✕", lambda: self._remove_module("c"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        # Boucle — coin supérieur droit
        loop_ctrl = tk.Frame(hdr, bg=HDR_BG)
        loop_ctrl.pack(side="right", padx=8)
        tk.Label(loop_ctrl, text="boucle", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left")
        self._loop_n_var = tk.IntVar(value=self._loop_n)
        tk.Spinbox(loop_ctrl, textvariable=self._loop_n_var,
                   from_=1, to=999, width=3,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_loop_n_change).pack(side="left", padx=(4, 2))
        tk.Label(loop_ctrl, text="×", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 6))
        self._loop_bounce_btn = self._btn(
            loop_ctrl, "⇄", self._toggle_loop_bounce,
            bg=HDR_BG, fg=FG2, font=("Helvetica", 11), padx=4, pady=1)
        self._loop_bounce_btn.pack(side="left")

        # ── Séparateur ────────────────────────────────────────────────────────
        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        # ── Carte Mandelbrot ──────────────────────────────────────────────────
        self._mandel_cv = tk.Canvas(module, width=MANDEL_W, height=MANDEL_H,
                                    bg="#000", highlightthickness=0,
                                    cursor="crosshair")
        self._mandel_cv.pack()
        self._mandel_cv.bind("<ButtonPress-1>",  self._loop_press)
        self._mandel_cv.bind("<B1-Motion>",       self._loop_drag)
        self._mandel_cv.bind("<ButtonRelease-1>", self._loop_release)

        # ── Séparateur ────────────────────────────────────────────────────────
        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        # ── Footer : valeur c + bouton lisser ─────────────────────────────────
        footer = tk.Frame(module, bg=MOD_BG)
        footer.pack(fill="x", padx=8, pady=5)

        self._c_lbl = tk.Label(footer, text=self._fmt_c(),
                               bg=MOD_BG, fg=FG2, font=("Courier", 9))
        self._c_lbl.pack(side="left")

        self._btn(footer, "Lisser", self._smooth_loop,
                  bg=BG2, fg=FG2, font=("Helvetica", 8),
                  padx=6, pady=2).pack(side="right")

        # ── Vitesse uniforme : reparamétrisation du chemin ────────────────────
        uni_row = tk.Frame(module, bg=MOD_BG)
        uni_row.pack(fill="x", padx=8, pady=(0, 2))
        self._uniform_var = tk.BooleanVar(value=self._uniform_speed)
        tk.Checkbutton(uni_row, text="Vitesse uniforme",
                       variable=self._uniform_var,
                       command=self._on_uniform_toggle,
                       bg=MOD_BG, fg=FG2,
                       activebackground=MOD_BG, activeforeground=FG,
                       selectcolor=BG2, highlightthickness=0, bd=0,
                       font=("Helvetica", 8)).pack(side="left")

        method_row = tk.Frame(module, bg=MOD_BG)
        method_row.pack(fill="x", padx=8, pady=(0, 5))
        self._reparam_method_var = tk.StringVar(
            value=self._REPARAM_LABELS_INV[self._reparam_method])
        om = tk.OptionMenu(method_row, self._reparam_method_var,
                           *self._REPARAM_LABELS,
                           command=self._on_reparam_method_change)
        om.config(bg=BG2, fg=FG, activebackground=BG3, activeforeground=FG,
                  highlightthickness=0, relief="flat", font=("Helvetica", 8))
        om["menu"].config(bg=BG2, fg=FG,
                          activebackground=BG3, activeforeground=FG)
        om.config(state="normal" if self._uniform_speed else "disabled")
        om.pack(side="left")
        self._reparam_method_menu = om
        self._reparam_prog_lbl = tk.Label(method_row, text="",
                                          bg=MOD_BG, fg=FG2,
                                          font=("Courier", 7))
        self._reparam_prog_lbl.pack(side="left", padx=(6, 0))

    # ── Palette + permutations ────────────────────────────────────────────────

    def _get_perm_colors(self) -> list:
        # Couleur aléatoire active → ses arrêts ; sinon palette Sanzo permutée.
        if "crandom" in self._mod_frames:
            return [tuple(c) for c in self._crand_colors]
        orig   = PALETTES[self._pal_idx][1]
        perms  = list(itertools.permutations(orig))
        return list(perms[self._perm_idx % len(perms)])

    def _color_state_key(self) -> tuple:
        """Identité de la source de couleurs (pour le cache métrique image)."""
        if "crandom" in self._mod_frames:
            return ("crandom", tuple(map(tuple, self._crand_colors)))
        return ("psanzo", self._pal_idx, self._perm_idx)

    def _n_perms(self) -> int:
        return math.factorial(len(PALETTES[self._pal_idx][1]))

    def _update_perm_ui(self):
        if not (hasattr(self, "_perm_lbl") and self._perm_lbl.winfo_exists()):
            return
        colors = self._get_perm_colors()
        self._perm_lbl.config(text=f"{self._perm_idx + 1} / {self._n_perms()}")
        self._fill_swatch_strip(self._psanzo_sw_row, colors)
        self._draw_gradient_band(self._psanzo_grad_cv, colors)

    def _prev_perm(self):
        self._perm_idx = (self._perm_idx - 1) % self._n_perms()
        self._update_perm_ui()
        self._request_preview()

    def _next_perm(self):
        self._perm_idx = (self._perm_idx + 1) % self._n_perms()
        self._update_perm_ui()
        self._request_preview()

    def _on_palette_select(self, idx: int):
        self._pal_idx  = idx
        self._perm_idx = 0
        self._update_perm_ui()
        self._request_preview()

    def _randomize_launch(self):
        self._pal_idx  = 300          # palette 301
        self._perm_idx = 10           # permutation 11
        if hasattr(self, "_pal_dd"):
            self._pal_dd.set_index(self._pal_idx)
        self._update_perm_ui()

    # ── Carte Mandelbrot ──────────────────────────────────────────────────────

    def _load_mandelbrot(self):
        if not MANDEL_NPY.exists():
            self._draw_mandelbrot()
            return
        raw  = np.load(str(MANDEL_NPY))
        pil  = Image.fromarray((raw * 255).astype(np.uint8), mode="L")
        pil  = pil.resize((MANDEL_W, MANDEL_H), Image.BILINEAR)
        gray = np.array(pil)
        rgb  = np.stack([gray] * 3, axis=-1)
        self._mandel_img = ImageTk.PhotoImage(
            Image.fromarray(rgb.astype(np.uint8)))
        self._draw_mandelbrot()

    def _c_to_mandel_pixel(self, c_re: float, c_im: float) -> tuple[int, int]:
        x = int((c_re + MANDEL_BORN) / (2 * MANDEL_BORN) * MANDEL_W)
        y = int((MANDEL_BORN - c_im) / (2 * MANDEL_BORN) * MANDEL_H)
        return (max(0, min(MANDEL_W - 1, x)),
                max(0, min(MANDEL_H - 1, y)))

    def _mandel_pixel_to_c(self, px: int, py: int) -> tuple[float, float]:
        c_re = px / MANDEL_W * (2 * MANDEL_BORN) - MANDEL_BORN
        c_im = MANDEL_BORN - py / MANDEL_H * (2 * MANDEL_BORN)
        return c_re, c_im

    def _draw_mandelbrot(self):
        if "c" not in self._mod_frames:
            return
        cv = self._mandel_cv
        cv.delete("all")
        if self._mandel_img:
            cv.create_image(0, 0, anchor="nw", image=self._mandel_img)

        if self._loop_pts:
            pxs = [self._c_to_mandel_pixel(re, im) for re, im in self._loop_pts]
            if self._loop_drawing and self._loop_ctrl:
                draw_pxs = [pxs[0], self._loop_drag_px]
                smooth   = False
            else:
                draw_pxs = list(pxs)
                if self._loop_closed:
                    draw_pxs.append(pxs[0])
                smooth = (not self._loop_ctrl) and len(draw_pxs) >= 3

            if len(draw_pxs) >= 2:
                flat = [coord for p in draw_pxs for coord in p]
                cv.create_line(*flat, fill="#00cccc", width=1, smooth=smooth)

            sx, sy = pxs[0]
            cv.create_oval(sx - 4, sy - 4, sx + 4, sy + 4,
                           outline="#00ff88", fill="", width=2)

        # Points de frame reparamétrés (densifiés près de ∂M)
        for rpx, rpy in self._reparam_overlay_px:
            cv.create_oval(rpx - 1, rpy - 1, rpx + 1, rpy + 1,
                           outline="", fill="#ffaa00")

        progress = self._t / max(self._duration, 1e-9)
        c  = self._loop_c_at(progress)
        px, py = self._c_to_mandel_pixel(c.real, c.imag)
        sz = 6
        cv.create_line(px - sz, py, px + sz, py, fill="#ff4444", width=1)
        cv.create_line(px, py - sz, px, py + sz, fill="#ff4444", width=1)
        cv.create_oval(px - 3, py - 3, px + 3, py + 3,
                       outline="#ff4444", width=1)

    # ── Module Rotation ───────────────────────────────────────────────────────

    def _build_rotation_section(self, parent):
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["rotation"] = module

        # En-tête
        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text="rotation", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"),
                      padx=8, pady=5)
        t1.pack(side="left")
        self._bind_module_drag("rotation", hdr, t1)

        self._btn(hdr, "✕", lambda: self._remove_module("rotation"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        rot_ctrl = tk.Frame(hdr, bg=HDR_BG)
        rot_ctrl.pack(side="right", padx=8)
        tk.Label(rot_ctrl, text="boucle", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left")
        self._rot_n_var = tk.IntVar(value=self._rot_n)
        tk.Spinbox(rot_ctrl, textvariable=self._rot_n_var,
                   from_=1, to=999, width=3,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_rot_n_change).pack(side="left", padx=(4, 2))
        tk.Label(rot_ctrl, text="×", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 6))
        self._rot_bounce_btn = self._btn(
            rot_ctrl, "⇄", self._toggle_rot_bounce,
            bg=HDR_BG, fg=FG2, font=("Helvetica", 11), padx=4, pady=1)
        self._rot_bounce_btn.pack(side="left")

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        # Champs angle de départ / d'arrivée
        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=10, pady=8)

        row1 = tk.Frame(body, bg=MOD_BG)
        row1.pack(anchor="w", pady=2)
        tk.Label(row1, text="départ", bg=MOD_BG, fg=FG2,
                 font=("Helvetica", 9), width=7, anchor="w").pack(side="left")
        self._rot_start_var = tk.StringVar(value=f"{self._rot_start:g}")
        e1 = tk.Entry(row1, textvariable=self._rot_start_var, width=6,
                      bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                      font=("Courier", 9), justify="right")
        e1.bind("<Return>",   lambda _e: self._on_rot_field_change())
        e1.bind("<FocusOut>", lambda _e: self._on_rot_field_change())
        e1.pack(side="left")
        tk.Label(row1, text="°", bg=MOD_BG, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(2, 0))

        row2 = tk.Frame(body, bg=MOD_BG)
        row2.pack(anchor="w", pady=2)
        tk.Label(row2, text="arrivée", bg=MOD_BG, fg=FG2,
                 font=("Helvetica", 9), width=7, anchor="w").pack(side="left")
        self._rot_end_var = tk.StringVar(value=f"{self._rot_end:g}")
        e2 = tk.Entry(row2, textvariable=self._rot_end_var, width=6,
                      bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                      font=("Courier", 9), justify="right")
        e2.bind("<Return>",   lambda _e: self._on_rot_field_change())
        e2.bind("<FocusOut>", lambda _e: self._on_rot_field_change())
        e2.pack(side="left")
        tk.Label(row2, text="°", bg=MOD_BG, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(2, 0))

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        footer = tk.Frame(module, bg=MOD_BG)
        footer.pack(fill="x", pady=4)
        self._rot_lbl = tk.Label(footer, text=self._fmt_rot(),
                                  bg=MOD_BG, fg=FG2, font=("Courier", 8),
                                  justify="center")
        self._rot_lbl.pack()

    def _on_rot_field_change(self):
        try:
            self._rot_start = float(self._rot_start_var.get().replace(",", "."))
            self._rot_end   = float(self._rot_end_var.get().replace(",", "."))
        except ValueError:
            # Valeur invalide : on remet les valeurs courantes
            self._rot_start_var.set(f"{self._rot_start:g}")
            self._rot_end_var.set(f"{self._rot_end:g}")
            return
        self._rot_lbl.config(text=self._fmt_rot())
        self._request_preview()

    def _fmt_rot(self) -> str:
        progress = self._t / max(self._duration, 1e-9)
        cur = self._rot_at(progress)
        return f"{cur:7.1f}°"

    def _rot_at(self, progress: float) -> float:
        p = self._anim_p("rotation", progress,
                         self._rot_n, self._rot_bounce)
        return self._rot_start + p * (self._rot_end - self._rot_start)
        self._request_preview()

    def _on_rot_n_change(self):
        try:
            self._rot_n = max(1, int(self._rot_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _toggle_rot_bounce(self):
        self._rot_bounce = not self._rot_bounce
        if self._rot_bounce:
            self._rot_bounce_btn.config(bg=ACCENT,       fg=FG,  font=("Helvetica", 11, "bold"))
        else:
            self._rot_bounce_btn.config(bg=self._HDR_BG, fg=FG2, font=("Helvetica", 11))
        self._request_preview()

    # ── Module Zoom ───────────────────────────────────────────────────────────

    def _build_zoom_section(self, parent):
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["zoom"] = module

        # En-tête : titre + boucle N× / ⇄ + ✕
        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text="zoom", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"), padx=8, pady=5)
        t1.pack(side="left")
        self._bind_module_drag("zoom", hdr, t1)

        self._btn(hdr, "✕", lambda: self._remove_module("zoom"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        z_ctrl = tk.Frame(hdr, bg=HDR_BG)
        z_ctrl.pack(side="right", padx=8)
        tk.Label(z_ctrl, text="boucle", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left")
        self._zoom_n_var = tk.IntVar(value=self._zoom_n)
        tk.Spinbox(z_ctrl, textvariable=self._zoom_n_var, from_=1, to=999,
                   width=3, bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_zoom_n_change).pack(side="left", padx=(4, 2))
        tk.Label(z_ctrl, text="×", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 6))
        self._zoom_bounce_btn = self._btn(
            z_ctrl, "⇄", self._toggle_zoom_bounce,
            bg=HDR_BG, fg=FG2, font=("Helvetica", 11), padx=4, pady=1)
        self._zoom_bounce_btn.pack(side="left")

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        # Carte : clic = centre du zoom, rectangle = vue à l'arrivée
        self._zoom_cv = tk.Canvas(module, width=ZOOM_W, height=ZOOM_H,
                                  bg="#0c0c0c", highlightthickness=0,
                                  cursor="crosshair")
        self._zoom_cv.pack()
        self._zoom_cv.bind("<ButtonPress-1>", self._zoom_set_center)
        self._zoom_cv.bind("<B1-Motion>",     self._zoom_set_center)

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        # Champs facteur de zoom départ → arrivée
        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=10, pady=6)
        for label, attr in (("départ", "_zoom_start"), ("arrivée", "_zoom_end")):
            row = tk.Frame(body, bg=MOD_BG)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=label, bg=MOD_BG, fg=FG2, font=("Helvetica", 9),
                     width=7, anchor="w").pack(side="left")
            var = tk.StringVar(value=f"{getattr(self, attr):g}")
            setattr(self, attr + "_var", var)
            e = tk.Entry(row, textvariable=var, width=6,
                         bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                         font=("Courier", 9), justify="right")
            e.bind("<Return>",   lambda _e: self._on_zoom_field_change())
            e.bind("<FocusOut>", lambda _e: self._on_zoom_field_change())
            e.pack(side="left")
            tk.Label(row, text="×", bg=MOD_BG, fg=FG2,
                     font=("Helvetica", 9)).pack(side="left", padx=(2, 0))

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        footer = tk.Frame(module, bg=MOD_BG)
        footer.pack(fill="x", pady=4)
        self._zoom_lbl = tk.Label(footer, text=self._fmt_zoom(),
                                  bg=MOD_BG, fg=FG2, font=("Courier", 8),
                                  justify="center")
        self._zoom_lbl.pack()
        self._draw_zoom()

    def _zoom_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        px = int((x + ZOOM_BORN) / (2 * ZOOM_BORN) * ZOOM_W)
        py = int((ZOOM_BORN - y) / (2 * ZOOM_BORN) * ZOOM_H)
        return px, py

    def _zoom_pixel_to_pos(self, px: int, py: int) -> tuple[float, float]:
        x = px / ZOOM_W * (2 * ZOOM_BORN) - ZOOM_BORN
        y = ZOOM_BORN - py / ZOOM_H * (2 * ZOOM_BORN)
        return x, y

    def _draw_zoom(self):
        if "zoom" not in self._mod_frames:
            return
        cv = self._zoom_cv
        cv.delete("all")
        # Grille aux entiers + axes
        for v in range(-int(ZOOM_BORN), int(ZOOM_BORN) + 1):
            gx, _ = self._zoom_to_pixel(v, 0)
            cv.create_line(gx, 0, gx, ZOOM_H,
                           fill="#3a3a3a" if v == 0 else "#1e1e1e")
            _, gy = self._zoom_to_pixel(0, v)
            cv.create_line(0, gy, ZOOM_W, gy,
                           fill="#3a3a3a" if v == 0 else "#1e1e1e")
        cx_px, cy_px = self._zoom_to_pixel(self._zoom_cx, self._zoom_cy)
        # Rectangles de vue : départ (gris) et arrivée/courant (accent)
        for fac, col, w in ((self._zoom_start, "#666666", 1),
                            (self._zoom_end,   ACCENT,    2)):
            half = (2.0 / max(fac, 1e-9)) / ZOOM_BORN * (ZOOM_W / 2)
            cv.create_rectangle(cx_px - half, cy_px - half,
                                cx_px + half, cy_px + half,
                                outline=col, width=w)
        # Playhead : vue courante
        zf, zx, zy = self._zoom_at(self._t / max(self._duration, 1e-9))
        pcx, pcy = self._zoom_to_pixel(zx, zy)
        half = (2.0 / max(zf, 1e-9)) / ZOOM_BORN * (ZOOM_W / 2)
        cv.create_rectangle(pcx - half, pcy - half, pcx + half, pcy + half,
                            outline="#ff4444", width=1)
        # Croix du centre
        cv.create_line(cx_px - 6, cy_px, cx_px + 6, cy_px, fill=GREEN)
        cv.create_line(cx_px, cy_px - 6, cx_px, cy_px + 6, fill=GREEN)

    def _zoom_set_center(self, event):
        x, y = self._zoom_pixel_to_pos(event.x, event.y)
        self._zoom_cx = max(-ZOOM_BORN, min(ZOOM_BORN, x))
        self._zoom_cy = max(-ZOOM_BORN, min(ZOOM_BORN, y))
        self._zoom_lbl.config(text=self._fmt_zoom())
        self._draw_zoom()
        self._request_preview()

    def _zoom_at(self, progress: float) -> tuple[float, float, float]:
        """(facteur, centre_re, centre_im) — interpolation log du facteur."""
        p = self._anim_p("zoom", progress, self._zoom_n, self._zoom_bounce)
        a = max(self._zoom_start, 1e-9)
        b = max(self._zoom_end, 1e-9)
        fac = a * (b / a) ** p
        return fac, self._zoom_cx, self._zoom_cy

    def _fmt_zoom(self) -> str:
        fac, _, _ = self._zoom_at(self._t / max(self._duration, 1e-9))
        return (f"{self._zoom_start:g}→{self._zoom_end:g}×\n"
                f"{fac:.2f}×  c {self._zoom_cx:+.2f} {self._zoom_cy:+.2f}")

    def _on_zoom_field_change(self):
        try:
            self._zoom_start = max(1e-6, float(self._zoom_start_var.get().replace(",", ".")))
            self._zoom_end   = max(1e-6, float(self._zoom_end_var.get().replace(",", ".")))
        except ValueError:
            self._zoom_start_var.set(f"{self._zoom_start:g}")
            self._zoom_end_var.set(f"{self._zoom_end:g}")
            return
        self._zoom_lbl.config(text=self._fmt_zoom())
        self._draw_zoom()
        self._request_preview()

    def _on_zoom_n_change(self):
        try:
            self._zoom_n = max(1, int(self._zoom_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _toggle_zoom_bounce(self):
        self._zoom_bounce = not self._zoom_bounce
        if self._zoom_bounce:
            self._zoom_bounce_btn.config(bg=ACCENT, fg=FG,
                                         font=("Helvetica", 11, "bold"))
        else:
            self._zoom_bounce_btn.config(bg=self._HDR_BG, fg=FG2,
                                         font=("Helvetica", 11))
        self._request_preview()

    # ── Module Biomorphe ──────────────────────────────────────────────────────

    def _build_biomorphe_section(self, parent):
        """Mode biomorphe : classification OU de Pickover sur le z final de la
        formule Julia active. c vient du module c ; f vient du champ Julia form
        (+ dé). Présence du module = mode actif."""
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["biomorphe"] = module

        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text="biomorphe", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"), padx=8, pady=5)
        t1.pack(side="left")
        self._bind_module_drag("biomorphe", hdr, t1)
        self._btn(hdr, "✕", lambda: self._remove_module("biomorphe"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=10, pady=8)

        def spin(label, attr, frm, to, inc):
            row = tk.Frame(body, bg=MOD_BG)
            row.pack(anchor="w", pady=2)
            tk.Label(row, text=label, bg=MOD_BG, fg=FG2, font=("Helvetica", 9),
                     width=9, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=getattr(self, attr))
            setattr(self, attr + "_var", var)
            tk.Spinbox(row, textvariable=var, from_=frm, to=to, increment=inc,
                       width=6, bg=BG2, fg=FG, insertbackground=FG,
                       relief="flat",
                       command=self._on_bio_change).pack(side="left")

        spin("L (OU)", "_bio_L", 1.0, 50.0, 1.0)
        spin("modulus", "_bio_modbail", 4.0, 1000.0, 10.0)
        spin("max_iter", "_bio_iter", 5, 500, 5)

        # Coloration : structure (min des composantes) ↔ escape time
        rowm = tk.Frame(body, bg=MOD_BG)
        rowm.pack(anchor="w", pady=(4, 0))
        tk.Label(rowm, text="couleur", bg=MOD_BG, fg=FG2, font=("Helvetica", 9),
                 width=9, anchor="w").pack(side="left")
        self._bio_cmode_btn = self._btn(
            rowm, "escape time" if self._bio_color_iter else "structure",
            self._toggle_bio_color,
            bg=BG2, fg=FG2, font=("Helvetica", 8), padx=6, pady=2)
        self._bio_cmode_btn.pack(side="left")

    def _on_bio_change(self, *_):
        try:
            self._bio_L       = float(self._bio_L_var.get())
            self._bio_modbail = float(self._bio_modbail_var.get())
            self._bio_iter    = max(1, int(self._bio_iter_var.get()))
        except (tk.TclError, ValueError):
            return
        self._img_cost_cache = None
        self._request_preview()

    def _toggle_bio_color(self):
        self._bio_color_iter = not self._bio_color_iter
        self._bio_cmode_btn.config(
            text="escape time" if self._bio_color_iter else "structure")
        self._request_preview()

    # ── Module Palette Sanzo ──────────────────────────────────────────────────

    def _build_psanzo_section(self, parent):
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["psanzo"] = module

        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text="palette sanzo", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"), padx=8, pady=5)
        t1.pack(side="left")
        self._bind_module_drag("psanzo", hdr, t1)
        self._btn(hdr, "✕", lambda: self._remove_module("psanzo"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=10, pady=8)

        self._pal_dd = _PaletteDropdown(
            body, self.root, PALETTES,
            initial_idx=self._pal_idx,
            on_select=self._on_palette_select)
        self._pal_dd.pack(anchor="w")

        # Navigation des permutations
        perm_row = tk.Frame(body, bg=MOD_BG)
        perm_row.pack(anchor="w", pady=(8, 0))
        tk.Label(perm_row, text="perm", bg=MOD_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 4))
        self._btn(perm_row, "<", self._prev_perm,
                  bg=BG2, font=("Helvetica", 9), padx=5, pady=2).pack(side="left")
        self._btn(perm_row, ">", self._next_perm,
                  bg=BG2, font=("Helvetica", 9), padx=5, pady=2
                  ).pack(side="left", padx=(4, 0))
        self._perm_lbl = tk.Label(perm_row, text="", bg=MOD_BG, fg=FG2,
                                   font=("Courier", 8), width=9, anchor="w")
        self._perm_lbl.pack(side="left", padx=(6, 0))

        # Squelette commun : rangée d'arrêts + bande de dégradé
        self._psanzo_sw_row = tk.Frame(body, bg=MOD_BG)
        self._psanzo_sw_row.pack(anchor="w", pady=(8, 0))
        self._psanzo_grad_cv = self._color_grad_canvas(body)
        self._psanzo_grad_cv.pack(anchor="w", pady=(8, 0))
        self._update_perm_ui()

    # ── Éléments partagés des modules couleur ─────────────────────────────────

    GRAD_W = 200       # largeur de la bande de dégradé
    GRAD_H = 26        # hauteur de la bande de dégradé

    def _color_grad_canvas(self, body):
        """Bande de dégradé commune aux deux modules couleur."""
        return tk.Canvas(body, width=self.GRAD_W, height=self.GRAD_H,
                         bg="#0c0c0c", highlightthickness=1,
                         highlightbackground=self._SUB_BOR)

    def _draw_gradient_band(self, cv, colors):
        if cv is None or not cv.winfo_exists():
            return
        cv.delete("all")
        n = len(colors)
        W = self.GRAD_W
        for x in range(W):
            t = x / max(1, W - 1) * (n - 1)
            k = min(int(t), n - 2) if n >= 2 else 0
            f = t - k
            a, b = colors[k], colors[min(k + 1, n - 1)]
            rgb = tuple(int(a[j] + f * (b[j] - a[j])) for j in range(3))
            cv.create_line(x, 0, x, self.GRAD_H, fill=_hexcolor(rgb))

    def _fill_swatch_strip(self, row, colors, on_click=None):
        """Rangée de pastilles d'arrêts (cliquables si on_click fourni)."""
        if not (row and row.winfo_exists()):
            return
        for w in row.winfo_children():
            w.destroy()
        for i, rgb in enumerate(colors):
            sw = tk.Label(row, bg=_hexcolor(rgb), width=2, height=1,
                          relief="solid", bd=1,
                          cursor="hand2" if on_click else "arrow")
            sw.pack(side="left", padx=2)
            if on_click:
                sw.bind("<Button-1>", lambda _e, i=i: on_click(i))

    # ── Module Couleur aléatoire ──────────────────────────────────────────────

    def _build_crandom_section(self, parent):
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["crandom"] = module

        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text="couleur aléatoire", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"), padx=8, pady=5)
        t1.pack(side="left")
        self._bind_module_drag("crandom", hdr, t1)
        self._btn(hdr, "✕", lambda: self._remove_module("crandom"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        ctrl = tk.Frame(hdr, bg=HDR_BG)
        ctrl.pack(side="right", padx=8)
        tk.Label(ctrl, text="arrêts", bg=HDR_BG, fg=FG2,
                 font=("Helvetica", 8)).pack(side="left")
        self._crand_n_var = tk.IntVar(value=self._crand_n)
        tk.Spinbox(ctrl, textvariable=self._crand_n_var, from_=2, to=8, width=2,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_crand_n_change).pack(side="left", padx=(4, 6))
        self._btn(ctrl, "🎲", self._randomize_crand,
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 11),
                  padx=4, pady=1).pack(side="left")

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=10, pady=8)
        # Squelette commun : rangée d'arrêts + bande de dégradé
        self._crand_sw_row = tk.Frame(body, bg=MOD_BG)
        self._crand_sw_row.pack(anchor="w")
        self._crand_grad_cv = self._color_grad_canvas(body)
        self._crand_grad_cv.pack(anchor="w", pady=(8, 0))
        tk.Label(body, text="clic sur une pastille → roue + pipette",
                 bg=MOD_BG, fg="#555555",
                 font=("Helvetica", 7)).pack(anchor="w", pady=(4, 0))
        self._rebuild_crand_swatches()

    def _rebuild_crand_swatches(self):
        self._fill_swatch_strip(getattr(self, "_crand_sw_row", None),
                                self._crand_colors, self._pick_crand_color)
        self._draw_gradient_band(getattr(self, "_crand_grad_cv", None),
                                 self._crand_colors)

    def _pick_crand_color(self, i: int):
        init = _hexcolor(self._crand_colors[i])
        rgb, _hex = tkcc.askcolor(color=init, title="Couleur",
                                  parent=self._mod_frames.get("crandom"))
        if rgb is None:
            return
        self._crand_colors[i] = tuple(int(v) for v in rgb)
        self._rebuild_crand_swatches()
        self._request_preview()

    def _randomize_crand(self):
        self._crand_colors = [tuple(random.randint(0, 255) for _ in range(3))
                              for _ in range(self._crand_n)]
        self._rebuild_crand_swatches()
        self._request_preview()

    def _on_crand_n_change(self):
        try:
            n = max(2, min(8, int(self._crand_n_var.get())))
        except (tk.TclError, ValueError):
            return
        cur = self._crand_colors
        if n > len(cur):
            cur = cur + [tuple(random.randint(0, 255) for _ in range(3))
                         for _ in range(n - len(cur))]
        else:
            cur = cur[:n]
        self._crand_n = n
        self._crand_colors = cur
        self._rebuild_crand_swatches()
        self._request_preview()

    # ── Dessin de la boucle ───────────────────────────────────────────────────

    def _loop_press(self, event):
        self._loop_drawing  = True
        self._loop_ctrl     = bool(event.state & CTRL_MASK)
        self._loop_closed   = False
        self._loop_start_px = (event.x, event.y)
        self._loop_last_px  = (event.x, event.y)
        self._loop_drag_px  = (event.x, event.y)
        c_re, c_im = self._mandel_pixel_to_c(event.x, event.y)
        self._loop_pts = [(c_re, c_im)]
        self._draw_mandelbrot()

    def _loop_drag(self, event):
        if not self._loop_drawing:
            return
        if self._loop_ctrl:
            self._loop_drag_px = (event.x, event.y)
            self._draw_mandelbrot()
            return
        lx, ly = self._loop_last_px
        if math.hypot(event.x - lx, event.y - ly) >= SAMPLE_DIST:
            c_re, c_im = self._mandel_pixel_to_c(event.x, event.y)
            self._loop_pts.append((c_re, c_im))
            self._loop_last_px = (event.x, event.y)
            self._draw_mandelbrot()

    def _loop_release(self, event):
        if not self._loop_drawing:
            return
        self._loop_drawing = False

        if self._loop_ctrl:
            if math.hypot(event.x - self._loop_start_px[0],
                          event.y - self._loop_start_px[1]) > 3:
                c_re, c_im = self._mandel_pixel_to_c(event.x, event.y)
                self._loop_pts = [self._loop_pts[0], (c_re, c_im)]
        else:
            c_re, c_im = self._mandel_pixel_to_c(event.x, event.y)
            self._loop_pts.append((c_re, c_im))

        if len(self._loop_pts) >= 3:
            px0, py0 = self._c_to_mandel_pixel(*self._loop_pts[0])
            pxn, pyn = self._c_to_mandel_pixel(*self._loop_pts[-1])
            if math.hypot(pxn - px0, pyn - py0) < CLOSE_THRESH:
                self._loop_closed = True
                self._loop_pts.pop()

        self._refresh_reparam()
        self._draw_mandelbrot()
        self._request_preview()

    def _smooth_loop(self):
        """Lisse la boucle (2 passes de moyennage 1/4-1/2-1/4)."""
        pts = self._loop_pts
        if len(pts) < 3:
            return
        for _ in range(2):
            n      = len(pts)
            result = []
            for i in range(n):
                if self._loop_closed:
                    p0 = pts[(i - 1) % n]
                    p1 = pts[i]
                    p2 = pts[(i + 1) % n]
                else:
                    if i == 0 or i == n - 1:
                        result.append(pts[i])
                        continue
                    p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
                result.append((
                    0.25 * p0[0] + 0.5 * p1[0] + 0.25 * p2[0],
                    0.25 * p0[1] + 0.5 * p1[1] + 0.25 * p2[1],
                ))
            pts = result
        self._loop_pts = pts
        self._refresh_reparam()
        self._draw_mandelbrot()
        self._request_preview()

    # ── Vitesse uniforme (reparamétrisation DEM / image / hybride) ────────────

    _REPARAM_METHODS = {"DEM (rapide)": "dem", "Image (précis)": "image",
                        "Hybride": "hybrid"}
    _REPARAM_LABELS = tuple(_REPARAM_METHODS)
    _REPARAM_LABELS_INV = {v: k for k, v in _REPARAM_METHODS.items()}

    def _on_uniform_toggle(self):
        self._uniform_speed = bool(self._uniform_var.get())
        self._reparam_method_menu.config(
            state="normal" if self._uniform_speed else "disabled")
        self._refresh_reparam()
        self._draw_mandelbrot()

    def _on_reparam_method_change(self, label: str):
        self._reparam_method = self._REPARAM_METHODS[label]
        self._refresh_reparam()
        self._draw_mandelbrot()

    def _refresh_reparam(self):
        """Met à jour l'overlay ; lance la passe image en arrière-plan si requise."""
        if (self._uniform_speed
                and self._reparam_method in ("image", "hybrid")
                and len(self._loop_pts) >= 2
                and self._image_rho_cached() is None):
            self._reparam_overlay_px = []
            self._launch_imgcost_pass()
            return
        self._update_reparam_overlay()

    def _update_reparam_overlay(self):
        """Recalcule les points de frame reparamétrés affichés sur la carte."""
        self._reparam_overlay_px = []
        if not (self._uniform_speed and len(self._loop_pts) >= 2):
            return
        import path_reparam
        method  = self._reparam_method
        img_rho = None
        if method in ("image", "hybrid"):
            img_rho = self._image_rho_cached()
            if img_rho is None:
                method = "dem"   # passe image indisponible : repli analytique
        n_frames = max(2, round(self._duration * self._fps))
        try:
            _, cs = path_reparam.reparametrize_path(
                self._loop_c_at, n_samples=1500, n_frames=n_frames,
                rho_cap=REPARAM_RHO_CAP, method=method, image_rho=img_rho)
        except Exception:
            return
        self._reparam_overlay_px = [
            self._c_to_mandel_pixel(c.real, c.imag) for c in cs]

    def _imgcost_key(self) -> tuple:
        """Clé de cache : chemin, boucles et paramètres de coloration actifs."""
        return (tuple(self._loop_pts), self._loop_closed, self._loop_n,
                self._loop_bounce, self._color_state_key(),
                self._mirror, self._equalize, self._clip_limit,
                self._color_mode, self._smooth, self._julia_formula_str,
                self._trap_mod_id(), self._img_name)

    def _image_rho_cached(self) -> "np.ndarray | None":
        if (self._img_cost_cache is not None
                and self._img_cost_cache[0] == self._imgcost_key()):
            return self._img_cost_cache[1]
        return None

    def _make_render_c_fn(self, size: int):
        """Preview Julia avec les paramètres actifs figés à t courant, c libre."""
        progress  = self._t / max(self._duration, 1e-9)
        trap_mod  = self._trap_mod_id()
        use_trap  = trap_mod is not None
        trap_pos  = self._otrap_pos_at(progress) if use_trap else (0.0, 0.0)
        norm      = self._norm_at(progress) if use_trap else 0.5
        trap_type, trap_rad, trap_tilt, trap_axis, trap_cnt = (
            self._trap_render_args(trap_mod, progress) if use_trap
            else (0, 0.0, 0.0, 0.0, 0.0))
        rot_deg = (self._rot_at(progress)
                   if "rotation" in self._mod_frames else 0.0)
        trap_tex = self._img_tex if trap_mod in ("oimage", "ogeom") else None
        trap_svg = self._svg_pack if trap_mod in ("oimage", "ogeom") else None
        colors   = self._get_perm_colors()
        mirror, equalize = self._mirror, self._equalize
        clip, color_mode = self._clip_limit, self._color_mode
        smooth, iter_fn  = self._smooth, self._julia_iter_fn
        julia_numba      = self._julia_numba
        mandelbrot       = self._mandelbrot
        bio_kw           = self._bio_kwargs()
        if "zoom" in self._mod_frames:
            zoom, zcx, zcy = self._zoom_at(progress)
        else:
            zoom, zcx, zcy = 1.0, 0.0, 0.0

        def render_c(c: complex) -> np.ndarray:
            with _RENDER_LOCK:
                return render_frame(0.0, 1.0, size, size, c, colors,
                                    mirror, equalize, clip,
                                    trap_pos[0], trap_pos[1], norm,
                                    color_mode, rot_deg, iter_fn,
                                    use_trap, smooth, trap_type, trap_rad,
                                    trap_tilt, trap_axis, trap_tex,
                                    julia_numba, trap_cnt, trap_svg,
                                    1, mandelbrot, zoom, zcx, zcy, **bio_kw)
        return render_c

    def _set_reparam_progress(self, text: str):
        if "c" in self._mod_frames and self._reparam_prog_lbl.winfo_exists():
            self._reparam_prog_lbl.config(text=text)

    def _launch_imgcost_pass(self):
        """Calcule image_cost dans un thread (la passe rend N previews)."""
        if self._imgcost_thread and self._imgcost_thread.is_alive():
            return
        import path_reparam
        key      = self._imgcost_key()
        render_c = self._make_render_c_fn(REPARAM_IMG_SIZE)

        def _after(*args):
            try:
                self.root.after(0, *args)
            except RuntimeError:
                pass   # fenêtre fermée pendant la passe

        def _cb(i, n):
            _after(self._set_reparam_progress, f"analyse {i}/{n}")

        def work():
            try:
                rho = path_reparam.image_cost(
                    self._loop_c_at, n_samples=REPARAM_IMG_SAMPLES,
                    render_fn=render_c, progress_cb=_cb)
            except Exception:
                rho = None
            _after(self._imgcost_done, key, rho)

        self._set_reparam_progress(f"analyse 0/{REPARAM_IMG_SAMPLES}")
        self._imgcost_thread = threading.Thread(target=work, daemon=True)
        self._imgcost_thread.start()

    def _imgcost_done(self, key: tuple, rho):
        self._set_reparam_progress("")
        if rho is not None and key == self._imgcost_key():
            self._img_cost_cache = (key, rho)
        elif rho is not None:
            # Chemin ou coloration modifiés pendant la passe : cache invalide,
            # on relance si la méthode image est toujours demandée.
            self._refresh_reparam()
            return
        self._update_reparam_overlay()
        self._draw_mandelbrot()

    # ── Orbit trap plan ──────────────────────────────────────────────────────

    def _otrap_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        # Pas de clamp : le canvas clippe lui-même ce qui déborde de la vue.
        px = int((x - self._otrap_view_cx + OTRAP_BORN) / (2 * OTRAP_BORN) * OTRAP_W)
        py = int((OTRAP_BORN - (y - self._otrap_view_cy)) / (2 * OTRAP_BORN) * OTRAP_H)
        return px, py

    def _otrap_pixel_to_pos(self, px: int, py: int) -> tuple[float, float]:
        x = px / OTRAP_W * (2 * OTRAP_BORN) - OTRAP_BORN + self._otrap_view_cx
        y = OTRAP_BORN - py / OTRAP_H * (2 * OTRAP_BORN) + self._otrap_view_cy
        return x, y

    _PAN_MARGIN = 6   # px : distance au bord qui déclenche le glissement

    def _otrap_autopan(self, ex: int, ey: int) -> tuple[int, int, bool]:
        """Glisse la vue pour garder le crayon dans le cadre (échelle fixe).

        Retourne la position du crayon dans la vue après glissement et un
        booléen indiquant si la vue a bougé."""
        m  = self._PAN_MARGIN
        dx = (ex - m) if ex < m else (ex - (OTRAP_W - m)) if ex > OTRAP_W - m else 0
        dy = (ey - m) if ey < m else (ey - (OTRAP_H - m)) if ey > OTRAP_H - m else 0
        if dx or dy:
            self._otrap_view_cx += dx * (2 * OTRAP_BORN) / OTRAP_W
            self._otrap_view_cy -= dy * (2 * OTRAP_BORN) / OTRAP_H
        return ex - dx, ey - dy, bool(dx or dy)

    def _otrap_recenter(self, _event=None):
        """Recentre la vue sur le chemin (clic droit), ou sur l'origine."""
        pts = self._otrap_pts
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            self._otrap_view_cx = (min(xs) + max(xs)) / 2
            self._otrap_view_cy = (min(ys) + max(ys)) / 2
        else:
            self._otrap_view_cx = self._otrap_view_cy = 0.0
        self._draw_otrap()

    def _draw_otrap(self):
        if self._trap_mod_id() is None:
            return
        cv = self._otrap_cv
        cv.delete("all")

        # Grille : lignes entières visibles dans la vue (axes 0 plus marqués)
        x_lo = self._otrap_view_cx - OTRAP_BORN
        x_hi = self._otrap_view_cx + OTRAP_BORN
        y_lo = self._otrap_view_cy - OTRAP_BORN
        y_hi = self._otrap_view_cy + OTRAP_BORN
        for v in range(math.ceil(x_lo), math.floor(x_hi) + 1):
            gx, _ = self._otrap_to_pixel(v, 0)
            cv.create_line(gx, 0, gx, OTRAP_H,
                           fill="#3a3a3a" if v == 0 else "#1e1e1e", width=1)
        for v in range(math.ceil(y_lo), math.floor(y_hi) + 1):
            _, gy = self._otrap_to_pixel(0, v)
            cv.create_line(0, gy, OTRAP_W, gy,
                           fill="#3a3a3a" if v == 0 else "#1e1e1e", width=1)

        # Indicateur de décentrage de la vue
        if self._otrap_view_cx or self._otrap_view_cy:
            cv.create_text(3, OTRAP_H - 3, anchor="sw",
                           text=f"vue {self._otrap_view_cx:+.2f} "
                                f"{self._otrap_view_cy:+.2f}",
                           fill="#555555", font=("Courier", 7))

        # Boucle tracée
        if self._otrap_pts:
            pxs = [self._otrap_to_pixel(x, y) for x, y in self._otrap_pts]
            if self._otrap_drawing and self._otrap_ctrl:
                draw_pxs = [pxs[0], self._otrap_drag_px]
                smooth   = False
            else:
                draw_pxs = list(pxs)
                if self._otrap_closed:
                    draw_pxs.append(pxs[0])
                smooth = (not self._otrap_ctrl) and len(draw_pxs) >= 3

            if len(draw_pxs) >= 2:
                flat = [coord for p in draw_pxs for coord in p]
                cv.create_line(*flat, fill="#00cccc", width=1, smooth=smooth)

            sx, sy = pxs[0]
            cv.create_oval(sx - 4, sy - 4, sx + 4, sy + 4,
                           outline="#00ff88", fill="", width=2)

        # Position courante
        progress = self._t / max(self._duration, 1e-9)
        pos = self._otrap_pos_at(progress)
        px, py = self._otrap_to_pixel(pos[0], pos[1])
        sz = 6
        cv.create_line(px - sz, py, px + sz, py, fill="#ff4444", width=1)
        cv.create_line(px, py - sz, px, py + sz, fill="#ff4444", width=1)
        cv.create_oval(px - 3, py - 3, px + 3, py + 3,
                       outline="#ff4444", width=1)

    def _otrap_press(self, event):
        self._otrap_drawing  = True
        self._otrap_ctrl     = bool(event.state & CTRL_MASK)
        self._otrap_closed   = False
        self._otrap_start_px = (event.x, event.y)
        self._otrap_last_px  = (event.x, event.y)
        self._otrap_drag_px  = (event.x, event.y)
        x, y = self._otrap_pixel_to_pos(event.x, event.y)
        self._otrap_pts = [(x, y)]
        self._draw_otrap()

    def _otrap_drag(self, event):
        if not self._otrap_drawing:
            return
        # Position réelle du crayon (avant glissement de la vue)
        x, y = self._otrap_pixel_to_pos(event.x, event.y)
        ex, ey, panned = self._otrap_autopan(event.x, event.y)
        if self._otrap_ctrl:
            self._otrap_drag_px = (ex, ey)
            self._draw_otrap()
            return
        if panned:
            # La vue a bougé : l'ancre pixel du dernier point est recalculée
            self._otrap_last_px = self._otrap_to_pixel(*self._otrap_pts[-1])
        lx, ly = self._otrap_last_px
        if math.hypot(ex - lx, ey - ly) >= SAMPLE_DIST:
            self._otrap_pts.append((x, y))
            self._otrap_last_px = (ex, ey)
            self._draw_otrap()
        elif panned:
            self._draw_otrap()

    def _otrap_release(self, event):
        if not self._otrap_drawing:
            return
        self._otrap_drawing = False

        if self._otrap_ctrl:
            if math.hypot(event.x - self._otrap_start_px[0],
                          event.y - self._otrap_start_px[1]) > 3:
                x, y = self._otrap_pixel_to_pos(event.x, event.y)
                self._otrap_pts = [self._otrap_pts[0], (x, y)]
        else:
            x, y = self._otrap_pixel_to_pos(event.x, event.y)
            self._otrap_pts.append((x, y))

        if len(self._otrap_pts) >= 3:
            px0, py0 = self._otrap_to_pixel(*self._otrap_pts[0])
            pxn, pyn = self._otrap_to_pixel(*self._otrap_pts[-1])
            if math.hypot(pxn - px0, pyn - py0) < CLOSE_THRESH:
                self._otrap_closed = True
                self._otrap_pts.pop()

        self._draw_otrap()
        self._request_preview()

    def _smooth_otrap(self):
        """Lisse la boucle orbit trap (2 passes de moyennage 1/4-1/2-1/4)."""
        pts = self._otrap_pts
        if len(pts) < 3:
            return
        for _ in range(2):
            n      = len(pts)
            result = []
            for i in range(n):
                if self._otrap_closed:
                    p0 = pts[(i - 1) % n]
                    p1 = pts[i]
                    p2 = pts[(i + 1) % n]
                else:
                    if i == 0 or i == n - 1:
                        result.append(pts[i])
                        continue
                    p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
                result.append((
                    0.25 * p0[0] + 0.5 * p1[0] + 0.25 * p2[0],
                    0.25 * p0[1] + 0.5 * p1[1] + 0.25 * p2[1],
                ))
            pts = result
        self._otrap_pts = pts
        self._draw_otrap()
        self._request_preview()

    def _otrap_pos_at(self, progress: float) -> tuple[float, float]:
        pts = self._otrap_pts
        if not pts:
            return (self._otrap_x, self._otrap_y)
        if len(pts) == 1:
            return pts[0]

        progress = self._anim_p("position", progress,
                                self._otrap_n, self._otrap_bounce)

        path = pts + [pts[0]] if self._otrap_closed else pts

        dists = [0.0]
        for i in range(1, len(path)):
            d = math.hypot(path[i][0] - path[i - 1][0],
                           path[i][1] - path[i - 1][1])
            dists.append(dists[-1] + d)
        total = dists[-1]
        if total == 0.0:
            return pts[0]

        target = max(0.0, min(1.0, progress)) * total
        for i in range(1, len(dists)):
            if dists[i] >= target or i == len(dists) - 1:
                seg   = dists[i] - dists[i - 1]
                alpha = (target - dists[i - 1]) / seg if seg > 0 else 0.0
                x     = path[i - 1][0] + alpha * (path[i][0] - path[i - 1][0])
                y     = path[i - 1][1] + alpha * (path[i][1] - path[i - 1][1])
                return (x, y)
        return pts[-1]

    def _fmt_otrap(self, pos: tuple | None = None) -> str:
        if pos is None:
            pos = (self._otrap_x, self._otrap_y)
        return f"trap  {pos[0]: .4f} / {pos[1]: .4f}"

    # ── Norme boucle (chemin libre sur ovale) ────────────────────────────────

    # ── Norm : helpers coordonnées slider ────────────────────────────────────

    def _norm_y_for_val(self, val: float) -> int:
        """Valeur → y pixel sur la piste (top = NORM_MIN, bottom = NORM_MAX)."""
        y = NORM_SLIDER_TOP + (val - NORM_MIN) / (NORM_MAX - NORM_MIN) * (NORM_SLIDER_BOT - NORM_SLIDER_TOP)
        return max(NORM_SLIDER_TOP, min(NORM_SLIDER_BOT, int(y)))

    def _norm_val_for_y(self, y: int) -> float:
        val = NORM_MIN + (y - NORM_SLIDER_TOP) / (NORM_SLIDER_BOT - NORM_SLIDER_TOP) * (NORM_MAX - NORM_MIN)
        return max(NORM_MIN, min(NORM_MAX, val))

    # ── Norm : dessin ─────────────────────────────────────────────────────────

    def _draw_norm(self):
        # oimage / ogeom : pas de sous-module norm
        if self._trap_mod_id() in (None, "oimage", "ogeom"):
            return
        cv = self._norm_cv
        cv.delete("all")
        progress = self._t / max(self._duration, 1e-9)
        cx = KNOB_CX

        # ── Slider de plage ──────────────────────────────────────────────────
        y0, y1 = NORM_SLIDER_TOP, NORM_SLIDER_BOT

        cv.create_line(cx, y0, cx, y1, fill="#2a2a2a", width=4, capstyle="round")

        y_lo = self._norm_y_for_val(self._norm_range_lo)
        y_hi = self._norm_y_for_val(self._norm_range_hi)
        cv.create_line(cx, y_lo, cx, y_hi, fill=ACCENT, width=4, capstyle="round")

        # Poignée lo (borne basse = vers le haut)
        cv.create_oval(cx - 7, y_lo - 7, cx + 7, y_lo + 7,
                       fill="#0c0c0c", outline=GREEN, width=2)
        # Poignée hi (borne haute = vers le bas)
        cv.create_oval(cx - 7, y_hi - 7, cx + 7, y_hi + 7,
                       fill="#0c0c0c", outline=ACCENT, width=2)

        # Labels extrêmes
        cv.create_text(cx, y0 - 5, text=f"{NORM_MIN:.2f}",
                       fill="#383838", font=("Courier", 6), anchor="s")
        cv.create_text(cx, y1 + 5, text=f"{NORM_MAX:.1f}",
                       fill="#383838", font=("Courier", 6), anchor="n")

        # Labels des poignées
        cv.create_text(cx + 10, y_lo, text=f"{self._norm_range_lo:.2f}",
                       fill=GREEN, font=("Courier", 6), anchor="w")
        cv.create_text(cx + 10, y_hi, text=f"{self._norm_range_hi:.2f}",
                       fill=ACCENT, font=("Courier", 6), anchor="w")

        # Playhead sur le slider
        cur_val = self._norm_at(progress)
        y_cur   = self._norm_y_for_val(cur_val)
        cv.create_line(cx - 10, y_cur, cx + 10, y_cur, fill="#ff4444", width=1)

        # ── Knob ─────────────────────────────────────────────────────────────
        kx, ky, kr = KNOB_CX, KNOB_CY, KNOB_R

        cv.create_oval(kx - kr, ky - kr, kx + kr, ky + kr,
                       fill="#181818", outline="#333", width=1)

        if self._knob_angle > 0.5:
            cv.create_arc(kx - kr + 3, ky - kr + 3, kx + kr - 3, ky + kr - 3,
                          start=90, extent=-min(359.9, self._knob_angle),
                          style="pieslice", fill="#162436", outline="")

        outline_col = ACCENT if self._norm_drag_what == "knob" else "#444"
        cv.create_oval(kx - kr, ky - kr, kx + kr, ky + kr,
                       fill="", outline=outline_col, width=1)

        # Marque 0° (12 h)
        cv.create_oval(kx - 2, ky - kr + 4, kx + 2, ky - kr + 8,
                       fill="#555", outline="")

        # Point cible (angle de fin du balayage)
        end_rad = math.radians(self._knob_angle - 90)
        ex = kx + (kr - 2) * math.cos(end_rad)
        ey = ky + (kr - 2) * math.sin(end_rad)
        cv.create_oval(ex - 3, ey - 3, ex + 3, ey + 3, fill=ACCENT, outline="")

        # Indicateur playhead (position animée en rouge)
        n        = max(1, self._norm_n)
        p        = (progress * n) % 1.0
        anim_rad = math.radians(p * self._knob_angle - 90)
        ix = kx + (kr - 5) * math.cos(anim_rad)
        iy = ky + (kr - 5) * math.sin(anim_rad)
        cv.create_line(kx, ky, ix, iy, fill="#ff4444", width=2)
        cv.create_oval(kx - 2, ky - 2, kx + 2, ky + 2, fill="#ff4444", outline="")

        # Étiquette angle
        cv.create_text(kx, ky + kr + 9, text=f"{self._knob_angle:.0f}°",
                       fill=FG2, font=("Courier", 7))

    # ── Norm : interactions ───────────────────────────────────────────────────

    def _norm_press(self, event):
        dx = event.x - KNOB_CX
        dy = event.y - KNOB_CY
        if math.hypot(dx, dy) <= KNOB_R + 5:
            self._norm_drag_what = "knob"
            self._knob_drag_ref  = (event.x, event.y)   # position précédente
            return
        # Piste du slider
        y_lo = self._norm_y_for_val(self._norm_range_lo)
        y_hi = self._norm_y_for_val(self._norm_range_hi)
        if abs(event.y - y_lo) <= 10:
            self._norm_drag_what = "lo"
        elif abs(event.y - y_hi) <= 10:
            self._norm_drag_what = "hi"
        elif NORM_SLIDER_TOP - 10 <= event.y <= NORM_SLIDER_BOT + 10:
            self._norm_drag_what = "lo" if abs(event.y - y_lo) < abs(event.y - y_hi) else "hi"
        self._draw_norm()

    def _norm_drag(self, event):
        if self._norm_drag_what == "knob":
            prev_x, prev_y = self._knob_drag_ref
            prev_a = (math.degrees(math.atan2(prev_y - KNOB_CY, prev_x - KNOB_CX)) + 90) % 360.0
            cur_a  = (math.degrees(math.atan2(event.y - KNOB_CY, event.x - KNOB_CX)) + 90) % 360.0
            delta  = cur_a - prev_a
            if delta >  180: delta -= 360
            if delta < -180: delta += 360
            self._knob_angle    = max(0.0, min(360.0, self._knob_angle + delta))
            self._knob_drag_ref = (event.x, event.y)
        elif self._norm_drag_what == "lo":
            self._norm_range_lo = min(self._norm_val_for_y(event.y), self._norm_range_hi)
        elif self._norm_drag_what == "hi":
            self._norm_range_hi = max(self._norm_val_for_y(event.y), self._norm_range_lo)
        else:
            return
        self._draw_norm()

    def _norm_release(self, event):
        self._norm_drag(event)
        self._norm_drag_what = None
        self._knob_drag_ref  = None
        self._norm_lbl.config(text=self._fmt_norm())
        self._draw_norm()
        self._request_preview()

    # ── Norm : logique d'animation ────────────────────────────────────────────

    def _norm_at(self, progress: float) -> float:
        """Retourne la valeur de norme pour progress ∈ [0, 1].

        Le knob définit l'angle de balayage total (0–360°) :
        - 180° = aller simple lo → hi
        - 360° = aller-retour lo → hi → lo
        Onde cosinusoïdale (période 360°) : vitesse nulle aux deux extrêmes,
        ce qui rend les rebonds au min et au max perceptuellement symétriques
        malgré la sensibilité non-linéaire de 1−exp(−d/norm_max).
        """
        p = self._anim_p("norm", progress, self._norm_n, self._norm_bounce)
        theta = p * self._knob_angle
        alpha = (1.0 - math.cos(math.radians(theta))) / 2.0
        return self._norm_range_lo + alpha * (self._norm_range_hi - self._norm_range_lo)

    def _fmt_norm(self) -> str:
        progress = self._t / max(self._duration, 1e-9)
        val = self._norm_at(progress)
        return f"{self._norm_range_lo:.2f}→{self._norm_range_hi:.2f}\n{val:.3f}"

    # ── Rayon (trap cercle) : helpers, dessin, interactions ──────────────────

    def _rad_at(self, progress: float) -> float:
        """Rayon du trap cercle pour progress ∈ [0, 1] (lo → hi, boucle N×, ⇄)."""
        p = self._anim_p("rayon", progress, self._rad_n, self._rad_bounce)
        return self._rad_lo + p * (self._rad_hi - self._rad_lo)

    def _fmt_rad(self) -> str:
        progress = self._t / max(self._duration, 1e-9)
        return (f"{self._rad_lo:.2f}→{self._rad_hi:.2f}\n"
                f"r {self._rad_at(progress):.3f}")

    def _rad_y_for_val(self, val: float) -> int:
        y = (NORM_SLIDER_TOP + (val - RAD_MIN) / (RAD_MAX - RAD_MIN)
             * (NORM_SLIDER_BOT - NORM_SLIDER_TOP))
        return max(NORM_SLIDER_TOP, min(NORM_SLIDER_BOT, int(y)))

    def _rad_val_for_y(self, y: int) -> float:
        val = (RAD_MIN + (y - NORM_SLIDER_TOP)
               / (NORM_SLIDER_BOT - NORM_SLIDER_TOP) * (RAD_MAX - RAD_MIN))
        return max(RAD_MIN, min(RAD_MAX, val))

    def _draw_rad(self):
        if self._trap_mod_id() not in ("ocircle", "oring"):
            return
        cv = self._rad_cv
        cv.delete("all")
        progress = self._t / max(self._duration, 1e-9)
        cx = KNOB_CX
        y0, y1 = NORM_SLIDER_TOP, NORM_SLIDER_BOT

        cv.create_line(cx, y0, cx, y1, fill="#2a2a2a", width=4, capstyle="round")

        y_lo = self._rad_y_for_val(self._rad_lo)
        y_hi = self._rad_y_for_val(self._rad_hi)
        cv.create_line(cx, y_lo, cx, y_hi, fill=ACCENT, width=4, capstyle="round")

        cv.create_oval(cx - 7, y_lo - 7, cx + 7, y_lo + 7,
                       fill="#0c0c0c", outline=GREEN, width=2)
        cv.create_oval(cx - 7, y_hi - 7, cx + 7, y_hi + 7,
                       fill="#0c0c0c", outline=ACCENT, width=2)

        cv.create_text(cx, y0 - 5, text=f"{RAD_MIN:.2f}",
                       fill="#383838", font=("Courier", 6), anchor="s")
        cv.create_text(cx, y1 + 5, text=f"{RAD_MAX:.1f}",
                       fill="#383838", font=("Courier", 6), anchor="n")

        cv.create_text(cx + 10, y_lo, text=f"{self._rad_lo:.2f}",
                       fill=GREEN, font=("Courier", 6), anchor="w")
        cv.create_text(cx + 10, y_hi, text=f"{self._rad_hi:.2f}",
                       fill=ACCENT, font=("Courier", 6), anchor="w")

        # Playhead (rayon courant)
        y_cur = self._rad_y_for_val(self._rad_at(progress))
        cv.create_line(cx - 10, y_cur, cx + 10, y_cur, fill="#ff4444", width=1)

        # Aperçu du cercle courant (en bas, à l'échelle RAD_MAX → rayon max dessiné)
        pr     = self._rad_at(progress) / RAD_MAX
        max_r  = 24
        ccx, ccy = cx, KNOB_CY
        cv.create_oval(ccx - max_r, ccy - max_r, ccx + max_r, ccy + max_r,
                       outline="#2a2a2a", width=1)
        r_px = max(1, int(pr * max_r))
        cv.create_oval(ccx - r_px, ccy - r_px, ccx + r_px, ccy + r_px,
                       outline=ACCENT, width=2)

    def _rad_press(self, event):
        y_lo = self._rad_y_for_val(self._rad_lo)
        y_hi = self._rad_y_for_val(self._rad_hi)
        if abs(event.y - y_lo) <= 10:
            self._rad_drag_what = "lo"
        elif abs(event.y - y_hi) <= 10:
            self._rad_drag_what = "hi"
        elif NORM_SLIDER_TOP - 10 <= event.y <= NORM_SLIDER_BOT + 10:
            self._rad_drag_what = ("lo" if abs(event.y - y_lo) < abs(event.y - y_hi)
                                   else "hi")
        self._draw_rad()

    def _rad_drag(self, event):
        if self._rad_drag_what == "lo":
            self._rad_lo = min(self._rad_val_for_y(event.y), self._rad_hi)
        elif self._rad_drag_what == "hi":
            self._rad_hi = max(self._rad_val_for_y(event.y), self._rad_lo)
        else:
            return
        self._draw_rad()

    def _rad_release(self, event):
        self._rad_drag(event)
        self._rad_drag_what = None
        self._rad_lbl.config(text=self._fmt_rad())
        self._request_preview()

    def _on_rad_n_change(self):
        try:
            self._rad_n = max(1, int(self._rad_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _toggle_rad_bounce(self):
        self._rad_bounce = not self._rad_bounce
        if self._rad_bounce:
            self._rad_bounce_btn.config(bg=ACCENT, fg=FG,
                                        font=("Helvetica", 10, "bold"))
        else:
            self._rad_bounce_btn.config(bg=self._SUB_HDR, fg=FG2,
                                        font=("Helvetica", 10))
        self._request_preview()

    # ── Orientation de l'anneau 3D ────────────────────────────────────────────

    def _ori_at(self, progress: float) -> tuple[float, float]:
        """(inclinaison, axe) en degrés pour progress ∈ [0, 1] (boucle N×, ⇄)."""
        p = self._anim_p("orientation", progress,
                         self._ori_n, self._ori_bounce)
        tilt = self._tilt_lo + p * (self._tilt_hi - self._tilt_lo)
        axis = self._axis_lo + p * (self._axis_hi - self._axis_lo)
        return tilt, axis

    def _fmt_ori(self) -> str:
        progress = self._t / max(self._duration, 1e-9)
        tilt, axis = self._ori_at(progress)
        return f"i {tilt:6.1f}°\na {axis:6.1f}°"

    def _draw_ori(self):
        """Aperçu de la projection de l'anneau : ellipse a=r, b=r·|cos i|."""
        if "oring" not in self._mod_frames:
            return
        cv = self._ori_cv
        cv.delete("all")
        progress   = self._t / max(self._duration, 1e-9)
        tilt, axis = self._ori_at(progress)
        tilt_r = math.radians(tilt)
        psi    = math.radians(axis) + math.pi / 2   # angle du grand axe
        ccx, ccy = NORM_W // 2, 52
        a_px = 40
        b_px = max(1.0, a_px * abs(math.cos(tilt_r)))
        cs, sn = math.cos(psi), math.sin(psi)
        # Cercle de référence (anneau vu de face)
        cv.create_oval(ccx - a_px, ccy - a_px, ccx + a_px, ccy + a_px,
                       outline="#2a2a2a", width=1)
        # Axe de bascule (diamètre fixe = grand axe de l'ellipse)
        cv.create_line(ccx - a_px * cs, ccy + a_px * sn,
                       ccx + a_px * cs, ccy - a_px * sn,
                       fill="#383838", width=1, dash=(2, 3))
        pts = []
        for k in range(36):
            t = 2 * math.pi * k / 36
            ex = a_px * math.cos(t)
            ey = b_px * math.sin(t)
            # y écran inversé
            pts.append(ccx + ex * cs - ey * sn)
            pts.append(ccy - (ex * sn + ey * cs))
        cv.create_polygon(*pts, outline=ACCENT, fill="", width=2, smooth=True)

    def _on_ori_field_change(self):
        try:
            tilt_lo = float(self._ori_vars["tilt"][0].get().replace(",", "."))
            tilt_hi = float(self._ori_vars["tilt"][1].get().replace(",", "."))
            axis_lo = float(self._ori_vars["axis"][0].get().replace(",", "."))
            axis_hi = float(self._ori_vars["axis"][1].get().replace(",", "."))
        except ValueError:
            # Valeur invalide : on remet les valeurs courantes
            self._ori_vars["tilt"][0].set(f"{self._tilt_lo:g}")
            self._ori_vars["tilt"][1].set(f"{self._tilt_hi:g}")
            self._ori_vars["axis"][0].set(f"{self._axis_lo:g}")
            self._ori_vars["axis"][1].set(f"{self._axis_hi:g}")
            return
        self._tilt_lo, self._tilt_hi = tilt_lo, tilt_hi
        self._axis_lo, self._axis_hi = axis_lo, axis_hi
        self._ori_lbl.config(text=self._fmt_ori())
        self._draw_ori()
        self._request_preview()

    def _on_ori_n_change(self):
        try:
            self._ori_n = max(1, int(self._ori_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _toggle_ori_bounce(self):
        self._ori_bounce = not self._ori_bounce
        if self._ori_bounce:
            self._ori_bounce_btn.config(bg=ACCENT, fg=FG,
                                        font=("Helvetica", 10, "bold"))
        else:
            self._ori_bounce_btn.config(bg=self._SUB_HDR, fg=FG2,
                                        font=("Helvetica", 10))
        self._request_preview()

    def _on_norm_n_change(self):
        try:
            self._norm_n = max(1, int(self._norm_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    # ── c(t) ─────────────────────────────────────────────────────────────────

    def _loop_c_at(self, progress: float) -> complex:
        pts = self._loop_pts
        if not pts:
            return complex(self._c_re, self._c_im)
        if len(pts) == 1:
            return complex(pts[0][0], pts[0][1])

        progress = self._anim_p("c", progress,
                                self._loop_n, self._loop_bounce)

        path = pts + [pts[0]] if self._loop_closed else pts

        dists = [0.0]
        for i in range(1, len(path)):
            d = math.hypot(path[i][0] - path[i - 1][0],
                           path[i][1] - path[i - 1][1])
            dists.append(dists[-1] + d)
        total = dists[-1]
        if total == 0.0:
            return complex(pts[0][0], pts[0][1])

        target = max(0.0, min(1.0, progress)) * total
        for i in range(1, len(dists)):
            if dists[i] >= target or i == len(dists) - 1:
                seg   = dists[i] - dists[i - 1]
                alpha = (target - dists[i - 1]) / seg if seg > 0 else 0.0
                re    = path[i - 1][0] + alpha * (path[i][0] - path[i - 1][0])
                im    = path[i - 1][1] + alpha * (path[i][1] - path[i - 1][1])
                return complex(re, im)
        return complex(pts[-1][0], pts[-1][1])

    def _fmt_c(self, c: complex | None = None) -> str:
        if c is None:
            c = complex(self._c_re, self._c_im)
        return f"c  {c.real: .4f}{c.imag:+.4f}i"

    # ── Helpers UI ────────────────────────────────────────────────────────────

    @staticmethod
    def _lighten(hex_color: str, amount: int = 20) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return "#{:02x}{:02x}{:02x}".format(
            min(255, r + amount), min(255, g + amount), min(255, b + amount))

    def _btn(self, parent, text, cmd, bg=BG2, fg=FG,
             font=("Helvetica", 10), padx=8, pady=3):
        lbl = tk.Label(parent, text=text, bg=bg, fg=fg, font=font,
                       padx=padx, pady=pady, cursor="hand2")
        lbl.bind("<Button-1>", lambda _e: cmd())
        lbl.bind("<Enter>",    lambda _e: lbl.config(bg=self._lighten(bg)))
        lbl.bind("<Leave>",    lambda _e: lbl.config(bg=bg))
        return lbl

    def _fmt_time(self) -> str:
        p = (self._current_pass % max(1, self._meta_n)) + 1
        total = "∞" if self._loop_infinite else str(self._meta_n)
        return f"t {self._t:5.2f}s/{self._duration:.0f}s  ×{p}/{total}"

    # ── Controles statiques ───────────────────────────────────────────────────

    def _on_mirror_change(self):
        try:
            self._mirror = max(1, int(self._mirror_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _on_clip_change(self):
        try:
            self._clip_limit = max(0.1, float(self._clip_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    def _toggle_equalize(self):
        self._equalize = not self._equalize
        if self._equalize:
            self._eq_btn.config(text="  ON  ", bg=GREEN, fg="#aaffaa")
        else:
            self._eq_btn.config(text="  OFF  ", bg=BG2, fg=FG2)
        self._request_preview()

    def _toggle_smooth(self):
        self._smooth = not self._smooth
        if self._smooth:
            self._smooth_btn.config(text="  ON  ", bg=GREEN, fg="#aaffaa")
        else:
            self._smooth_btn.config(text="  OFF  ", bg=BG2, fg=FG2)
        self._request_preview()

    def _toggle_mandelbrot(self):
        self._mandelbrot = not self._mandelbrot
        if self._mandelbrot:
            self._mandel_btn.config(text="Mandelbrot", bg=ACCENT, fg=FG)
        else:
            self._mandel_btn.config(text="Julia", bg=BG3, fg=FG)
        # En mode Mandelbrot le chemin c(t) du module c n'a plus d'effet
        # (c = grille par pixel) ; le cache de la métrique image dépend du mode.
        self._img_cost_cache = None
        self._request_preview()

    _CMODE_MAP       = {"OkLab": "oklab", "RGB": "rgb", "HSV": "hsv", "Cyclic": "cyclic"}


    def _on_color_mode_change(self, _=None):
        self._color_mode = self._CMODE_MAP.get(self._cmode_var.get(), "oklab")
        self._request_preview()

    def _on_julia_formula_change(self, _=None):
        raw = self._julia_entry_var.get().strip()
        if not raw:
            return
        self._julia_formula_str = raw
        self._julia_formula_compiled = None   # force recompile in worker
        self._julia_entry.config(bg=BG2)
        self._request_preview()

    def _on_loop_n_change(self):
        try:
            self._loop_n = max(1, int(self._loop_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._refresh_reparam()
        self._draw_mandelbrot()
        self._request_preview()

    def _on_otrap_n_change(self):
        try:
            self._otrap_n = max(1, int(self._otrap_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._request_preview()

    # ── Controles transport ───────────────────────────────────────────────────

    def _on_fps_change(self):
        try:
            self._fps = max(1, int(self._fps_var.get()))
        except (tk.TclError, ValueError):
            pass
        # Le nombre de points de l'overlay vitesse uniforme dépend de fps×durée
        self._refresh_reparam()
        self._draw_mandelbrot()

    def _on_dur_change(self):
        try:
            self._duration = max(1.0, float(self._dur_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._scrub.config(to=self._duration)
        self._refresh_reparam()
        self._update_labels()

    def _on_scrub(self, val):
        try:
            t = float(val)
        except ValueError:
            return
        self._t = max(0.0, min(self._duration, t))
        self._update_labels()
        self._schedule_preview(delay=80)

    def _seek(self, t: float):
        self._t = max(0.0, min(self._duration, t))
        self._scrub_var.set(self._t)
        self._update_labels()
        self._request_preview()

    def _rewind(self):
        self._current_pass = 0
        self._seek(0)

    def _step(self, d: int):
        self._seek(self._t + d / max(1, self._fps))

    def _on_meta_n_change(self):
        try:
            self._meta_n = max(1, int(self._meta_n_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._update_labels()

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._play_btn.config(text=" PAUSE ", bg=RED, fg="#ffcccc")
            self._tick()
        else:
            self._play_btn.config(text="  PLAY  ", bg=GREEN, fg="#aaffaa")
            if self._play_after:
                self.root.after_cancel(self._play_after)

    _HDR_BG = "#282828"
    _SUB_HDR = "#2e2e2e"   # header des sous-modules (position, norm)

    def _toggle_infinite(self):
        self._loop_infinite = not self._loop_infinite
        if self._loop_infinite:
            self._inf_btn.config(bg=ACCENT,       fg=FG,  font=("Helvetica", 13, "bold"))
        else:
            self._inf_btn.config(bg=self._HDR_BG, fg=FG2, font=("Helvetica", 13))
        self._update_labels()

    def _toggle_loop_bounce(self):
        self._loop_bounce = not self._loop_bounce
        if self._loop_bounce:
            self._loop_bounce_btn.config(bg=ACCENT,       fg=FG,  font=("Helvetica", 11, "bold"))
        else:
            self._loop_bounce_btn.config(bg=self._HDR_BG, fg=FG2, font=("Helvetica", 11))
        self._refresh_reparam()
        self._draw_mandelbrot()
        self._request_preview()

    def _toggle_otrap_bounce(self):
        self._otrap_bounce = not self._otrap_bounce
        if self._otrap_bounce:
            self._otrap_bounce_btn.config(bg=ACCENT,       fg=FG,  font=("Helvetica", 11, "bold"))
        else:
            self._otrap_bounce_btn.config(bg=self._SUB_HDR, fg=FG2, font=("Helvetica", 11))
        self._request_preview()

    def _toggle_norm_bounce(self):
        self._norm_bounce = not self._norm_bounce
        if self._norm_bounce:
            self._norm_bounce_btn.config(bg=ACCENT,       fg=FG,  font=("Helvetica", 10, "bold"))
        else:
            self._norm_bounce_btn.config(bg=self._SUB_HDR, fg=FG2, font=("Helvetica", 10))
        self._request_preview()

    def _tick(self):
        if not self._playing:
            return
        t   = self._t + 1.0 / max(1, self._fps)
        dur = self._duration

        if t >= dur:
            # On évite t=dur exactement : progress=1.0 → (1.0×n)%1.0=0.0 (image subliminale)
            t = dur - 1e-6
            self._current_pass += 1

            more = self._loop_infinite or self._current_pass < self._meta_n
            if not more:
                self._seek(0)
                self._current_pass = 0
                self._toggle_play()
                return
            t = 0.0   # nouvelle passe depuis le début

        self._seek(t)
        self._play_after = self.root.after(
            max(1, int(1000 / self._fps)), self._tick)

    def _update_labels(self):
        self._time_lbl.config(text=self._fmt_time())
        progress = self._t / max(self._duration, 1e-9)
        if "c" in self._mod_frames:
            c = self._loop_c_at(progress)
            self._c_lbl.config(text=self._fmt_c(c))
            self._draw_mandelbrot()
        trap_mod = self._trap_mod_id()
        if trap_mod is not None:
            pos = self._otrap_pos_at(progress)
            self._otrap_lbl.config(text=self._fmt_otrap(pos))
            self._draw_otrap()
            if trap_mod not in ("oimage", "ogeom"):   # pas de sous-module norm
                self._draw_norm()
                self._norm_lbl.config(text=self._fmt_norm())
        if self._trap_mod_id() in ("ocircle", "oring"):
            self._draw_rad()
            self._rad_lbl.config(text=self._fmt_rad())
        if "oring" in self._mod_frames:
            self._draw_ori()
            self._ori_lbl.config(text=self._fmt_ori())
        pp_key = {"odroite": "lin", "osinus": "sin",
                  "oimage": "img", "ogeom": "geo"}.get(self._trap_mod_id())
        if pp_key:
            self._pp[pp_key]["lbl"].config(text=self._fmt_pp(pp_key))
        if "rotation" in self._mod_frames:
            self._rot_lbl.config(text=self._fmt_rot())
        if "zoom" in self._mod_frames:
            self._draw_zoom()
            self._zoom_lbl.config(text=self._fmt_zoom())
        self._update_velocity_playheads()

    # ── Preview (thread) ─────────────────────────────────────────────────────

    def _schedule_preview(self, delay: int = 120):
        if self._refresh_id:
            self.root.after_cancel(self._refresh_id)
        self._refresh_id = self.root.after(delay, self._request_preview)

    def _request_preview(self):
        self._prev_dirty = True
        if self._prev_thread and self._prev_thread.is_alive():
            return
        self._prev_thread = threading.Thread(
            target=self._preview_worker, daemon=True)
        self._prev_thread.start()

    def _preview_worker(self):
        while True:
            self._prev_dirty = False
            t          = self._t
            dur        = self._duration
            progress   = t / max(dur, 1e-9)
            has_c      = "c" in self._mod_frames
            trap_mod   = self._trap_mod_id()
            use_trap   = trap_mod is not None
            has_rot    = "rotation" in self._mod_frames
            c          = (self._loop_c_at(progress) if has_c
                          else complex(self._c_re, self._c_im))
            trap_pos   = self._otrap_pos_at(progress) if use_trap else (0.0, 0.0)
            norm       = self._norm_at(progress) if use_trap else 0.5
            trap_type, trap_rad, trap_tilt, trap_axis, trap_cnt = (
                self._trap_render_args(trap_mod, progress) if use_trap
                else (0, 0.0, 0.0, 0.0, 0.0))
            trap_tex   = (self._img_tex
                          if trap_mod in ("oimage", "ogeom") else None)
            trap_svg   = (self._svg_pack
                          if trap_mod in ("oimage", "ogeom") else None)
            rot_deg    = self._rot_at(progress) if has_rot else 0.0
            if "zoom" in self._mod_frames:
                zoom, zcx, zcy = self._zoom_at(progress)
            else:
                zoom, zcx, zcy = 1.0, 0.0, 0.0
            colors     = self._get_perm_colors()
            mirror     = self._mirror
            equalize   = self._equalize
            clip_limit = self._clip_limit
            color_mode = self._color_mode
            smooth     = self._smooth
            # Compile la formule si elle a changé
            formula = self._julia_formula_str
            if formula != self._julia_formula_compiled:
                normalized = _normalize_formula(formula).replace(' ', '')
                if normalized == _DEFAULT_FORMULA_NORMALIZED:
                    self._julia_iter_fn = None
                    self._julia_numba   = None
                else:
                    try:
                        self._julia_iter_fn = _compile_julia_iter(formula)
                        self.root.after(0, lambda: self._julia_entry.config(bg=BG2))
                    except Exception:
                        self.root.after(0, lambda: self._julia_entry.config(bg='#3a1515'))
                    else:
                        try:
                            self._julia_numba = _compile_julia_numba(formula)
                        except Exception:
                            self._julia_numba = None   # repli moteur numpy
                self._julia_formula_compiled = formula
            julia_iter_fn = self._julia_iter_fn
            julia_numba   = self._julia_numba
            # En pause : raffinement anti-crénelage 2× (4 orbites par pixel)
            aa = 1 if self._playing else 2
            try:
                with _RENDER_LOCK:
                    arr = render_frame(t, dur, PREV_W, PREV_H, c,
                                       colors, mirror, equalize, clip_limit,
                                       trap_pos[0], trap_pos[1], norm,
                                       color_mode, rot_deg, julia_iter_fn,
                                       use_trap, smooth, trap_type, trap_rad,
                                       trap_tilt, trap_axis, trap_tex,
                                       julia_numba, trap_cnt, trap_svg, aa,
                                       self._mandelbrot, zoom, zcx, zcy,
                                       **self._bio_kwargs())
                img = ImageTk.PhotoImage(Image.fromarray(arr))
                self.root.after(0, self._show_preview, img)
            except Exception:
                pass
            if not self._prev_dirty:
                break

    # ── Render / export MP4 ───────────────────────────────────────────────────

    _RENDER_RESOLUTIONS = [
        ("540p  — 832×540",   832,  540),
        ("1080p — 1080×702",  1080, 702),
        ("1920p — 1920×1248", 1920, 1248),
    ]

    def _open_render_dialog(self):
        if hasattr(self, '_render_win') and self._render_win and self._render_win.winfo_exists():
            self._render_win.lift()
            return

        # Synchronise les valeurs éventuellement tapées dans les spinbox FPS /
        # passe avant de calculer les infos et le nombre de frames.
        self._on_fps_change()
        self._on_dur_change()

        win = tk.Toplevel(self.root)
        win.title("Render")
        win.configure(bg=BG)
        win.resizable(False, False)
        self._render_win = win

        pad = {"padx": 14, "pady": 5}

        # — Résolution —
        tk.Label(win, text="Résolution", bg=BG, fg=FG2,
                 font=("Helvetica", 9, "bold"), **pad).grid(row=0, column=0, sticky="w")
        self._render_res_var = tk.IntVar(value=1)   # index dans _RENDER_RESOLUTIONS
        for i, (label, _, _h) in enumerate(self._RENDER_RESOLUTIONS):
            tk.Radiobutton(win, text=label, variable=self._render_res_var, value=i,
                           bg=BG, fg=FG, selectcolor=BG2, activebackground=BG,
                           font=("Courier", 9)).grid(row=0, column=i + 1, padx=6, pady=5)

        # — Sortie —
        tk.Label(win, text="Fichier", bg=BG, fg=FG2,
                 font=("Helvetica", 9, "bold"), **pad).grid(row=1, column=0, sticky="w")
        self._render_path_var = tk.StringVar(value=str(Path.home() / "fractal_studio.mp4"))
        path_entry = tk.Entry(win, textvariable=self._render_path_var,
                              bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                              font=("Courier", 9), width=38)
        path_entry.grid(row=1, column=1, columnspan=2, padx=(0, 4), pady=5, sticky="w")
        self._btn(win, "…", self._browse_render_path,
                  bg=BG3, font=("Helvetica", 9), padx=6, pady=2
                  ).grid(row=1, column=3, padx=(0, 14), pady=5)

        # — Infos —
        n_passes = self._meta_n if not self._loop_infinite else 1
        total_s  = self._duration * n_passes
        n_frames = math.ceil(total_s * self._fps)
        info = f"{total_s:.1f}s  ({n_passes} passe{'s' if n_passes > 1 else ''} × {self._duration:.1f}s)  @{self._fps} fps  →  {n_frames} frames"
        self._render_info_lbl = tk.Label(win, text=info, bg=BG, fg=FG2,
                                          font=("Helvetica", 8))
        self._render_info_lbl.grid(row=2, column=0, columnspan=4, padx=14, pady=(0, 4), sticky="w")

        # — Barre de progression —
        tk.Frame(win, bg=BG3, height=1).grid(row=3, column=0, columnspan=4, sticky="ew", padx=10)
        self._render_prog = ttk.Progressbar(win, length=460, maximum=n_frames)
        self._render_prog.grid(row=4, column=0, columnspan=4, padx=14, pady=(8, 4))
        self._render_prog_lbl = tk.Label(win, text="", bg=BG, fg=FG2,
                                          font=("Courier", 8))
        self._render_prog_lbl.grid(row=5, column=0, columnspan=4)

        # — Boutons —
        btn_row = tk.Frame(win, bg=BG)
        btn_row.grid(row=6, column=0, columnspan=4, pady=(8, 12))
        self._render_aa_var = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_row, text="Anti-crénelage 2×",
                       variable=self._render_aa_var,
                       bg=BG, fg=FG2, activebackground=BG,
                       activeforeground=FG, selectcolor=BG2,
                       highlightthickness=0, bd=0,
                       font=("Helvetica", 9)).pack(side="left", padx=(0, 10))
        self._render_go_btn = self._btn(btn_row, "  RENDER  ", self._start_render,
                                         bg="#1e3d1e", fg="#88ee88",
                                         font=("Helvetica", 10, "bold"), padx=14, pady=4)
        self._render_go_btn.pack(side="left", padx=8)
        self._btn(btn_row, "Fermer", win.destroy,
                  bg=BG3, font=("Helvetica", 9), padx=10, pady=4).pack(side="left", padx=4)

        self._render_thread: threading.Thread | None = None
        self._render_cancel = False

    def _browse_render_path(self):
        path = tkfd.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("Tous", "*.*")],
            initialfile="fractal_studio.mp4",
        )
        if path:
            self._render_path_var.set(path)

    def _start_render(self):
        if self._render_thread and self._render_thread.is_alive():
            self._render_cancel = True
            return
        self._render_cancel = False
        self._render_go_btn.config(text="  STOP  ", bg="#4a1a1a", fg="#ff8888")
        idx    = self._render_res_var.get()
        _, W, H = self._RENDER_RESOLUTIONS[idx]
        path   = self._render_path_var.get()
        self._render_thread = threading.Thread(
            target=self._run_render, args=(path, W, H, self._uniform_speed),
            kwargs={"rho_cap": REPARAM_RHO_CAP,
                    "method": self._reparam_method,
                    "image_rho": self._image_rho_cached(),
                    "aa": 2 if self._render_aa_var.get() else 1},
            daemon=True)
        self._render_thread.start()

    def _run_render(self, path: str, width: int, height: int,
                    reparametrize: bool = False,
                    n_samples: int = 4000,
                    eps: float = 1e-4,
                    rho_cap: float | None = None,
                    smooth_sigma: float = 2.0,
                    method: str = "dem",
                    smooth_sigma_image: float = 4.0,
                    image_rho: "np.ndarray | None" = None,
                    aa: int = 1):
        import imageio
        fps      = self._fps
        dur      = self._duration
        n_passes = self._meta_n if not self._loop_infinite else 1
        n_frames = math.ceil(dur * n_passes * fps)

        def _upd_text(msg):
            if (hasattr(self, '_render_prog_lbl')
                    and self._render_prog_lbl.winfo_exists()):
                self._render_prog_lbl.config(text=msg)

        # Reparamétrisation : warp progress → t pour une vitesse de
        # déformation uniforme le long du chemin c. Sans effet si désactivée.
        warp = None
        if reparametrize and "c" in self._mod_frames and len(self._loop_pts) >= 2:
            import path_reparam
            if method in ("image", "hybrid") and image_rho is None:
                # Pas de cache valide : passe image dans ce thread de rendu.
                image_rho = path_reparam.image_cost(
                    self._loop_c_at, n_samples=REPARAM_IMG_SAMPLES,
                    render_fn=self._make_render_c_fn(REPARAM_IMG_SIZE),
                    progress_cb=lambda i, n: self.root.after(
                        0, _upd_text, f"analyse image {i}/{n}"))
            warp = path_reparam.reparam_warp(
                self._loop_c_at, n_samples=n_samples, eps=eps,
                rho_cap=rho_cap, smooth_sigma=smooth_sigma,
                method=method, smooth_sigma_image=smooth_sigma_image,
                image_rho=image_rho)

        colors        = self._get_perm_colors()
        mirror        = self._mirror
        equalize      = self._equalize
        clip_limit    = self._clip_limit
        color_mode    = self._color_mode
        julia_iter_fn = self._julia_iter_fn
        julia_numba   = self._julia_numba
        has_c         = "c" in self._mod_frames
        trap_mod      = self._trap_mod_id()
        use_trap      = trap_mod is not None
        trap_tex      = (self._img_tex
                         if trap_mod in ("oimage", "ogeom") else None)
        trap_svg      = (self._svg_pack
                         if trap_mod in ("oimage", "ogeom") else None)
        has_rot       = "rotation" in self._mod_frames
        has_zoom      = "zoom" in self._mod_frames
        smooth        = self._smooth
        mandelbrot    = self._mandelbrot
        bio_kw        = self._bio_kwargs()

        def _upd(i):
            if not (hasattr(self, '_render_prog') and self._render_prog.winfo_exists()):
                return
            self._render_prog['value'] = i
            self._render_prog_lbl.config(text=f"frame {i} / {n_frames}")

        try:
            with imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=8, pixelformat="yuv420p",
                                    macro_block_size=None) as writer:
                for i in range(n_frames):
                    if self._render_cancel:
                        break
                    t_total  = i / fps
                    t        = t_total % dur
                    progress = t / max(dur, 1e-9)

                    c_prog   = (float(np.interp(progress, warp[0], warp[1]))
                                if warp is not None else progress)
                    c        = (self._loop_c_at(c_prog) if has_c
                                else complex(self._c_re, self._c_im))
                    pos      = self._otrap_pos_at(progress) if use_trap else (0.0, 0.0)
                    norm     = self._norm_at(progress) if use_trap else 0.5
                    trap_type, trap_rad, trap_tilt, trap_axis, trap_cnt = (
                        self._trap_render_args(trap_mod, progress) if use_trap
                        else (0, 0.0, 0.0, 0.0, 0.0))
                    rot_deg  = self._rot_at(progress) if has_rot else 0.0
                    if has_zoom:
                        zoom, zcx, zcy = self._zoom_at(progress)
                    else:
                        zoom, zcx, zcy = 1.0, 0.0, 0.0

                    with _RENDER_LOCK:
                        arr = render_frame(t, dur, width, height, c,
                                           colors, mirror, equalize, clip_limit,
                                           pos[0], pos[1], norm,
                                           color_mode, rot_deg, julia_iter_fn,
                                           use_trap, smooth, trap_type, trap_rad,
                                           trap_tilt, trap_axis, trap_tex,
                                           julia_numba, trap_cnt, trap_svg, aa,
                                           mandelbrot, zoom, zcx, zcy, **bio_kw)
                    writer.append_data(arr)
                    self.root.after(0, _upd, i + 1)

            status = "Annulé." if self._render_cancel else f"✓ Sauvegardé → {path}"
        except Exception as exc:
            status = f"Erreur : {exc}"

        def _done():
            if not (hasattr(self, '_render_prog_lbl') and self._render_prog_lbl.winfo_exists()):
                return
            self._render_prog_lbl.config(text=status)
            self._render_go_btn.config(text="  RENDER  ", bg="#1e3d1e", fg="#88ee88")
        self.root.after(0, _done)

    def _show_preview(self, img):
        self._img_ref = img
        self._canvas.config(image=img)

    # ── Raccourcis ────────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<space>", lambda _e: self._toggle_play())
        self.root.bind("<Left>",  lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))

    # ── Lancement ────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    FractalStudio().run()
