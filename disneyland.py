#!/usr/bin/env python3
"""
disneyland.py  —  Generateur de videos fractales par boucles de parametres.
Etape 3 : parametres statiques a droite, carte Mandelbrot + boucles en bas.
"""

from __future__ import annotations

import itertools
import math
import random
import re
import threading
import tkinter as tk
import tkinter.filedialog as tkfd
import tkinter.ttk as ttk
from pathlib import Path

import numpy as np
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

NORM_W          = 132    # largeur de la colonne norme
NORM_H          = 200    # hauteur = même que la carte OTRAP
NORM_MIN        = 0.05
NORM_MAX        = 2.0
NORM_SLIDER_TOP = 16     # y haut de la piste du slider
NORM_SLIDER_BOT = 122    # y bas de la piste du slider
KNOB_CX         = NORM_W // 2   # centre x du knob
KNOB_CY         = 160    # centre y du knob
KNOB_R          = 22     # rayon du knob


CLOSE_THRESH = 18
SAMPLE_DIST  =  4
CTRL_MASK    = 0x4

FPS_DEFAULT = 24
DUR_DEFAULT = 10.0

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_palette(colors: list) -> list:
    n   = len(colors)
    pos = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return [pos, [list(c) for c in colors]]


def _hexcolor(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


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

def _trap_julia_numpy(Z: np.ndarray, iter_fn, c: complex,
                      trap_params: np.ndarray,
                      n: int = 80, B: float = 256.0,
                      norm_max: float = 1.0) -> np.ndarray:
    """Orbit trap avec formule arbitraire — numpy vectorisé, supporte toute expression."""
    z = Z.copy()
    trap = complex(trap_params[0], trap_params[1])
    min_d = np.full(Z.shape, 1e18, dtype=np.float64)
    with np.errstate(all='ignore'):
        for _ in range(n):
            z = np.asarray(iter_fn(z, c), dtype=np.complex128)
            np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            d = np.abs(z - trap)
            np.minimum(min_d, d, out=min_d)
    return 1.0 - np.exp(-min_d / norm_max)


def _escape_julia_numpy(Z: np.ndarray, iter_fn, c: complex,
                        n: int = 80, B: float = 256.0,
                        smooth: bool = True) -> np.ndarray:
    """Escape time avec formule arbitraire — numpy vectorisé.
    Même convention que iteration.escape_speed : V dans [0,1], 0 = non échappé."""
    z     = Z.astype(np.complex128)
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
                 smooth: bool = True) -> np.ndarray:
    borne   = 2.0
    borne_y = borne * height / width
    xs = np.linspace(-borne, borne, width)
    ys = np.linspace(-borne_y, borne_y, height)
    Z  = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    if view_rot_deg:
        theta = math.radians(view_rot_deg)
        Z = Z * complex(math.cos(theta), -math.sin(theta))

    if use_trap:
        trap_params = np.array([trap_x, trap_y, 0.0, 0.5])
        if julia_iter_fn is None:
            V = orbit_trap.trap_julia(Z, 1 + 0j, 0 + 0j, c,
                                      0, trap_params, N_ITER, 256.0, trap_norm)
        else:
            V = _trap_julia_numpy(Z, julia_iter_fn, c, trap_params,
                                  N_ITER, 256.0, trap_norm)
    else:
        # Mode escape time classique (pas de module orbit trap)
        if julia_iter_fn is None:
            V = iteration.escape_speed(Z, 1 + 0j, 0 + 0j, c,
                                       N_ITER, 256.0, smooth)
        else:
            V = _escape_julia_numpy(Z, julia_iter_fn, c,
                                    N_ITER, 256.0, smooth)
    palette  = _make_palette(colors)
    renderer = render.FractalRenderer(
        palette, mode=color_mode, n_iter=N_ITER,
        repeat=mirror, equalize=equalize, clip_limit=clip_limit,
        # Escape time : plage fixe pour que l'égalisation ne saute pas
        # d'une frame à l'autre quand les extrêmes de V fluctuent.
        eq_range=None if use_trap else (0.0, 1.0),
    )
    return renderer.render(V)


# ── Application ───────────────────────────────────────────────────────────────

class Disneyland:

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

        # Boucle norme — knob + slider
        self._norm_range_lo: float = 0.5    # borne basse
        self._norm_range_hi: float = 1.5    # borne haute
        self._knob_angle:    float = 0.0    # balayage en degrés (0–360°)
        self._norm_n:        int   = 1
        self._norm_bounce:   bool  = False
        self._norm_drag_what: str | None   = None

        # Module rotation
        self._rot_start:    float = 0.0     # angle de départ (degrés)
        self._rot_end:      float = 360.0   # angle d'arrivée (degrés)
        self._rot_n:        int   = 1
        self._rot_bounce:   bool  = False
        self._knob_drag_ref:  tuple | None = None   # (start_mouse_angle, start_knob)

        # Parametres statiques
        self._perm_idx    = 10   # permutation 11
        self._mirror      = 1
        self._equalize    = True
        self._clip_limit  = 3.0
        self._color_mode  = "oklab"
        self._smooth      = True    # lissage escape time
        self._julia_formula_str      = "z^2 + c"
        self._julia_formula_compiled = _DEFAULT_FORMULA_NORMALIZED
        self._julia_iter_fn          = None   # None = z^2+c fast numba path

        self._prev_thread: threading.Thread | None = None
        self._prev_dirty  = False
        self._refresh_id  = None
        self._mandel_img: ImageTk.PhotoImage | None = None

        # Registre des modules dynamiques (zone du bas)
        self._mod_frames:   dict[str, tk.Frame] = {}
        self._module_order: list[str] = []
        self._dragging_mod: str | None = None

        self.root = tk.Tk()
        self.root.title("Disneyland")
        self.root.configure(bg=BG)

        self._build_ui()
        self.root.update_idletasks()
        # Fige la hauteur de la zone modules sur la configuration la plus
        # haute (tous les modules construits) pour que l'ajout/retrait de
        # modules ne redimensionne jamais la fenêtre.
        self._bottom.config(height=self._bottom.winfo_reqheight())
        self._bottom.pack_propagate(False)
        # Lancement par défaut : seul le module c est actif
        self._remove_module("otrap",    refresh=False)
        self._remove_module("rotation", refresh=False)
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

        tk.Label(top, text="FPS", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(20, 2))
        self._fps_var = tk.IntVar(value=self._fps)
        tk.Spinbox(top, textvariable=self._fps_var, from_=1, to=60, width=3,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_fps_change).pack(side="left")

        tk.Label(top, text="passe", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(10, 2))
        self._dur_var = tk.DoubleVar(value=self._duration)
        tk.Spinbox(top, textvariable=self._dur_var, from_=1, to=600,
                   increment=1, width=4,
                   bg=BG2, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_dur_change).pack(side="left")
        tk.Label(top, text="s", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(2, 2))

        self._btn(top, "  ▶  RENDER  ", self._open_render_dialog,
                  bg="#1e3d1e", fg="#88ee88",
                  font=("Helvetica", 10, "bold"), padx=10, pady=4
                  ).pack(side="right", padx=12, pady=3)

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
        # puis otrap et rotation sont retirés pour le lancement par défaut).
        for mid in ("c", "otrap", "rotation"):
            self._build_module(mid)
            self._module_order.append(mid)
        self._repack_modules()

    # ── Gestion des modules dynamiques ────────────────────────────────────────

    _MOD_LABELS = {"c": "c (Mandelbrot)", "otrap": "orbit trap", "rotation": "rotation"}

    def _build_module(self, mod_id: str):
        if mod_id == "c":
            self._build_mandelbrot_section(self._bottom)
        elif mod_id == "otrap":
            self._build_otrap_section(self._bottom)
        elif mod_id == "rotation":
            self._build_rotation_section(self._bottom)

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
        self._build_module(mod_id)
        self._module_order.append(mod_id)
        self._repack_modules()
        if mod_id == "c":
            self._draw_mandelbrot()
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
            self._request_preview()

    def _show_module_menu(self, event):
        avail = [m for m in ("c", "otrap", "rotation")
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
        tk.Label(parent, text="PALETTE", bg=BG, fg=FG2,
                 font=("Helvetica", 8, "bold")).pack(anchor="w")

        self._pal_dd = _PaletteDropdown(
            parent, self.root, PALETTES,
            initial_idx=0,
            on_select=self._on_palette_select)
        self._pal_dd.pack(anchor="w", pady=(4, 0))

        # Permutation
        perm_row = tk.Frame(parent, bg=BG)
        perm_row.pack(anchor="w", pady=(6, 0))

        self._btn(perm_row, "<", self._prev_perm,
                  bg=BG2, font=("Helvetica", 9), padx=5, pady=2
                  ).pack(side="left")

        self._perm_sw_cv = tk.Canvas(perm_row, bg=BG,
                                      width=_PaletteDropdown.TRIGGER_SW_W,
                                      height=12, highlightthickness=0)
        self._perm_sw_cv.pack(side="left", padx=4)

        self._btn(perm_row, ">", self._next_perm,
                  bg=BG2, font=("Helvetica", 9), padx=5, pady=2
                  ).pack(side="left")

        self._perm_lbl = tk.Label(perm_row, text="", bg=BG, fg=FG2,
                                   font=("Courier", 8), width=9, anchor="w")
        self._perm_lbl.pack(side="left", padx=(6, 0))

        tk.Frame(parent, bg=BG3, height=1).pack(fill="x", pady=(12, 8))

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
                                     relief="flat", font=("Courier", 9), width=18)
        self._julia_entry.bind("<Return>",   lambda _e: self._on_julia_formula_change())
        self._julia_entry.bind("<FocusOut>", lambda _e: self._on_julia_formula_change())
        self._julia_entry.pack(side="left")

    def _build_otrap_section(self, parent):
        """Sur-module orbit trap : contient les sous-modules position et norm."""
        MOD_BG  = "#202020"
        MOD_BOR = "#3a3a3a"
        HDR_BG  = "#282828"
        SUB_BOR = "#454545"
        SUB_HDR = "#2e2e2e"

        module = tk.Frame(parent, bg=MOD_BG,
                          highlightthickness=1, highlightbackground=MOD_BOR)
        self._mod_frames["otrap"] = module

        # ── En-tête du sur-module ─────────────────────────────────────────────
        hdr = tk.Frame(module, bg=HDR_BG)
        hdr.pack(fill="x")
        t1 = tk.Label(hdr, text="orbit trap", bg=HDR_BG, fg=ACCENT,
                      font=("Helvetica", 10, "bold"),
                      padx=8, pady=3)
        t1.pack(side="left")
        self._bind_module_drag("otrap", hdr, t1)

        self._btn(hdr, "✕", lambda: self._remove_module("otrap"),
                  bg=HDR_BG, fg=FG2, font=("Helvetica", 9),
                  padx=5, pady=1).pack(side="right", padx=(0, 4))

        tk.Frame(module, bg=MOD_BOR, height=1).pack(fill="x")

        # ── Corps : les deux sous-modules côte à côte ─────────────────────────
        body = tk.Frame(module, bg=MOD_BG)
        body.pack(padx=4, pady=4)

        # ── Sous-module position ──────────────────────────────────────────────
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

        tk.Frame(pos_mod, bg=SUB_BOR, height=1).pack(fill="x")

        footer = tk.Frame(pos_mod, bg=MOD_BG)
        footer.pack(fill="x", padx=8, pady=5)
        self._otrap_lbl = tk.Label(footer, text=self._fmt_otrap(),
                                    bg=MOD_BG, fg=FG2, font=("Courier", 9))
        self._otrap_lbl.pack(side="left")
        self._btn(footer, "Lisser", self._smooth_otrap,
                  bg=BG2, fg=FG2, font=("Helvetica", 8),
                  padx=6, pady=2).pack(side="right")

        # ── Sous-module norm ──────────────────────────────────────────────────
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

        self._draw_otrap()
        self._draw_norm()

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

    # ── Palette + permutations ────────────────────────────────────────────────

    def _get_perm_colors(self) -> list:
        idx    = self._pal_dd.selected_index
        orig   = PALETTES[idx][1]
        perms  = list(itertools.permutations(orig))
        return list(perms[self._perm_idx % len(perms)])

    def _n_perms(self) -> int:
        idx = self._pal_dd.selected_index
        return math.factorial(len(PALETTES[idx][1]))

    def _update_perm_ui(self):
        colors = self._get_perm_colors()
        n      = self._n_perms()
        self._perm_lbl.config(text=f"{self._perm_idx + 1} / {n}")
        sw = self._perm_sw_cv
        sw.delete("all")
        w, h = _PaletteDropdown.SW_W, _PaletteDropdown.SW_H
        p    = _PaletteDropdown.SW_PAD
        for i, rgb in enumerate(colors):
            x0 = i * (w + p)
            sw.create_rectangle(x0, 1, x0 + w, 1 + h,
                                  fill=_hexcolor(rgb), outline="")

    def _prev_perm(self):
        self._perm_idx = (self._perm_idx - 1) % self._n_perms()
        self._update_perm_ui()
        self._request_preview()

    def _next_perm(self):
        self._perm_idx = (self._perm_idx + 1) % self._n_perms()
        self._update_perm_ui()
        self._request_preview()

    def _on_palette_select(self, idx: int):
        self._perm_idx = 0
        self._update_perm_ui()
        self._request_preview()

    def _randomize_launch(self):
        self._pal_dd.set_index(300)   # palette 301
        self._perm_idx = 10           # permutation 11
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
        n   = max(1, self._rot_n)
        raw = progress * n
        p   = raw % 1.0
        if self._rot_bounce and int(raw) % 2 == 1:
            p = 1.0 - p
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
        self._draw_mandelbrot()
        self._request_preview()

    # ── Orbit trap plan ──────────────────────────────────────────────────────

    def _otrap_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        px = int((x + OTRAP_BORN) / (2 * OTRAP_BORN) * OTRAP_W)
        py = int((OTRAP_BORN - y) / (2 * OTRAP_BORN) * OTRAP_H)
        return (max(0, min(OTRAP_W - 1, px)),
                max(0, min(OTRAP_H - 1, py)))

    def _otrap_pixel_to_pos(self, px: int, py: int) -> tuple[float, float]:
        x = px / OTRAP_W * (2 * OTRAP_BORN) - OTRAP_BORN
        y = OTRAP_BORN - py / OTRAP_H * (2 * OTRAP_BORN)
        return x, y

    def _draw_otrap(self):
        if "otrap" not in self._mod_frames:
            return
        cv = self._otrap_cv
        cv.delete("all")

        # Grille légère à ±1
        for v in (-1.0, 1.0):
            gx, _ = self._otrap_to_pixel(v, 0)
            cv.create_line(gx, 0, gx, OTRAP_H, fill="#1e1e1e")
            _, gy = self._otrap_to_pixel(0, v)
            cv.create_line(0, gy, OTRAP_W, gy, fill="#1e1e1e")

        # Axes principaux
        ax, ay = self._otrap_to_pixel(0, 0)
        cv.create_line(0, ay, OTRAP_W, ay, fill="#3a3a3a", width=1)
        cv.create_line(ax, 0, ax, OTRAP_H, fill="#3a3a3a", width=1)

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
        if self._otrap_ctrl:
            self._otrap_drag_px = (event.x, event.y)
            self._draw_otrap()
            return
        lx, ly = self._otrap_last_px
        if math.hypot(event.x - lx, event.y - ly) >= SAMPLE_DIST:
            x, y = self._otrap_pixel_to_pos(event.x, event.y)
            self._otrap_pts.append((x, y))
            self._otrap_last_px = (event.x, event.y)
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

        n   = max(1, self._otrap_n)
        raw = progress * n
        p   = raw % 1.0
        if self._otrap_bounce and int(raw) % 2 == 1:
            p = 1.0 - p
        progress = p

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
        if "otrap" not in self._mod_frames:
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
        n   = max(1, self._norm_n)
        raw = progress * n
        p   = raw % 1.0
        if self._norm_bounce and int(raw) % 2 == 1:
            p = 1.0 - p
        theta = p * self._knob_angle
        alpha = (1.0 - math.cos(math.radians(theta))) / 2.0
        return self._norm_range_lo + alpha * (self._norm_range_hi - self._norm_range_lo)

    def _fmt_norm(self) -> str:
        progress = self._t / max(self._duration, 1e-9)
        val = self._norm_at(progress)
        return f"{self._norm_range_lo:.2f}→{self._norm_range_hi:.2f}\n{val:.3f}"

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

        n   = max(1, self._loop_n)
        raw = progress * n
        p   = raw % 1.0
        if self._loop_bounce and int(raw) % 2 == 1:
            p = 1.0 - p
        progress = p

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

    def _on_dur_change(self):
        try:
            self._duration = max(1.0, float(self._dur_var.get()))
        except (tk.TclError, ValueError):
            pass
        self._scrub.config(to=self._duration)
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
        if "otrap" in self._mod_frames:
            pos = self._otrap_pos_at(progress)
            self._otrap_lbl.config(text=self._fmt_otrap(pos))
            self._draw_otrap()
            self._draw_norm()
            self._norm_lbl.config(text=self._fmt_norm())
        if "rotation" in self._mod_frames:
            self._rot_lbl.config(text=self._fmt_rot())

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
            use_trap   = "otrap" in self._mod_frames
            has_rot    = "rotation" in self._mod_frames
            c          = (self._loop_c_at(progress) if has_c
                          else complex(self._c_re, self._c_im))
            trap_pos   = self._otrap_pos_at(progress) if use_trap else (0.0, 0.0)
            norm       = self._norm_at(progress) if use_trap else 0.5
            rot_deg    = self._rot_at(progress) if has_rot else 0.0
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
                else:
                    try:
                        self._julia_iter_fn = _compile_julia_iter(formula)
                        self.root.after(0, lambda: self._julia_entry.config(bg=BG2))
                    except Exception:
                        self.root.after(0, lambda: self._julia_entry.config(bg='#3a1515'))
                self._julia_formula_compiled = formula
            julia_iter_fn = self._julia_iter_fn
            try:
                arr = render_frame(t, dur, PREV_W, PREV_H, c,
                                   colors, mirror, equalize, clip_limit,
                                   trap_pos[0], trap_pos[1], norm,
                                   color_mode, rot_deg, julia_iter_fn,
                                   use_trap, smooth)
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
        self._render_path_var = tk.StringVar(value=str(Path.home() / "disneyland.mp4"))
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
            initialfile="disneyland.mp4",
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
            target=self._run_render, args=(path, W, H), daemon=True)
        self._render_thread.start()

    def _run_render(self, path: str, width: int, height: int):
        import imageio
        fps      = self._fps
        dur      = self._duration
        n_passes = self._meta_n if not self._loop_infinite else 1
        n_frames = math.ceil(dur * n_passes * fps)

        colors        = self._get_perm_colors()
        mirror        = self._mirror
        equalize      = self._equalize
        clip_limit    = self._clip_limit
        color_mode    = self._color_mode
        julia_iter_fn = self._julia_iter_fn
        has_c         = "c" in self._mod_frames
        use_trap      = "otrap" in self._mod_frames
        has_rot       = "rotation" in self._mod_frames
        smooth        = self._smooth

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

                    c       = (self._loop_c_at(progress) if has_c
                               else complex(self._c_re, self._c_im))
                    pos     = self._otrap_pos_at(progress) if use_trap else (0.0, 0.0)
                    norm    = self._norm_at(progress) if use_trap else 0.5
                    rot_deg = self._rot_at(progress) if has_rot else 0.0

                    arr = render_frame(t, dur, width, height, c,
                                       colors, mirror, equalize, clip_limit,
                                       pos[0], pos[1], norm,
                                       color_mode, rot_deg, julia_iter_fn,
                                       use_trap, smooth)
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
    Disneyland().run()
