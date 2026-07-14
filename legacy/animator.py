#!/usr/bin/env python3
"""
animator.py — Generateur de videos fractales avec timeline a keyframes.

Lancement :  myenv/bin/python animator.py

Raccourcis :
  Espace        lecture / pause
  <- / ->       avance de 1 frame
  K             ajouter un keyframe au temps courant
  Suppr         supprimer le keyframe selectionne
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import numpy as np
from PIL import Image, ImageTk

import orbit_trap
import render

# ── Constantes ────────────────────────────────────────────────────────────────
N_ITER   = 80
PREV_W   = 460
PREV_H   = int(PREV_W * 1964 / 3024)

MANDEL_W = 240
MANDEL_H = 240
MANDEL_BORNE = 2.0           # map couvre Re/Im in [-2, 2]

TRAP_TYPE_MAP = {
    "point": 0, "line": 1, "cross": 2,
    "circle": 3, "square": 4, "sine": 5,
}
EASING_MODES = ["linear", "ease-in", "ease-out", "ease-in-out", "step"]

BG     = "#1a1a1a"
BG2    = "#252525"
BG3    = "#2e2e2e"
FG     = "#dddddd"
FG2    = "#888888"
ACCENT = "#4a8abf"
GREEN  = "#3a8a3a"
RED    = "#8a3a3a"
GOLD   = "#c8a040"

PALETTES      = render.load_sanzo_palettes()
PALETTE_NAMES = render.load_sanzo_names()

# Groupes de parametres : (nom_interne, label, type, min, max, resolution, easing_applicable)
PARAM_GROUPS: list[tuple[str, list]] = [
    ("Julia c", [
        ("julia_c_re",     "Re",        float, -2.0,  2.0,   0.001, True),
        ("julia_c_im",     "Im",        float, -2.0,  2.0,   0.001, True),
    ]),
    ("Zoom / Pan", [
        ("borne",          "Borne",     float,  0.05, 4.0,   0.005, True),
        ("center_re",      "Center Re", float, -3.0,  3.0,   0.005, True),
        ("center_im",      "Center Im", float, -3.0,  3.0,   0.005, True),
    ]),
    ("Orbit Trap", [
        ("trap_cx",        "CX",        float, -3.0,  3.0,   0.005, True),
        ("trap_cy",        "CY",        float, -3.0,  3.0,   0.005, True),
        ("trap_angle_deg", "Angle",     float, -360., 360.,  1.0,   True),
        ("trap_radius",    "Radius",    float,  0.01, 5.0,   0.01,  True),
        ("trap_norm_max",  "Norm max",  float,  0.05, 5.0,   0.05,  True),
    ]),
    ("Rendu", [
        ("clip_limit",     "Clip",      float,  0.1,  20.0,  0.1,   True),
        ("mirror_n",       "Mirror",    int,    1,    20,    1,     False),
    ]),
    ("Palette", [
        ("sanzo_index",    "Index",     int,    0,    347,   1,     False),
        ("perm_index",     "Permut.",   int,    0,    23,    1,     False),
    ]),
]
ALL_PARAM_NAMES = [name for _, rows in PARAM_GROUPS for name, *_ in rows]


# ── Easing ────────────────────────────────────────────────────────────────────

def _apply_easing(t: float, mode: str) -> float:
    if mode == "ease-in":
        return t * t
    if mode == "ease-out":
        return 1.0 - (1.0 - t) ** 2
    if mode == "ease-in-out":
        return (1.0 - math.cos(math.pi * t)) * 0.5
    if mode == "step":
        return 0.0 if t < 0.5 else 1.0
    return t   # linear


# ── Helpers UI ────────────────────────────────────────────────────────────────

def _lighten(hex_color: str, amount: int = 22) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        min(255, r + amount), min(255, g + amount), min(255, b + amount))


def _btn(parent, text, cmd, bg=BG2, fg=FG,
         font=("Helvetica", 10), padx=8, pady=3):
    lbl = tk.Label(parent, text=text, bg=bg, fg=fg, font=font,
                   padx=padx, pady=pady, cursor="hand2")
    lbl.bind("<Button-1>", lambda _e: cmd())
    lbl.bind("<Enter>",    lambda _e: lbl.config(bg=_lighten(bg)))
    lbl.bind("<Leave>",    lambda _e: lbl.config(bg=bg))
    return lbl


def _sep(parent, color="#333"):
    tk.Frame(parent, bg=color, height=1).pack(fill="x", pady=(6, 2))


def _section_label(parent, text: str):
    f = tk.Frame(parent, bg=BG2)
    f.pack(fill="x", pady=(4, 1))
    tk.Label(f, text=text, bg=BG2, fg=ACCENT,
             font=("Helvetica", 9, "bold"), padx=6, pady=2).pack(side="left")


def _make_palette(colors: list) -> list:
    n   = len(colors)
    pos = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    return [pos, [list(c) for c in colors]]


# ── Mandelbrot map ────────────────────────────────────────────────────────────

def _load_mandel_image() -> ImageTk.PhotoImage | None:
    path = Path(__file__).parent / "mandelbrot_map.npy"
    if not path.exists():
        return None, None
    m  = np.load(path)
    # Inverser : interieur (0) = blanc, exterieur (1) = noir
    gray = (255 * (1.0 - np.clip(m, 0, 1))).astype(np.uint8)
    rgb  = np.stack([gray, gray, gray], axis=-1)
    img  = Image.fromarray(rgb).resize((MANDEL_W, MANDEL_H), Image.LANCZOS)
    return img, m.shape   # (PIL.Image, (H, W))


_MANDEL_PIL, _MANDEL_ORIG_SHAPE = _load_mandel_image()


def _c_to_mandel_pixel(c_re: float, c_im: float) -> tuple[int, int]:
    """Convertit un complexe c en coordonnees pixel sur le canvas Mandelbrot."""
    orig_H, orig_W = _MANDEL_ORIG_SHAPE if _MANDEL_ORIG_SHAPE else (1000, 1000)
    borne_y = MANDEL_BORNE * orig_H / orig_W
    x = (c_re + MANDEL_BORNE) / (2 * MANDEL_BORNE) * MANDEL_W
    y = (borne_y - c_im)      / (2 * borne_y)      * MANDEL_H
    return int(x), int(y)


def _mandel_pixel_to_c(px: int, py: int) -> tuple[float, float]:
    """Convertit un pixel du canvas Mandelbrot en complexe c."""
    orig_H, orig_W = _MANDEL_ORIG_SHAPE if _MANDEL_ORIG_SHAPE else (1000, 1000)
    borne_y = MANDEL_BORNE * orig_H / orig_W
    c_re = -MANDEL_BORNE + px / MANDEL_W * (2 * MANDEL_BORNE)
    c_im =  borne_y       - py / MANDEL_H * (2 * borne_y)
    return c_re, c_im


# ── Modele de donnees ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class ParamSet:
    julia_c_re:     float = -0.700
    julia_c_im:     float =  0.270
    borne:          float =  2.000
    center_re:      float =  0.000
    center_im:      float =  0.000
    trap_cx:        float =  0.000
    trap_cy:        float =  0.000
    trap_angle_deg: float =  0.000
    trap_radius:    float =  0.500
    trap_norm_max:  float =  1.000
    clip_limit:     float =  3.000
    mirror_n:       int   =  3
    sanzo_index:    int   =  0
    perm_index:     int   =  0

    def lerp(self, other: "ParamSet", t: float,
             easings: dict[str, str] | None = None) -> "ParamSet":
        easings = easings or {}
        vals: dict[str, Any] = {}
        for name, *meta in [row for _, rows in PARAM_GROUPS for row in rows]:
            typ = meta[1]
            mode = easings.get(name, "ease-in-out")
            alpha = _apply_easing(t, mode)
            a = getattr(self, name)
            b = getattr(other, name)
            if typ == float:
                vals[name] = a + (b - a) * alpha
            else:   # int — snap at midpoint
                vals[name] = a if alpha < 0.5 else b
        return ParamSet(**vals)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParamSet":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclasses.dataclass
class Keyframe:
    time_s:  float
    params:  ParamSet
    easings: dict = dataclasses.field(default_factory=dict)
    # easings[param_name] = "linear"|"ease-in"|"ease-out"|"ease-in-out"|"step"

    def to_dict(self) -> dict:
        return {"time_s": self.time_s,
                "params": self.params.to_dict(),
                "easings": self.easings}

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        return cls(time_s=d["time_s"],
                   params=ParamSet.from_dict(d["params"]),
                   easings=d.get("easings", {}))


@dataclasses.dataclass
class GlobalConfig:
    fps:          int   = 24
    duration_s:   float = 10.0
    width:        int   = 1080
    height:       int   = 702
    trap_enabled: bool  = True
    trap_type:    str   = "point"
    equalize:     bool  = True
    smooth:       bool  = True
    mode:         str   = "oklab"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


class Timeline:
    def __init__(self, cfg: GlobalConfig | None = None):
        self.cfg = cfg or GlobalConfig()
        p0 = ParamSet()
        p1 = dataclasses.replace(p0, julia_c_re=-0.4, julia_c_im=0.6,
                                   trap_cx=0.3, trap_cy=-0.2)
        self.keyframes: list[Keyframe] = [
            Keyframe(0.0, p0),
            Keyframe(self.cfg.duration_s, p1),
        ]

    def sorted_kf(self) -> list[Keyframe]:
        return sorted(self.keyframes, key=lambda k: k.time_s)

    def interpolate_at(self, t: float) -> ParamSet:
        kfs = self.sorted_kf()
        t = max(0.0, min(self.cfg.duration_s, t))
        if t <= kfs[0].time_s:
            return kfs[0].params
        if t >= kfs[-1].time_s:
            return kfs[-1].params
        for i in range(len(kfs) - 1):
            a, b = kfs[i], kfs[i + 1]
            if a.time_s <= t <= b.time_s:
                span   = b.time_s - a.time_s
                local_t = (t - a.time_s) / span if span > 0 else 1.0
                return a.params.lerp(b.params, local_t, a.easings)
        return kfs[-1].params

    def add_keyframe(self, t: float, params: ParamSet, easings: dict | None = None):
        for kf in self.keyframes:
            if abs(kf.time_s - t) < 0.05:
                kf.params  = params
                if easings is not None:
                    kf.easings = easings
                return
        new_kf = Keyframe(t, params, easings or {})
        self.keyframes.append(new_kf)
        self.keyframes.sort(key=lambda k: k.time_s)

    def remove_keyframe(self, kf: Keyframe):
        if len(self.keyframes) > 2:
            self.keyframes.remove(kf)

    def to_dict(self) -> dict:
        return {"cfg": self.cfg.to_dict(),
                "keyframes": [k.to_dict() for k in self.sorted_kf()]}

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        tl = cls(GlobalConfig.from_dict(d["cfg"]))
        tl.keyframes = [Keyframe.from_dict(k) for k in d["keyframes"]]
        return tl


# ── Rendu ─────────────────────────────────────────────────────────────────────

def render_frame(params: ParamSet, cfg: GlobalConfig) -> np.ndarray:
    borne_y = params.borne * cfg.height / cfg.width
    xs = np.linspace(params.center_re - params.borne,
                     params.center_re + params.borne, cfg.width)
    ys = np.linspace(params.center_im - borne_y,
                     params.center_im + borne_y, cfg.height)
    Z  = xs[np.newaxis, :] + 1j * ys[:, np.newaxis]
    c  = complex(params.julia_c_re, params.julia_c_im)

    trap_params_arr = np.array([
        params.trap_cx, params.trap_cy,
        math.radians(params.trap_angle_deg),
        params.trap_radius,
    ])
    V = orbit_trap.trap_julia(
        Z, 1 + 0j, 0 + 0j, c,
        TRAP_TYPE_MAP.get(cfg.trap_type, 0),
        trap_params_arr, N_ITER, 256.0, params.trap_norm_max,
    )

    all_perms = list(itertools.permutations(PALETTES[params.sanzo_index][1]))
    colors    = list(all_perms[min(params.perm_index, len(all_perms) - 1)])
    renderer  = render.FractalRenderer(
        _make_palette(colors), mode=cfg.mode, n_iter=N_ITER,
        repeat=params.mirror_n, equalize=cfg.equalize,
        clip_limit=params.clip_limit,
    )
    return renderer.render(V)


def _render_preview(params: ParamSet, cfg: GlobalConfig) -> np.ndarray:
    return render_frame(params, dataclasses.replace(cfg, width=PREV_W, height=PREV_H))


# ── Export MP4 ────────────────────────────────────────────────────────────────

def export_mp4(timeline: Timeline, output_path: Path,
               progress_cb, cancel_flag: threading.Event):
    import imageio
    cfg      = timeline.cfg
    n_frames = int(cfg.duration_s * cfg.fps)
    writer   = imageio.get_writer(
        str(output_path), fps=cfg.fps, codec="libx264",
        quality=8, pixelformat="yuv420p", macro_block_size=2,
    )
    try:
        for i in range(n_frames):
            if cancel_flag.is_set():
                break
            frame = render_frame(timeline.interpolate_at(i / cfg.fps), cfg)
            writer.append_data(frame)
            progress_cb(i + 1, n_frames)
    finally:
        writer.close()


# ── ScrollFrame ───────────────────────────────────────────────────────────────

class _ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        bg = kw.get("bg", BG)
        cv = tk.Canvas(self, bg=bg, highlightthickness=0)
        sb = tk.Scrollbar(self, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(cv, bg=bg)
        wid = cv.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda _e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(int(-1 * e.delta / 120), "units"))


# ── Application ───────────────────────────────────────────────────────────────

class AnimatorApp:
    TL_H         = 80
    TL_PAD_L     = 28
    TL_PAD_R     = 16
    KF_SIZE      = 8
    RIGHT_W      = 330

    def __init__(self):
        self.timeline     = Timeline()
        self.selected_kf  = self.timeline.keyframes[0]
        self.playhead_t   = 0.0
        self._playing     = False
        self._play_after  = None
        self._drag_kf     = None

        self._preview_thread: threading.Thread | None = None
        self._preview_dirty  = False
        self._refresh_after  = None

        # Per-param tkinter variables
        self._pvars:     dict[str, tk.Variable]  = {}
        self._scales:    dict[str, tk.Scale]     = {}
        self._ease_vars: dict[str, tk.StringVar] = {}

        self._mandel_tk: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("Fractal Animator")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self._build_ui()
        self._bind_keys()
        self._load_kf_to_ui(self.selected_kf)
        self._refresh_preview()
        self._draw_timeline()

    # ── Construction ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Barre haute ──────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG2)
        top.pack(fill="x", padx=0, pady=0)

        _btn(top, " Sauvegarder ", self._save_json,
             bg=BG3, pady=5).pack(side="left", padx=1)
        _btn(top, " Charger ", self._load_json,
             bg=BG3, pady=5).pack(side="left", padx=1)

        tk.Label(top, text="FPS:", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(14, 2))
        self._fps_var = tk.IntVar(value=self.timeline.cfg.fps)
        tk.Spinbox(top, textvariable=self._fps_var, from_=1, to=60, width=3,
                   bg=BG3, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_cfg_change).pack(side="left")

        tk.Label(top, text="Duree (s):", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(10, 2))
        self._dur_var = tk.DoubleVar(value=self.timeline.cfg.duration_s)
        tk.Spinbox(top, textvariable=self._dur_var, from_=1, to=600, increment=1,
                   width=4, bg=BG3, fg=FG, insertbackground=FG, relief="flat",
                   command=self._on_cfg_change).pack(side="left")

        tk.Label(top, text="Trap:", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(14, 2))
        self._trap_type_var = tk.StringVar(value=self.timeline.cfg.trap_type)
        om = tk.OptionMenu(top, self._trap_type_var, *TRAP_TYPE_MAP.keys(),
                           command=lambda _: self._on_cfg_change())
        om.config(bg=BG3, fg=FG, activebackground=BG2,
                  highlightthickness=0, relief="flat")
        om["menu"].config(bg=BG3, fg=FG)
        om.pack(side="left")

        self._trap_en_var = tk.BooleanVar(value=self.timeline.cfg.trap_enabled)
        tk.Checkbutton(top, text="On", variable=self._trap_en_var,
                       bg=BG2, fg=FG, selectcolor=BG3,
                       activebackground=BG2,
                       command=self._on_cfg_change).pack(side="left")

        tk.Label(top, text="Mode:", bg=BG2, fg=FG2,
                 font=("Helvetica", 9)).pack(side="left", padx=(14, 2))
        self._mode_var = tk.StringVar(value=self.timeline.cfg.mode)
        om2 = tk.OptionMenu(top, self._mode_var, "oklab", "rgb", "hsv",
                            command=lambda _: self._on_cfg_change())
        om2.config(bg=BG3, fg=FG, activebackground=BG2,
                   highlightthickness=0, relief="flat")
        om2["menu"].config(bg=BG3, fg=FG)
        om2.pack(side="left")

        self._eq_var = tk.BooleanVar(value=self.timeline.cfg.equalize)
        tk.Checkbutton(top, text="EQ", variable=self._eq_var,
                       bg=BG2, fg=FG, selectcolor=BG3,
                       activebackground=BG2,
                       command=self._on_cfg_change).pack(side="left")

        _btn(top, "  EXPORTER MP4  ", self._export_dialog,
             bg="#1e4060", fg="#aaddff",
             font=("Helvetica", 10, "bold"), pady=5).pack(side="right", padx=6)

        # ── Corps ────────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", padx=8, pady=6)

        # ── Colonne gauche : previews ─────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", anchor="n")

        # Preview fractale
        pf = tk.Frame(left, bg="#000", bd=1, relief="solid")
        pf.pack()
        self._preview_lbl = tk.Label(pf, bg="#000",
                                      width=PREV_W, height=PREV_H)
        self._preview_lbl.pack()

        # Transport
        tr = tk.Frame(left, bg=BG)
        tr.pack(fill="x", pady=(4, 0))
        self._time_lbl = tk.Label(tr, text="t: 0.00s / 10.00s",
                                   bg=BG, fg=FG2, font=("Courier", 9))
        self._time_lbl.pack(side="left")

        transport_r = tk.Frame(tr, bg=BG)
        transport_r.pack(side="right")
        _btn(transport_r, "<<", lambda: self._seek(0), bg=BG3, padx=6).pack(side="left", padx=1)
        _btn(transport_r, " < ", lambda: self._step(-1), bg=BG3, padx=6).pack(side="left", padx=1)
        self._play_btn = _btn(transport_r, " PLAY ", self._toggle_play,
                              bg=GREEN, fg="#ccffcc",
                              font=("Helvetica", 10, "bold"), padx=10)
        self._play_btn.pack(side="left", padx=3)
        _btn(transport_r, " > ", lambda: self._step(1), bg=BG3, padx=6).pack(side="left", padx=1)
        _btn(transport_r, ">>", lambda: self._seek(self.timeline.cfg.duration_s), bg=BG3, padx=6).pack(side="left", padx=1)

        # Preview Mandelbrot
        tk.Label(left, text="Mandelbrot  (clic = positionne c)",
                 bg=BG, fg=FG2, font=("Helvetica", 8, "italic")).pack(
                     anchor="w", pady=(10, 2))
        mf = tk.Frame(left, bg="#000", bd=1, relief="solid")
        mf.pack(anchor="w")
        self._mandel_canvas = tk.Canvas(mf, width=MANDEL_W, height=MANDEL_H,
                                         bg="#000", highlightthickness=0,
                                         cursor="crosshair")
        self._mandel_canvas.pack()
        self._mandel_canvas.bind("<ButtonPress-1>",  self._mandel_click)
        self._mandel_canvas.bind("<B1-Motion>",      self._mandel_click)
        self._draw_mandel()

        # ── Colonne droite : parametres ───────────────────────────────────
        right_outer = tk.Frame(body, bg=BG, width=self.RIGHT_W)
        right_outer.pack(side="left", fill="y", padx=(10, 0))
        right_outer.pack_propagate(False)

        self._kf_lbl = tk.Label(right_outer, text="", bg=BG, fg=ACCENT,
                                 font=("Helvetica", 10, "bold"))
        self._kf_lbl.pack(anchor="w", padx=4, pady=(2, 4))

        scroll = _ScrollFrame(right_outer, bg=BG)
        scroll.pack(fill="both", expand=True)
        self._params_inner = scroll.inner

        self._build_param_panel(self._params_inner)

        # Boutons KF
        kf_btns = tk.Frame(right_outer, bg=BG)
        kf_btns.pack(fill="x", pady=(6, 0), padx=4)
        _btn(kf_btns, "+ Ajouter KF ici", self._add_kf_at_playhead,
             bg="#2a4a2a", fg="#aaffaa",
             font=("Helvetica", 10, "bold")).pack(side="left", padx=2)
        _btn(kf_btns, "Supprimer", self._delete_selected_kf,
             bg="#4a2a2a", fg="#ffaaaa").pack(side="left", padx=2)

        # ── Timeline ─────────────────────────────────────────────────────
        tl_frame = tk.Frame(self.root, bg=BG)
        tl_frame.pack(fill="x", padx=8, pady=(0, 8))
        self._tl = tk.Canvas(tl_frame, height=self.TL_H,
                              bg="#141414", highlightthickness=0)
        self._tl.pack(fill="x")
        self._tl.bind("<Configure>",     lambda _e: self._draw_timeline())
        self._tl.bind("<ButtonPress-1>",  self._tl_press)
        self._tl.bind("<B1-Motion>",      self._tl_drag)
        self._tl.bind("<ButtonRelease-1>", self._tl_release)
        self._tl_width = 800

    def _build_param_panel(self, parent: tk.Frame):
        for group_name, rows in PARAM_GROUPS:
            _section_label(parent, group_name)
            for name, label, typ, mn, mx, res, ease_ok in rows:
                self._make_param_row(parent, name, label, typ, mn, mx, res, ease_ok)

        # Nom de palette
        self._pal_name_lbl = tk.Label(parent, text="", bg=BG, fg=FG2,
                                       font=("Helvetica", 8, "italic"))
        self._pal_name_lbl.pack(anchor="w", padx=6, pady=(4, 0))

    def _make_param_row(self, parent, name, label, typ, mn, mx, res, ease_ok):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", padx=4, pady=1)

        tk.Label(row, text=label, width=9, anchor="w",
                 bg=BG, fg=FG, font=("Courier", 8)).pack(side="left")

        var = tk.DoubleVar() if typ == float else tk.IntVar()
        self._pvars[name] = var

        scale = tk.Scale(
            row, variable=var, from_=mn, to=mx, resolution=res,
            orient="horizontal", bg=BG, fg=FG, troughcolor=BG3,
            highlightthickness=0, showvalue=False, length=100,
            command=lambda v, n=name: self._on_scale_move(n),
        )
        scale.pack(side="left")
        self._scales[name] = scale

        entry = tk.Entry(row, textvariable=var, width=7, bg=BG2, fg=FG,
                         insertbackground=FG, relief="flat",
                         font=("Courier", 8))
        entry.pack(side="left", padx=2)
        entry.bind("<Return>",   lambda e, n=name: self._on_entry_commit(n))
        entry.bind("<FocusOut>", lambda e, n=name: self._on_entry_commit(n))

        ease_var = tk.StringVar(value="ease-in-out")
        self._ease_vars[name] = ease_var
        if ease_ok:
            om = tk.OptionMenu(row, ease_var, *EASING_MODES,
                               command=lambda _, n=name: self._on_easing_change(n))
            om.config(bg=BG2, fg=FG2, activebackground=BG3,
                      highlightthickness=0, relief="flat",
                      font=("Helvetica", 7), padx=2, pady=0,
                      indicatoron=True)
            om["menu"].config(bg=BG2, fg=FG)
            om.pack(side="left")

    # ── Mandelbrot ────────────────────────────────────────────────────────────

    def _draw_mandel(self):
        c = self._mandel_canvas
        if _MANDEL_PIL:
            self._mandel_tk = ImageTk.PhotoImage(_MANDEL_PIL)
            c.create_image(0, 0, anchor="nw", image=self._mandel_tk)
        else:
            c.create_rectangle(0, 0, MANDEL_W, MANDEL_H, fill="#111")
        self._update_mandel_crosshair()

    def _update_mandel_crosshair(self):
        c    = self._mandel_canvas
        c_re = self._pvars.get("julia_c_re")
        c_im = self._pvars.get("julia_c_im")
        if c_re is None or c_im is None:
            return
        try:
            px, py = _c_to_mandel_pixel(c_re.get(), c_im.get())
        except (tk.TclError, ValueError):
            return
        c.delete("crosshair")
        R = 6
        c.create_line(px - R, py, px + R, py,
                      fill="#ff4444", width=1, tags="crosshair")
        c.create_line(px, py - R, px, py + R,
                      fill="#ff4444", width=1, tags="crosshair")
        c.create_oval(px - 3, py - 3, px + 3, py + 3,
                      outline="#ff4444", width=1, tags="crosshair")

    def _mandel_click(self, e):
        c_re, c_im = _mandel_pixel_to_c(e.x, e.y)
        self._pvars["julia_c_re"].set(round(c_re, 4))
        self._pvars["julia_c_im"].set(round(c_im, 4))
        self._commit_ui_to_kf()
        self._update_mandel_crosshair()
        self._schedule_refresh()

    # ── Paramètres UI ─────────────────────────────────────────────────────────

    def _load_kf_to_ui(self, kf: Keyframe):
        self.selected_kf = kf
        p = kf.params
        for name in ALL_PARAM_NAMES:
            self._pvars[name].set(getattr(p, name))
        for name, ease_var in self._ease_vars.items():
            ease_var.set(kf.easings.get(name, "ease-in-out"))
        self._kf_lbl.config(text=f"KF @ {kf.time_s:.2f}s")
        self._update_pal_name()
        self._update_mandel_crosshair()

    def _load_interp_to_ui(self, t: float):
        p = self.timeline.interpolate_at(t)
        for name in ALL_PARAM_NAMES:
            self._pvars[name].set(getattr(p, name))
        self._kf_lbl.config(text=f"t = {t:.2f}s  (interpole)")
        self._update_pal_name()
        self._update_mandel_crosshair()

    def _read_params_from_ui(self) -> ParamSet:
        INT_FIELDS = {"mirror_n", "sanzo_index", "perm_index"}
        vals: dict[str, Any] = {}
        for name in ALL_PARAM_NAMES:
            try:
                v = self._pvars[name].get()
                vals[name] = int(v) if name in INT_FIELDS else float(v)
            except (tk.TclError, ValueError):
                vals[name] = getattr(ParamSet(), name)
        return ParamSet(**vals)

    def _read_easings_from_ui(self) -> dict:
        return {name: var.get() for name, var in self._ease_vars.items()}

    def _commit_ui_to_kf(self):
        if self.selected_kf:
            self.selected_kf.params  = self._read_params_from_ui()
            self.selected_kf.easings = self._read_easings_from_ui()

    def _on_scale_move(self, name: str):
        self._commit_ui_to_kf()
        self._update_pal_name()
        if name in ("julia_c_re", "julia_c_im"):
            self._update_mandel_crosshair()
        self._schedule_refresh()

    def _on_entry_commit(self, name: str):
        self._commit_ui_to_kf()
        self._update_pal_name()
        if name in ("julia_c_re", "julia_c_im"):
            self._update_mandel_crosshair()
        self._schedule_refresh()

    def _on_easing_change(self, name: str):
        if self.selected_kf:
            self.selected_kf.easings[name] = self._ease_vars[name].get()

    def _on_cfg_change(self):
        cfg             = self.timeline.cfg
        cfg.fps         = int(self._fps_var.get())
        cfg.duration_s  = float(self._dur_var.get())
        cfg.trap_type   = self._trap_type_var.get()
        cfg.trap_enabled = self._trap_en_var.get()
        cfg.mode        = self._mode_var.get()
        cfg.equalize    = self._eq_var.get()
        self._draw_timeline()
        self._schedule_refresh()

    def _update_pal_name(self):
        try:
            idx = int(self._pvars["sanzo_index"].get())
            if 0 <= idx < len(PALETTE_NAMES):
                self._pal_name_lbl.config(text=PALETTE_NAMES[idx])
        except (tk.TclError, ValueError, AttributeError):
            pass

    # ── Timeline ──────────────────────────────────────────────────────────────

    def _tl_x(self, t: float) -> int:
        dur = self.timeline.cfg.duration_s
        w   = self._tl_width - self.TL_PAD_L - self.TL_PAD_R
        return self.TL_PAD_L + int(t / dur * w)

    def _x_to_t(self, x: int) -> float:
        dur = self.timeline.cfg.duration_s
        w   = self._tl_width - self.TL_PAD_L - self.TL_PAD_R
        return max(0.0, min(dur, (x - self.TL_PAD_L) / w * dur))

    def _draw_timeline(self):
        c    = self._tl
        self._tl_width = c.winfo_width() or 800
        c.delete("all")
        H    = self.TL_H
        dur  = self.timeline.cfg.duration_s
        lx   = self.TL_PAD_L
        rx   = self._tl_width - self.TL_PAD_R

        # Piste
        c.create_rectangle(lx, H//2 - 4, rx, H//2 + 4, fill="#2a2a2a", outline="#333")

        # Ticks
        for s in range(int(dur) + 1):
            x = self._tl_x(s)
            c.create_line(x, H - 14, x, H - 8, fill="#444")
            c.create_text(x, H - 4, text=f"{s}s", fill="#555",
                          font=("Courier", 7), anchor="s")

        # Keyframes
        for kf in self.timeline.keyframes:
            x   = self._tl_x(kf.time_s)
            mid = H // 2
            ks  = self.KF_SIZE
            sel = (kf is self.selected_kf)
            col = ACCENT if sel else "#777"
            c.create_polygon(x, mid - ks, x + ks, mid,
                              x, mid + ks, x - ks, mid,
                              fill=col, outline="white" if sel else "#999",
                              tags="kf")
            c.create_text(x, mid - ks - 3, text=f"{kf.time_s:.1f}s",
                          fill=col, font=("Courier", 7), anchor="s")

        # Playhead
        px = self._tl_x(self.playhead_t)
        c.create_line(px, 4, px, H - 4, fill="#ff5555", width=2, tags="ph")

    def _tl_press(self, e):
        for kf in self.timeline.keyframes:
            if abs(e.x - self._tl_x(kf.time_s)) < self.KF_SIZE + 4:
                self._drag_kf = kf
                self._load_kf_to_ui(kf)
                self._draw_timeline()
                return
        self._drag_kf = None
        self._seek(self._x_to_t(e.x))

    def _tl_drag(self, e):
        if self._drag_kf:
            t   = self._x_to_t(e.x)
            kfs = self.timeline.sorted_kf()
            if self._drag_kf is kfs[0]:
                t = max(0.0, min(t, kfs[1].time_s - 0.05))
            elif self._drag_kf is kfs[-1]:
                t = max(kfs[-2].time_s + 0.05, min(t, self.timeline.cfg.duration_s))
            self._drag_kf.time_s = round(t, 3)
            self._draw_timeline()
        else:
            self._seek(self._x_to_t(e.x))

    def _tl_release(self, _e):
        self._drag_kf = None

    # ── Transport ─────────────────────────────────────────────────────────────

    def _seek(self, t: float):
        self.playhead_t = max(0.0, min(self.timeline.cfg.duration_s, t))
        self._time_lbl.config(
            text=f"t: {self.playhead_t:.2f}s / {self.timeline.cfg.duration_s:.1f}s")
        self._draw_timeline()
        self._load_interp_to_ui(self.playhead_t)
        self._schedule_refresh()

    def _step(self, d: int):
        self._seek(self.playhead_t + d / max(1, self.timeline.cfg.fps))

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._play_btn.config(text=" PAUSE ", bg=RED, fg="#ffcccc")
            self._tick_play()
        else:
            self._play_btn.config(text=" PLAY ", bg=GREEN, fg="#ccffcc")
            if self._play_after:
                self.root.after_cancel(self._play_after)

    def _tick_play(self):
        if not self._playing:
            return
        fps = max(1, self.timeline.cfg.fps)
        t   = self.playhead_t + 1 / fps
        if t >= self.timeline.cfg.duration_s:
            self._toggle_play()
            self._seek(0)
            return
        self._seek(t)
        self._play_after = self.root.after(max(1, int(1000 / fps)), self._tick_play)

    # ── Keyframes ─────────────────────────────────────────────────────────────

    def _add_kf_at_playhead(self):
        params  = self._read_params_from_ui()
        easings = self._read_easings_from_ui()
        self.timeline.add_keyframe(self.playhead_t, params, easings)
        for kf in self.timeline.keyframes:
            if abs(kf.time_s - self.playhead_t) < 0.05:
                self.selected_kf = kf
                break
        self._draw_timeline()

    def _delete_selected_kf(self):
        if self.selected_kf:
            self.timeline.remove_keyframe(self.selected_kf)
            self.selected_kf = self.timeline.sorted_kf()[0]
            self._load_kf_to_ui(self.selected_kf)
            self._draw_timeline()

    # ── Preview ───────────────────────────────────────────────────────────────

    def _schedule_refresh(self):
        if self._refresh_after:
            self.root.after_cancel(self._refresh_after)
        self._refresh_after = self.root.after(120, self._refresh_preview)

    def _refresh_preview(self):
        self._preview_dirty = True
        if self._preview_thread and self._preview_thread.is_alive():
            return
        self._preview_thread = threading.Thread(
            target=self._preview_worker, daemon=True)
        self._preview_thread.start()

    def _preview_worker(self):
        while True:
            self._preview_dirty = False
            try:
                params = self.timeline.interpolate_at(self.playhead_t)
                arr    = _render_preview(params, self.timeline.cfg)
                img    = ImageTk.PhotoImage(Image.fromarray(arr))
                self.root.after(0, self._set_preview, img)
            except Exception:
                pass
            if not self._preview_dirty:
                break

    def _set_preview(self, img):
        self._preview_img = img
        self._preview_lbl.config(image=img)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialdir=Path(__file__).parent,
        )
        if path:
            Path(path).write_text(json.dumps(self.timeline.to_dict(), indent=2))

    def _load_json(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")],
            initialdir=Path(__file__).parent,
        )
        if not path:
            return
        try:
            self.timeline = Timeline.from_dict(
                json.loads(Path(path).read_text()))
        except Exception as exc:
            messagebox.showerror("Erreur", str(exc))
            return
        cfg = self.timeline.cfg
        self._fps_var.set(cfg.fps)
        self._dur_var.set(cfg.duration_s)
        self._trap_type_var.set(cfg.trap_type)
        self._trap_en_var.set(cfg.trap_enabled)
        self._mode_var.set(cfg.mode)
        self._eq_var.set(cfg.equalize)
        self.selected_kf = self.timeline.sorted_kf()[0]
        self._load_kf_to_ui(self.selected_kf)
        self._seek(0)
        self._draw_timeline()

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_dialog(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Exporter MP4")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)

        cfg      = self.timeline.cfg
        n_frames = int(cfg.duration_s * cfg.fps)
        tk.Label(dlg, text=f"{cfg.width}x{cfg.height}  |  {n_frames} frames  "
                           f"({cfg.fps} fps x {cfg.duration_s:.0f}s)",
                 bg=BG, fg=FG2, font=("Helvetica", 9)).pack(padx=16, pady=(12, 4))

        res_var = tk.StringVar(value="1080x702")
        for res in ("1080x702", "1920x1248", "540x351"):
            tk.Radiobutton(dlg, text=res, variable=res_var, value=res,
                           bg=BG, fg=FG, selectcolor=BG2,
                           activebackground=BG).pack(anchor="w", padx=16)

        path_var = tk.StringVar(
            value=str(Path.home() / "Desktop" / "fractal_anim.mp4"))
        pr = tk.Frame(dlg, bg=BG)
        pr.pack(fill="x", padx=16, pady=(10, 4))
        tk.Entry(pr, textvariable=path_var, width=34, bg=BG2,
                 fg=FG, relief="flat").pack(side="left")
        _btn(pr, "...", lambda: path_var.set(
            filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4", "*.mp4")]) or path_var.get()),
             bg=BG3, padx=4).pack(side="left", padx=4)

        prog_cv = tk.Canvas(dlg, height=10, bg="#2a2a2a", highlightthickness=0)
        prog_cv.pack(fill="x", padx=16, pady=(8, 2))
        prog_lbl = tk.Label(dlg, text="", bg=BG, fg=FG2,
                             font=("Helvetica", 9))
        prog_lbl.pack()

        cancel_flag = threading.Event()

        def _progress(done: int, total: int):
            pct = done / total
            w   = prog_cv.winfo_width()
            prog_cv.delete("all")
            prog_cv.create_rectangle(0, 0, int(w * pct), 10, fill=ACCENT)
            prog_lbl.config(text=f"{done}/{total} frames  ({100*pct:.0f}%)")
            dlg.update_idletasks()

        def _run():
            res  = res_var.get()
            rw, rh = (int(x) for x in res.split("x"))
            tl  = Timeline.from_dict(self.timeline.to_dict())
            tl.cfg = dataclasses.replace(tl.cfg, width=rw, height=rh)
            try:
                export_mp4(tl, Path(path_var.get()), _progress, cancel_flag)
                if not cancel_flag.is_set():
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Export", f"Video sauvegardee :\n{path_var.get()}"))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Erreur", str(exc)))
            dlg.after(0, dlg.destroy)

        start = _btn(dlg, "  LANCER LE RENDU  ", lambda: None,
                     bg="#1e4060", fg="#aaddff",
                     font=("Helvetica", 11, "bold"), pady=6)
        start.pack(pady=8)
        _btn(dlg, "Annuler", lambda: cancel_flag.set(),
             bg=RED, fg="#ffaaaa").pack(pady=(0, 12))
        start.bind("<Button-1>", lambda _e: threading.Thread(
            target=_run, daemon=True).start())

    # ── Touches ───────────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<space>",  lambda _e: self._toggle_play())
        self.root.bind("<Left>",   lambda _e: self._step(-1))
        self.root.bind("<Right>",  lambda _e: self._step(1))
        self.root.bind("k",        lambda _e: self._add_kf_at_playhead())
        self.root.bind("<Delete>",  lambda _e: self._delete_selected_kf())

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AnimatorApp().run()
