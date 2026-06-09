from pathlib import Path
from datetime import datetime
import math
import random

import numpy as np
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog

import iteration
import render
import fractal
from transform import parse_transform


class MainWindow:
    FULL_W, FULL_H = 3024, 1964
    PREVIEW_W = 450
    PREVIEW_H = round(PREVIEW_W * FULL_H / FULL_W)
    N_ITER = 80
    BORNE = 2
    GRAD_H = 28
    _MANDEL_PALETTE = [[0.0, 1.0], [[0, 0, 0], [255, 255, 255]]]  # carte de Mandelbrot N&B

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fractale de Julia + palette")
        # exports manuels du GUI : dossier dédié, distinct des wallpapers quotidiens
        self.export_dir = Path(__file__).parent / "Exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.V_preview = None
        self.barre = np.linspace(np.zeros(self.GRAD_H), np.ones(self.GRAD_H), self.PREVIEW_W).T
        self.swatches: list[tk.Label] = []

        # combinaisons de couleurs Sanzo Wada ; on démarre sur une au hasard
        self.sanzo_palettes = render.load_sanzo_palettes()
        self.sanzo_names = render.load_sanzo_names()
        self.sanzo_index = random.randrange(len(self.sanzo_palettes))
        self.palette = render.copy_palette(self.sanzo_palettes[self.sanzo_index])

        c0 = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER).pick_interesting_c()
        self.c_re = tk.DoubleVar(value=round(c0.real, 4))
        self.c_im = tk.DoubleVar(value=round(c0.imag, 4))
        self.mode_var = tk.StringVar(value="rgb")
        self.smooth_var = tk.BooleanVar(value=True)   # lissage logarithmique vs comptage classique
        self.cyclic_var = tk.BooleanVar(value=False)  # bandes cycliques (couleur = itérations % N)
        self.equalize_var = tk.BooleanVar(value=False)  # égalisation d'histogramme
        self.clip_limit = tk.DoubleVar(value=3.0)       # limite de contraste de l'égalisation
        self.mirror_var = tk.BooleanVar(value=False)  # dégradé répété en miroir n fois
        self.mirror_n = tk.IntVar(value=3)            # nombre de répétitions du dégradé
        self.ssaa = tk.IntVar(value=1)                # supersampling (anti-aliasing) : 1 = off
        self.transform_var = tk.StringVar(value="z")  # transfo du plan f(z) (pullback)
        self.transform = None                         # callable courant (None = identité)
        self.c_label_var = tk.StringVar()
        self.sanzo_label_var = tk.StringVar()
        self.sanzo_combo_var = tk.StringVar()

        # image trap
        self.img_trap_enabled  = tk.BooleanVar(value=False)
        self.img_trap_tex      = None          # numpy uint8 (TH, TW, 4) ou None
        self.img_trap_path_var = tk.StringVar(value="(aucune image)")
        self.img_trap_re_min   = tk.DoubleVar(value=-2.0)
        self.img_trap_re_max   = tk.DoubleVar(value=2.0)
        self.img_trap_im_min   = tk.DoubleVar(value=-2.0)
        self.img_trap_im_max   = tk.DoubleVar(value=2.0)
        self.img_trap_min_iter = tk.IntVar(value=2)
        # --- geometric series image trap ---
        self.img_trap_mode    = tk.StringVar(value="classique")  # "classique" | "geom"
        self.trap_geom_N      = tk.IntVar(value=4)
        self.trap_geom_r      = tk.DoubleVar(value=0.5)
        self.trap_geom_size   = tk.DoubleVar(value=1.0)
        self.trap_geom_cx     = tk.DoubleVar(value=0.0)
        self.trap_geom_cy     = tk.DoubleVar(value=0.0)
        self.trap_geom_angle  = tk.DoubleVar(value=0.0)   # degrés
        self.trap_geom_bg     = tk.StringVar(value="#000000")  # fond en mode geom (hex)
        self._img_geom_active = False  # tells _composite_img_trap to use geom bg
        self.img_smooth_var    = tk.BooleanVar(value=False)
        self.img_rgba          = None          # RGBA brut du noyau Numba (alpha=0 → non piégé)
        self.img_preview       = None          # RGB composité prêt à l'affichage
        self.img_trap_pil      = None          # PIL Image RGBA pour miniature canvas
        self.img_trap_angle_deg = tk.DoubleVar(value=0.0)  # degrés (−180 … 180)
        self._trap_shape_drag_start = None         # (mx0, my0, cx0, cy0)
        self._trap_drag_mode   = None          # "move" | "resize_nw/ne/sw/se" | "rotate"
        self._trap_drag_start  = None          # (mx, my, re_min, re_max, im_min, im_max)
        self._trap_drag_anchor = (0.0, 0.0)    # coin fixe lors d'un resize
        self.trap_rand_mu    = tk.DoubleVar(value=0.0)
        self.trap_rand_sigma = tk.DoubleVar(value=0.5)

        # interior coloring
        self.coloring_mode  = tk.StringVar(value="escape")
        self.attractor_norm = tk.DoubleVar(value=0.5)
        self.lambda_burn_in = tk.IntVar(value=100)

        # orbit trap
        self.trap_enabled  = tk.BooleanVar(value=False)
        self.trap_type_var = tk.StringVar(value="point")
        self.trap_cx       = tk.DoubleVar(value=0.0)
        self.trap_cy       = tk.DoubleVar(value=0.0)
        self.trap_angle    = tk.DoubleVar(value=0.0)   # degrés (ligne/croix)
        self.trap_radius   = tk.DoubleVar(value=0.5)   # rayon (cercle/carré) ou amplitude (sinus)
        self.trap_freq     = tk.DoubleVar(value=2.0)   # fréquence (sinus)
        self.trap_norm_max = tk.DoubleVar(value=1.0)

        self._build_scroll_container()
        self._build_top_row()
        self._build_mode_controls()
        self._build_tabs()
        self._build_c_controls()
        self._build_transform_controls()
        self._build_buttons()
        self._fit_window()

        _demo = Path(__file__).parent / "trap_demo.png"
        if _demo.exists():
            self._load_trap_image_path(str(_demo))

        self._recompute_fractal()
        self._update_gradient()

    # ------------------------------------------------------------------ #
    #  Palette helpers
    # ------------------------------------------------------------------ #
    def _mirror_repeat_count(self) -> int:
        if not self.mirror_var.get():
            return 1
        try:
            return max(1, int(self.mirror_n.get()))
        except (tk.TclError, ValueError):
            return 1

    def _ssaa_factor(self) -> int:
        try:
            return max(1, int(self.ssaa.get()))
        except (tk.TclError, ValueError):
            return 1

    def _clip_limit_value(self) -> float:
        try:
            return max(1.0, float(self.clip_limit.get()))
        except (tk.TclError, ValueError):
            return 3.0

    def _coloriser(self, V: np.ndarray) -> np.ndarray:
        mode = "cyclic" if self.cyclic_var.get() else self.mode_var.get()
        renderer = render.FractalRenderer(
            self.palette, mode, self.N_ITER,
            repeat=self._mirror_repeat_count(),
            equalize=self.equalize_var.get(),
            clip_limit=self._clip_limit_value())
        return renderer.render(V)

    def _on_cyclic_toggle(self):
        # bascule cyclique : change à la fois le champ (classique) et la couleur
        self._recompute_fractal()
        self._update_gradient()

    # ------------------------------------------------------------------ #
    #  UI builders
    # ------------------------------------------------------------------ #
    def _build_scroll_container(self):
        # tout le contenu va dans un Frame scrollable verticalement (self.content)
        self.scroll_canvas = tk.Canvas(self.root, borderwidth=0, highlightthickness=0)
        vbar = tk.Scrollbar(self.root, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.content = tk.Frame(self.scroll_canvas)
        self._content_win = self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
        )
        # étire le frame de contenu à la largeur réelle du canvas quand la fenêtre change de taille
        self.scroll_canvas.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.itemconfig(self._content_win, width=e.width),
        )
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):   # mac/windows + linux
            self.scroll_canvas.bind_all(seq, self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.scroll_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.scroll_canvas.yview_scroll(1, "units")
        else:
            self.scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _fit_window(self):
        self.content.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        min_w = self.content.winfo_reqwidth() + 24
        w = max(min_w, screen_w - 40)
        h = min(self.content.winfo_reqheight(), screen_h - 120)
        self.root.geometry(f"{w}x{h}")

    def _build_top_row(self):
        # colonne gauche [ Julia + barre de gradient dessous ] | Mandelbrot (droite)
        self.top_frame = tk.Frame(self.content)
        self.top_frame.pack(padx=10, pady=(10, 4))
        self.julia_col = tk.Frame(self.top_frame)          # gauche : Julia au-dessus du gradient
        self.julia_col.pack(side="left", anchor="n")
        self._build_fractal_canvas()
        self._build_gradient_canvas()
        self._build_mandelbrot_canvas()                    # droite

    def _trap_rand_generate(self):
        import numpy as np
        mu    = self.trap_rand_mu.get()
        sigma = max(1e-3, self.trap_rand_sigma.get())
        cx = float(np.random.normal(mu, sigma))
        cy = float(np.random.normal(mu, sigma))
        gen = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER)
        nc = gen.pick_interesting_c()
        self.c_re.set(round(nc.real, 4))
        self.c_im.set(round(nc.imag, 4))
        self.trap_cx.set(round(cx, 3))
        self.trap_cy.set(round(cy, 3))
        self.trap_type_var.set("point")
        self.trap_enabled.set(True)
        self._select_sanzo(random.randrange(len(self.sanzo_palettes)))
        self._recompute_fractal()

    def _build_mandelbrot_canvas(self):
        mandel_col = tk.Frame(self.top_frame)
        mandel_col.pack(side="left", padx=(8, 0), anchor="n")
        self.mandel_canvas = tk.Canvas(mandel_col, width=self.PREVIEW_W, height=self.PREVIEW_H,
                                       cursor="cross")
        self.mandel_canvas.pack()
        self.mandel_item = self.mandel_canvas.create_image(0, 0, anchor="nw")
        self._render_mandelbrot_base()
        # point rouge indiquant la position du c de la Julia courante
        self.mandel_dot = self.mandel_canvas.create_oval(0, 0, 0, 0, outline="red", fill="red")
        # clic sur la carte -> choisit c et recalcule la Julia
        self.mandel_canvas.bind("<Button-1>", self._on_mandel_click)
        tk.Button(mandel_col, text="c aléatoire", command=self._c_aleatoire).pack(pady=(4, 0))

    def _render_mandelbrot_base(self):
        # carte de référence du plan des c : calculée une seule fois (le set ne change pas)
        gen = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER)
        V = gen.generate_mandelbrot()
        C = render.FractalRenderer(self._MANDEL_PALETTE).render(V)
        self.mandel_photo = ImageTk.PhotoImage(Image.fromarray(C))
        self.mandel_canvas.itemconfig(self.mandel_item, image=self.mandel_photo)

    def _build_fractal_canvas(self):
        self.fractal_canvas = tk.Canvas(self.julia_col, width=self.PREVIEW_W, height=self.PREVIEW_H,
                                        cursor="hand2")
        self.fractal_canvas.pack()
        self.fractal_item = self.fractal_canvas.create_image(0, 0, anchor="nw")
        # clic sur l'aperçu -> ouvre la fractale en pleine résolution HD
        self.fractal_canvas.bind("<Button-1>", self._show_hd_popup)

    def _build_gradient_canvas(self):
        # pile sous la Julia (même colonne, même largeur)
        self.grad_canvas = tk.Canvas(self.julia_col, width=self.PREVIEW_W, height=self.GRAD_H)
        self.grad_canvas.pack(pady=(4, 0))
        self.grad_item = self.grad_canvas.create_image(0, 0, anchor="nw")

    def _build_mode_controls(self):
        mode_frame = tk.Frame(self.content)
        mode_frame.pack(padx=10, pady=(4, 0), anchor="w")
        tk.Label(mode_frame, text="Interpolation :").pack(side="left")
        tk.Radiobutton(mode_frame, text="RGB", variable=self.mode_var, value="rgb",
                       command=self._apply_palette).pack(side="left")
        tk.Radiobutton(mode_frame, text="HSV", variable=self.mode_var, value="hsv",
                       command=self._apply_palette).pack(side="left")
        tk.Radiobutton(mode_frame, text="Oklab", variable=self.mode_var, value="oklab",
                       command=self._apply_palette).pack(side="left")
        # lissage : change le champ d'itérations -> recalcul nécessaire (pas juste la couleur)
        tk.Label(mode_frame, text="    ").pack(side="left")
        tk.Checkbutton(mode_frame, text="Lissage", variable=self.smooth_var,
                       command=self._recompute_fractal).pack(side="left")
        # bandes cycliques : couleur indexée par (itérations % N), ignore l'interpolation
        tk.Checkbutton(mode_frame, text="Bandes (mod N)", variable=self.cyclic_var,
                       command=self._on_cyclic_toggle).pack(side="left")
        # égalisation d'histogramme à contraste limité : recoloration seule
        tk.Checkbutton(mode_frame, text="Histogramme", variable=self.equalize_var,
                       command=self._apply_palette).pack(side="left")
        tk.Label(mode_frame, text="lim.").pack(side="left")
        sp = ttk.Spinbox(mode_frame, from_=1.0, to=20.0, increment=0.5, width=4,
                         textvariable=self.clip_limit, command=self._apply_palette)
        sp.pack(side="left")
        self.clip_limit.trace_add("write", lambda *_: self._apply_palette())
        # dégradé miroir : replie le dégradé n fois (recoloration seule, pas de recalcul)
        tk.Checkbutton(mode_frame, text="Miroir ×", variable=self.mirror_var,
                       command=self._apply_palette).pack(side="left")
        sp = ttk.Spinbox(mode_frame, from_=1, to=20, width=3, textvariable=self.mirror_n,
                         command=self._apply_palette)
        sp.pack(side="left")
        self.mirror_n.trace_add("write", lambda *_: self._apply_palette())
        # supersampling (anti-aliasing) : calcul à k× la résolution -> recalcul nécessaire
        tk.Label(mode_frame, text="    AA×").pack(side="left")
        ttk.Spinbox(mode_frame, from_=1, to=4, width=3, textvariable=self.ssaa,
                    command=self._recompute_fractal).pack(side="left")

    def _build_tabs(self):
        nb = ttk.Notebook(self.content)
        nb.pack(padx=10, pady=4, fill="x")
        tab_pal  = tk.Frame(nb)
        tab_trap = tk.Frame(nb)
        tab_img  = tk.Frame(nb)
        nb.add(tab_pal,  text="  Palette  ")
        nb.add(tab_trap, text="  Orbit Trap  ")
        nb.add(tab_img,  text="  Trap image  ")
        self._build_sanzo_controls(tab_pal)
        self._build_interior_controls(tab_pal)
        self._build_palette_controls(tab_pal)
        self._build_trap_controls(tab_trap)
        self._build_img_trap_controls(tab_img)

    def _build_sanzo_controls(self, parent=None):
        if parent is None:
            parent = self.content
        f = tk.LabelFrame(parent, text="Combinaison Sanzo Wada")
        f.pack(padx=10, pady=4, fill="x")
        self.sanzo_combo = ttk.Combobox(f, values=self.sanzo_names,
                                         textvariable=self.sanzo_combo_var,
                                         state="readonly", width=36)
        self.sanzo_combo.pack(side="left", padx=(6, 4), pady=4)
        self.sanzo_combo.bind("<<ComboboxSelected>>", self._on_sanzo_combo_select)
        tk.Button(f, text="◀", command=self._sanzo_prev).pack(side="left", padx=2, pady=4)
        tk.Button(f, text="▶", command=self._sanzo_next).pack(side="left", padx=2)
        tk.Button(f, text="Aléatoire", command=self._sanzo_random).pack(side="left", padx=6)
        tk.Button(f, text="⇅ Inverser", command=self._reverse_palette).pack(side="left", padx=6)
        tk.Button(f, text="⇄ Mélanger", command=self._shuffle_palette).pack(side="left", padx=6)
        self._update_sanzo_label()

    def _build_interior_controls(self, parent):
        f = tk.LabelFrame(parent, text="Coloriage intérieur")
        f.pack(padx=10, pady=4, fill="x")
        radio_row = tk.Frame(f)
        radio_row.pack(fill="x", padx=6, pady=(4, 2))
        for label, value in (("Escape", "escape"), ("Période", "period"),
                              ("Attracteur", "attractor"), ("Lambda λ", "lambda")):
            tk.Radiobutton(radio_row, text=label, variable=self.coloring_mode,
                           value=value, command=self._recompute_fractal).pack(side="left", padx=8)
        norm_row = tk.Frame(f)
        norm_row.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(norm_row, text="norm", width=6).pack(side="left")
        tk.Scale(norm_row, from_=0.01, to=1000.0, resolution=1.0, orient="horizontal",
                 length=220, variable=self.attractor_norm,
                 command=self._recompute_fractal).pack(side="left")
        burn_row = tk.Frame(f)
        burn_row.pack(fill="x", padx=6, pady=(0, 4))
        tk.Label(burn_row, text="burn-in", width=6).pack(side="left")
        tk.Scale(burn_row, from_=10, to=500, resolution=10, orient="horizontal",
                 length=220, variable=self.lambda_burn_in,
                 command=self._recompute_fractal).pack(side="left")

    def _build_palette_controls(self, parent=None):
        if parent is None:
            parent = self.content
        self.pal_frame = tk.LabelFrame(parent, text="Palette")
        self.pal_frame.pack(padx=10, pady=4, fill="x")
        self._populate_palette_rows()

    def _populate_palette_rows(self):
        for child in self.pal_frame.winfo_children():
            child.destroy()
        self.swatches = []
        for i in range(len(self.palette[0])):
            row = tk.Frame(self.pal_frame)
            row.pack(fill="x", pady=3, padx=6)
            sw = tk.Label(row, width=3, relief="solid", bd=1, bg=self._hexcolor(self.palette[1][i]))
            sw.pack(side="left", padx=(0, 8))
            self.swatches.append(sw)
            for channel, name in enumerate(("R", "G", "B")):
                tk.Label(row, text=name).pack(side="left")
                var = tk.StringVar(value=str(self.palette[1][i][channel]))
                var.trace_add("write", self._make_rgb_cb(i, channel, var))
                ttk.Spinbox(row, from_=0, to=255, width=4, textvariable=var).pack(side="left", padx=(0, 6))
            tk.Label(row, text="pos").pack(side="left", padx=(8, 4))
            sc = tk.Scale(row, from_=0, to=1, resolution=0.01, orient="horizontal", length=150)
            sc.set(self.palette[0][i])
            sc.config(command=self._make_pos_cb(i))
            sc.pack(side="left")

    def _build_c_controls(self):
        c_frame = tk.LabelFrame(self.content, text="Paramètre c")
        c_frame.pack(padx=10, pady=4, fill="x")
        self.c_re.trace_add("write", self._update_c_label)
        self.c_im.trace_add("write", self._update_c_label)
        self._update_c_label()
        re_row = tk.Frame(c_frame)
        re_row.pack(fill="x", padx=6, pady=2)
        tk.Label(re_row, text="Re(c)", width=6).pack(side="left")
        sc_re = tk.Scale(re_row, from_=-1.5, to=1.5, resolution=0.001,
                         orient="horizontal", length=320, variable=self.c_re, showvalue=False)
        sc_re.pack(side="left", fill="x", expand=True)
        im_row = tk.Frame(c_frame)
        im_row.pack(fill="x", padx=6, pady=2)
        tk.Label(im_row, text="Im(c)", width=6).pack(side="left")
        sc_im = tk.Scale(im_row, from_=-1.5, to=1.5, resolution=0.001,
                         orient="horizontal", length=320, variable=self.c_im, showvalue=False)
        sc_im.pack(side="left", fill="x", expand=True)
        sc_re.bind("<ButtonRelease-1>", self._recompute_fractal)
        sc_im.bind("<ButtonRelease-1>", self._recompute_fractal)
        tk.Label(c_frame, textvariable=self.c_label_var).pack(pady=(2, 4))

    def _build_transform_controls(self):
        f = tk.LabelFrame(self.content, text="Transformation du plan  (pullback : pixel → point échantillonné)")
        f.pack(padx=10, pady=4, fill="x")
        row = tk.Frame(f)
        row.pack(fill="x", padx=6, pady=4)
        tk.Label(row, text="f(z) =").pack(side="left")
        entry = tk.Entry(row, textvariable=self.transform_var, width=22)
        entry.pack(side="left", padx=(4, 8))
        entry.bind("<Return>", self._apply_transform)
        tk.Button(row, text="Appliquer", command=self._apply_transform).pack(side="left")
        # préréglages : remplissent le champ puis appliquent
        presets = tk.Frame(f)
        presets.pack(fill="x", padx=6, pady=(0, 4))
        for expr in ("z", "i*z", "z^2", "e^z", "1/z"):
            tk.Button(presets, text=expr, width=4,
                      command=lambda e=expr: self._set_transform(e)).pack(side="left", padx=2)

    def _build_trap_controls(self, parent=None):
        if parent is None:
            parent = self.content
        f = tk.LabelFrame(parent, text="Orbit Trap")
        f.pack(padx=10, pady=4, fill="x")

        body = tk.Frame(f)
        body.pack(fill="x", padx=4, pady=4)

        # — contrôles gauche —
        left = tk.Frame(body)
        left.pack(side="left", anchor="n")

        row0 = tk.Frame(left)
        row0.pack(fill="x", padx=6, pady=4)
        tk.Checkbutton(row0, text="Activer", variable=self.trap_enabled,
                       command=self._recompute_fractal).pack(side="left")
        tk.Label(row0, text="  Forme :").pack(side="left")
        om = ttk.OptionMenu(row0, self.trap_type_var,
                            self.trap_type_var.get(),
                            "point", "ligne", "croix", "cercle", "carré", "sinus")
        om.pack(side="left", padx=4)
        self.trap_type_var.trace_add("write", lambda *_: self._on_trap_params_change())

        for label, var, from_, to_, res in (
            ("cx",         self.trap_cx,      -2.0,  2.0,  0.01),
            ("cy",         self.trap_cy,      -2.0,  2.0,  0.01),
            ("angle°",     self.trap_angle,    0,    180,   1.0),
            ("rayon/amp.", self.trap_radius,   0.0,   2.0,  0.01),
            ("fréq.",      self.trap_freq,     0.1,  12.0,  0.1),
            ("norm_max",   self.trap_norm_max, 0.05,  5.0,  0.05),
        ):
            row = tk.Frame(left)
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=label, width=10).pack(side="left")
            tk.Scale(row, from_=from_, to=to_, resolution=res, orient="horizontal",
                     length=240, variable=var,
                     command=self._on_trap_params_change).pack(side="left")

        tk.Frame(left, height=6).pack()
        tk.Button(left, text="Trap aléatoire",
                  command=self._trap_rand_generate).pack(padx=6, pady=(0, 4), fill="x")
        for label, var, from_, to_, res in (
            ("μ", self.trap_rand_mu,   -2.0, 2.0, 0.05),
            ("σ", self.trap_rand_sigma, 0.01, 2.0, 0.01),
        ):
            row = tk.Frame(left)
            row.pack(fill="x", padx=6, pady=1)
            tk.Label(row, text=label, width=10).pack(side="left")
            tk.Scale(row, from_=from_, to=to_, resolution=res, orient="horizontal",
                     length=240, variable=var, showvalue=True).pack(side="left")

        # — aperçu plan complexe à droite —
        cf = tk.Frame(body)
        cf.pack(side="left", fill="both", expand=True, padx=(10, 2))
        self.trap_shape_canvas = tk.Canvas(cf, highlightthickness=0)
        self.trap_shape_canvas.pack(fill="both", expand=True)
        self.trap_shape_canvas.bind("<Configure>",      lambda e: self._update_trap_shape_display())
        self.trap_shape_canvas.bind("<Motion>",         self._trap_shape_hover)
        self.trap_shape_canvas.bind("<ButtonPress-1>",  self._trap_shape_press)
        self.trap_shape_canvas.bind("<B1-Motion>",      self._trap_shape_drag)
        self.trap_shape_canvas.bind("<ButtonRelease-1>",self._trap_shape_release)
        tk.Label(cf, text="plan  [−3, 3]", fg="#666666", font=("", 8)).pack(pady=(2, 0))

        self._update_trap_shape_display()

    def _on_trap_params_change(self, *_):
        self._update_trap_shape_display()
        self._recompute_fractal()

    def _update_trap_shape_display(self):
        if not hasattr(self, "trap_shape_canvas"):
            return
        c = self.trap_shape_canvas
        W, H = c.winfo_width(), c.winfo_height()
        if W < 2 or H < 2:
            return
        R = 3.0
        scale = min(W, H) / (2 * R)
        ox, oy = W / 2, H / 2
        xl = ox - R * scale;  xr = ox + R * scale
        yt = oy - R * scale;  yb = oy + R * scale

        def px(re, im):
            return ox + re * scale, oy - im * scale

        c.delete("all")
        c.create_rectangle(xl, yt, xr, yb, fill="#111111", outline="#333333")

        for v in range(-2, 3):
            if v == 0:
                continue
            c.create_line(ox + v * scale, yt, ox + v * scale, yb, fill="#252525")
            c.create_line(xl, oy - v * scale, xr, oy - v * scale, fill="#252525")
        c.create_line(xl, oy, xr, oy, fill="#555555")
        c.create_line(ox, yt, ox, yb, fill="#555555")

        t    = self.trap_type_var.get()
        cxv  = self.trap_cx.get()
        cyv  = self.trap_cy.get()
        arad = math.radians(self.trap_angle.get())
        r    = self.trap_radius.get()
        freq = self.trap_freq.get()

        if t == "point":
            x0, y0 = px(cxv, cyv)
            nm = self.trap_norm_max.get()
            c.create_oval(x0 - nm * scale, y0 - nm * scale,
                          x0 + nm * scale, y0 + nm * scale,
                          outline="#4488ff", dash=(3, 3))
            c.create_oval(x0 - 6, y0 - 6, x0 + 6, y0 + 6,
                          fill="#4488ff", outline="#aaddff")

        elif t == "ligne":
            ext = R * 1.5
            x1, y1 = px(-ext * math.cos(arad), -ext * math.sin(arad))
            x2, y2 = px( ext * math.cos(arad),  ext * math.sin(arad))
            c.create_line(x1, y1, x2, y2, fill="#4488ff", width=2)

        elif t == "croix":
            c.create_line(xl, oy, xr, oy, fill="#4488ff", width=2)
            c.create_line(ox, yt, ox, yb, fill="#4488ff", width=2)

        elif t == "cercle":
            x0, y0 = px(cxv, cyv)
            rp = r * scale
            c.create_oval(x0 - rp, y0 - rp, x0 + rp, y0 + rp,
                          outline="#4488ff", width=2)
            c.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4, fill="#4488ff")

        elif t == "carré":
            x0, y0 = px(cxv, cyv)
            rp = r * scale
            c.create_rectangle(x0 - rp, y0 - rp, x0 + rp, y0 + rp,
                                outline="#4488ff", width=2)
            c.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4, fill="#4488ff")

        elif t == "sinus":
            pts = []
            for i in range(201):
                xre = -R + 2 * R * i / 200
                yim = cyv + r * math.sin(freq * (xre - cxv))
                pts.extend(px(xre, yim))
            if len(pts) >= 4:
                c.create_line(pts, fill="#4488ff", width=2)
            x0, y0 = px(cxv, cyv)
            c.create_oval(x0 - 4, y0 - 4, x0 + 4, y0 + 4, fill="#4488ff")

    def _trap_shape_hover(self, event):
        t = self.trap_type_var.get()
        self.trap_shape_canvas.configure(
            cursor="fleur" if t != "croix" else "")

    def _trap_shape_press(self, event):
        t = self.trap_type_var.get()
        if t == "croix":
            self._trap_shape_drag_start = None
            return
        self._trap_shape_drag_start = (event.x, event.y,
                                        self.trap_cx.get(), self.trap_cy.get())

    def _trap_shape_drag(self, event):
        if self._trap_shape_drag_start is None:
            return
        c = self.trap_shape_canvas
        W, H = c.winfo_width(), c.winfo_height()
        scale = min(W, H) / (2 * 3.0)
        ox, oy = W / 2, H / 2
        mx0, my0, cx0, cy0 = self._trap_shape_drag_start
        t = self.trap_type_var.get()

        if t == "ligne":
            mre = (event.x - ox) / scale
            mim = -(event.y - oy) / scale
            new_deg = math.degrees(math.atan2(mim, mre)) % 180
            self.trap_angle.set(round(new_deg, 1))
        else:
            new_cx = max(-2.0, min(2.0, cx0 + (event.x - mx0) / scale))
            new_cy = max(-2.0, min(2.0, cy0 - (event.y - my0) / scale))
            self.trap_cx.set(round(new_cx, 3))
            self.trap_cy.set(round(new_cy, 3))

        self._update_trap_shape_display()

    def _trap_shape_release(self, _event):
        if self._trap_shape_drag_start is not None:
            self._trap_shape_drag_start = None
            self._recompute_fractal()

    def _build_img_trap_controls(self, parent=None):
        if parent is None:
            parent = self.content
        f = tk.LabelFrame(parent, text="Trap par image (PNG détouré)")
        f.pack(padx=10, pady=4, fill="x")

        # ── Always-visible: enable toggle + image loader ───────────────────
        top_row = tk.Frame(f)
        top_row.pack(fill="x", padx=6, pady=(4, 0))
        tk.Checkbutton(top_row, text="Activer", variable=self.img_trap_enabled,
                       command=self._recompute_fractal).pack(side="left")
        tk.Button(top_row, text="Charger PNG…", command=self._load_trap_image).pack(side="left", padx=8)
        tk.Label(top_row, textvariable=self.img_trap_path_var, anchor="w").pack(side="left")

        # ── Mode selector ──────────────────────────────────────────────────
        mode_row = tk.Frame(f)
        mode_row.pack(fill="x", padx=6, pady=(2, 0))
        tk.Label(mode_row, text="Mode :").pack(side="left")
        for val, label in [("classique", "Classique"), ("geom", "Série géométrique")]:
            tk.Radiobutton(mode_row, text=label, variable=self.img_trap_mode, value=val,
                           command=self._on_img_trap_mode_change).pack(side="left", padx=4)

        # ── Classic controls (existing layout) ─────────────────────────────
        self._img_classic_frame = tk.Frame(f)
        self._img_classic_frame.pack(fill="x")
        self._build_img_classic_controls(self._img_classic_frame)

        # ── Geom controls (new) ────────────────────────────────────────────
        self._img_geom_frame = tk.Frame(f)
        # not packed yet — shown only when mode == "geom"
        self._build_img_geom_controls(self._img_geom_frame)

    def _build_img_classic_controls(self, parent):
        body = tk.Frame(parent)
        body.pack(fill="x", padx=4, pady=4)

        # — contrôles gauche (largeur naturelle) —
        right = tk.Frame(body)
        right.pack(side="left", anchor="n")

        # — aperçu plan complexe : prend tout l'espace restant —
        canvas_frame = tk.Frame(body)
        canvas_frame.pack(side="left", fill="both", expand=True, padx=(10, 2))
        self.trap_rect_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        self.trap_rect_canvas.pack(fill="both", expand=True)
        self.trap_rect_canvas.bind("<Configure>",      lambda e: self._update_trap_rect_display())
        self.trap_rect_canvas.bind("<Motion>",         self._trap_canvas_hover)
        self.trap_rect_canvas.bind("<ButtonPress-1>",  self._trap_canvas_press)
        self.trap_rect_canvas.bind("<B1-Motion>",      self._trap_canvas_drag)
        self.trap_rect_canvas.bind("<ButtonRelease-1>",self._trap_canvas_release)
        tk.Label(canvas_frame, text="plan  [−5, 5]", fg="#666666", font=("", 8)).pack(pady=(2, 0))

        for label, var in (("Re min", self.img_trap_re_min), ("Re max", self.img_trap_re_max),
                           ("Im min", self.img_trap_im_min), ("Im max", self.img_trap_im_max)):
            row = tk.Frame(right)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, width=8).pack(side="left")
            tk.Scale(row, from_=-5.0, to=5.0, resolution=0.05, orient="horizontal",
                     length=240, variable=var,
                     command=self._on_img_rect_change).pack(side="left")

        row_ang = tk.Frame(right)
        row_ang.pack(fill="x", pady=1)
        tk.Label(row_ang, text="rotation°", width=8).pack(side="left")
        tk.Scale(row_ang, from_=-180, to=180, resolution=1, orient="horizontal",
                 length=240, variable=self.img_trap_angle_deg,
                 command=self._on_img_rect_change).pack(side="left")

        row_bot = tk.Frame(right)
        row_bot.pack(fill="x", pady=(2, 4))
        tk.Label(row_bot, text="min iter").pack(side="left")
        ttk.Spinbox(row_bot, from_=0, to=20, width=4,
                    textvariable=self.img_trap_min_iter,
                    command=self._recompute_fractal).pack(side="left", padx=(2, 12))
        tk.Checkbutton(row_bot, text="Lissage sortie",
                       variable=self.img_smooth_var,
                       command=self._on_img_smooth_toggle).pack(side="left")

        self._update_trap_rect_display()

    def _build_img_geom_controls(self, parent):
        geom_sliders = [
            ("N (copies)",      self.trap_geom_N,     1,    8,    1),
            ("r (ratio)",       self.trap_geom_r,     0.20, 0.90, 0.05),
            ("Taille base",     self.trap_geom_size,  0.1,  4.0,  0.05),
            ("Centre X",        self.trap_geom_cx,   -2.0,  2.0,  0.05),
            ("Centre Y",        self.trap_geom_cy,   -2.0,  2.0,  0.05),
            ("Rotation/copie°", self.trap_geom_angle, 0,    360,  5),
        ]
        for label, var, lo, hi, res in geom_sliders:
            row = tk.Frame(parent)
            row.pack(fill="x", padx=6)
            tk.Label(row, text=label, width=14, anchor="w").pack(side="left")
            tk.Scale(row, from_=lo, to=hi, resolution=res, orient="horizontal",
                     length=220, variable=var,
                     command=self._recompute_fractal).pack(side="left")
        # Colour picker for the background
        bg_row = tk.Frame(parent)
        bg_row.pack(fill="x", padx=6, pady=(2, 0))
        tk.Label(bg_row, text="Fond", width=14, anchor="w").pack(side="left")
        self._geom_bg_btn = tk.Button(bg_row, width=4,
                                       bg=self.trap_geom_bg.get(),
                                       command=self._pick_geom_bg)
        self._geom_bg_btn.pack(side="left")

    def _on_img_trap_mode_change(self):
        if self.img_trap_mode.get() == "geom":
            self._img_classic_frame.pack_forget()
            self._img_geom_frame.pack(fill="x")
        else:
            self._img_geom_frame.pack_forget()
            self._img_classic_frame.pack(fill="x")
        self._recompute_fractal()

    def _pick_geom_bg(self):
        from tkinter import colorchooser
        result = colorchooser.askcolor(color=self.trap_geom_bg.get(),
                                       title="Couleur de fond (mode géométrique)")
        if result[1]:
            self.trap_geom_bg.set(result[1])
            self._geom_bg_btn.configure(bg=result[1])
            self._recompute_fractal()

    def _on_img_rect_change(self, *_):
        self._update_trap_rect_display()
        self._recompute_fractal()

    def _update_trap_rect_display(self):
        if not hasattr(self, "trap_rect_canvas"):
            return
        c = self.trap_rect_canvas
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 2 or H < 2:
            return
        R = 5.0  # plan affiché : [-R, R] × [-R, R]

        # échelle uniforme : 1 unité Re = 1 unité Im
        scale = min(W, H) / (2 * R)
        ox, oy = W / 2, H / 2  # origine (0+0j) au centre du canvas

        def px(re, im):
            return ox + re * scale, oy - im * scale

        # bornes pixel de la zone de coordonnées
        xl, xr = ox - R * scale, ox + R * scale
        yt, yb = oy - R * scale, oy + R * scale

        c.delete("all")

        # fond sombre limité au carré de coordonnées (pas de noir hors zone)
        c.create_rectangle(xl, yt, xr, yb, fill="#111111", outline="#333333")

        # grille entière très discrète
        for v in range(-4, 5):
            if v == 0:
                continue
            c.create_line(ox + v * scale, yt, ox + v * scale, yb, fill="#252525")
            c.create_line(xl, oy - v * scale, xr, oy - v * scale, fill="#252525")

        # axes Re = 0 et Im = 0
        c.create_line(xl, oy, xr, oy, fill="#555555")
        c.create_line(ox, yt, ox, yb, fill="#555555")

        # rectangle de la vue Julia (borne = 2)
        by = self.BORNE * self.PREVIEW_H / self.PREVIEW_W
        x1, y1 = px(-self.BORNE, by)
        x2, y2 = px(self.BORNE, -by)
        c.create_rectangle(x1, y1, x2, y2, outline="#888888", width=1)

        # rectangle du trap image — avec rotation
        re_min = self.img_trap_re_min.get()
        re_max = self.img_trap_re_max.get()
        im_min = self.img_trap_im_min.get()
        im_max = self.img_trap_im_max.get()
        angle_rad = math.radians(self.img_trap_angle_deg.get())
        re_c = (re_min + re_max) / 2
        im_c = (im_min + im_max) / 2
        rw = re_max - re_min
        rh = im_max - im_min
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        def rot(lre, lim):
            return px(re_c + lre * cos_a - lim * sin_a,
                      im_c + lre * sin_a + lim * cos_a)

        corners = [rot(-rw/2,  rh/2), rot( rw/2,  rh/2),
                   rot( rw/2, -rh/2), rot(-rw/2, -rh/2)]

        # miniature PIL rotée au centre
        if self.img_trap_pil is not None:
            dw_local = max(1, int(round(rw * scale)))
            dh_local = max(1, int(round(rh * scale)))
            thumb = self.img_trap_pil.resize((dw_local, dh_local), Image.LANCZOS)
            thumb_r = thumb.rotate(math.degrees(angle_rad), expand=True,
                                   resample=Image.BICUBIC)
            self._trap_canvas_thumb = ImageTk.PhotoImage(thumb_r)
            cxp, cyp = px(re_c, im_c)
            c.create_image(int(cxp), int(cyp), image=self._trap_canvas_thumb, anchor="center")

        # contour du polygone rotatif
        pts = [v for pt in corners for v in pt]
        c.create_polygon(pts, fill="", outline="#4488ff", width=2)

        # poignées de coin
        hr = 5
        for hx, hy in corners:
            c.create_rectangle(hx - hr, hy - hr, hx + hr, hy + hr,
                                fill="#4488ff", outline="#aaddff")

        # poignée de rotation (cercle orange au-dessus du bord supérieur)
        margin = 20 / scale
        h_re = re_c - (rh / 2 + margin) * sin_a
        h_im = im_c + (rh / 2 + margin) * cos_a
        tm_re = re_c - (rh / 2) * sin_a
        tm_im = im_c + (rh / 2) * cos_a
        hx, hy = px(h_re, h_im)
        tmx, tmy = px(tm_re, tm_im)
        c.create_line(tmx, tmy, hx, hy, fill="#4488ff", dash=(3, 3))
        c.create_oval(hx - 7, hy - 7, hx + 7, hy + 7, fill="#ff8844", outline="#ffcc88")

    # ---- helpers drag/resize du canvas ---------------------------------- #

    def _trap_canvas_transform(self):
        """Retourne (ox, oy, scale) : mapping plan complexe ↔ canvas."""
        c = self.trap_rect_canvas
        W, H = c.winfo_width(), c.winfo_height()
        scale = min(W, H) / (2 * 5.0)
        return W / 2, H / 2, scale

    def _trap_aspect_ratio(self):
        if self.img_trap_pil is not None:
            w, h = self.img_trap_pil.size
            return w / h
        re_w = self.img_trap_re_max.get() - self.img_trap_re_min.get()
        im_h = self.img_trap_im_max.get() - self.img_trap_im_min.get()
        return re_w / im_h if im_h > 0 else 1.0

    def _trap_canvas_hit_test(self, cx, cy):
        """Renvoie "rotate", "move", "resize_XX" ou None selon la position."""
        ox, oy, scale = self._trap_canvas_transform()
        rm = self.img_trap_re_min.get(); rx = self.img_trap_re_max.get()
        im = self.img_trap_im_min.get(); ix = self.img_trap_im_max.get()
        angle_rad = math.radians(self.img_trap_angle_deg.get())
        re_c = (rm + rx) / 2;  im_c = (im + ix) / 2
        rw = rx - rm;          rh = ix - im
        cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
        tol = 10

        def ppx(re, imp): return ox + re * scale, oy - imp * scale
        def rot(lre, lim):
            return ppx(re_c + lre * cos_a - lim * sin_a,
                       im_c + lre * sin_a + lim * cos_a)

        # poignée de rotation
        margin = 20 / scale
        hx, hy = ppx(re_c - (rh / 2 + margin) * sin_a,
                     im_c + (rh / 2 + margin) * cos_a)
        if (cx - hx) ** 2 + (cy - hy) ** 2 < 64:
            return "rotate"

        # coins
        corners = {"nw": rot(-rw/2, rh/2), "ne": rot(rw/2, rh/2),
                   "sw": rot(-rw/2, -rh/2), "se": rot(rw/2, -rh/2)}
        for name, (px0, py0) in corners.items():
            if abs(cx - px0) < tol and abs(cy - py0) < tol:
                return f"resize_{name}"

        # intérieur (test en coordonnées locales)
        mouse_re = (cx - ox) / scale
        mouse_im = -(cy - oy) / scale
        dre = mouse_re - re_c;  dim = mouse_im - im_c
        local_re = dre * cos_a + dim * sin_a
        local_im = -dre * sin_a + dim * cos_a
        if -rw/2 <= local_re <= rw/2 and -rh/2 <= local_im <= rh/2:
            return "move"
        return None

    def _trap_canvas_hover(self, event):
        cursors = {"move": "fleur", "rotate": "exchange",
                   "resize_nw": "top_left_corner", "resize_ne": "top_right_corner",
                   "resize_sw": "bottom_left_corner", "resize_se": "bottom_right_corner"}
        mode = self._trap_canvas_hit_test(event.x, event.y)
        self.trap_rect_canvas.configure(cursor=cursors.get(mode, ""))

    def _trap_canvas_press(self, event):
        mode = self._trap_canvas_hit_test(event.x, event.y)
        if mode is None:
            self._trap_drag_mode = None
            return
        self._trap_drag_mode = mode
        rm, rx = self.img_trap_re_min.get(), self.img_trap_re_max.get()
        im, ix = self.img_trap_im_min.get(), self.img_trap_im_max.get()
        self._trap_drag_start = (event.x, event.y, rm, rx, im, ix)
        self._trap_drag_anchor = {
            "resize_nw": (rx, im), "resize_ne": (rm, im),
            "resize_sw": (rx, ix), "resize_se": (rm, ix),
        }.get(mode, (0.0, 0.0))

    def _trap_canvas_drag(self, event):
        if self._trap_drag_mode is None or self._trap_drag_start is None:
            return
        ox, oy, scale = self._trap_canvas_transform()
        mx0, my0, rm0, rx0, im0, ix0 = self._trap_drag_start

        if self._trap_drag_mode == "rotate":
            re_c = (rm0 + rx0) / 2
            im_c = (im0 + ix0) / 2
            mouse_re = (event.x - ox) / scale
            mouse_im = -(event.y - oy) / scale
            new_angle = math.atan2(-(mouse_re - re_c), mouse_im - im_c)
            self.img_trap_angle_deg.set(round(math.degrees(new_angle), 1))
        elif self._trap_drag_mode == "move":
            d_re = (event.x - mx0) / scale
            d_im = -(event.y - my0) / scale
            w, h = rx0 - rm0, ix0 - im0
            new_rm = max(-5.0, min(5.0 - w, rm0 + d_re))
            new_im = max(-5.0, min(5.0 - h, im0 + d_im))
            self.img_trap_re_min.set(round(new_rm, 3))
            self.img_trap_re_max.set(round(new_rm + w, 3))
            self.img_trap_im_min.set(round(new_im, 3))
            self.img_trap_im_max.set(round(new_im + h, 3))
        else:
            ar = self._trap_aspect_ratio()
            anc_re, anc_im = self._trap_drag_anchor
            drag_re = (event.x - ox) / scale
            if self._trap_drag_mode in ("resize_ne", "resize_se"):
                new_w = max(0.05, drag_re - anc_re)
            else:
                new_w = max(0.05, anc_re - drag_re)
            new_h = new_w / ar
            if self._trap_drag_mode == "resize_se":
                rm, rx, im, ix = anc_re, anc_re + new_w, anc_im - new_h, anc_im
            elif self._trap_drag_mode == "resize_nw":
                rm, rx, im, ix = anc_re - new_w, anc_re, anc_im, anc_im + new_h
            elif self._trap_drag_mode == "resize_ne":
                rm, rx, im, ix = anc_re, anc_re + new_w, anc_im, anc_im + new_h
            else:  # sw
                rm, rx, im, ix = anc_re - new_w, anc_re, anc_im - new_h, anc_im
            self.img_trap_re_min.set(round(rm, 3))
            self.img_trap_re_max.set(round(rx, 3))
            self.img_trap_im_min.set(round(im, 3))
            self.img_trap_im_max.set(round(ix, 3))

        self._update_trap_rect_display()

    def _trap_canvas_release(self, _event):
        if self._trap_drag_mode is not None:
            self._trap_drag_mode = None
            self._trap_drag_start = None
            self._recompute_fractal()

    # ---------------------------------------------------------------------- #

    def _load_trap_image_path(self, path: str):
        img = Image.open(path).convert("RGBA")
        self.img_trap_pil = img
        self.img_trap_tex = np.array(img, dtype=np.uint8)
        name = Path(path).name
        self.img_trap_path_var.set(name[:40] + ("…" if len(name) > 40 else ""))
        self._update_trap_rect_display()
        if self.img_trap_enabled.get():
            self._recompute_fractal()

    def _load_trap_image(self):
        path = filedialog.askopenfilename(
            title="Charger une image avec transparence",
            filetypes=[("PNG", "*.png"), ("Tous les fichiers", "*.*")])
        if not path:
            return
        self._load_trap_image_path(path)

    def _hex_to_rgb(self, hex_color: str) -> np.ndarray:
        h = hex_color.lstrip("#")
        return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.uint8)

    def _img_trap_rect(self) -> np.ndarray:
        return np.array([self.img_trap_re_min.get(), self.img_trap_re_max.get(),
                         self.img_trap_im_min.get(), self.img_trap_im_max.get()])

    def _composite_img_trap(self):
        """Composite self.img_rgba avec la couleur de fond; applique le lissage."""
        if self.img_rgba is None:
            return
        if self._img_geom_active:
            bg = self._hex_to_rgb(self.trap_geom_bg.get())
        else:
            bg = np.array(self.palette[1][0], dtype=np.uint8)
        rgb = self.img_rgba[:, :, :3].copy()
        rgb[self.img_rgba[:, :, 3] == 0] = bg
        if self.img_smooth_var.get():
            rgb = np.array(Image.fromarray(rgb).filter(ImageFilter.SMOOTH))
        self.img_preview = rgb

    def _on_img_smooth_toggle(self):
        if self.img_rgba is not None:
            self._composite_img_trap()
            self._redraw_fractal()
        # si pas encore de calcul, rien à faire

    def _build_buttons(self):
        btns = tk.Frame(self.content)
        btns.pack(pady=8)
        tk.Button(btns, text="Régénérer", command=self._recompute_fractal).pack(side="left", padx=4)
        tk.Button(btns, text="Exporter en HD", command=self._export_hd).pack(side="left", padx=4)

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #
    def _hexcolor(self, rgb) -> str:
        r, g, b = (int(v) for v in rgb)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _refresh_swatch(self, i: int):
        self.swatches[i].config(bg=self._hexcolor(self.palette[1][i]))

    def _make_pos_cb(self, i: int):
        def cb(val):
            self.palette[0][i] = float(val)
            self._apply_palette()
        return cb

    def _make_rgb_cb(self, i: int, channel: int, var: tk.StringVar):
        def cb(*_):
            try:
                v = int(float(var.get()))
            except (ValueError, tk.TclError):
                return
            self.palette[1][i][channel] = max(0, min(255, v))
            self._refresh_swatch(i)
            self._apply_palette()
        return cb

    def _update_c_label(self, *_):
        signe = "+" if self.c_im.get() >= 0 else "-"
        self.c_label_var.set(f"c = {self.c_re.get():.3f} {signe} {abs(self.c_im.get()):.3f} i")
        if hasattr(self, "mandel_dot"):     # suit c en direct sur la carte de Mandelbrot
            self._update_mandel_dot()

    def _c_to_pixel(self, c: complex):
        # même cadrage que generate_mandelbrot : borne=BORNE, borne_y = BORNE*H/W
        borne = self.BORNE
        borne_y = borne * self.PREVIEW_H / self.PREVIEW_W
        px = (c.real + borne) / (2 * borne) * (self.PREVIEW_W - 1)
        py = (c.imag + borne_y) / (2 * borne_y) * (self.PREVIEW_H - 1)
        return px, py

    def _update_mandel_dot(self):
        px, py = self._c_to_pixel(self._current_c())
        r = 4
        self.mandel_canvas.coords(self.mandel_dot, px - r, py - r, px + r, py + r)

    def _on_mandel_click(self, event):
        # pixel -> c (inverse de _c_to_pixel), puis on recalcule la Julia
        borne = self.BORNE
        borne_y = borne * self.PREVIEW_H / self.PREVIEW_W
        re = -borne + event.x / (self.PREVIEW_W - 1) * (2 * borne)
        im = -borne_y + event.y / (self.PREVIEW_H - 1) * (2 * borne_y)
        self.c_re.set(round(re, 4))     # déclenche aussi le déplacement du point rouge
        self.c_im.set(round(im, 4))
        self._recompute_fractal()

    def _current_c(self) -> complex:
        return complex(self.c_re.get(), self.c_im.get())

    # ------------------------------------------------------------------ #
    #  Transformation du plan
    # ------------------------------------------------------------------ #
    def _set_transform(self, expr: str):
        self.transform_var.set(expr)
        self._apply_transform()

    def _apply_transform(self, *_):
        expr = self.transform_var.get()
        try:
            # "z" / vide = identité -> on garde None (rapide, comportement par défaut)
            self.transform = None if expr.strip() in ("", "z") else parse_transform(expr)
        except ValueError as e:
            messagebox.showerror("Transformation invalide", str(e))
            return                       # garde la transfo précédente
        self._recompute_fractal()

    # ------------------------------------------------------------------ #
    #  Navigation des combinaisons Sanzo Wada
    # ------------------------------------------------------------------ #
    def _update_sanzo_label(self):
        n = len(self.sanzo_palettes)
        self.sanzo_label_var.set(f"{self.sanzo_index} / {n - 1}")
        self.sanzo_combo_var.set(self.sanzo_names[self.sanzo_index])

    def _on_sanzo_combo_select(self, *_):
        name = self.sanzo_combo_var.get()
        self._select_sanzo(self.sanzo_names.index(name))

    def _select_sanzo(self, index: int):
        self.sanzo_index = index % len(self.sanzo_palettes)
        self.palette = render.copy_palette(self.sanzo_palettes[self.sanzo_index])
        self._update_sanzo_label()
        self._populate_palette_rows()   # le nombre de couleurs peut changer
        self._apply_palette()

    def _sanzo_prev(self):
        self._select_sanzo(self.sanzo_index - 1)

    def _sanzo_next(self):
        self._select_sanzo(self.sanzo_index + 1)

    def _sanzo_random(self):
        self._select_sanzo(random.randrange(len(self.sanzo_palettes)))

    def _reverse_palette(self):
        pos = self.palette[0]
        col = self.palette[1]
        n = len(pos)
        self.palette[0] = [1.0 - pos[n - 1 - i] for i in range(n)]
        self.palette[1] = list(reversed(col))
        self._populate_palette_rows()
        self._apply_palette()

    def _shuffle_palette(self):
        col = self.palette[1][:]
        random.shuffle(col)
        self.palette[1] = col
        self._populate_palette_rows()
        self._apply_palette()

    # ------------------------------------------------------------------ #
    #  Rendu
    # ------------------------------------------------------------------ #
    def _smooth_now(self) -> bool:
        # le mode bandes a besoin du champ classique (comptes d'itérations entiers)
        return False if self.cyclic_var.get() else self.smooth_var.get()

    def _trap_params_now(self):
        t = self.trap_type_var.get()
        if t == "sinus":
            trap_type = 5
            params = np.array([self.trap_cx.get(), self.trap_cy.get(),
                               self.trap_radius.get(), self.trap_freq.get()])
        elif t in ("cercle", "carré"):
            trap_type = {"cercle": 3, "carré": 4}[t]
            params = np.array([self.trap_cx.get(), self.trap_cy.get(),
                               self.trap_radius.get()])
        else:
            trap_type = {"point": 0, "ligne": 1, "croix": 2}[t]
            params = np.array([self.trap_cx.get(), self.trap_cy.get(),
                               math.radians(self.trap_angle.get())])
        return trap_type, params

    def _recompute_fractal(self, *_):
        k = self._ssaa_factor()
        gen = fractal.FractalGenerator(self.PREVIEW_H * k, self.PREVIEW_W * k, self.N_ITER,
                                       smooth=self._smooth_now(), transform=self.transform)
        poly = iteration.Poly(1, 0, self._current_c())
        if self.img_trap_enabled.get() and self.img_trap_tex is not None:
            if self.img_trap_mode.get() == "geom":
                self._img_geom_active = True
                rgba_hi = gen.generate_julia_geom_trap(
                    poly, self.img_trap_tex,
                    N=self.trap_geom_N.get(),
                    r=self.trap_geom_r.get(),
                    cx=self.trap_geom_cx.get(),
                    cy=self.trap_geom_cy.get(),
                    base_size=self.trap_geom_size.get(),
                    angle_step=math.radians(self.trap_geom_angle.get()))
            else:
                self._img_geom_active = False
                rect = self._img_trap_rect()
                rgba_hi = gen.generate_julia_image_trap(
                    poly, self.img_trap_tex, rect,
                    min_iter=self.img_trap_min_iter.get(),
                    angle=math.radians(self.img_trap_angle_deg.get()))
            if k > 1:
                pil = Image.fromarray(rgba_hi).resize(
                    (self.PREVIEW_W, self.PREVIEW_H), Image.LANCZOS)
                self.img_rgba = np.array(pil)
            else:
                self.img_rgba = rgba_hi
            self.V_preview = None
            self.img_preview = None
            self._composite_img_trap()
        elif self.trap_enabled.get():
            trap_type, trap_params = self._trap_params_now()
            V = gen.generate_julia_trap(poly, trap_type, trap_params,
                                        norm_max=self.trap_norm_max.get())
            self.img_rgba = None
            self.img_preview = None
            self._img_geom_active = False
            self.V_preview = render.downscale_field(V, k)
        else:
            self.img_rgba = None
            self.img_preview = None
            self._img_geom_active = False
            mode = self.coloring_mode.get()
            if mode == "period":
                V = gen.generate_julia_period(poly)
            elif mode == "attractor":
                V = gen.generate_julia_attractor(poly, norm_max=self.attractor_norm.get())
            elif mode == "lambda":
                V = gen.generate_julia_lambda(poly,
                                               burn_in=self.lambda_burn_in.get(),
                                               norm_max=self.attractor_norm.get())
            else:
                V = gen.generate_julia(poly)
            self.V_preview = render.downscale_field(V, k)
        self._redraw_fractal()

    def _redraw_fractal(self):
        if self.img_preview is not None:
            arr = self.img_preview
        elif self.V_preview is not None:
            arr = self._coloriser(self.V_preview)
        else:
            return
        photo = ImageTk.PhotoImage(Image.fromarray(arr.astype(np.uint8)))
        self.fractal_canvas.itemconfig(self.fractal_item, image=photo)
        self.fractal_canvas.image = photo

    def _update_gradient(self):
        C = self._coloriser(self.barre)
        photo = ImageTk.PhotoImage(Image.fromarray(C))
        self.grad_canvas.itemconfig(self.grad_item, image=photo)
        self.grad_canvas.image = photo

    def _apply_palette(self):
        self._update_gradient()
        if self.img_rgba is not None:
            self._composite_img_trap()   # re-composite avec la nouvelle couleur de fond
        self._redraw_fractal()

    def _c_aleatoire(self):
        gen = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER)
        nc = gen.pick_interesting_c()
        self.c_re.set(round(nc.real, 4))
        self.c_im.set(round(nc.imag, 4))
        self._recompute_fractal()

    def _compute_hd(self) -> np.ndarray:
        k = self._ssaa_factor()
        gen = fractal.FractalGenerator(self.FULL_H * k, self.FULL_W * k, self.N_ITER,
                                       smooth=self._smooth_now(), transform=self.transform)
        poly = iteration.Poly(1, 0, self._current_c())
        if self.img_trap_enabled.get() and self.img_trap_tex is not None:
            if self.img_trap_mode.get() == "geom":
                rgba_hi = gen.generate_julia_geom_trap(
                    poly, self.img_trap_tex,
                    N=self.trap_geom_N.get(),
                    r=self.trap_geom_r.get(),
                    cx=self.trap_geom_cx.get(),
                    cy=self.trap_geom_cy.get(),
                    base_size=self.trap_geom_size.get(),
                    angle_step=math.radians(self.trap_geom_angle.get()))
                bg = self._hex_to_rgb(self.trap_geom_bg.get())
            else:
                rect = self._img_trap_rect()
                rgba_hi = gen.generate_julia_image_trap(
                    poly, self.img_trap_tex, rect,
                    min_iter=self.img_trap_min_iter.get(),
                    angle=math.radians(self.img_trap_angle_deg.get()))
                bg = np.array(self.palette[1][0], dtype=np.uint8)
            if k > 1:
                pil = Image.fromarray(rgba_hi).resize(
                    (self.FULL_W, self.FULL_H), Image.LANCZOS)
                rgba = np.array(pil)
            else:
                rgba = rgba_hi
            rgb = rgba[:, :, :3].copy()
            rgb[rgba[:, :, 3] == 0] = bg
            if self.img_smooth_var.get():
                rgb = np.array(Image.fromarray(rgb).filter(ImageFilter.SMOOTH))
            return rgb
        elif self.trap_enabled.get():
            trap_type, trap_params = self._trap_params_now()
            V_full = render.downscale_field(
                gen.generate_julia_trap(poly, trap_type, trap_params,
                                        norm_max=self.trap_norm_max.get()), k)
        else:
            mode = self.coloring_mode.get()
            if mode == "period":
                V_full = render.downscale_field(gen.generate_julia_period(poly), k)
            elif mode == "attractor":
                V_full = render.downscale_field(
                    gen.generate_julia_attractor(poly, norm_max=self.attractor_norm.get()), k)
            elif mode == "lambda":
                V_full = render.downscale_field(
                    gen.generate_julia_lambda(poly,
                                              burn_in=self.lambda_burn_in.get(),
                                              norm_max=self.attractor_norm.get()), k)
            else:
                V_full = render.downscale_field(gen.generate_julia(poly), k)
        return self._coloriser(V_full)

    def _show_hd_popup(self, *_):
        # le calcul HD prend ~1 s : curseur d'attente pendant le rendu
        self.root.config(cursor="watch")
        self.root.update()
        try:
            C = self._compute_hd()
        finally:
            self.root.config(cursor="")

        top = tk.Toplevel(self.root)
        top.title(f"Fractale HD — {self.FULL_W}×{self.FULL_H}")

        # fenêtre plafonnée à la taille de l'écran ; image scrollable en taille réelle
        win_w = min(self.FULL_W, top.winfo_screenwidth() - 80)
        win_h = min(self.FULL_H, top.winfo_screenheight() - 120)
        top.geometry(f"{win_w}x{win_h}")

        canvas = tk.Canvas(top, width=win_w, height=win_h)
        hbar = tk.Scrollbar(top, orient="horizontal", command=canvas.xview)
        vbar = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set,
                         scrollregion=(0, 0, self.FULL_W, self.FULL_H))
        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        photo = ImageTk.PhotoImage(Image.fromarray(C))
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo   # référence gardée (sinon le GC efface l'image)

    def _export_hd(self):
        # l'utilisateur choisit le nom ; enregistre dans Exports/ (sans toucher au fond d'écran)
        default = "fractale_" + datetime.today().strftime("%d_%m_%Y")
        name = simpledialog.askstring("Exporter en HD", "Nom du fichier :",
                                      initialvalue=default, parent=self.root)
        if name is None:
            return                      # annulé
        name = name.strip()
        if not name:
            return
        if not name.lower().endswith(".png"):
            name += ".png"
        path = self.export_dir / Path(name).name   # .name empêche d'échapper le dossier

        self.root.config(cursor="watch")
        self.root.update()
        try:
            C = self._compute_hd()
        finally:
            self.root.config(cursor="")
        Image.fromarray(C).save(path)
        print(f"image exportée : {path}")
        messagebox.showinfo("Export", f"Image enregistrée :\n{path}")

    # ------------------------------------------------------------------ #
    #  Point d'entrée
    # ------------------------------------------------------------------ #
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root).run()
