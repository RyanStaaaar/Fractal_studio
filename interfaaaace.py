from pathlib import Path
from datetime import datetime
import random

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

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

        self._build_scroll_container()
        self._build_top_row()
        self._build_mode_controls()
        self._build_sanzo_controls()
        self._build_palette_controls()
        self._build_c_controls()
        self._build_transform_controls()
        self._build_buttons()
        self._fit_window()

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
        if self.cyclic_var.get():
            renderer = render.FractalRenderer(self.palette, "cyclic", self.N_ITER)
        else:
            renderer = render.FractalRenderer(self.palette, self.mode_var.get(),
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
        self.scroll_canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
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
        # largeur = contenu ; hauteur plafonnée à l'écran -> le reste se scrolle
        self.content.update_idletasks()
        w = self.content.winfo_reqwidth() + 24      # + barre de défilement
        h = min(self.content.winfo_reqheight(), self.root.winfo_screenheight() - 120)
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

    def _build_mandelbrot_canvas(self):
        self.mandel_canvas = tk.Canvas(self.top_frame, width=self.PREVIEW_W, height=self.PREVIEW_H,
                                       cursor="cross")
        self.mandel_canvas.pack(side="left", padx=(8, 0), anchor="n")
        self.mandel_item = self.mandel_canvas.create_image(0, 0, anchor="nw")
        self._render_mandelbrot_base()
        # point rouge indiquant la position du c de la Julia courante
        self.mandel_dot = self.mandel_canvas.create_oval(0, 0, 0, 0, outline="red", fill="red")
        # clic sur la carte -> choisit c et recalcule la Julia
        self.mandel_canvas.bind("<Button-1>", self._on_mandel_click)

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

    def _build_sanzo_controls(self):
        f = tk.LabelFrame(self.content, text="Combinaison Sanzo Wada")
        f.pack(padx=10, pady=4, fill="x")
        tk.Button(f, text="◀ Précédent", command=self._sanzo_prev).pack(side="left", padx=4, pady=4)
        tk.Label(f, textvariable=self.sanzo_label_var, width=14).pack(side="left", padx=4)
        tk.Button(f, text="Suivant ▶", command=self._sanzo_next).pack(side="left", padx=4)
        tk.Button(f, text="Aléatoire", command=self._sanzo_random).pack(side="left", padx=4)
        self._update_sanzo_label()

    def _build_palette_controls(self):
        # cadre persistant : son contenu (les lignes) est reconstruit à chaque
        # changement de combinaison, car le nombre de stops varie (2 à 4 couleurs)
        self.pal_frame = tk.LabelFrame(self.content, text="Palette")
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

    def _build_buttons(self):
        btns = tk.Frame(self.content)
        btns.pack(pady=8)
        tk.Button(btns, text="Régénérer", command=self._recompute_fractal).pack(side="left", padx=4)
        tk.Button(btns, text="c aléatoire", command=self._c_aleatoire).pack(side="left", padx=4)
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
        self.sanzo_label_var.set(f"Combo {self.sanzo_index + 1} / {len(self.sanzo_palettes)}")

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

    # ------------------------------------------------------------------ #
    #  Rendu
    # ------------------------------------------------------------------ #
    def _smooth_now(self) -> bool:
        # le mode bandes a besoin du champ classique (comptes d'itérations entiers)
        return False if self.cyclic_var.get() else self.smooth_var.get()

    def _recompute_fractal(self, *_):
        # V calculé à k× la résolution d'aperçu ; réduit au moment de l'affichage
        k = self._ssaa_factor()
        gen = fractal.FractalGenerator(self.PREVIEW_H * k, self.PREVIEW_W * k, self.N_ITER,
                                       smooth=self._smooth_now(), transform=self.transform)
        poly = iteration.Poly(1, 0, self._current_c())
        self.V_preview = gen.generate_julia(poly)
        self._redraw_fractal()

    def _redraw_fractal(self):
        if self.V_preview is None:
            return
        C = render.downscale(self._coloriser(self.V_preview), self.PREVIEW_W, self.PREVIEW_H)
        photo = ImageTk.PhotoImage(Image.fromarray(C))
        self.fractal_canvas.itemconfig(self.fractal_item, image=photo)
        self.fractal_canvas.image = photo

    def _update_gradient(self):
        C = self._coloriser(self.barre)
        photo = ImageTk.PhotoImage(Image.fromarray(C))
        self.grad_canvas.itemconfig(self.grad_item, image=photo)
        self.grad_canvas.image = photo

    def _apply_palette(self):
        self._update_gradient()
        self._redraw_fractal()

    def _c_aleatoire(self):
        gen = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER)
        nc = gen.pick_interesting_c()
        self.c_re.set(round(nc.real, 4))
        self.c_im.set(round(nc.imag, 4))
        self._recompute_fractal()

    def _compute_hd(self) -> np.ndarray:
        # calcule la fractale en pleine résolution (FULL_W x FULL_H) pour le c courant,
        # avec supersampling k× puis réduction moyennée (anti-aliasing)
        k = self._ssaa_factor()
        gen = fractal.FractalGenerator(self.FULL_H * k, self.FULL_W * k, self.N_ITER,
                                       smooth=self._smooth_now(), transform=self.transform)
        poly = iteration.Poly(1, 0, self._current_c())
        V_full = gen.generate_julia(poly)
        return render.downscale(self._coloriser(V_full), self.FULL_W, self.FULL_H)

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
