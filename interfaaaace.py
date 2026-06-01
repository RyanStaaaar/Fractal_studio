from pathlib import Path
from datetime import datetime
import subprocess

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk

import iteration
import render
import fractal


class MainWindow:
    FULL_W, FULL_H = 3024, 1964
    PREVIEW_W = 450
    PREVIEW_H = round(PREVIEW_W * FULL_H / FULL_W)
    N_ITER = 80
    BORNE = 2
    GRAD_H = 28

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fractale de Julia + palette")
        self.output_dir = Path(__file__).parent / "Wallpapers"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.V_preview = None
        self.palette = render.make_random_palette()
        self.barre = np.linspace(np.zeros(self.GRAD_H), np.ones(self.GRAD_H), self.PREVIEW_W).T
        self.swatches: list[tk.Label] = []

        c0 = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER).pick_interesting_c()
        self.c_re = tk.DoubleVar(value=round(c0.real, 4))
        self.c_im = tk.DoubleVar(value=round(c0.imag, 4))
        self.mode_var = tk.StringVar(value="rgb")
        self.c_label_var = tk.StringVar()

        self._build_fractal_canvas()
        self._build_gradient_canvas()
        self._build_mode_controls()
        self._build_palette_controls()
        self._build_c_controls()
        self._build_buttons()

        self._recompute_fractal()
        self._update_gradient()

    # ------------------------------------------------------------------ #
    #  Palette helpers
    # ------------------------------------------------------------------ #
    def _coloriser(self, V: np.ndarray) -> np.ndarray:
        renderer = render.FractalRenderer(self.palette, self.mode_var.get())
        return renderer.render(V)

    # ------------------------------------------------------------------ #
    #  UI builders
    # ------------------------------------------------------------------ #
    def _build_fractal_canvas(self):
        self.fractal_canvas = tk.Canvas(self.root, width=self.PREVIEW_W, height=self.PREVIEW_H)
        self.fractal_canvas.pack(padx=10, pady=(10, 4))
        self.fractal_item = self.fractal_canvas.create_image(0, 0, anchor="nw")

    def _build_gradient_canvas(self):
        self.grad_canvas = tk.Canvas(self.root, width=self.PREVIEW_W, height=self.GRAD_H)
        self.grad_canvas.pack(padx=10, pady=(0, 10))
        self.grad_item = self.grad_canvas.create_image(0, 0, anchor="nw")

    def _build_mode_controls(self):
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(padx=10, pady=(4, 0), anchor="w")
        tk.Label(mode_frame, text="Interpolation :").pack(side="left")
        tk.Radiobutton(mode_frame, text="RGB", variable=self.mode_var, value="rgb",
                       command=self._apply_palette).pack(side="left")
        tk.Radiobutton(mode_frame, text="HSV", variable=self.mode_var, value="hsv",
                       command=self._apply_palette).pack(side="left")

    def _build_palette_controls(self):
        pal_frame = tk.LabelFrame(self.root, text="Palette")
        pal_frame.pack(padx=10, pady=4, fill="x")
        for i in range(len(self.palette[0])):
            row = tk.Frame(pal_frame)
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
        c_frame = tk.LabelFrame(self.root, text="Paramètre c")
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

    def _build_buttons(self):
        btns = tk.Frame(self.root)
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

    def _current_c(self) -> complex:
        return complex(self.c_re.get(), self.c_im.get())

    # ------------------------------------------------------------------ #
    #  Rendu
    # ------------------------------------------------------------------ #
    def _recompute_fractal(self, *_):
        gen = fractal.FractalGenerator(self.PREVIEW_H, self.PREVIEW_W, self.N_ITER)
        poly = iteration.Poly(1, 0, self._current_c())
        self.V_preview = gen.generate_julia(poly)
        self._redraw_fractal()

    def _redraw_fractal(self):
        if self.V_preview is None:
            return
        C = self._coloriser(self.V_preview)
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

    def _export_hd(self):
        gen = fractal.FractalGenerator(self.FULL_H, self.FULL_W, self.N_ITER)
        poly = iteration.Poly(1, 0, self._current_c())
        V_full = gen.generate_julia(poly)
        C = self._coloriser(V_full)
        today = datetime.today().strftime("%d_%m_%Y")
        path = self.output_dir / f"wallpaper_{today}.png"
        Image.fromarray(C).save(path)
        print(f"image sauvegardée : {path}")
        try:
            result = subprocess.run(["/usr/local/bin/desktoppr", str(path)], capture_output=True, text=True)
            print(f"desktoppr : {result.returncode} | {result.stdout} | {result.stderr}")
            subprocess.run(["killall", "Dock"])
        except Exception as e:
            print(f"pose du wallpaper ignorée : {e}")

    # ------------------------------------------------------------------ #
    #  Point d'entrée
    # ------------------------------------------------------------------ #
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root).run()
