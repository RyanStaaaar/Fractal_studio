from pathlib import Path
from datetime import datetime
import subprocess
import random

import numpy as np

import iteration
import render
import fractal
from transform import parse_transform


class WallpaperApp:
    # ------------------------------------------------------------------ #
    #  RÉGLAGES — éditer ces valeurs pour changer le rendu quotidien
    # ------------------------------------------------------------------ #
    MODE = "oklab"        # interpolation des couleurs : "rgb" | "hsv" | "oklab" | "cyclic"
    SMOOTH = True         # lissage logarithmique (ignoré si MODE == "cyclic")
    REPEAT = random.randint(1, 2)
    REPEAT = 2
    EQUALIZE = True
    CLIP_LIMIT = 5.5
    SSAA = 4
    TRANSFORM = "z"
    N_ITER = 100
    WIDTH, HEIGHT = 3024, 1964

    # Orbit trap (point) — position tirée d'une loi normale N(mu, sigma)
    TRAP_MU    = 0.0
    TRAP_SIGMA = 0.5
    TRAP_NORM_MAX = 0.5
    # ------------------------------------------------------------------ #

    _DESKTOPPR = Path("/usr/local/bin/desktoppr")

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate(self, mu: float | None = None, sigma: float | None = None) -> Path:
        mu    = self.TRAP_MU    if mu    is None else mu
        sigma = self.TRAP_SIGMA if sigma is None else sigma

        smooth = False if self.MODE == "cyclic" else self.SMOOTH
        k = max(1, self.SSAA)
        transform = parse_transform(self.TRANSFORM)
        gen = fractal.FractalGenerator(self.HEIGHT * k, self.WIDTH * k, self.N_ITER,
                                       smooth=smooth, transform=transform)
        c = gen.pick_interesting_c()
        poly = iteration.Poly(1, 0, c)

        # position du point de trap : distribution normale N(mu, sigma)
        cx = float(np.random.normal(mu, sigma))
        cy = float(np.random.normal(mu, sigma))
        trap_params = np.array([cx, cy, 0.0])
        V = render.downscale_field(
            gen.generate_julia_trap(poly, trap_type=0,
                                    trap_params=trap_params,
                                    norm_max=self.TRAP_NORM_MAX), k)

        palette = render.make_random_palette()
        renderer = render.FractalRenderer(palette, mode=self.MODE, n_iter=self.N_ITER,
                                          repeat=self.REPEAT, equalize=self.EQUALIZE,
                                          clip_limit=self.CLIP_LIMIT)
        image = renderer.render(V)
        today = random.randint(0, 100)
        path = self.output_dir / f"wallpaper_{today}.png"
        renderer.save(image, path)
        print(f"[trap point ({cx:.3f}, {cy:.3f})]  image sauvegardée : {path}")
        return path

    def set_as_wallpaper(self, path: Path) -> None:
        result = subprocess.run([str(self._DESKTOPPR), str(path)], capture_output=True, text=True)
        print(f"desktoppr retour : {result.returncode} | {result.stdout} | {result.stderr}")
        subprocess.run(["killall", "Dock"])
        print("dock relancé")

    def run(self) -> None:
        path = self.generate()
        self.set_as_wallpaper(path)


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "Wallpapers"
    WallpaperApp(output_dir).run()
    print(WallpaperApp.REPEAT)
