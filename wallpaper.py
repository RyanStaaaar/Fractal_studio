from pathlib import Path
from datetime import datetime
import subprocess
import random

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
    REPEAT = random.randint(2,4)      # > 1 : dégradé replié en miroir n fois (ignoré si MODE == "cyclic")
    EQUALIZE = True      # True : égalisation d'histogramme (répartit les couleurs uniformément)
    CLIP_LIMIT = 5.5      # limite de contraste de l'égalisation (bas = + de détail, haut = égalisation pure)
    SSAA = 4             # supersampling anti-aliasing : 1 = off, 2 = calcul ×2 puis réduit
    TRANSFORM = "z"      # transformation du plan f(z) (pullback) : "z" = aucune ; ex "i*z", "z^2", "e^z"
    N_ITER = 100           # nombre d'itérations
    WIDTH, HEIGHT = 3024, 1964   # résolution de l'image
    # ------------------------------------------------------------------ #

    _DESKTOPPR = Path("/usr/local/bin/desktoppr")

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate(self) -> Path:
        # le mode bandes a besoin du champ classique (comptes d'itérations entiers)
        smooth = False if self.MODE == "cyclic" else self.SMOOTH
        k = max(1, self.SSAA)   # supersampling : calcul à k× puis réduction moyennée
        transform = parse_transform(self.TRANSFORM)
        gen = fractal.FractalGenerator(self.HEIGHT * k, self.WIDTH * k, self.N_ITER,
                                       smooth=smooth, transform=transform)
        c = gen.pick_interesting_c()
        poly = iteration.Poly(1, 0, c)
        V = render.downscale_field(gen.generate_julia(poly), k)
        palette = render.make_random_palette()
        renderer = render.FractalRenderer(palette, mode=self.MODE, n_iter=self.N_ITER,
                                          repeat=self.REPEAT, equalize=self.EQUALIZE,
                                          clip_limit=self.CLIP_LIMIT)
        image = renderer.render(V)
        today = datetime.today().strftime("%d_%m_%Y")
        path = self.output_dir / f"wallpaper_{today}.png"
        renderer.save(image, path)
        print(f"image sauvegardée : {path}")
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
