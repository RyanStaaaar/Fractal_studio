from pathlib import Path
from datetime import datetime
import subprocess

import iteration
import render
import fractal


class WallpaperApp:
    # ------------------------------------------------------------------ #
    #  RÉGLAGES — éditer ces valeurs pour changer le rendu quotidien
    # ------------------------------------------------------------------ #
    MODE = "oklab"        # interpolation des couleurs : "rgb" | "hsv" | "oklab" | "cyclic"
    SMOOTH = True         # lissage logarithmique (ignoré si MODE == "cyclic")
    REPEAT = 1            # > 1 : dégradé replié en miroir n fois (ignoré si MODE == "cyclic")
    SSAA = 2             # supersampling anti-aliasing : 1 = off, 2 = calcul ×2 puis réduit
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
        gen = fractal.FractalGenerator(self.HEIGHT * k, self.WIDTH * k, self.N_ITER, smooth=smooth)
        c = gen.pick_interesting_c()
        poly = iteration.Poly(1, 0, c)
        V = gen.generate_julia(poly)
        palette = render.make_random_palette()
        renderer = render.FractalRenderer(palette, mode=self.MODE, n_iter=self.N_ITER, repeat=self.REPEAT)
        image = render.downscale(renderer.render(V), self.WIDTH, self.HEIGHT)
        #today = datetime.today().strftime("%d_%m_%Y")
        today = "au"
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
