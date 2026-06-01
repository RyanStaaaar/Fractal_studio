from pathlib import Path
from datetime import datetime
import subprocess
import random

import iteration
import render
import fractal


class WallpaperApp:
    _DESKTOPPR = Path("/usr/local/bin/desktoppr")

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def _random_palette(self) -> list:
        nuancier = [
            render.Color(0, 0, 0), render.Color(255, 255, 255),
            render.Color(116, 0, 184), render.Color(105, 48, 195),
            render.Color(94, 96, 206), render.Color(83, 144, 217),
            render.Color(78, 168, 222), render.Color(72, 191, 227),
            render.Color(86, 207, 225), render.Color(100, 223, 223),
            render.Color(114, 239, 221), render.Color(128, 255, 219),
        ]
        positions = sorted([0.0, random.uniform(0, 0.8), random.uniform(0, 0.8), random.uniform(0, 0.8), 1.0])
        colors = [list(c.get_rgb()) for c in random.sample(nuancier, k=5)]
        return [positions, colors]

    def generate(self) -> Path:
        gen = fractal.FractalGenerator(height=1964, width=3024, n_iter=80)
        c = gen.pick_interesting_c()
        poly = iteration.Poly(1, 0, c)
        V = gen.generate_julia(poly)
        palette = self._random_palette()
        renderer = render.FractalRenderer(palette)
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
