from pathlib import Path
from datetime import datetime
import subprocess

import iteration
import render
import fractal


class WallpaperApp:
    _DESKTOPPR = Path("/usr/local/bin/desktoppr")

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def generate(self) -> Path:
        gen = fractal.FractalGenerator(height=1964, width=3024, n_iter=80)
        c = gen.pick_interesting_c()
        poly = iteration.Poly(1, 0, c)
        V = gen.generate_julia(poly)
        palette = render.make_random_palette()
        renderer = render.FractalRenderer(palette, mode="oklab")
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
