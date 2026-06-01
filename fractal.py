from pathlib import Path
import numpy as np
import iteration


class FractalGenerator:
    _MAP_PATH = Path(__file__).parent / "mandelbrot_map.npy"

    def __init__(self, height: int, width: int, n_iter: int = 80):
        self.height = height
        self.width = width
        self.n_iter = n_iter

    def pick_interesting_c(self, borne: float = 2, seuil_bas: float = 0.01, seuil_haut: float = 0.89) -> complex:
        V = np.load(self._MAP_PATH)
        H, W = V.shape
        masque = (V >= seuil_bas) & (V <= seuil_haut)
        ys, xs = np.nonzero(masque)
        idx = np.random.randint(len(xs))
        borne_y = borne * H / W
        cx = -borne + xs[idx] * (2 * borne) / (W - 1)
        cy = -borne_y + ys[idx] * (2 * borne_y) / (H - 1)
        return complex(cx, cy)

    def generate_julia(self, poly: iteration.Poly, borne: float = 2) -> np.ndarray:
        borne_y = borne * self.height / self.width
        xs = np.linspace(-borne, borne, self.width)
        ys = np.linspace(-borne_y, borne_y, self.height)
        X, Y = np.meshgrid(xs, ys)
        Z = X + 1j * Y
        return iteration.escape_speed(Z, poly.a, poly.b, poly.c, self.n_iter)

    def generate_mandelbrot(self, seed: int = 0, borne: float = 2) -> np.ndarray:
        borne_y = borne * self.height / self.width
        xs = np.linspace(-borne, borne, self.width)
        ys = np.linspace(-borne_y, borne_y, self.height)
        X, Y = np.meshgrid(xs, ys)
        C = X + 1j * Y
        return iteration.mandelbrot(C, seed, self.n_iter)

    def generate_mosaic(self, n_sub: int = 11, tile_size: int = 100, borne_julia: float = 2, borne_c: float = 2) -> np.ndarray:
        meta_size = n_sub * tile_size
        V = np.zeros((meta_size, meta_size))
        cs = np.linspace(-borne_c, borne_c, n_sub)
        tile_gen = FractalGenerator(tile_size, tile_size, self.n_iter)
        for i, cy in enumerate(cs):
            for j, cx in enumerate(cs):
                c = complex(cx, cy)
                f = iteration.Poly(1, 0, c)
                V[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile_gen.generate_julia(f, borne_julia)
        return V
