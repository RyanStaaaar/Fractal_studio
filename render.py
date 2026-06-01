from __future__ import annotations
from PIL import Image
import numpy as np
import random

class Color : 
    def __init__(self, r=255, g=255, b=255) :
        self.r=r
        self.g=g
        self.b=b
    def __str__(self):
        return f"({self.r},{self.g},{self.b})"
    def __add__(self, autre_color : Color) :
        return Color(min(255, self.r + autre_color.r),min(255, self.g + autre_color.g),min(255, self.b + autre_color.b))

    def __mul__(self, coeff) :
        if 0 <= coeff <=1 :
            return Color(coeff*self.r, coeff*self.g, coeff*self.b) 
        else :
            return NotImplemented
    def __rmul__(self,coeff) :
        return self.__mul__(coeff)
    
    def blend(self, autre_color, lambda_=0.5) :
            return lambda_*self + (1-lambda_)*autre_color

    def get_rgb(self) :
        return (int(self.r), int(self.g), int(self.b))
    
Color.black = Color(0,0,0)
Color.white = Color(255,255,255)
Color.red = Color(255,0,0)
Color.green = Color(0,255,0)
Color.blue = Color(0,0,255)

Color.magenta= Color.red + Color.blue
Color.yellow = Color.red + Color.green
Color.cyan= Color.green + Color.blue

Color.violet = Color.red.blend(Color.blue)
Color.orange = Color.red.blend(Color.yellow, 0.5)
Color.green2 = Color.cyan.blend(Color.yellow,0.5)


class ColorScale:
    def __init__(self, color_0: Color, color_1: Color) :
        self.color_0 = color_0
        self.color_1 = color_1
    def level(self, lambda_) :
        return self.color_0.blend(self.color_1, lambda_)

        
ColorScale.interfaaace_green = ColorScale(Color(80, 255, 120),Color(255,255,255))
ColorScale.lava_ocean = ColorScale(Color(120, 0, 0),Color(102, 155, 188))
ColorScale.red_flag_blue = ColorScale(Color(193, 18, 31),Color(0, 48, 73))
ColorScale.fresh_meadow = ColorScale(Color(181, 228, 140),Color(22, 138, 173))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc == 0, 0.0, delta / np.where(maxc == 0, 1.0, maxc))
    deltac = np.where(delta == 0, 1.0, delta)
    rc = (maxc - r) / deltac
    gc = (maxc - g) / deltac
    bc = (maxc - b) / deltac
    h = np.where(maxc == r, bc - gc, 0.0)
    h = np.where(maxc == g, 2.0 + rc - bc, h)
    h = np.where(maxc == b, 4.0 + gc - rc, h)
    h = np.where(delta == 0, 0.0, (h / 6.0) % 1.0)
    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    w = v * (1.0 - s * (1.0 - f))
    i = i % 6
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, w, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [w, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, w, v, v, q])
    rgb = np.stack([r, g, b], axis=-1)
    gray = np.stack([v, v, v], axis=-1)
    return np.where(s[..., None] == 0, gray, rgb)


def coloriser_rgb(V: np.ndarray, palette: list) -> np.ndarray:
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    colors = colors[order]
    V_clamped = np.clip(V, positions[0], positions[-1])
    idx_right = np.clip(np.searchsorted(positions, V_clamped, side="right"), 1, len(positions) - 1)
    idx_left = idx_right - 1
    span = positions[idx_right] - positions[idx_left]
    span[span == 0] = 1
    t = (V_clamped - positions[idx_left]) / span
    t = t[:, :, None]
    return (colors[idx_left] * (1 - t) + colors[idx_right] * t).astype(np.uint8)


def coloriser_hsv(V: np.ndarray, palette: list) -> np.ndarray:
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    colors = colors[order]
    stops_hsv = rgb_to_hsv(colors / 255.0)
    V_clamped = np.clip(V, positions[0], positions[-1])
    idx_right = np.clip(np.searchsorted(positions, V_clamped, side="right"), 1, len(positions) - 1)
    idx_left = idx_right - 1
    span = positions[idx_right] - positions[idx_left]
    span[span == 0] = 1
    t = (V_clamped - positions[idx_left]) / span
    lhsv = stops_hsv[idx_left]
    rhsv = stops_hsv[idx_right]
    lh, rh = lhsv[..., 0], rhsv[..., 0]
    dh = (rh - lh + 0.5) % 1.0 - 0.5
    h = (lh + t * dh) % 1.0
    s = lhsv[..., 1] * (1 - t) + rhsv[..., 1] * t
    v = lhsv[..., 2] * (1 - t) + rhsv[..., 2] * t
    return (hsv_to_rgb(np.stack([h, s, v], axis=-1)) * 255.0).astype(np.uint8)


class FractalRenderer:
    def __init__(self, palette: list, mode: str = "rgb"):
        self.palette = palette
        self.mode = mode

    def render(self, V: np.ndarray) -> np.ndarray:
        if self.mode == "hsv":
            return coloriser_hsv(V, self.palette)
        return coloriser_rgb(V, self.palette)

    def save(self, image: np.ndarray, path) -> None:
        Image.fromarray(image).save(path)


def make_random_palette() -> list:
    """Tire une palette aléatoire : 5 positions triées + 5 couleurs (listes RGB mutables)."""
    nuancier = [
        Color(0, 0, 0), Color(255, 255, 255),
        Color(116, 0, 184), Color(105, 48, 195),
        Color(94, 96, 206), Color(83, 144, 217),
        Color(78, 168, 222), Color(72, 191, 227),
        Color(86, 207, 225), Color(100, 223, 223),
        Color(114, 239, 221), Color(128, 255, 219),
    ]
    positions = sorted([0.0, random.uniform(0, 0.8), random.uniform(0, 0.8), random.uniform(0, 0.8), 1.0])
    colors = [list(c.get_rgb()) for c in random.sample(nuancier, k=5)]
    return [positions, colors]
