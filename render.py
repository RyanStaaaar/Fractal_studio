from __future__ import annotations
from pathlib import Path
import json
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


# ------------------------------------------------------------------ #
#  Oklab (Björn Ottosson) : espace perceptuellement uniforme,
#  donne des dégradés plus naturels que RGB ou HSV.
# ------------------------------------------------------------------ #
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c > 0.0031308, 1.055 * (c ** (1 / 2.4)) - 0.055, 12.92 * c)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB [0,1], shape (...,3) -> Oklab (L, a, b)."""
    lin = _srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack([L, a, bb], axis=-1)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Oklab (L, a, b) -> sRGB [0,1], shape (...,3)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return _linear_to_srgb(np.stack([r, g, bb], axis=-1))


def coloriser_oklab(V: np.ndarray, palette: list) -> np.ndarray:
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    colors = colors[order]
    stops_lab = srgb_to_oklab(colors / 255.0)
    V_clamped = np.clip(V, positions[0], positions[-1])
    idx_right = np.clip(np.searchsorted(positions, V_clamped, side="right"), 1, len(positions) - 1)
    idx_left = idx_right - 1
    span = positions[idx_right] - positions[idx_left]
    span[span == 0] = 1
    t = (V_clamped - positions[idx_left]) / span
    t = t[:, :, None]
    lab = stops_lab[idx_left] * (1 - t) + stops_lab[idx_right] * t
    return (oklab_to_srgb(lab) * 255.0).astype(np.uint8)


# ------------------------------------------------------------------ #
#  Coloration cyclique « limited color » : on indexe une couleur par
#  (itérations modulo N), N = nombre de couleurs de la palette. Pas
#  d'interpolation -> bandes nettes qui recyclent les couleurs.
# ------------------------------------------------------------------ #
def coloriser_cyclic(V: np.ndarray, palette: list, n_iter: int) -> np.ndarray:
    """V est le champ classique (n-i)/n -> on récupère i = n*(1-V) et indexe colors[i % N]."""
    colors = np.array(palette[1], dtype=np.float64)
    N = len(colors)
    i = np.rint(n_iter * (1.0 - V)).astype(np.int64)
    idx = np.mod(i, N)
    return colors[idx].astype(np.uint8)


def downscale(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Réduit une image RGB à (out_w, out_h) par moyennage (supersampling / SSAA).
    Calculer à k× la résolution puis appeler ceci lisse l'aliasing et le Moiré."""
    if image.shape[1] == out_w and image.shape[0] == out_h:
        return image
    return np.asarray(Image.fromarray(image).resize((out_w, out_h), Image.LANCZOS))


def equalize_field(V: np.ndarray) -> np.ndarray:
    """Égalisation d'histogramme : remappe V par sa CDF pour répartir les couleurs
    uniformément (chaque valeur devient son rang relatif dans l'image). Les points
    de l'ensemble (V == 0, jamais échappés) restent à 0."""
    eq = np.zeros_like(V, dtype=np.float64)
    mask = V > 0
    vals = V[mask]
    if vals.size == 0:
        return eq
    _, inv, counts = np.unique(vals, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts) / vals.size          # CDF dans (0, 1], égales valeurs -> même couleur
    eq[mask] = cdf[inv.ravel()]
    return eq


def mirror_repeat(V: np.ndarray, n: int) -> np.ndarray:
    """Remappe V (dans [0,1]) en onde triangulaire : le dégradé se répète n fois
    en miroir (0->1->0->1...), sans couture. n=1 redonne le dégradé normal."""
    u = V * n
    return 1.0 - np.abs((u % 2.0) - 1.0)


class FractalRenderer:
    def __init__(self, palette: list, mode: str = "rgb", n_iter: int = 80, repeat: int = 1,
                 equalize: bool = False):
        self.palette = palette
        self.mode = mode
        self.n_iter = n_iter      # utilisé seulement par le mode "cyclic"
        self.repeat = repeat      # > 1 : dégradé répété en miroir (cyclic gradient)
        self.equalize = equalize  # True : égalisation d'histogramme avant l'interpolation

    def render(self, V: np.ndarray) -> np.ndarray:
        # bandes indexées : pas d'interpolation ni de répétition de dégradé
        if self.mode == "cyclic":
            return coloriser_cyclic(V, self.palette, self.n_iter)
        # dégradé : égalisation d'histogramme puis éventuel repli miroir, avant interpolation
        if self.equalize:
            V = equalize_field(V)
        if self.repeat > 1:
            V = mirror_repeat(V, self.repeat)
        if self.mode == "hsv":
            return coloriser_hsv(V, self.palette)
        if self.mode == "oklab":
            return coloriser_oklab(V, self.palette)
        return coloriser_rgb(V, self.palette)

    def save(self, image: np.ndarray, path) -> None:
        Image.fromarray(image).save(path)


# ------------------------------------------------------------------ #
#  Palettes « A Dictionary of Color Combinations » (Sanzo Wada)
#  Données : github.com/mattdesl/dictionary-of-colour-combinations
# ------------------------------------------------------------------ #
_SANZO_PATH = Path(__file__).parent / "data" / "sanzo_colors.json"
_sanzo_palettes_cache = None


def copy_palette(palette: list) -> list:
    """Copie profonde d'une palette [positions, colors] (évite de muter le cache partagé)."""
    return [list(palette[0]), [list(c) for c in palette[1]]]


def load_sanzo_palettes(path=_SANZO_PATH) -> list:
    """Charge les combinaisons Sanzo Wada -> liste de palettes [positions, colors].

    Le JSON est centré-couleur : chaque couleur indique les combinaisons (1-348)
    auxquelles elle appartient. On regroupe par combinaison pour reconstruire
    chaque palette, avec des positions de dégradé réparties uniformément sur [0, 1].
    Le résultat est mis en cache (liste partagée — utiliser copy_palette pour muter).
    """
    global _sanzo_palettes_cache
    if _sanzo_palettes_cache is None:
        data = json.loads(Path(path).read_text())
        combos: dict[int, list] = {}
        for color in data:
            for cid in color["combinations"]:
                combos.setdefault(cid, []).append(color["rgb"])
        palettes = []
        for cid in sorted(combos):
            colors = combos[cid]
            n = len(colors)
            positions = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
            palettes.append([positions, [list(c) for c in colors]])
        _sanzo_palettes_cache = palettes
    return _sanzo_palettes_cache


def make_random_palette() -> list:
    """Tire une combinaison de couleurs Sanzo Wada au hasard (copie mutable)."""
    return copy_palette(random.choice(load_sanzo_palettes()))
