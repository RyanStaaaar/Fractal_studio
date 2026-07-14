"""DEM-based arc-length reparametrization of Julia parameter paths.

The deformation speed of a Julia set J_c diverges as the path c(t) approaches
the Mandelbrot boundary.  Weighting arc length by 1/distance-to-boundary
(distance estimation method, DEM) and resampling t uniformly in that weighted
length yields frames with roughly constant visual change.
"""

from typing import Callable

import math
import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d

import iteration
import render


@njit(cache=True)
def mandelbrot_distance(c_re: float, c_im: float,
                        max_iter: int = 2000,
                        escape_radius: float = 1e6) -> float:
    """Distance estimate from c to the Mandelbrot boundary (0.0 if interior).

    Iterates z <- z^2 + c with the derivative w <- 2*z*w + 1 and returns
    d ~ |z| * ln|z| / |w| at escape time.
    """
    zr = 0.0
    zi = 0.0
    wr = 0.0
    wi = 0.0
    r2 = escape_radius * escape_radius
    for _ in range(max_iter):
        # w_{n+1} = 2*z_n*w_n + 1   (uses z_n, before the z update)
        wr, wi = 2.0 * (zr * wr - zi * wi) + 1.0, 2.0 * (zr * wi + zi * wr)
        zr, zi = zr * zr - zi * zi + c_re, 2.0 * zr * zi + c_im
        m2 = zr * zr + zi * zi
        if m2 > r2:
            mag = math.sqrt(m2)
            wm  = math.sqrt(wr * wr + wi * wi)
            if wm == 0.0:
                return 0.0
            return mag * math.log(mag) / wm
    return 0.0


# Volontairement non parallèle : appelé depuis le thread Tk pendant que le
# worker de preview exécute des kernels numba parallel=True (workqueue non
# thread-safe — deux régions parallèles concurrentes font avorter le process).
@njit(cache=True)
def _distances(c_re: np.ndarray, c_im: np.ndarray,
               max_iter: int = 2000,
               escape_radius: float = 1e6) -> np.ndarray:
    """Vectorized mandelbrot_distance over arrays of coordinates."""
    n = c_re.shape[0]
    out = np.empty(n)
    for i in range(n):
        out[i] = mandelbrot_distance(c_re[i], c_im[i], max_iter, escape_radius)
    return out


def sample_rho(path_fn: Callable[[float], complex],
               n_samples: int = 4000,
               eps: float = 1e-4,
               rho_cap: float | None = None,
               smooth_sigma: float = 2.0,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the local cost rho = 1/DEM along the path.

    Returns (t, cs, rho_raw, rho_smooth) over n_samples uniform t in [0, 1].
    """
    t  = np.linspace(0.0, 1.0, n_samples)
    cs = np.array([path_fn(ti) for ti in t], dtype=np.complex128)
    d  = _distances(np.ascontiguousarray(cs.real),
                    np.ascontiguousarray(cs.imag))
    d  = np.maximum(d, eps)            # interior points (d == 0) -> eps
    rho_raw = 1.0 / d
    if rho_cap is not None:
        rho_raw = np.minimum(rho_raw, rho_cap)
    closed = abs(path_fn(0.0) - path_fn(1.0)) < 1e-12
    rho_smooth = gaussian_filter1d(rho_raw, smooth_sigma,
                                   mode="wrap" if closed else "nearest")
    return t, cs, rho_raw, rho_smooth


def _default_render_fn(preview_size: int,
                       julia_max_iter: int) -> Callable[[complex], np.ndarray]:
    """Pipeline-default Julia preview: escape time + Sanzo palette in OKLab."""
    colors = render.load_sanzo_palettes()[0][1]
    pos = [i / (len(colors) - 1) for i in range(len(colors))]
    renderer = render.FractalRenderer([pos, [list(c) for c in colors]],
                                      mode="oklab", n_iter=julia_max_iter,
                                      repeat=1, equalize=True, clip_limit=3.0,
                                      eq_range=(0.0, 1.0))
    xs = np.linspace(-2.0, 2.0, preview_size)
    Z = xs[np.newaxis, :] + 1j * xs[:, np.newaxis]

    def fn(c: complex) -> np.ndarray:
        V = iteration.escape_speed(Z, 1 + 0j, 0 + 0j, c,
                                   julia_max_iter, 256.0, True)
        return renderer.render(V)
    return fn


def image_cost(path_fn: Callable[[float], complex],
               n_samples: int = 400,
               preview_size: int = 128,
               julia_max_iter: int = 300,
               color_space: str = "oklab",
               render_fn: Callable[[complex], np.ndarray] | None = None,
               progress_cb: Callable[[int, int], None] | None = None,
               ) -> np.ndarray:
    """Empirical cost: RMS perceptual delta between consecutive previews.

    Renders a small Julia preview for each of n_samples uniform t and measures
    the frame-to-frame RMS difference of the OKLab channels (color_space='rgb'
    falls back to normalized RGB).  Pass render_fn(c) -> RGB uint8 to mirror
    the active UI pipeline; preview_size / julia_max_iter only apply to the
    default pipeline.  Returns n_samples costs; for a closed loop the last
    cost wraps to the first sample, otherwise the end cost is duplicated.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")
    closed = abs(path_fn(0.0) - path_fn(1.0)) < 1e-12
    t = np.linspace(0.0, 1.0, n_samples, endpoint=not closed)
    if render_fn is None:
        render_fn = _default_render_fn(preview_size, julia_max_iter)

    def to_space(frame: np.ndarray) -> np.ndarray:
        rgb = frame.astype(np.float64) / 255.0
        return render.srgb_to_oklab(rgb) if color_space == "oklab" else rgb

    step  = max(1, n_samples // 10)
    costs: list[float] = []
    first = prev = None
    for i, ti in enumerate(t):
        cur = to_space(render_fn(path_fn(float(ti))))
        if i == 0:
            first = cur
        else:
            costs.append(float(np.sqrt(np.mean((cur - prev) ** 2))))
        prev = cur
        if progress_cb is not None:
            progress_cb(i + 1, n_samples)
        elif (i + 1) % step == 0:
            print(f"image_cost: {i + 1}/{n_samples}")
    if closed:
        costs.append(float(np.sqrt(np.mean((first - prev) ** 2))))
    else:
        costs.append(costs[-1])
    return np.asarray(costs)


def build_rho(path_fn: Callable[[float], complex],
              n_samples: int = 4000,
              eps: float = 1e-4,
              rho_cap: float | None = None,
              smooth_sigma: float = 2.0,
              method: str = "dem",
              smooth_sigma_image: float = 4.0,
              image_rho: np.ndarray | None = None,
              image_kwargs: dict | None = None,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build the final cost rho on the fine t grid for the chosen method.

    'dem'    : analytic 1/distance cost (unchanged historical behaviour).
    'image'  : empirical perceptual cost from image_cost, interpolated onto
               the fine grid then smoothed (stronger default sigma, the
               measurement is noisy), median-normalized so rho_cap reads as
               a multiple of the median cost.
    'hybrid' : geometric mean of the two median-normalized costs — the DEM
               provides the smooth analytic structure, the image metric
               corrects for coloring-pipeline effects it cannot see.

    image_rho lets the caller pass a cached/precomputed image_cost array.
    Returns (t, cs, rho, parts); parts maps curve names to arrays for
    diagnostics.
    """
    if method not in ("dem", "image", "hybrid"):
        raise ValueError(f"unknown method: {method!r}")
    t, cs, rho_raw, rho_dem = sample_rho(
        path_fn, n_samples, eps,
        rho_cap if method == "dem" else None, smooth_sigma)
    parts: dict[str, np.ndarray] = {}
    if method == "dem":
        parts["dem_raw"], parts["dem"] = rho_raw, rho_dem
        return t, cs, rho_dem, parts

    if image_rho is None:
        image_rho = image_cost(path_fn, **(image_kwargs or {}))
    image_rho = np.asarray(image_rho, dtype=np.float64)
    closed = abs(path_fn(0.0) - path_fn(1.0)) < 1e-12
    t_img  = np.linspace(0.0, 1.0, len(image_rho), endpoint=not closed)
    img    = (np.interp(t, t_img, image_rho, period=1.0) if closed
              else np.interp(t, t_img, image_rho))
    parts["image_raw"] = img
    img = gaussian_filter1d(img, smooth_sigma_image,
                            mode="wrap" if closed else "nearest")
    img_norm = img / max(float(np.median(img)), 1e-12)
    parts["image"] = img_norm

    if method == "image":
        rho = img_norm
    else:
        dem_norm = rho_dem / max(float(np.median(rho_dem)), 1e-12)
        rho = np.sqrt(dem_norm * img_norm)
        parts["dem"] = dem_norm
        parts["hybrid"] = rho
    if rho_cap is not None:
        rho = np.minimum(rho, rho_cap)
    return t, cs, rho, parts


def reparam_warp(path_fn: Callable[[float], complex],
                 n_samples: int = 4000,
                 eps: float = 1e-4,
                 rho_cap: float | None = None,
                 smooth_sigma: float = 2.0,
                 method: str = "dem",
                 smooth_sigma_image: float = 4.0,
                 image_rho: np.ndarray | None = None,
                 image_kwargs: dict | None = None,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Build the warp (S_norm, t_grid) mapping uniform progress to path time.

    S_norm is the normalized rho-weighted cumulative arc length; interpolating
    np.interp(s, S_norm, t_grid) gives the t achieving constant visual change.
    """
    t, cs, rho, _ = build_rho(path_fn, n_samples, eps, rho_cap, smooth_sigma,
                              method, smooth_sigma_image, image_rho,
                              image_kwargs)
    seg = np.abs(np.diff(cs))
    S = np.concatenate(([0.0], np.cumsum(rho[:-1] * seg)))
    if S[-1] <= 0.0:
        return np.linspace(0.0, 1.0, n_samples), t
    return S / S[-1], t


def reparametrize_path(path_fn: Callable[[float], complex],
                       n_samples: int = 4000,
                       n_frames: int = 300,
                       eps: float = 1e-4,
                       rho_cap: float | None = None,
                       smooth_sigma: float = 2.0,
                       method: str = "dem",
                       smooth_sigma_image: float = 4.0,
                       image_rho: np.ndarray | None = None,
                       image_kwargs: dict | None = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Resample t so each of n_frames carries ~constant visual change.

    method selects the cost: 'dem' (analytic, fast), 'image' (empirical
    perceptual), 'hybrid' (geometric mean of both) — see build_rho.
    Returns (ts, cs): the n_frames warped times in [0, 1] and the matching
    complex parameter values.
    """
    S, t_grid = reparam_warp(path_fn, n_samples, eps, rho_cap, smooth_sigma,
                             method, smooth_sigma_image, image_rho,
                             image_kwargs)
    s_targets = np.linspace(0.0, 1.0, n_frames)
    ts = np.interp(s_targets, S, t_grid)
    cs = np.array([path_fn(ti) for ti in ts], dtype=np.complex128)
    return ts, cs


if __name__ == "__main__":
    # Quick sanity checks of the distance estimator
    for c, expect in ((0.0 + 0.0j, "0 (intérieur)"),
                      (-1.0 + 0.0j, "0 (intérieur)"),
                      (0.26 + 0.0j, "petit > 0 (proche cardioïde)"),
                      (1.0 + 0.0j, "plus grand")):
        d = mandelbrot_distance(c.real, c.imag)
        print(f"DEM({c}) = {d:.6g}   attendu : {expect}")

    ts, cs = reparametrize_path(lambda t: 0.3 * np.exp(2j * np.pi * t) - 0.1,
                                n_samples=1000, n_frames=60)
    print(f"reparametrize_path : {len(ts)} frames, "
          f"t monotone : {bool(np.all(np.diff(ts) >= 0))}")

    # Cohérence DEM / métrique image (Spearman) sur un chemin traversant ∂M.
    # Restreinte à l'extérieur de M : à l'intérieur le DEM sature à 1/eps
    # (proxy aveugle) alors que le coût image mesure le vrai changement —
    # c'est précisément ce que la méthode 'image' corrige.
    from scipy.stats import spearmanr
    fn = lambda t: -0.4 + 0.45 * np.exp(2j * np.pi * t)
    n  = 160
    rho_img = image_cost(fn, n_samples=n, preview_size=64,
                         julia_max_iter=150, progress_cb=lambda i, m: None)
    tg = np.linspace(0.0, 1.0, n, endpoint=False)
    pts = np.array([fn(ti) for ti in tg])
    d   = _distances(np.ascontiguousarray(pts.real),
                     np.ascontiguousarray(pts.imag))
    ext = d > 0
    r = spearmanr(1.0 / np.maximum(d[ext], 1e-4), rho_img[ext]).statistic
    print(f"Spearman DEM/image (extérieur de M) : r = {r:.3f} "
          f"{'OK' if r > 0.5 else 'ÉCHEC'} (> 0.5 attendu)")
