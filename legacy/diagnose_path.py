"""Diagnostic plots for DEM-based path reparametrization.

Renders three panels for a given c(t) path:
  (a) the path over a quick grayscale Mandelbrot render, with the
      reparametrized frame points marked (denser near the boundary),
  (b) the cost curves — raw/smoothed for 'dem' and 'image'; for 'hybrid'
      both median-normalized costs overlaid plus the combined curve, to
      visualize where they diverge,
  (c) the warped t(frame) curve showing where the animation slows down.

Usage: python diagnose_path.py [--method {dem,image,hybrid}] [output.png]
Edit PATH_FN below (or import and call diagnose() with your own path).
"""

import argparse
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit, prange

from path_reparam import build_rho, reparam_warp


def PATH_FN(t: float) -> complex:
    """Demo path: a circle crossing the Mandelbrot boundary."""
    return -0.4 + 0.0j + 0.45 * np.exp(2j * np.pi * t)


@njit(cache=True, parallel=True)
def _mandel_field(re0: float, re1: float, im0: float, im1: float,
                  w: int, h: int, n: int = 200) -> np.ndarray:
    """Quick escape-time field of M for the background render."""
    out = np.zeros((h, w))
    for y in prange(h):
        for x in range(w):
            c = complex(re0 + (re1 - re0) * x / (w - 1),
                        im0 + (im1 - im0) * y / (h - 1))
            z = 0j
            for i in range(n):
                z = z * z + c
                if z.real * z.real + z.imag * z.imag > 4.0:
                    out[y, x] = 1.0 - i / n
                    break
    return out


def diagnose(path_fn: Callable[[float], complex] = PATH_FN,
             n_samples: int = 4000,
             n_frames: int = 300,
             eps: float = 1e-4,
             rho_cap: float | None = None,
             smooth_sigma: float = 2.0,
             method: str = "dem",
             smooth_sigma_image: float = 4.0,
             image_kwargs: dict | None = None,
             out_path: str = "diagnose_path.png") -> None:
    """Build and save the three diagnostic panels for the chosen method."""
    t, cs, rho, parts = build_rho(path_fn, n_samples, eps, rho_cap,
                                  smooth_sigma, method, smooth_sigma_image,
                                  image_kwargs=image_kwargs)
    seg = np.abs(np.diff(cs))
    S = np.concatenate(([0.0], np.cumsum(rho[:-1] * seg)))
    S = S / S[-1] if S[-1] > 0 else np.linspace(0.0, 1.0, len(t))
    ts = np.interp(np.linspace(0.0, 1.0, n_frames), S, t)
    cs_frames = np.array([path_fn(ti) for ti in ts])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) path + reparametrized frame points over M
    ax = axes[0]
    pad = 0.3
    re0, re1 = cs.real.min() - pad, cs.real.max() + pad
    im0, im1 = cs.imag.min() - pad, cs.imag.max() + pad
    M = _mandel_field(re0, re1, im0, im1, 600, 450)
    ax.imshow(M, extent=(re0, re1, im0, im1), origin="lower",
              cmap="gray", aspect="equal")
    ax.plot(cs.real, cs.imag, color="#00cccc", lw=1.0, label="chemin c(t)")
    ax.scatter(cs_frames.real, cs_frames.imag, s=6, color="#ffaa00",
               zorder=3, label=f"{n_frames} frames reparamétrées")
    ax.set_title(f"(a) chemin et frames sur M — méthode {method}")
    ax.legend(loc="upper right", fontsize=8)

    # (b) cost curves, per method
    ax = axes[1]
    if method == "dem":
        ax.semilogy(t, parts["dem_raw"], color="#888888", lw=0.6,
                    label="rho DEM brut")
        ax.semilogy(t, parts["dem"], color="#cc4444", lw=1.4,
                    label=f"rho DEM lissé (σ={smooth_sigma})")
    elif method == "image":
        ax.semilogy(t, parts["image_raw"], color="#888888", lw=0.6,
                    label="rho image brut (interpolé)")
        ax.semilogy(t, parts["image"] * np.median(parts["image_raw"]),
                    color="#cc4444", lw=1.4,
                    label=f"rho image lissé (σ={smooth_sigma_image})")
    else:  # hybrid : les deux coûts normalisés + la combinaison
        ax.semilogy(t, parts["dem"], color="#4488cc", lw=1.1,
                    label="DEM (normalisé médiane)")
        ax.semilogy(t, parts["image"], color="#44aa66", lw=1.1,
                    label="image (normalisé médiane)")
        ax.semilogy(t, parts["hybrid"], color="#cc4444", lw=1.6,
                    label="hybride = √(DEM·image)")
    ax.set_xlabel("t")
    ax.set_ylabel("rho")
    ax.set_title("(b) coût local le long du chemin")
    ax.legend(fontsize=8)

    # (c) t(frame): plateaus = slowdown near the boundary
    ax = axes[2]
    ax.plot(np.arange(n_frames), ts, color="#4488cc", lw=1.4)
    ax.plot([0, n_frames - 1], [0, 1], color="#555555", lw=0.8,
            ls="--", label="sans reparamétrisation")
    ax.set_xlabel("frame")
    ax.set_ylabel("t")
    ax.set_title("(c) t(frame) — pentes faibles = ralentissement")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"diagnostic sauvegardé → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=("dem", "image", "hybrid"),
                    default="dem")
    ap.add_argument("out", nargs="?", default="diagnose_path.png")
    args = ap.parse_args()
    # La passe image (400 previews par défaut) est coûteuse : paramètres
    # réduits pour le diagnostic.
    img_kw = {"n_samples": 200, "preview_size": 96, "julia_max_iter": 200}
    diagnose(method=args.method, out_path=args.out,
             image_kwargs=img_kw if args.method != "dem" else None)
