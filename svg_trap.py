"""Vector SVG orbit traps.

Instead of sampling a rasterized texture, orbit points are tested directly
against the SVG geometry (non-zero winding over flattened paths), so the trap
boundary stays exact at any scale or trap size.  Curves are flattened to
polylines with a tolerance relative to the document size (1/1024th), far
below one pixel of any render.

Gradient fills are resolved to the mean color of their stops; shapes whose
effective opacity is below ~40 % (e.g. vignette overlays) are skipped.
"""

import math
import re
import xml.etree.ElementTree as ET

import numpy as np
from numba import njit, prange
from svgelements import SVG, Shape, Path as SvgPath, Move, Close, Line

_ALPHA_MIN = 102   # opacité effective minimale (sur 255) pour piéger


def _parse_css_color(s: str) -> tuple[int, int, int] | None:
    s = s.strip()
    m = re.match(r"#([0-9a-fA-F]{6})$", s)
    if m:
        v = int(m.group(1), 16)
        return (v >> 16) & 255, (v >> 8) & 255, v & 255
    m = re.match(r"#([0-9a-fA-F]{3})$", s)
    if m:
        h = m.group(1)
        return tuple(int(ch * 2, 16) for ch in h)
    return None


def _gradient_table(path: str) -> dict[str, tuple[tuple[int, int, int], float]]:
    """id de dégradé → (couleur moyenne des stops, opacité moyenne)."""
    table: dict[str, tuple[tuple[int, int, int], float]] = {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return table
    for grad in root.iter():
        tag = grad.tag.rsplit('}', 1)[-1]
        if tag not in ("linearGradient", "radialGradient"):
            continue
        gid = grad.get("id")
        if not gid:
            continue
        cols, alphas = [], []
        for stop in grad:
            if stop.tag.rsplit('}', 1)[-1] != "stop":
                continue
            style = stop.get("style", "")
            props = dict(p.split(':', 1) for p in style.split(';')
                         if ':' in p)
            col = (_parse_css_color(props.get("stop-color", "").strip())
                   or _parse_css_color(stop.get("stop-color", "") or ""))
            if col is None and props.get("stop-color", "").strip() == "#000":
                col = (0, 0, 0)
            try:
                alpha = float(props.get("stop-opacity",
                                        stop.get("stop-opacity", 1.0)))
            except (TypeError, ValueError):
                alpha = 1.0
            if col is not None:
                cols.append(col)
            alphas.append(alpha)
        if cols:
            mean = tuple(int(sum(c[i] for c in cols) / len(cols))
                         for i in range(3))
            mean_a = sum(alphas) / len(alphas) if alphas else 1.0
            table[gid] = (mean, mean_a)
    return table


def _shape_color(el, gradients) -> tuple[tuple[int, int, int], float] | None:
    """(RGB, opacité 0-1) effectifs d'une forme, ou None si non remplie."""
    raw = (el.values.get("fill") or "").strip()
    m = re.match(r"url\(#([^)]+)\)", raw)
    if m:
        entry = gradients.get(m.group(1))
        if entry is None:
            return (0, 0, 0), 1.0   # dégradé inconnu : noir opaque
        return entry
    fill = getattr(el, "fill", None)
    if fill is None or fill.value is None:
        return None
    try:
        rgb   = (int(fill.red), int(fill.green), int(fill.blue))
        alpha = (fill.alpha if fill.alpha is not None else 255) / 255.0
    except (TypeError, ValueError):
        return (255, 255, 255), 1.0
    return rgb, alpha


def _parse(path: str, flatten_res: int = 1024):
    """SVG → liste de (polygones, couleur RGB) + rect du viewport."""
    svg = SVG.parse(path, reify=True)
    gradients = _gradient_table(path)

    shapes: list[tuple[list[list[tuple[float, float]]],
                       tuple[int, int, int]]] = []
    for el in svg.elements():
        if not isinstance(el, Shape):
            continue
        col = _shape_color(el, gradients)
        if col is None or col[1] * 255 < _ALPHA_MIN:
            continue   # non remplie ou quasi transparente (vignettes…)
        p = SvgPath(el)
        p.reify()
        try:
            seg_len_ref = p.length(error=1e-3)
        except Exception:
            seg_len_ref = 0.0
        polys: list[list[tuple[float, float]]] = []
        cur: list[tuple[float, float]] = []
        for seg in p:
            if isinstance(seg, Move):
                if len(cur) >= 3:
                    polys.append(cur)
                cur = [(seg.end.x, seg.end.y)] if seg.end is not None else []
            elif isinstance(seg, Close):
                if len(cur) >= 3:
                    polys.append(cur)
                cur = []
            elif isinstance(seg, Line):
                if seg.end is not None:
                    cur.append((seg.end.x, seg.end.y))
            else:
                # Courbe (Bézier, arc) : échantillonnée selon sa longueur
                try:
                    slen = seg.length(error=1e-3)
                except Exception:
                    slen = 0.0
                k = max(8, int(flatten_res * slen / max(seg_len_ref, 1e-9)))
                for j in range(1, k + 1):
                    pt = seg.point(j / k)
                    cur.append((pt.x, pt.y))
        if len(cur) >= 3:
            polys.append(cur)
        if polys:
            shapes.append((polys, col[0]))

    if not shapes:
        raise ValueError("aucune forme remplie dans le SVG")

    # Rect du viewport : svgelements reifie la géométrie dans l'espace
    # width/height quand ils existent (le viewBox est déjà appliqué).
    if svg.width and svg.height:
        view = (0.0, 0.0, float(svg.width), float(svg.height))
    elif svg.viewbox is not None:
        vb = svg.viewbox
        view = (float(vb.x), float(vb.y), float(vb.width), float(vb.height))
    else:
        xs = [p[0] for polys, _ in shapes for poly in polys for p in poly]
        ys = [p[1] for polys, _ in shapes for poly in polys for p in poly]
        view = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return shapes, view


def _pack(shapes, view):
    edges_list: list[tuple[float, float, float, float]] = []
    offsets = [0]
    bboxes  = []
    colors  = []
    for polys, col in shapes:
        xs: list[float] = []
        ys: list[float] = []
        for poly in polys:
            m = len(poly)
            for i in range(m):
                x0, y0 = poly[i]
                x1, y1 = poly[(i + 1) % m]   # fermeture implicite
                edges_list.append((x0, y0, x1, y1))
            xs.extend(p[0] for p in poly)
            ys.extend(p[1] for p in poly)
        offsets.append(len(edges_list))
        bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        colors.append(col)
    return (np.array(edges_list, dtype=np.float64),
            np.array(offsets, dtype=np.int64),
            np.array(bboxes, dtype=np.float64),
            np.array(colors, dtype=np.uint8),
            np.array(view, dtype=np.float64))


def rasterize_shapes(shapes, view, side: int = 1024) -> np.ndarray:
    """Raster RGBA depuis la géométrie vectorielle (vignette + repli des
    formules libres). Limite : les trous de sous-chemins sont remplis."""
    from PIL import Image, ImageDraw
    x0, y0, w, h = view
    scale = side / max(w, h, 1e-9)
    W, H = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    drw  = ImageDraw.Draw(img)
    for polys, col in shapes:
        for poly in polys:
            pts = [((px - x0) * scale, (py - y0) * scale) for px, py in poly]
            if len(pts) >= 3:
                drw.polygon(pts, fill=(col[0], col[1], col[2], 255))
    return np.array(img, dtype=np.uint8)


def load_svg(path: str, flatten_res: int = 1024, raster_side: int = 1024
             ) -> tuple[tuple, np.ndarray]:
    """Parse unique → (pack vectoriel pour les kernels, raster RGBA)."""
    shapes, view = _parse(path, flatten_res)
    return _pack(shapes, view), rasterize_shapes(shapes, view, raster_side)


def load_svg_shapes(path: str, flatten_res: int = 1024):
    """Pack vectoriel seul : (edges, offsets, bboxes, colors, view)."""
    shapes, view = _parse(path, flatten_res)
    return _pack(shapes, view)


@njit(cache=True)
def _winding(edges, s0, s1, x, y):
    """Nombre d'enroulement (règle non-zero) du contour [s0, s1) autour de (x, y)."""
    wn = 0
    for k in range(s0, s1):
        x0 = edges[k, 0]; y0 = edges[k, 1]
        x1 = edges[k, 2]; y1 = edges[k, 3]
        cross = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)
        if y0 <= y:
            if y1 > y and cross > 0.0:
                wn += 1
        elif y1 <= y and cross < 0.0:
            wn -= 1
    return wn


@njit(cache=True)
def _svg_shape_at(edges, offs, bboxes, x, y):
    """Index du shape le plus haut (dernier dessiné) contenant (x, y), ou -1."""
    for s in range(offs.shape[0] - 2, -1, -1):
        if (x < bboxes[s, 0] or x > bboxes[s, 2]
                or y < bboxes[s, 1] or y > bboxes[s, 3]):
            continue
        if _winding(edges, offs[s], offs[s + 1], x, y) != 0:
            return s
    return -1


@njit(parallel=True, cache=True)
def trap_svg_julia(Z, a, b, c, edges, offs, bboxes, colors, view, rect,
                   n=100, B=256.0, min_iter=2, angle=0.0, mandel=False):
    """Vector counterpart of trap_image_julia: exact point-in-SVG tests.

    rect, angle: placement of the SVG document rect in the complex plane
    (same convention as the raster kernel).  mandel=True iterates the
    Mandelbrot orbit (z0=0, c=grid).  Returns uint8 (H, W, 4) RGBA.
    """
    H, W = Z.shape
    re_min = rect[0]; re_max = rect[1]; im_min = rect[2]; im_max = rect[3]
    re_c = (re_min + re_max) * 0.5
    im_c = (im_min + im_max) * 0.5
    re_w = re_max - re_min
    im_h = im_max - im_min
    half_w = re_w * 0.5
    half_h = im_h * 0.5
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    out = np.zeros((H, W, 4), dtype=np.uint8)
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            for it in range(n):
                z = a * z * z + b * z + c
                zr = z.real
                zi = z.imag
                if it >= min_iter:
                    dre = zr - re_c
                    dim = zi - im_c
                    local_re = dre * cos_a + dim * sin_a
                    local_im = -dre * sin_a + dim * cos_a
                    if (-half_w <= local_re <= half_w
                            and -half_h <= local_im <= half_h):
                        u = (local_re + half_w) / re_w
                        v = (half_h - local_im) / im_h   # y SVG vers le bas
                        sx = view[0] + u * view[2]
                        sy = view[1] + v * view[3]
                        s = _svg_shape_at(edges, offs, bboxes, sx, sy)
                        if s >= 0:
                            out[y, x, 0] = colors[s, 0]
                            out[y, x, 1] = colors[s, 1]
                            out[y, x, 2] = colors[s, 2]
                            out[y, x, 3] = 255
                            break
                if zr * zr + zi * zi > B2:
                    break
    return out


@njit(parallel=True, cache=True)
def trap_svg_geom_julia(Z, a, b, c, edges, offs, bboxes, colors, view,
                        N, r, cx, cy, base_size, angle_step,
                        n=100, B=256.0, mandel=False):
    """Vector counterpart of trap_image_geom_series_julia (spiral copies).
    mandel=True iterates the Mandelbrot orbit (z0=0, c=grid)."""
    H, W = Z.shape
    aspect = view[3] / view[2]
    scales = np.empty(N, dtype=np.float64)
    cx_ks  = np.empty(N, dtype=np.float64)
    cy_ks  = np.empty(N, dtype=np.float64)
    sc = 1.0
    for k in range(N):
        scales[k] = sc
        ang = float(k + 1) * angle_step
        cx_ks[k] = cx + base_size * sc * math.sin(ang)
        cy_ks[k] = cy + base_size * sc * (1.0 - math.cos(ang))
        sc *= r
    out = np.zeros((H, W, 4), dtype=np.uint8)
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            if mandel:
                c = Z[y, x]
                z = 0j
            else:
                z = Z[y, x]
            hit = False
            for it in range(n):
                z = a * z * z + b * z + c
                zr = z.real
                zi = z.imag
                for k in range(N):
                    hw = base_size * scales[k] * 0.5
                    hh = hw * aspect
                    dre = zr - cx_ks[k]
                    dim = zi - cy_ks[k]
                    if -hw <= dre <= hw and -hh <= dim <= hh:
                        u = (dre + hw) / (2.0 * hw)
                        v = (hh - dim) / (2.0 * hh)
                        sx = view[0] + u * view[2]
                        sy = view[1] + v * view[3]
                        s = _svg_shape_at(edges, offs, bboxes, sx, sy)
                        if s >= 0:
                            out[y, x, 0] = colors[s, 0]
                            out[y, x, 1] = colors[s, 1]
                            out[y, x, 2] = colors[s, 2]
                            out[y, x, 3] = 255
                            hit = True
                            break
                if hit:
                    break
                if zr * zr + zi * zi > B2:
                    break
    return out
