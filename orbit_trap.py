import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def _dist_point(zr, zi, cx, cy):
    dx = zr - cx
    dy = zi - cy
    return math.sqrt(dx * dx + dy * dy)


@njit(cache=True)
def _dist_line(zr, zi, angle):
    return abs(zr * math.sin(angle) - zi * math.cos(angle))


@njit(cache=True)
def _dist_cross(zr, zi):
    ar = abs(zr)
    ai = abs(zi)
    return ar if ar < ai else ai


@njit(cache=True)
def _dist_circle(zr, zi, cx, cy, r):
    dx = zr - cx
    dy = zi - cy
    return abs(math.sqrt(dx * dx + dy * dy) - r)


@njit(cache=True)
def _dist_square(zr, zi, cx, cy, r):
    dx = abs(zr - cx)
    dy = abs(zi - cy)
    cheby = dx if dx > dy else dy   # norme Chebyshev = demi-côté effectif
    return abs(cheby - r)


@njit(cache=True)
def _dist_sinus(zr, zi, cx, cy, amp, freq):
    # distance verticale au point de la courbe  y = cy + amp·sin(freq·(x − cx))
    return abs(zi - cy - amp * math.sin(freq * (zr - cx)))


@njit(parallel=True, cache=True)
def trap_mandelbrot(C, trap_type, trap_params, n=100, B=256.0, norm_max=1.0):
    """Mandelbrot orbit trap: V[y,x] = min distance to trap shape over the orbit, normalised."""
    H, W = C.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            c = C[y, x]
            z = 0j
            min_d = 1e18
            for _ in range(n):
                z = z * z + c
                zr = z.real
                zi = z.imag
                if trap_type == 0:
                    d = _dist_point(zr, zi, trap_params[0], trap_params[1])
                elif trap_type == 1:
                    d = _dist_line(zr, zi, trap_params[2])
                elif trap_type == 2:
                    d = _dist_cross(zr, zi)
                elif trap_type == 3:
                    d = _dist_circle(zr, zi, trap_params[0], trap_params[1], trap_params[2])
                elif trap_type == 4:
                    d = _dist_square(zr, zi, trap_params[0], trap_params[1], trap_params[2])
                else:
                    d = _dist_sinus(zr, zi, trap_params[0], trap_params[1],
                                    trap_params[2], trap_params[3])
                if d < min_d:
                    min_d = d
                if zr * zr + zi * zi > B2:
                    break
            V[y, x] = 1.0 - math.exp(-min_d / norm_max)
    return V


@njit(parallel=True, cache=True)
def trap_image_julia(Z, a, b, c, tex, rect, n=100, B=256.0, min_iter=2, angle=0.0):
    """Image orbit trap for Julia sets.

    tex   : uint8 (TH, TW, 4) RGBA — transparent pixels are not traps.
    rect  : float64[4] = [re_min, re_max, im_min, im_max] axis-aligned bounding rect.
    angle : rotation of the rect CCW in the complex plane (radians).
    Returns uint8 (H, W, 4) RGBA; untrapped pixels have alpha = 0.
    """
    H, W = Z.shape
    TH, TW = tex.shape[0], tex.shape[1]
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
            z = Z[y, x]
            for it in range(n):
                z = a * z * z + b * z + c
                zr = z.real
                zi = z.imag
                if it >= min_iter:
                    dre = zr - re_c
                    dim = zi - im_c
                    # rotate by -angle into local rect frame
                    local_re = dre * cos_a + dim * sin_a
                    local_im = -dre * sin_a + dim * cos_a
                    if -half_w <= local_re <= half_w and -half_h <= local_im <= half_h:
                        fj = (local_re + half_w) / re_w * (TW - 1)
                        fi = (half_h - local_im) / im_h * (TH - 1)
                        tj = int(math.floor(fj + 0.5))
                        ti = int(math.floor(fi + 0.5))
                        if 0 <= ti < TH and 0 <= tj < TW:
                            if tex[ti, tj, 3] != 0:
                                out[y, x, 0] = tex[ti, tj, 0]
                                out[y, x, 1] = tex[ti, tj, 1]
                                out[y, x, 2] = tex[ti, tj, 2]
                                out[y, x, 3] = 255
                                break
                if zr * zr + zi * zi > B2:
                    break
    return out


@njit(parallel=True, cache=True)
def trap_image_mandelbrot(C, tex, rect, n=100, B=256.0, min_iter=2, angle=0.0):
    """Image orbit trap for Mandelbrot set. Same output format as trap_image_julia."""
    H, W = C.shape
    TH, TW = tex.shape[0], tex.shape[1]
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
            c = C[y, x]
            z = 0j
            for it in range(n):
                z = z * z + c
                zr = z.real
                zi = z.imag
                if it >= min_iter:
                    dre = zr - re_c
                    dim = zi - im_c
                    local_re = dre * cos_a + dim * sin_a
                    local_im = -dre * sin_a + dim * cos_a
                    if -half_w <= local_re <= half_w and -half_h <= local_im <= half_h:
                        fj = (local_re + half_w) / re_w * (TW - 1)
                        fi = (half_h - local_im) / im_h * (TH - 1)
                        tj = int(math.floor(fj + 0.5))
                        ti = int(math.floor(fi + 0.5))
                        if 0 <= ti < TH and 0 <= tj < TW:
                            if tex[ti, tj, 3] != 0:
                                out[y, x, 0] = tex[ti, tj, 0]
                                out[y, x, 1] = tex[ti, tj, 1]
                                out[y, x, 2] = tex[ti, tj, 2]
                                out[y, x, 3] = 255
                                break
                if zr * zr + zi * zi > B2:
                    break
    return out


@njit(parallel=True, cache=True)
def trap_julia(Z, a, b, c, trap_type, trap_params, n=100, B=256.0, norm_max=1.0):
    """Julia orbit trap: V[y,x] = min distance to trap shape over the orbit, normalised."""
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            z = Z[y, x]
            min_d = 1e18
            for _ in range(n):
                z = a * z * z + b * z + c
                zr = z.real
                zi = z.imag
                if trap_type == 0:
                    d = _dist_point(zr, zi, trap_params[0], trap_params[1])
                elif trap_type == 1:
                    d = _dist_line(zr, zi, trap_params[2])
                elif trap_type == 2:
                    d = _dist_cross(zr, zi)
                elif trap_type == 3:
                    d = _dist_circle(zr, zi, trap_params[0], trap_params[1], trap_params[2])
                elif trap_type == 4:
                    d = _dist_square(zr, zi, trap_params[0], trap_params[1], trap_params[2])
                else:
                    d = _dist_sinus(zr, zi, trap_params[0], trap_params[1],
                                    trap_params[2], trap_params[3])
                if d < min_d:
                    min_d = d
                if zr * zr + zi * zi > B2:
                    break
            V[y, x] = 1.0 - math.exp(-min_d / norm_max)
    return V
