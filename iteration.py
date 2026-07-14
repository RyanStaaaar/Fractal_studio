import math
import numpy as np
from numba import njit, prange
class Poly :
    def __init__(self, a, b, c) :
        self.a= a
        self.b=b
        self.c=c
    def evaluate(self,z) :
        return self.a*z*z + self.b*z + self.c


# smooth=True : coloration lissée (escape time logarithmique, dégradés continus)
# smooth=False : version classique (compte d'itérations avant échappement)
@njit(parallel=True, cache=True)
def escape_speed(Z, a, b, c, n=100, B=256, smooth=True):
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            z = Z[y, x]
            for i in range(n):
                z = a*z*z + b*z + c
                if z.real*z.real + z.imag*z.imag > B2:
                    if smooth:
                        log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                        smooth_i = i + 1 - np.log2(log_zn / np.log(B))
                        V[y, x] = max(0.0, min(1.0, 1.0 - smooth_i / n))
                    else:
                        V[y, x] = (n - i) / n
                    break
    return V

@njit(parallel=True, cache=True)
def mandelbrot(C, seed, n=100, B=2.0, smooth=True):
    H, W = C.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            c = C[y, x]
            z = seed
            for i in range(n):
                z = z*z + c
                if z.real*z.real + z.imag*z.imag > B2:
                    if smooth:
                        log_zn = np.log(z.real*z.real + z.imag*z.imag) / 2
                        smooth_i = i + 1 - np.log2(log_zn / np.log(B))
                        V[y, x] = max(0.0, min(1.0, 1.0 - smooth_i / n))
                    else:
                        V[y, x] = (n - i) / n
                    break
    return V


@njit(parallel=True, cache=True)
def julia_period(Z, a, b, c, n=200, B=256.0, eps=1e-6, max_period=64):
    """Period coloring for Julia sets.
    Exterior: smooth escape value. Interior: period / (max_period+1)."""
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    eps2 = eps * eps
    for y in prange(H):
        for x in range(W):
            z = Z[y, x]
            escaped = False
            smooth_val = 0.0
            for i in range(n):
                z = a * z * z + b * z + c
                r2 = z.real * z.real + z.imag * z.imag
                if r2 > B2:
                    escaped = True
                    log_zn = np.log(r2) / 2.0
                    nu = i + 1.0 - np.log2(log_zn / np.log(B))
                    smooth_val = max(0.0, min(1.0, 1.0 - nu / n))
                    break
            if escaped:
                V[y, x] = smooth_val
                continue
            # z is now near the attractor — count steps to return close to it
            z_target = z
            z_test = a * z_target * z_target + b * z_target + c
            period = 1
            while period <= max_period:
                dr = z_test.real - z_target.real
                di = z_test.imag - z_target.imag
                if dr * dr + di * di < eps2:
                    break
                z_test = a * z_test * z_test + b * z_test + c
                period += 1
            V[y, x] = float(period) / float(max_period + 1)
    return V


@njit(parallel=True, cache=True)
def mandelbrot_period(C, n=200, B=256.0, eps=1e-6, max_period=64):
    """Period coloring for Mandelbrot set."""
    H, W = C.shape
    V = np.zeros((H, W))
    B2 = B * B
    eps2 = eps * eps
    for y in prange(H):
        for x in range(W):
            c = C[y, x]
            z = 0j
            escaped = False
            smooth_val = 0.0
            for i in range(n):
                z = z * z + c
                r2 = z.real * z.real + z.imag * z.imag
                if r2 > B2:
                    escaped = True
                    log_zn = np.log(r2) / 2.0
                    nu = i + 1.0 - np.log2(log_zn / np.log(B))
                    smooth_val = max(0.0, min(1.0, 1.0 - nu / n))
                    break
            if escaped:
                V[y, x] = smooth_val
                continue
            z_target = z
            z_test = z_target * z_target + c
            period = 1
            while period <= max_period:
                dr = z_test.real - z_target.real
                di = z_test.imag - z_target.imag
                if dr * dr + di * di < eps2:
                    break
                z_test = z_test * z_test + c
                period += 1
            V[y, x] = float(period) / float(max_period + 1)
    return V


@njit(parallel=True, cache=True)
def julia_attractor(Z, a, b, c, n=100, B=256.0, norm_max=0.5):
    """Interior distance estimation for Julia sets via derivative tracking.
    Tracks |dz_n/dz_0| alongside the orbit.
    Interior value: |z_n| / |dz_n/dz_0|, normalised with exp decay."""
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            z = Z[y, x]
            dz_abs = 1.0
            escaped = False
            smooth_val = 0.0
            for i in range(n):
                zr = z.real
                zi = z.imag
                # |d/dz_0 (a*z^2 + b*z + c)| = |2a*z + b|  (real a, b)
                fr = 2.0 * a * zr + b
                fi = 2.0 * a * zi
                dz_abs *= np.sqrt(fr * fr + fi * fi)
                z = a * z * z + b * z + c
                r2 = z.real * z.real + z.imag * z.imag
                if r2 > B2:
                    escaped = True
                    log_zn = np.log(r2) / 2.0
                    nu = i + 1.0 - np.log2(log_zn / np.log(B))
                    smooth_val = max(0.0, min(1.0, 1.0 - nu / n))
                    break
            if escaped:
                V[y, x] = smooth_val
            else:
                zabs = np.sqrt(z.real * z.real + z.imag * z.imag)
                d = zabs / (dz_abs + 1e-30)
                V[y, x] = 1.0 - np.exp(-d / norm_max)
    return V


@njit(parallel=True, cache=True)
def mandelbrot_attractor(C, n=100, B=256.0, norm_max=0.5):
    """Interior distance estimation for Mandelbrot set via derivative tracking.
    Tracks dz_n/dc as (dz_r, dz_i). Interior value: |z_n| / |dz_n/dc|, normalised."""
    H, W = C.shape
    V = np.zeros((H, W))
    B2 = B * B
    for y in prange(H):
        for x in range(W):
            c = C[y, x]
            z = 0j
            dz_r = 0.0
            dz_i = 0.0
            escaped = False
            smooth_val = 0.0
            for i in range(n):
                zr = z.real
                zi = z.imag
                # dz_{k+1}/dc = 2*z_k * dz_k/dc + 1
                new_dz_r = 2.0 * (zr * dz_r - zi * dz_i) + 1.0
                new_dz_i = 2.0 * (zr * dz_i + zi * dz_r)
                dz_r = new_dz_r
                dz_i = new_dz_i
                z = z * z + c
                r2 = z.real * z.real + z.imag * z.imag
                if r2 > B2:
                    escaped = True
                    log_zn = np.log(r2) / 2.0
                    nu = i + 1.0 - np.log2(log_zn / np.log(B))
                    smooth_val = max(0.0, min(1.0, 1.0 - nu / n))
                    break
            if escaped:
                V[y, x] = smooth_val
            else:
                zabs = np.sqrt(z.real * z.real + z.imag * z.imag)
                dzabs = np.sqrt(dz_r * dz_r + dz_i * dz_i)
                d = zabs / (dzabs + 1e-30)
                V[y, x] = 1.0 - np.exp(-d / norm_max)
    return V


@njit(parallel=True, cache=True)
def julia_lambda(Z, a, b, c, n=100, B=256.0,
                 burn_in=100, max_period=64, eps=1e-6, delta=1e-6, norm_max=500.0):
    """Interior coloring by cycle multiplier λ (true convergence speed).

    Phase 1 — burn_in iterations, escape → smooth exterior.
    Phase 2 — period p: smallest p≤max_period with |f^p(w)−w|<ε.
    Phase 3 — λ = ∏_{k=0}^{p-1} f'(z_k) = ∏ (2az_k+b); if |λ|≥1 → V=0.
    Phase 4 — ν = log(δ/d₀) / log|λ|  (d₀ = |f^p(w)−w|, closed-form smooth count).
              V = 1 − exp(−max(0,ν) / norm_max).
    """
    H, W = Z.shape
    V = np.zeros((H, W))
    B2 = B * B
    eps2 = eps * eps

    for y in prange(H):
        for x in range(W):
            z = Z[y, x]

            # --- Phase 1: burn-in + escape ---
            escaped = False
            smooth_val = 0.0
            for i in range(burn_in):
                z = a * z * z + b * z + c
                r2 = z.real * z.real + z.imag * z.imag
                if r2 > B2:
                    escaped = True
                    log_zn = np.log(r2) / 2.0
                    nu_e = i + 1.0 - np.log2(log_zn / np.log(B))
                    smooth_val = max(0.0, min(1.0, 1.0 - nu_e / n))
                    break
            if escaped:
                V[y, x] = smooth_val
                continue

            w = z  # near the attractor after burn_in steps

            # --- Phase 2: period detection ---
            z_test = a * w * w + b * w + c
            period = 0
            d0 = 0.0
            for p in range(1, max_period + 1):
                dr = z_test.real - w.real
                di = z_test.imag - w.imag
                dist2 = dr * dr + di * di
                if dist2 < eps2:
                    period = p
                    d0 = np.sqrt(dist2)
                    break
                z_test = a * z_test * z_test + b * z_test + c

            if period == 0:
                V[y, x] = 0.0
                continue

            # --- Phase 3: multiplier λ = ∏ f'(z_k) over one cycle ---
            lam_r = 1.0
            lam_i = 0.0
            z_k = w
            for _ in range(period):
                fp_r = 2.0 * a * z_k.real + b
                fp_i = 2.0 * a * z_k.imag
                new_r = lam_r * fp_r - lam_i * fp_i
                new_i = lam_r * fp_i + lam_i * fp_r
                lam_r = new_r
                lam_i = new_i
                z_k = a * z_k * z_k + b * z_k + c

            lam_abs = np.sqrt(lam_r * lam_r + lam_i * lam_i)

            if lam_abs >= 1.0:
                V[y, x] = 0.0
                continue

            # --- Phase 4: smooth convergence index ---
            # d0 = |f^p(w) - w| ≈ d_initial * |λ|^(burn_in/p)  after burn_in steps.
            # Using d0 directly gives ν < 0 for most pixels (already below δ).
            # We back-compute d_initial to get the count from the original pixel z_0.
            exponent = float(burn_in) / float(period)
            lam_pow = lam_abs ** exponent   # |λ|^(burn_in/p)

            # If lam_pow is negligible, the orbit converged too fast to back-compute:
            # d_initial would overflow → classify as fast-converging (V=0, dark interior).
            if lam_pow < 1e-8 or d0 < 1e-30:
                V[y, x] = 0.0
                continue

            d_initial = d0 / lam_pow   # distance of z_0 from the attractor (estimated)

            # Guard against back-computation noise for super-small d0
            if d_initial > 10.0 or d_initial <= delta:
                V[y, x] = 0.0
                continue

            # ν = total p-step count from z_0 to reach distance δ (continuous)
            nu = np.log(delta / d_initial) / np.log(lam_abs)
            if nu < 0.0:
                nu = 0.0

            V[y, x] = 1.0 - np.exp(-nu / norm_max)

    return V


# ── Biomorphes de Pickover ────────────────────────────────────────────────────
# Chemin rapide z²+c (formule par défaut). Les autres fonctions du biomorphe
# passent par le moteur de formules libres de fractal_studio.py (kernel généré /
# numpy), qui applique la même classification OU sur le z final.
@njit(parallel=True, cache=True)
def biomorph(Z, c, n=50, mod_bail=100.0, L=10.0,
             mandel=False, color_by_iter=False):
    """Rendu biomorphe de Pickover pour z² + c.

    Itère z <- z*z + c jusqu'à |z| > mod_bail ou n.  Julia (défaut) :
    z0 = Z[pixel], c constante. Mandelbrot : c = Z[pixel], z0 = 0.
    Classification appliquée au z FINAL (le OU composante crée les
    cils/appendices) : membre si  |re| < L  OU  |im| < L.

    Renvoie (V, mask) :
      mask : uint8, 1 = membre biomorphe (test OU).
      V    : champ de coloration continu pour la palette OKLab existante —
             color_by_iter=False → log(1 + min(|re|, |im|)) (la structure),
             color_by_iter=True  → escape time lissé (vitesse de fuite).
    """
    H, W = Z.shape
    V    = np.zeros((H, W))
    mask = np.zeros((H, W), dtype=np.uint8)
    mb2  = mod_bail * mod_bail
    logB = math.log(mod_bail)
    for y in prange(H):
        for x in range(W):
            if mandel:
                cc = Z[y, x]
                z  = 0j
            else:
                cc = c
                z  = Z[y, x]
            escaped = False
            it_esc  = n
            for it in range(n):
                z = z * z + cc
                zr = z.real
                zi = z.imag
                if zr != zr or zi != zi:       # NaN : orbite morte
                    break
                if zr * zr + zi * zi > mb2:
                    escaped = True
                    it_esc  = it
                    break
            are = abs(z.real)
            aim = abs(z.imag)
            # Classification biomorphe : OU composante (NE PAS remplacer par |z|)
            if are < L or aim < L:
                mask[y, x] = 1
            q = are if are < aim else aim       # min(|re|, |im|)
            if color_by_iter:
                if escaped:
                    m2 = z.real * z.real + z.imag * z.imag
                    if m2 > 1.0:
                        lz = 0.5 * math.log(m2)
                        if lz / logB > 1e-12:
                            si = it_esc + 1 - math.log2(lz / logB)
                        else:
                            si = it_esc + 1.0
                    else:
                        si = it_esc + 1.0
                    V[y, x] = max(0.0, min(1.0, 1.0 - si / n))
                else:
                    V[y, x] = 0.0
            else:
                V[y, x] = math.log(1.0 + q)
    return V, mask
