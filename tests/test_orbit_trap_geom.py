import numpy as np
import pytest


def _make_tex(r=255, g=0, b=0, alpha=255, size=4):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[:, :] = [r, g, b, alpha]
    return img


def test_output_shape_and_dtype():
    from orbit_trap import trap_image_geom_series_julia
    H, W = 6, 8
    Z = np.zeros((H, W), dtype=complex)
    img = _make_tex()
    out = trap_image_geom_series_julia(
        Z, 1+0j, 0+0j, -0.75+0j, img,
        N=3, r=0.5, cx=0.0, cy=0.0,
        base_size=1.0, angle_step=0.0, n=20, B=256.0
    )
    assert out.shape == (H, W, 4)
    assert out.dtype == np.uint8


def test_far_pixels_are_untrapped():
    """Pixels starting far from origin escape instantly; tiny trap near origin → alpha=0."""
    from orbit_trap import trap_image_geom_series_julia
    H, W = 4, 4
    Z = np.full((H, W), 100.0 + 100.0j, dtype=complex)
    img = _make_tex(alpha=255)
    out = trap_image_geom_series_julia(
        Z, 1+0j, 0+0j, -0.75+0j, img,
        N=2, r=0.5, cx=0.0, cy=0.0,
        base_size=0.01, angle_step=0.0, n=5, B=256.0
    )
    assert np.all(out[:, :, 3] == 0), "Far pixels should all be untrapped (alpha=0)"


def test_transparent_tex_pixels_not_trapped():
    """Fully transparent image pixels (alpha=0) must not trigger a trap hit."""
    from orbit_trap import trap_image_geom_series_julia
    H, W = 4, 4
    Z = np.zeros((H, W), dtype=complex)
    img = _make_tex(alpha=0)  # fully transparent — no valid hit
    out = trap_image_geom_series_julia(
        Z, 1+0j, 0+0j, -0.75+0j, img,
        N=1, r=0.5, cx=0.0, cy=0.0,
        base_size=8.0, angle_step=0.0, n=50, B=256.0
    )
    assert np.all(out[:, :, 3] == 0), "Transparent tex pixels must not produce hits"


def test_rotation_displaces_copies():
    """angle_step > 0 must produce a different result than angle_step=0."""
    from orbit_trap import trap_image_geom_series_julia
    H, W = 16, 16
    xs = np.linspace(-1.5, 1.5, W)
    ys = np.linspace(-1.5, 1.5, H)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    img = _make_tex(r=200, g=100, b=50, alpha=255, size=8)
    common = dict(a=1+0j, b=0+0j, c=-0.4+0.6j, tex=img,
                  N=4, r=0.6, cx=0.0, cy=0.0, base_size=2.0, n=50, B=256.0)
    out0 = trap_image_geom_series_julia(Z, **common, angle_step=0.0)
    out1 = trap_image_geom_series_julia(Z, **common, angle_step=1.0)
    assert not np.array_equal(out0, out1), "angle_step=1.0 must produce a different output than angle_step=0"


def test_large_trap_produces_hits():
    """With a very large base_size, orbit points land inside and should be trapped."""
    from orbit_trap import trap_image_geom_series_julia
    H, W = 8, 8
    xs = np.linspace(-0.3, 0.3, W)
    ys = np.linspace(-0.3, 0.3, H)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    img = _make_tex(r=255, g=0, b=0, alpha=255)
    out = trap_image_geom_series_julia(
        Z, 1+0j, 0+0j, -0.4+0.6j, img,
        N=1, r=0.5, cx=0.0, cy=0.0,
        base_size=8.0, angle_step=0.0, n=100, B=256.0
    )
    hit_mask = out[:, :, 3] == 255
    assert hit_mask.any(), "Large trap should catch at least some orbit points"
    assert np.all(out[hit_mask, 0] == 255), "Trapped pixels should have R=255 (red image)"
    assert np.all(out[hit_mask, 1] == 0),   "Trapped pixels should have G=0 (red image)"
