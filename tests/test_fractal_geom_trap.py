import numpy as np
import pytest
from fractal import FractalGenerator
from iteration import Poly


def _red_tex(size=4):
    img = np.zeros((size, size, 4), dtype=np.uint8)
    img[:, :] = [255, 0, 0, 255]
    return img


def test_generate_julia_geom_trap_output_shape():
    gen = FractalGenerator(height=16, width=20, n_iter=30)
    poly = Poly(1, 0, -0.75 + 0j)
    tex = _red_tex()
    out = gen.generate_julia_geom_trap(poly, tex, N=2, r=0.5, base_size=4.0)
    assert out.shape == (16, 20, 4)
    assert out.dtype == np.uint8


def test_generate_julia_geom_trap_has_hits():
    gen = FractalGenerator(height=16, width=20, n_iter=100)
    poly = Poly(1, 0, -0.4 + 0.6j)
    tex = _red_tex(size=8)
    out = gen.generate_julia_geom_trap(poly, tex, N=1, r=0.5, base_size=8.0)
    assert out[:, :, 3].any(), "Expected at least some trapped pixels"
