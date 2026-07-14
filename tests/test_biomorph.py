import math

import numpy as np


def _grid(b=2.0, n=48):
    xs = np.linspace(-b, b, n)
    return (xs[np.newaxis, :] + 1j * xs[:, np.newaxis]).astype(np.complex128)


def test_or_classification_is_component_wise():
    """Le masque biomorphe = |re|<L OU |im|<L, ce qui équivaut à min(|re|,|im|)<L.

    C'est ce OU (et non un test de module) qui crée les cils : un point loin de
    l'origine mais aligné sur un axe reste membre.
    """
    from iteration import biomorph
    Z = _grid()
    L = 10.0
    V, mask = biomorph(Z, 0.5 + 0j, 50, 100.0, L, False, False)
    # V = log(1 + min(|re|,|im|)) → min = exp(V) - 1, et membre ⇔ min < L
    min_comp = np.exp(V) - 1.0
    assert np.array_equal(mask.astype(bool), min_comp < L)
    # Le masque est non trivial (ni vide ni plein) sur cette vue
    assert 0.0 < mask.mean() < 1.0


def test_L_controls_membership_extent():
    """L plus grand ⇒ au moins autant de membres (seuil du OU plus permissif)."""
    from iteration import biomorph
    Z = _grid()
    _, small = biomorph(Z, 0.5 + 0j, 50, 100.0, 2.0, False, False)
    _, big = biomorph(Z, 0.5 + 0j, 50, 100.0, 20.0, False, False)
    assert big.sum() >= small.sum()


def test_fields_are_finite_and_color_by_iter_normalised():
    from iteration import biomorph
    Z = _grid()
    V, _ = biomorph(Z, 0.5 + 0j, 50, 100.0, 10.0, False, False)
    assert np.isfinite(V).all()
    Vi, _ = biomorph(Z, 0.5 + 0j, 50, 100.0, 10.0, False, True)
    assert np.isfinite(Vi).all()
    assert Vi.min() >= 0.0 and Vi.max() <= 1.0


def test_julia_and_mandelbrot_modes_differ():
    from iteration import biomorph
    Z = _grid()
    Vj, _ = biomorph(Z, 0.5 + 0j, 50, 100.0, 10.0, False, False)
    Vm, _ = biomorph(Z, 0.5 + 0j, 50, 100.0, 10.0, True, False)
    assert not np.array_equal(Vj, Vm)


def test_generated_numba_kernel_matches_reference_for_z2():
    """Le kernel généré pour la formule « z^2+c » doit reproduire iteration.biomorph."""
    import fractal_studio as fs
    from iteration import biomorph
    Z = _grid()
    kernels = fs._compile_julia_numba("z^2 + c")
    assert len(kernels) == 5          # esc, trap, svg, svg_geom, bio
    with fs._RENDER_LOCK:
        V_gen = kernels[4](np.ascontiguousarray(Z), 0.5 + 0j,
                           50, 100.0, 10.0, False, False)
    V_ref, _ = biomorph(Z, 0.5 + 0j, 50, 100.0, 10.0, False, False)
    assert np.allclose(V_gen, V_ref, atol=1e-9)


def test_numpy_fallback_matches_numba_kernel():
    """Repli NumPy ≡ kernel Numba généré, pour une formule transcendante."""
    import fractal_studio as fs
    Z = _grid()
    fn = fs._compile_julia_iter("sin(z) + c")
    kernels = fs._compile_julia_numba("sin(z) + c")
    with fs._RENDER_LOCK:
        V_nb = kernels[4](np.ascontiguousarray(Z), 0.4 + 0j,
                          40, 100.0, 12.0, False, False)
    V_np = fs._biomorph_formula_numpy(Z, fn, 0.4 + 0j, 40, 100.0, 12.0,
                                      False, False)
    assert np.allclose(V_nb, V_np, atol=1e-6)
