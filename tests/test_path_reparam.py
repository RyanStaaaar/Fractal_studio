import numpy as np


def test_dem_is_zero_inside_mandelbrot():
    """c = 0 et c = -1 sont intérieurs (cardioïde, bulbe) : DEM = 0."""
    from path_reparam import mandelbrot_distance
    assert mandelbrot_distance(0.0, 0.0) == 0.0
    assert mandelbrot_distance(-1.0, 0.0) == 0.0


def test_dem_is_small_just_outside_and_larger_far_away():
    """c = 0.26 frôle la cardioïde ; c = 1.0 en est loin."""
    from path_reparam import mandelbrot_distance
    near = mandelbrot_distance(0.26, 0.0)
    far = mandelbrot_distance(1.0, 0.0)
    assert 0.0 < near < 0.01
    assert far > near
    assert far > 0.1


def test_reparametrize_path_is_monotonic_and_sized():
    """Le reparamétrage renvoie n_frames temps croissants dans [0, 1]."""
    from path_reparam import reparametrize_path
    fn = lambda t: -0.4 + 0.45 * np.exp(2j * np.pi * t)   # traverse la frontière
    ts, cs = reparametrize_path(fn, n_samples=800, n_frames=60, rho_cap=25.0)
    assert len(ts) == 60 and len(cs) == 60
    assert np.all(np.diff(ts) >= -1e-12)
    assert ts.min() >= 0.0 and ts.max() <= 1.0


def test_rho_cap_prevents_frame_starvation():
    """Sans plafond, un chemin traversant M monopolise les frames près de ∂M.

    Le plafond garde un espacement régulier : aucun grand saut entre frames.
    """
    from path_reparam import reparametrize_path
    fn = lambda t: -0.4 + 0.45 * np.exp(2j * np.pi * t)
    _, capped = reparametrize_path(fn, n_samples=800, n_frames=120, rho_cap=25.0)
    _, uncapped = reparametrize_path(fn, n_samples=800, n_frames=120)
    gap_capped = np.abs(np.diff(capped)).max()
    gap_uncapped = np.abs(np.diff(uncapped)).max()
    assert gap_capped < gap_uncapped


def test_warp_is_normalised_and_monotonic():
    from path_reparam import reparam_warp
    fn = lambda t: -0.4 + 0.45 * np.exp(2j * np.pi * t)
    S, t_grid = reparam_warp(fn, n_samples=500, rho_cap=25.0)
    assert np.all(np.diff(S) >= -1e-12)
    assert np.isclose(S[0], 0.0) and np.isclose(S[-1], 1.0)
    assert len(S) == len(t_grid)
