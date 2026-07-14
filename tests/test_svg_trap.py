import numpy as np
import pytest

# Un SVG minimal : un cercle orange à gauche, un rectangle bleu à droite.
_SVG_TWO_SHAPES = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="60">
  <circle cx="30" cy="30" r="25" fill="#ff5500"/>
  <rect x="60" y="10" width="35" height="40" fill="#0088ff"/>
</svg>"""

# Deux cercles qui se recouvrent : le second (vert) est peint par-dessus.
_SVG_OVERLAP = """<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
  <circle cx="50" cy="50" r="40" fill="#ff0000"/>
  <circle cx="60" cy="50" r="20" fill="#00ff00"/>
</svg>"""


@pytest.fixture
def two_shapes(tmp_path):
    p = tmp_path / "two.svg"
    p.write_text(_SVG_TWO_SHAPES)
    import svg_trap
    return svg_trap.load_svg_shapes(str(p))


def test_point_in_shape_inside_outside(two_shapes):
    """Test point-dans-forme exact (nombre d'enroulement), pas d'échantillonnage."""
    import svg_trap
    edges, offs, bboxes, colors, view = two_shapes
    # Au centre du cercle → sa couleur de remplissage exacte
    s = svg_trap._svg_shape_at(edges, offs, bboxes, 30.0, 30.0)
    assert s >= 0 and tuple(colors[s]) == (255, 85, 0)
    # Au centre du rectangle
    s = svg_trap._svg_shape_at(edges, offs, bboxes, 75.0, 30.0)
    assert s >= 0 and tuple(colors[s]) == (0, 136, 255)
    # Dans un coin vide → aucune forme
    assert svg_trap._svg_shape_at(edges, offs, bboxes, 2.0, 58.0) == -1


def test_viewport_matches_document(two_shapes):
    _, _, _, _, view = two_shapes
    assert tuple(view) == (0.0, 0.0, 100.0, 60.0)


def test_overlap_last_painted_wins(tmp_path):
    """Ordre de peinture SVG : la dernière forme dessinée gagne."""
    import svg_trap
    p = tmp_path / "overlap.svg"
    p.write_text(_SVG_OVERLAP)
    edges, offs, bboxes, colors, _ = svg_trap.load_svg_shapes(str(p))
    s = svg_trap._svg_shape_at(edges, offs, bboxes, 60.0, 50.0)
    assert tuple(colors[s]) == (0, 255, 0)


def test_winding_number_zero_outside(two_shapes):
    import svg_trap
    edges, offs, _, _, _ = two_shapes
    # Loin de tout : enroulement nul pour la première forme
    assert svg_trap._winding(edges, offs[0], offs[1], 500.0, 500.0) == 0


def test_trap_svg_julia_paints_fill_colors(two_shapes):
    """Le rendu ne pose que des couleurs de fill du SVG (aucune interpolation)."""
    import svg_trap
    edges, offs, bboxes, colors, view = two_shapes
    xs = np.linspace(-2, 2, 60)
    Z = (xs[np.newaxis, :] + 1j * xs[:, np.newaxis]).astype(np.complex128)
    rect = np.array([-1.0, 1.0, -0.6, 0.6])
    rgba = svg_trap.trap_svg_julia(Z, 1 + 0j, 0 + 0j, complex(-0.7, 0.27),
                                   edges, offs, bboxes, colors, view, rect,
                                   60, 256.0, 2, 0.0)
    assert rgba.shape == (60, 60, 4)
    hit = rgba[..., 3] > 0
    if hit.any():
        painted = {tuple(px) for px in rgba[hit][:, :3]}
        allowed = {tuple(c) for c in colors}
        assert painted <= allowed
