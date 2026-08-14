"""Purge du dossier Wallpapers/ : on ne garde que les N fonds les plus récents."""
import os
from pathlib import Path

from daily_wallpaper import WallpaperApp


def _fake_wallpaper(directory: Path, name: str, mtime: float) -> Path:
    """Crée un faux PNG avec une date de modification imposée."""
    path = directory / name
    path.write_bytes(b"fake png")
    os.utime(path, (mtime, mtime))
    return path


def test_garde_les_n_plus_recents(tmp_path):
    app = WallpaperApp(tmp_path)
    app.KEEP_LAST = 3
    for i in range(6):
        _fake_wallpaper(tmp_path, f"wallpaper_jour{i}.png", 1_000 + i)

    removed = app._prune()

    assert {p.name for p in removed} == {
        "wallpaper_jour0.png", "wallpaper_jour1.png", "wallpaper_jour2.png"}
    assert {p.name for p in tmp_path.glob("wallpaper_*.png")} == {
        "wallpaper_jour3.png", "wallpaper_jour4.png", "wallpaper_jour5.png"}


def test_trie_par_date_pas_par_nom(tmp_path):
    """Les anciens noms aléatoires (wallpaper_7) ne doivent pas piéger le tri."""
    app = WallpaperApp(tmp_path)
    app.KEEP_LAST = 1
    _fake_wallpaper(tmp_path, "wallpaper_7.png", 1_000)             # vieux, nom "grand"
    _fake_wallpaper(tmp_path, "wallpaper_2026-08-14.png", 2_000)    # récent, nom "petit"

    app._prune()

    assert [p.name for p in tmp_path.glob("wallpaper_*.png")] == ["wallpaper_2026-08-14.png"]


def test_ne_supprime_jamais_le_plus_recent(tmp_path):
    app = WallpaperApp(tmp_path)
    app.KEEP_LAST = 1
    _fake_wallpaper(tmp_path, "wallpaper_vieux.png", 1_000)
    actif = _fake_wallpaper(tmp_path, "wallpaper_actif.png", 2_000)

    app._prune()

    assert actif.exists()


def test_ne_fait_rien_si_moins_de_fichiers_que_la_limite(tmp_path):
    app = WallpaperApp(tmp_path)
    app.KEEP_LAST = 30
    for i in range(4):
        _fake_wallpaper(tmp_path, f"wallpaper_jour{i}.png", 1_000 + i)

    assert app._prune() == []
    assert len(list(tmp_path.glob("wallpaper_*.png"))) == 4


def test_ignore_les_fichiers_etrangers(tmp_path):
    app = WallpaperApp(tmp_path)
    app.KEEP_LAST = 1
    (tmp_path / "notes.txt").write_text("à garder")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    _fake_wallpaper(tmp_path, "wallpaper_vieux.png", 1_000)
    _fake_wallpaper(tmp_path, "wallpaper_neuf.png", 2_000)

    app._prune()

    assert (tmp_path / "notes.txt").exists()
    assert (tmp_path / ".DS_Store").exists()
