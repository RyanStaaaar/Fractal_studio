# legacy/ — scripts archivés

Étapes antérieures du projet et outils ponctuels. Ils ne sont **pas** nécessaires pour
`fractal_studio.py` ni `daily_wallpaper.py`, mais restent ici parce qu'ils fonctionnent
et documentent l'évolution du projet.

| Script | Rôle |
|---|---|
| `interfaaaace.py` | L'interface Tkinter d'origine (exploration : traps, transformations, trap image). Ancêtre direct de `fractal_studio.py`. |
| `animator.py` | Premier prototype d'animation par keyframes, avant les modules et l'éditeur de vélocités. |
| `main.py` | Script initial : la mosaïque de Julia sur la grille de Mandelbrot. |
| `precalcule.py` | Pré-calcule `mandelbrot_map.npy` (la carte affichée dans le module « c »). |
| `palette_calibrator.py` | Passe en revue les 348 palettes Sanzo Wada et exporte un tableur de calibration. |
| `diagnose_path.py` | Tracés de diagnostic du reparamétrage DEM (« vitesse uniforme ») : coût ρ(t), t(frame). |
| `ssaa_compare.py` | Compare le suréchantillonnage 1×→4× sur trois fractales. |

## Exécution

Leurs imports (`fractal`, `render`, `iteration`…) supposent la racine du dépôt :

```bash
PYTHONPATH=. python legacy/<script>.py
```

Dépendances supplémentaires (matplotlib, openpyxl) :

```bash
pip install -r requirements-legacy.txt
```
