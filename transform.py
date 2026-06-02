"""Transformations du plan complexe (pullback / gather).

Compile une expression de fonction complexe saisie par l'utilisateur en un
callable vectorisé `g(Z) -> Z`. La fonction est appliquée au meshgrid AVANT
l'itération de la fractale : le pixel de coordonnée `w` échantillonne la
fractale en `f(w)`. C'est un pullback (transformation de coordonnées), ce qui
rend le rendu sans trou et n'exige jamais d'inverser `f`.
"""
import numpy as np

# fonctions et constantes autorisées dans les expressions
_NAMESPACE = {
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "abs": np.abs, "conj": np.conj,
    "i": 1j, "e": np.e, "pi": np.pi,
}


def parse_transform(expr: str):
    """Compile `expr` (fonction complexe de `z`) en callable g(Z: ndarray) -> ndarray.

    Notation conviviale : `^` = puissance, `i` = unité imaginaire, `e`/`pi`
    constantes, fonctions numpy (`exp`, `sin`, `log`, ...). Chaîne vide ou `"z"`
    = identité. Lève ValueError si l'expression est invalide.
    """
    expr = (expr or "z").strip() or "z"
    py = expr.replace("^", "**")

    def g(Z: np.ndarray) -> np.ndarray:
        local = dict(_NAMESPACE)
        local["z"] = Z
        # errstate : 1/z, log(0), e^z (overflow)... renvoient inf/nan sans warning
        # (l'itération les traite comme du fond) ; évite aussi un lookup __import__
        # déclenché par la machinerie de warnings avec __builtins__ vidé.
        with np.errstate(all="ignore"):
            return np.asarray(eval(py, {"__builtins__": {}}, local), dtype=np.complex128)

    # validation immédiate sur un petit échantillon pour repérer les erreurs tout de suite
    try:
        g(np.array([0 + 0j, 1 + 1j]))
    except Exception as exc:
        raise ValueError(f"Transformation invalide : {expr!r} ({exc})") from exc
    return g
