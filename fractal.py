import iteration
import numpy as np

# --- chaque jour : tirer un c intéressant (instantané) ---
def choisir_c(borne=2, seuil_bas=0.01, seuil_haut=0.89):
    V = np.load("mandelbrot_map.npy")
    H, W = V.shape
    
    # les cases entre seuil_bas et seuil_haut = près de la frontière
    masque = (V >= seuil_bas) & (V <= seuil_haut)
    ys, xs = np.nonzero(masque)
    
    # tirer un pixel au hasard parmi les candidats
    idx = np.random.randint(len(xs))
    
    # convertir l'index pixel en coordonnée complexe
    borne_y = borne * H / W
    cx = -borne   + xs[idx] * (2 * borne)   / (W - 1)
    cy = -borne_y + ys[idx] * (2 * borne_y) / (H - 1)
    
    return complex(cx, cy)

def julia(f, height, width, borne=2, n=100):
    borne_y = borne * height / width
    xs = np.linspace(-borne, borne, width)
    ys = np.linspace(-borne_y, borne_y, height)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y
    return iteration.escape_speed(Z, f.a, f.b, f.c, n)
def mandelbrot(height, width, seed = 0, borne=2, n=100):
    borne_y = borne * height / width
    xs = np.linspace(-borne, borne, width)
    ys = np.linspace(-borne_y, borne_y, height)
    X, Y = np.meshgrid(xs, ys)
    C= X + 1j * Y
    return iteration.mandelbrot(C, seed, n,)

def mosaique_mandelbrot(size=100, n_sub=11, borne_julia=2, borne_c=2, n=100):
    meta_size = n_sub * size
    V = np.zeros((meta_size, meta_size))

    cs = np.linspace(-borne_c, borne_c, n_sub)
    for i, cy in enumerate(cs):
        for j, cx in enumerate(cs):
            c = complex(cx, cy)
            f = iteration.Poly(1, 0, c)
            V[i*size:(i+1)*size, j*size:(j+1)*size] = julia(f, size, size, borne_julia, n)
    return V
#if __name__=="__main__" :



