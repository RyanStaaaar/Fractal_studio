from __future__ import annotations
from PIL import Image
import numpy as np

class Color : 
    def __init__(self, r=255, g=255, b=255) :
        self.r=r
        self.g=g
        self.b=b
    def __str__(self):
        return f"({self.r},{self.g},{self.b})"
    def __add__(self, autre_color : Color) :
        return Color(min(255, self.r + autre_color.r),min(255, self.g + autre_color.g),min(255, self.b + autre_color.b))

    def __mul__(self, coeff) :
        if 0 <= coeff <=1 :
            return Color(coeff*self.r, coeff*self.g, coeff*self.b) 
        else :
            return NotImplemented
    def __rmul__(self,coeff) :
        return self.__mul__(coeff)
    
    def blend(self, autre_color, lambda_=0.5) :
            return lambda_*self + (1-lambda_)*autre_color

    def get_rgb(self) :
        return (int(self.r), int(self.g), int(self.b))
    
Color.black = Color(0,0,0)
Color.white = Color(255,255,255)
Color.red = Color(255,0,0)
Color.green = Color(0,255,0)
Color.blue = Color(0,0,255)

Color.magenta= Color.red + Color.blue
Color.yellow = Color.red + Color.green
Color.cyan= Color.green + Color.blue

Color.violet = Color.red.blend(Color.blue)
Color.orange = Color.red.blend(Color.yellow, 0.5)
Color.green2 = Color.cyan.blend(Color.yellow,0.5)


class ColorScale:
    def __init__(self, color_0: Color, color_1: Color) :
        self.color_0 = color_0
        self.color_1 = color_1
    def level(self, lambda_) :
        return self.color_0.blend(self.color_1, lambda_)

        
ColorScale.interfaaace_green = ColorScale(Color(80, 255, 120),Color(255,255,255))
ColorScale.lava_ocean = ColorScale(Color(120, 0, 0),Color(102, 155, 188))
ColorScale.red_flag_blue = ColorScale(Color(193, 18, 31),Color(0, 48, 73))
ColorScale.fresh_meadow = ColorScale(Color(181, 228, 140),Color(22, 138, 173))


def coloriser(V, palette):
    positions = np.array(palette[0], dtype=np.float64)
    colors = np.array(palette[1], dtype=np.float64)

    V_clamped = np.clip(V, positions[0], positions[-1])

    # pour chaque pixel, trouver dans quel segment il tombe
    idx_right = np.searchsorted(positions, V_clamped, side='right')
    idx_right = np.clip(idx_right, 1, len(positions) - 1)
    idx_left = idx_right - 1

    # positions des stops gauche et droit
    lpos = positions[idx_left]
    rpos = positions[idx_right]

    # poids d'interpolation (même logique que ton lweight/rweight)
    span = rpos - lpos
    span[span == 0] = 1          # évite la division par zéro (pile sur un stop)
    t = (V_clamped - lpos) / span

    # couleurs aux stops gauche et droit
    lcolor = colors[idx_left]    # (H, W, 3)
    rcolor = colors[idx_right]   # (H, W, 3)

    # interpolation
    t = t[:, :, None]            # (H, W, 1) pour diffuser sur les 3 canaux
    return (lcolor * (1 - t) + rcolor * t).astype(np.uint8)  

# def coloriser(V, scale):
#     c0 = np.array(scale.color_0.get_rgb())
#     c1 = np.array(scale.color_1.get_rgb())
#     s = V[:, :, None]                       # (H, W, 1) pour diffuser sur les 3 canaux ?
#     return (c1 * (1 - s) + c0 * s).astype(np.uint8)       


#if __name__== "__main__" :
