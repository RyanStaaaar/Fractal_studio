import complex
import iteration
import render
from PIL import ImageDraw, ImageFont
from math import pi


def julia_picture(scale ,f: iteration.Poly, size = 10, n_iter = 10, borne = 2, ) :
    cons=2
    step = pi/size
    # initialisation du plan
    plan = render.Plan(size)
    for x in range(-size//cons, size//cons) :
        for y in range(-size//cons, size//cons) :
            z= complex.Complex(step*x,step*y)
            escape_speed = f.orbit_behaviour(z, n_iter, n_tot=n_iter, B=borne)
            plan.draw_point((x,y),color=(scale.level(escape_speed/255)))
    return plan
if __name__=="__main__" :
    a=1
    b=0
    c=complex.Complex(-0.4, -0.6)
    f=iteration.Poly(a,b,c)
    color_0=render.Color.white
    color_1= render.Color.black
    scale= render.Color_scale(color_0, color_1)

    j=julia_picture(scale, f,size=500, n_iter = 50)
    j.display()


