import complex
import iteration
import render
from PIL import ImageDraw, ImageFont
from math import pi


def julia_picture(f: Poly, size = 10, n_iter = 10, borne = 2) :
    cons=2
    step = pi/size
    # initialisation du plan
    plan = render.Plan(size)
    for x in range(-size//cons, size//cons) :
        for y in range(-size//cons, size//cons) :
            z= complex.Complex(step*x,step*y)
            escape_speed = f.orbit_behaviour(z, n_iter, n_tot=n_iter, B=borne)
            plan.draw_point((x,y),color=(escape_speed, escape_speed,escape_speed))
    return plan
if __name__=="__main__" :
    a=1
    b=0
    c=complex.Complex(0, -0.75)
    f=iteration.Poly(a,b,c)
    j=julia_picture(f,size=100, n_iter = 100)
    j.display()


