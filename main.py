import complex
import iteration
import julia
import render
from PIL import ImageDraw, ImageFont

#parametres julia
size = 200
n_iter = 50
borne = 2
color_0 = render.Color(255,0,30)
color_1 = render.Color(0,50,200)
scale=render.Color_scale(color_0, color_1)

#parametres meta
meta_size = 8000

# initialisation du meta-plan
meta_plan = render.Plan(meta_size)



for x in range(-meta_size//(size*2),meta_size//(size*2)+1) :
    
    for y in range(-meta_size//(size*2),meta_size//(size*2)+1) :
        #print(-meta_size//(size*2),meta_size//(size*2))
        c=complex.Complex(x/(meta_size/size)*4, y/(meta_size/size)*4)
        print(c)
        f=iteration.Poly(1,0,c)
        j=julia.julia_picture(scale, f=f, size=size, n_iter=n_iter, borne=borne)
        meta_plan.paste(j,(x*size- size//2,y*size- size//2))
meta_plan.display()

""""
# parametres poly
a=1
b=0
c=complex.Complex(0, -0.75)
f=iteration.Poly(a,b,c)

j=julia.julia_picture(f=f, size,n_iter, borne )
j.display()
"""
"""
# initialisation du plan
plan = render.Plan(size)
# initialisation du poly
f=iteration.Poly(a,b,c)

for x in range(-size//2, size//2) :
    for y in range(-size//2, size//2) :
        z= complex.Complex(pas*x,pas*y)
        plan.draw_point((x,y),color=f.orbit_behaviour(z, n_iter, n_tot=n_iter, B=borne))

#plan.figure.text((0,0), f"size={size},n_iter={n_iter}, pas={pas}")
plan.display()
"""