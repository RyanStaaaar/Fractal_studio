import complex
import iteration
import render

#parametres generaux
size = 4000
n_iter = 50
borne = 2
pas=4/size


# parametres poly
a=1
b=0
c=0.285

# initialisation du plan
plan = render.Plan(size)
# initialisation du poly
f=iteration.Poly(a,b,c)

for x in range(-size//2, size//2) :
    for y in range(-size//2, size//2) :
        z= complex.Complex(pas*x,pas*y)
        if f.orbit_bounded(z,n_iter,borne):
            plan.draw_point((x,y))
plan.display()
