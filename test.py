import random

x= 0.2


def left_neigbour(x,ls) :
    index=0
    if x<ls[0] : 
        raise ValueError("x too smalled compared to the list")
    for i in range(len(ls)) :
        if x-ls[index]>=x-ls[i]>=0 :
            index = i
    return index

def right_neigbour(x,ls) :
    index=len(ls)-1
    if x>ls[index] : 
        raise ValueError("x too big compared to the list")
    for i in range(len(ls)) :
        if ls[index]-x>=ls[i]-x>=0 :
            index =i
    return index

def neighbourhood(x,ls) :
    return (left_neigbour(x,ls), right_neigbour(x,ls))

palette=[[0, 0.8, 1], [(0,0,0), (116, 0, 184),(83, 144, 217)]]

def couleur(x,palette) :
    neighbours = neighbourhood(x,palette[0])
    lpos = palette[0][neighbours[0]]
    rpos = palette[0][neighbours[1]]
    lweight = (1-(x-lpos)/(rpos-lpos))
    rweight = (1-(rpos-x)/(rpos-lpos))
    lcolor = palette[1][neighbours[0]]
    rcolor = palette[1][neighbours[1]]
    return (round(lweight*lcolor[0]+rweight*rcolor[0]),round(lweight*lcolor[1]+rweight*rcolor[1]),round(lweight*lcolor[2]+rweight*rcolor[2]))

print(couleur(x,palette))
print(random.uniform(0,1))