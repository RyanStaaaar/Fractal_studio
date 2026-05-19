from PIL import Image, ImageDraw


class Color : 
    def __init__(self, r=255, g=255, b=255) :
        self.r=r
        self.g=g
        self.b=b
    def __add__(self, autre_color : Color) :
        somme= Color((self.r + autre_color.r)//2, (self.g + autre_color.g)//2,(self.b + autre_color.b)//2)
        return somme
    def __mul__(self, coeff) :
        if 0 <= coeff <=1 :
            return Color(coeff*self.r, coeff*self.g, coeff*self.b) 
        else :
            return NotImplemented
    def __rmul__(self,coeff) :
        return self.__mul__(coeff)

    def get_rgb(self) :
        return (int(self.r), int(self.g), int(self.b))
    
Color.black = Color(0,0,0)
Color.white = Color(255,255,255)
Color.red = Color(255,0,0)
Color.green = Color(0,255,0)
Color.blue = Color(0,0,255)

Color.violet= Color.red + Color.blue
Color.magenta= Color(255,0,255)

Color.olive = Color.red + Color.green
Color.yellow = Color(255,255,0)

Color.turquoise= Color.green + Color.blue
Color.cyan = Color(0,255,255)



class Color_scale:
    def __init__(self, color_0: Color, color_1: color) :
        self.color_0 = color_0
        self.color_1 = color_1
    def level(self, lambda_) :
        return self.color_0*lambda_ + self.color_1*(1-lambda_)
    
        

class Plan :
    def __init__(self, size) :
        self.canvas = Image.new("RGB", (size,size), color = (Color.cyan).get_rgb())
        self.centre = (size//2, size//2) # le centre est plus ou moins le vrai centre
        self.figure = ImageDraw.Draw(self.canvas)
    def draw_point(self,coordinates, color = Color.black) :
        self.figure.point((coordinates[0] + self.centre[0], coordinates[1] + self.centre[1]), fill=color.get_rgb())
    def display(self) :
        return self.canvas.show()
    def paste(self, autre_plan, coordinates) :
        self.canvas.paste(autre_plan.canvas, (coordinates[0] + self.centre[0], coordinates[1] + self.centre[1]))

       
    
if __name__== "__main__" :
    test = Plan(100)
    test.draw_point((-50,0))
    test.display()

