from PIL import Image, ImageDraw

class Plan :
    def __init__(self, size) :
        self.canvas = Image.new("L", (size,size), color = 255)
        self.centre = (size//2, size//2) # le centre est plus ou moins le vrai centre
        self.figure = ImageDraw.Draw(self.canvas)
    def draw_point(self,coordinates) :
        self.figure.point((coordinates[0] + self.centre[0], coordinates[1] + self.centre[1]))
    def display(self) :
        return self.canvas.show()
    
if __name__== "__main__" :
    test = Plan(100)
    test.draw_point((-50,0))
    test.display()

