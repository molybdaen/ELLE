__author__ = 'Johannes'

from Tkinter import *

canvas_width = 500
canvas_height = 150


class DrawUtility(object):
    def drawCircle(self, w, x, y, r):
        python_green = "#476042"
        x1, y1 = ( x - r ), ( y - r )
        x2, y2 = ( x + r ), ( y + r )
        w.create_oval( x1, y1, x2, y2, fill = python_green )

def paint( event ):
   du.drawCircle(event.widget, event.x, event.y, 5)

du = DrawUtility()
master = Tk()
master.title("Input Space")
w = Canvas(master, width=canvas_width, height=canvas_height)
w.pack(expand = YES, fill = BOTH)
w.bind( "<B1-Motion>", paint )

message = Label( master, text = "Press and Drag the mouse to draw" )
message.pack( side = BOTTOM )

mainloop()
#
# from Tkinter import Tk, Canvas, PhotoImage, mainloop
# from math import sin
#
# WIDTH, HEIGHT = 640, 480
#
# window = Tk()
# canvas = Canvas(window, width=WIDTH, height=HEIGHT, bg="#000000")
# canvas.pack()
# img = PhotoImage(width=WIDTH, height=HEIGHT)
# canvas.create_image((WIDTH/2, HEIGHT/2), image=img, state="normal")
#
# for x in range(4 * WIDTH):
#     y = int(HEIGHT/2 + HEIGHT/4 * sin(x/80.0))
#     img.put("#ffffff", (x//4,y))
#
# mainloop()