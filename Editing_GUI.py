import tkinter as tk # this is in python 3.4. For python 2.x import Tkinter
from tkFileDialog   import askopenfilename
from PIL import Image, ImageTk
import sys,os
import numpy as np 



class Edit_window:

    def __init__(self, master, filename):

        self.master=master
        master.title('grabcut_window')
        self.x = self.y = 0
        self.im=Image.open(filename)
        width,height=self.im.size
        print width, height
        self.canvas = tk.Canvas(self.master, width=width, height=height, cursor="cross")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.rect = None
        self.start_x = None
        self.start_y = None

        self._draw_image(self.im)


    def _draw_image(self, img):
         self.tk_im = ImageTk.PhotoImage(img)
         self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)


    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        # create rectangle if not yet exist
        #if not self.rect:
        self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1, fill='')

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)
        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)

    def on_button_release(self, event):
        pass



class Grabcut_dialog:

    def __init__(self, master):
        self.master=master
    def help(self):
        return None





class Main_app:

    def __init__(self, master):
        self.master=master
        master.title('inital window')


        #### menu for different task
        menu = tk.Menu(master)
        self.master.config(menu=menu)
        filemenu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open example image", command=self.OpenFile)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)
        helpmenu = tk.Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About)


    def NewFile(self):
        print "New File!"
    def OpenFile(self):
        self.name = askopenfilename()
        print self.name
        # self.newWindow=tk.Toplevel(self.master)
        app=Edit_window(self.master, self.name)

    def About(self):
        print "This is a simple example of a menu"
        







if __name__ == "__main__":

    # base_dir = os.path.split( os.path.realpath(__file__) )[0]

    root=tk.Tk()
    app = Main_app(root)
    root.mainloop()


