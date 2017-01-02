import Tkinter as tk
from Tkinter import *
from tkFileDialog   import askopenfilename
from PIL import Image, ImageTk
import sys,os
import numpy as np 

from Grabcut_app_tkinter import *
from Copy_Paste_Insert_Delete_app_tkinter import *
# from Recoloring_app_tkinter import *
from Local_Alpha_Matting_app_tkinter import *



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
        filemenu.add_command(label="Grabcut", command=self.Grabcut)
        
        filemenu.add_separator()
        filemenu.add_command(label="Copy_Paste_Insert_Delete", command=self.Copy_Paste_Insert_Delete)
        
        # filemenu.add_separator()
        # filemenu.add_command(label="Recoloring", command=self.Recoloring)

        filemenu.add_separator()
        filemenu.add_command(label="Local_Alpha_Matting", command=self.Local_Alpha_Matting)

        # filemenu.add_separator()
        # filemenu.add_command(label="Edges_Detection_and_Enhancement", command=self.Edges_Detection_and_Enhancement)

        # filemenu.add_separator()
        # filemenu.add_command(label="Primay_Pigments_Extraction", command=self.Primay_Pigments_Extraction)


        filemenu.add_separator()
        filemenu.add_command(label="Reset", command=self.Reset)

        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)


        helpmenu = tk.Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About...", command=self.About)



    def Show_image(self, img, option=0):
        self.master.title('Main window')
        width,height=img.size
        # print width, height
        if option==0:
            self.canvas = tk.Canvas(self.master, width=width, height=height, cursor="cross")
            self.canvas.pack(side="top", fill="both", expand=True)
        self.tk_im = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)



    def About(self):
        print "This is a simple example of a menu"


    def OpenFile(self):
        self.name = askopenfilename()
        print self.name
        self.im=Image.open(self.name).convert('RGB')

        # self.newWindow=tk.Toplevel(self.master)
        # main_window=Main_window(self.newWindow, self.name)
        self.Show_image(self.im, option=0)


    def Reset(self):
        self.Show_image(self.im, option=1)


    def Grabcut(self):
        grabcut_app=Grabcut_app(self.master, self.im, self.canvas)
        self.mask=grabcut_app.finalmask

    def Copy_Paste_Insert_Delete(self):
        copy_paste_insert_delete_app=Copy_Paste_Insert_Delete_app(self.master, self.im, self.canvas, self.mask)
        

    # def Recoloring(self): ### can local or global
    #     recoloring_app=Recoloring_app(self.master, self.im, self.canvas, self.mask)


    def Local_Alpha_Matting(self):
        local_alpha_matte_app=Local_Alpha_Matting_app(self.master, self.im, self.canvas)


    # def Edges_Detection_and_Enhancement(self):
    #     edges_detection_and_enhancement_app=Edges_Detection_and_Enhancement_app(self.master, self.im, self.canvas)
        

    # def Primay_Pigments_Extraction(self):
    #     primay_pigments_extraction_app=Primay_Pigments_Extraction_app(self.master, self.im, self.canvas)
        






if __name__ == "__main__":

    # base_dir = os.path.split( os.path.realpath(__file__) )[0]

    root=tk.Tk()
    app = Main_app(root)
    root.mainloop()



