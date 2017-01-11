import Tkinter as tk
from Tkinter import *
from tkFileDialog   import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk
import sys,os,io
import numpy as np 
import cv2
from ImageMatting import *
import subprocess


class Local_Alpha_Matting_app:

    def __init__(self, controller):
        self.master=controller.master
        self.canvas=controller.canvas

        
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.im=controller.im
        self.img2=np.asarray(self.im)
        self.mask=np.ones((self.im.size[1],self.im.size[0]), dtype=np.uint8)*128
        self.x = self.y = 0
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.showing=0
        self.solve_flag=1
        self.data=controller.AllData.KM_weights #### our data here is using weights, not thickness
        
        self.new_master=None
        self.new_master1=None
        self.new_master2=None
        self.new_canvas=None



        
        self.newWindow=tk.Toplevel(self.master)
        self.newWindow.geometry("300x300")
        self.shift(self.newWindow, position='right_top', scale=1.1)

        self.newWindow.title("Alpha Matting Window")

        self.var3 = IntVar()
        Checkbutton(self.newWindow, text="UseOurData (default is RGB)", variable=self.var3, command=self.update_status).grid(row=0, sticky=W, pady=5)

        self.var1 = IntVar()
        Checkbutton(self.newWindow, text="foreground", variable=self.var1, command=self.update_txt1).grid(row=1, sticky=W)
        self.var2 = IntVar()
        Checkbutton(self.newWindow, text="background", variable=self.var2, command=self.update_txt2).grid(row=2, sticky=W)
        
        self.thickness=IntVar()
        Label(self.newWindow, text="Scribble_size").grid(row=3,sticky=W, pady=10)
        self.thickness=Scale(self.newWindow, from_=2, to=10, orient=HORIZONTAL)
        self.thickness.grid(row=3, sticky=W, padx=100, pady=10)
        self.thickness.set(5)

        Button(self.newWindow, text='Execute', command=self.Execute).grid(row=4, sticky=W, pady=4)
        
        Button(self.newWindow, text='Reset', command=self.Reset).grid(row=5, sticky=W, pady=4)

        Button(self.newWindow, text='Save', command=self.save_as).grid(row=6, sticky=W, pady=4)

        Button(self.newWindow, text='Quit', command=self.Quit).grid(row=7, sticky=W, pady=4)
    
    
    def update_txt1(self):
        if self.var1.get()==1:
            self.var2.set(0)
    def update_txt2(self):
        if self.var2.get()==1:
            self.var1.set(0)

    def update_status(self):
        self.solve_flag=1
    
    
    def save_as(self):

        if self.new_canvas!=None:
            ps=self.new_canvas.postscript(colormode='color')
            hen = asksaveasfilename(defaultextension = '.png')
            im = Image.open(io.BytesIO(ps.encode('utf-8')))
            im.save(hen)
        


    def Reset(self):
        self.mask=np.ones((self.im.size[1],self.im.size[0]), dtype=np.uint8)*128
        self.img2=np.asarray(self.im)
        self.x = self.y = 0
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.rectangle=(0,0,1,1)
        self.showing=0
        self.solve_flag=1

        self.var1.set(0)
        self.var2.set(0)
        self.var3.set(0)


        self.Show_image(self.master, self.im, option=1)

        if self.new_master!=None:
            self.new_master.destroy()
        if self.new_master1!=None:
            self.new_master1.destroy()
        if self.new_master2!=None:
            self.new_master2.destroy()


    def Quit(self):
        self.Reset()
        self.newWindow.destroy()
        


    def Show_image(self, master, img, option=0):
        
        width,height=img.size
        # print width, height
        if option==0:
            self.canvas = tk.Canvas(master, width=width, height=height, highlightthickness=0, borderwidth=0, cursor="cross")
            self.canvas.pack(side="top", fill="both", expand=True)
        self.tk_im = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)



    def Execute(self):


        rect=self.rectangle
        Trimap=np.ones((rect[3],rect[2]),dtype=np.uint8)*128
        small_mask=self.mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        Trimap[small_mask==1]=255
        Trimap[small_mask==0]=0
        # cv2.imshow('small_mask', small_mask*255)
        # cv2.imshow('trimap',Trimap)
        small_data=self.data[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],:]
        WinSize=7
        Option='Linear'
        small_img=self.img2[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2],:]

        # cv2.imwrite(output_prefix+"-selected_region.png", small_img)
        # cv2.imwrite(output_prefix+"-trimap.png", Trimap)
        # np.savetxt(output_prefix+"-small_weights.txt",small_mixing_Weights.reshape((rect[3]*rect[2],-1)))
        # print (RGB_or_Weights_flag)
        # print (solve_flag)
         


        self.new_master=1 ### temporary


        if self.solve_flag==1 and self.var3.get()==0:
            Alpha=Image_Matting_By_Learning(Get_newformat_data(small_img,'rgb'), Trimap, WinSize=WinSize, Options=Option)
            # Alpha=Image_Matting_By_Learning(small_img/255.0, Trimap, WinSize=WinSize, Options=Option)

            ### filter ALpha
            # Alpha=cv2.ximgproc.guidedFilter(small_img, Alpha.astype(np.float32), 10, 1e-5)
            # Alpha[Trimap==255]=1.0
            # Alpha[Trimap==0]=0.0
            
            Alpha=(Alpha*255).round().astype(np.uint8)

            # Alpha_3=np.zeros((Alpha.shape[0],Alpha.shape[1],3), dtype=np.uint8)
            # Alpha_3[:,:,:]=Alpha.reshape((Alpha.shape[0],Alpha.shape[1],1))
            # Alpha_3=cv2.ximgproc.jointBilateralFilter(small_img, Alpha_3, 5, 5.0, 3 )
            # Alpha=Alpha_3[:,:,0]


            
            # cv2.imwrite(output_prefix+Option+"-alphamatting_on_RGB.png", Alpha)
            self.solve_flag=0
            
            self.new_master1=tk.Toplevel()
            self.new_master1.title('Alpha_Matting_Results-using_rgb_data')
            self.new_master=self.new_master1
  


        if self.solve_flag==1 and self.var3.get()==1:
            Alpha=Image_Matting_By_Learning(small_data, Trimap, WinSize=WinSize, Options=Option)

            ### filter ALpha
            # Alpha=cv2.ximgproc.guidedFilter(small_img, Alpha.astype(np.float32), 10, 1e-5)
            # Alpha[Trimap==255]=1.0
            # Alpha[Trimap==0]=0.0
            

            Alpha=(Alpha*255).round().astype(np.uint8)
            # Alpha_3=np.zeros((Alpha.shape[0],Alpha.shape[1],3), dtype=np.uint8)
            # Alpha_3[:,:,:]=Alpha.reshape((Alpha.shape[0],Alpha.shape[1],1))
            # Alpha_3=cv2.ximgproc.jointBilateralFilter(small_img, Alpha_3, 5, 5.0, 3 )
            # Alpha=Alpha_3[:,:,0]
     

            self.solve_flag=0
            
            self.new_master2=tk.Toplevel()
            self.new_master2.title('Alpha_Matting_Results-using_our_data')
            self.new_master=self.new_master2



        self.show_results(small_img, Trimap, Alpha)

        

    def show_results(self, small_img, Trimap, Alpha):

        # #### show results in other window.
        img1=Image.fromarray(small_img)
        img2=Image.fromarray(Trimap)
        img3=Image.fromarray(Alpha)
        width,height=img1.size


        self.new_canvas = tk.Canvas(self.new_master, width=width*3+20, height=height, highlightthickness=0, borderwidth=0, cursor="cross")
        self.new_canvas.pack(side="top", fill="both", expand=True)

        tk_im1 = ImageTk.PhotoImage(img1)
        tk_im2 = ImageTk.PhotoImage(img2)
        tk_im3 = ImageTk.PhotoImage(img3)

        self.new_canvas.create_image(0,0,anchor="nw",image=tk_im1)
        self.new_canvas.create_image(width+10,0,anchor="nw",image=tk_im2)
        self.new_canvas.create_image(width*2+20,0,anchor="nw",image=tk_im3)

        self.shift(self.new_master, position='center', scale=1.0)
        self.new_master.mainloop()






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
        
        if self.var1.get()==0 and self.var2.get()==0:

            self.rectangle = (min(curX,self.start_x),min(curY,self.start_y),abs(curX-self.start_x),abs(curY-self.start_y))
            self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)
            self.rect_or_mask=0

        elif self.var1.get()==1:####foreground

            cv2.circle(self.mask,(curX,curY),self.thickness.get(),1,-1)
            self.canvas.create_oval(curX-self.thickness.get(), curY-self.thickness.get(), curX+self.thickness.get(), curY+self.thickness.get(), outline="white", fill = "white" )

        elif self.var2.get()==1:#### background

            cv2.circle(self.mask,(curX,curY),self.thickness.get(),0,-1)
            self.canvas.create_oval(curX-self.thickness.get(), curY-self.thickness.get(), curX+self.thickness.get(), curY+self.thickness.get(), outline="black", fill = "black" )
        





    def on_button_release(self, event):
        pass

    def shift(self,toplevel, position='right_top', scale=1.0):
        toplevel.update_idletasks()
        w = toplevel.winfo_screenwidth()
        h = toplevel.winfo_screenheight()
        size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
        size2= (size[0]*scale, size[1]*scale)

        if position=='center':
            x = w/2 - size2[0]/2
            y = h/2 - size2[1]/2
        if position=='right_top':
            x=w-size2[0]
            y=0

        toplevel.geometry("%dx%d+%d+%d" % (size2 + (x, y)))

