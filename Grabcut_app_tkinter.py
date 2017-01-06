import Tkinter as tk
from Tkinter import *
from tkFileDialog   import askopenfilename,asksaveasfilename 
from PIL import Image, ImageTk
import sys,os,io
import numpy as np 
import cv2


class Grabcut_app:

    def __init__(self, controller):
        self.controller=controller
        self.master=controller.master
        self.canvas=controller.canvas
        
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.im=controller.im
        self.img2=np.asarray(self.im)
        self.mask=np.zeros((self.im.size[1],self.im.size[0]), dtype=np.uint8)
        self.finalmask=np.zeros((self.im.size[1],self.im.size[0]), dtype=np.uint8)
        self.x = self.y = 0
        self.rect = None
        self.start_x = 0
        self.start_y = 0
        self.rect_or_mask=0
        self.thickness=5
        self.showing=0

        self.data=controller.AllData.KM_weights ##### our data here is using weights, not thickness.
        self.data_imgs=(self.data*255.0).round().clip(0,255).astype(np.uint8)
        print self.data_imgs.shape

        row, col, M=self.data_imgs.shape
        if (M%3)!=0:
            N=M/3+1
        else:
            N=M/3

        self.mask_copy=self.mask.copy()
        self.masklist = np.zeros((row,col,N), dtype=np.uint8)
        self.masklist[:,:,:]=self.mask_copy.reshape((self.mask_copy.shape[0],self.mask_copy.shape[1],1))
        # print self.masklist.shape
        self.finalmasklist=self.masklist.copy()

        self.new_master=None
        self.new_canvas=None


        
        self.newWindow=tk.Toplevel(self.master)
        self.shift(self.newWindow, position='right_top', scale=1.1)

        self.newWindow.title("Grabcut Window")

        self.var3 = IntVar()
        Checkbutton(self.newWindow, text="UseOurData (default is RGB)", variable=self.var3, command=self.update_status).grid(row=0, sticky=W, pady=5)

        self.var1 = IntVar()
        Checkbutton(self.newWindow, text="foreground", variable=self.var1, command=self.update_txt1).grid(row=1, sticky=W)
        self.var2 = IntVar()
        Checkbutton(self.newWindow, text="background", variable=self.var2, command=self.update_txt2).grid(row=2, sticky=W)
        

        Button(self.newWindow, text='Execute (Can press multi times)', command=self.Execute).grid(row=4, sticky=W, pady=4)
        
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
        if self.rect_or_mask==1:
            self.rect_or_mask=0
    
    def save_as(self):
        if self.new_canvas!=None:
            ps=self.new_canvas.postscript(colormode='color')
            hen = asksaveasfilename(defaultextension = '.png')
            im = Image.open(io.BytesIO(ps.encode('utf-8')))
            im.save(hen)

    def Reset(self):
        self.mask=np.zeros((self.im.size[1],self.im.size[0]), dtype=np.uint8)
        self.img2=np.asarray(self.im)
        self.x = self.y = 0
        self.rect = None
        self.start_x = 0
        self.start_y = 0
        self.rect_or_mask=0
        self.rectangle=(0,0,1,1)
        self.thickness=5
        self.showing=0

        self.var1.set(0)
        self.var2.set(0)
        self.var3.set(0)

        row, col, M=self.data_imgs.shape
        if (M%3)!=0:
            N=M/3+1
        else:
            N=M/3

        self.mask_copy=self.mask.copy()
        self.masklist = np.zeros((row,col,N), dtype=np.uint8)
        self.masklist[:,:,:]=self.mask_copy.reshape((self.mask_copy.shape[0],self.mask_copy.shape[1],1))
        self.finalmasklist=self.masklist.copy()


        self.Show_image(self.master, self.im, option=1)
        
        if self.new_master!=None:
            self.new_master.destroy()


    def Quit(self):
        self.Reset()
        self.newWindow.destroy()
        self.controller.mask=self.finalmask.copy()
       
    def return_results(self):
        return self.finalmask


    def Show_image(self, master, img, option=0):
        
        width,height=img.size
        # print width, height
        if option==0:
            self.canvas = tk.Canvas(master, width=width, height=height, cursor="cross")
            self.canvas.pack(side="top", fill="both", expand=True)
        self.tk_im = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)



    def Execute(self):


        if self.var3.get()==0: ### use rgb image as input

            if self.rect_or_mask == 0:         # grabcut with rect
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(self.img2,self.mask,self.rectangle,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                self.rect_or_mask = 1

            elif self.rect_or_mask == 1:         # grabcut with mask
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv2.grabCut(self.img2,self.mask,self.rectangle,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
     
            self.finalmask = np.where((self.mask==1) + (self.mask==3),255,0).astype('uint8')

            # Image.fromarray(self.finalmask).save('grabcut_output-mask.png')
            # output = cv2.bitwise_and(self.img2,self.img2,mask=self.finalmask)


        elif self.var3.get()==1: ### use our weights or thickness map as input
            # print self.var3.get()

            row,col,M=self.data_imgs.shape
            # print self.rect_or_mask
            
            if (M%3)!=0:
                N=M/3+1
            else:
                N=M/3


            if self.rect_or_mask == 0:         # grabcut with rect


                for i in range(N):
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)

                    temp_img=np.ones((row,col,3),dtype=np.uint8)
                    index_list=np.array([ (i*3)%M, (i*3+1)%M, (i*3+2)%M ])
                    temp_img[:,:,:]=self.data_imgs[:,:,index_list].copy()

                    temp_mask=self.masklist[:,:,i].copy()
 
                    cv2.grabCut(temp_img,temp_mask,self.rectangle,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
                    self.masklist[:,:,i]=temp_mask.copy()


                self.rect_or_mask = 1

            elif self.rect_or_mask == 1:         # grabcut with mask
                


                for i in range(N):
                    bgdmodel = np.zeros((1,65),np.float64)
                    fgdmodel = np.zeros((1,65),np.float64)

                    temp_img=np.ones((row,col,3),dtype=np.uint8)
                    index_list=np.array([ (i*3)%M, (i*3+1)%M, (i*3+2)%M ])
                    temp_img[:,:,:]=self.data_imgs[:,:,index_list].copy()
                    temp_mask=self.masklist[:,:,i].copy()
                    cv2.grabCut(temp_img,temp_mask,self.rectangle,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
                    self.masklist[:,:,i]=temp_mask.copy()
                    

            for i in range(N):
                self.finalmasklist[:,:,i]=np.where((self.masklist[:,:,i]==1) + (self.masklist[:,:,i]==3),255,0).astype('uint8')
            
            self.finalmask = self.finalmasklist[:,:,0]
            for i in range(1,N):
                self.finalmask=cv2.bitwise_or(self.finalmask,self.finalmasklist[:,:,i])


            

        


        # #### show results in other window.
        img=Image.fromarray(self.finalmask).convert('L')
        width,height=img.size
        # print img.size

        if self.showing==0:
            self.new_master=tk.Toplevel()
            self.new_master.title('Grabcut_results')
            self.new_canvas = tk.Canvas(self.new_master, width=width, height=height, cursor="cross")
            self.new_canvas.pack(side="top", fill="both", expand=True)
            self.showing=1

        tk_im = ImageTk.PhotoImage(img)
        self.new_canvas.create_image(0,0,anchor="nw",image=tk_im)
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

            cv2.circle(self.mask,(curX,curY),self.thickness,1,-1)

            for i in range(self.masklist.shape[-1]):
                temp_mask=self.masklist[:,:,i].copy()
                cv2.circle(temp_mask,(curX,curY),self.thickness,1,-1)
                self.masklist[:,:,i]=temp_mask

            self.canvas.create_oval(curX-self.thickness, curY-self.thickness, curX+self.thickness, curY+self.thickness, outline="white", fill = "white" )

        elif self.var2.get()==1:#### background

            cv2.circle(self.mask,(curX,curY),self.thickness,0,-1)

            for i in range(self.masklist.shape[-1]):
                temp_mask=self.masklist[:,:,i].copy()
                cv2.circle(temp_mask,(curX,curY),self.thickness,0,-1)
                self.masklist[:,:,i]=temp_mask

            self.canvas.create_oval(curX-self.thickness, curY-self.thickness, curX+self.thickness, curY+self.thickness, outline="black", fill = "black" )
        





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

