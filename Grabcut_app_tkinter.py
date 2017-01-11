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
        self.mask=np.zeros(self.img2.shape[:2], dtype=np.uint8)
        self.finalmask=np.zeros(self.img2.shape[:2], dtype=np.uint8)
        self.x = self.y = 0
        self.rect = None
        self.start_x = 0
        self.start_y = 0
        self.rect_or_mask=0
        self.showing=0

        self.AllData=controller.AllData ##### our data here is using weights, not thickness.
        self.data=self.AllData.KM_weights
        self.PigNum=self.data.shape[-1]
        self.data_imgs=(self.data*255.0).round().clip(0,255).astype(np.uint8)
        self.data_imgs_copy=self.data_imgs.copy()
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
        self.user_select_indices=np.zeros(self.PigNum, dtype=np.uint8)
        self.user_select_indices_copy=np.zeros(self.PigNum, dtype=np.uint8)
        self.current_Extracted_pigments_index=0

        
        self.newWindow=tk.Toplevel(self.master)
        self.newWindow.geometry("650x250")
        self.shift(self.newWindow, position='right_top', scale=1.1)

        self.newWindow.title("Grabcut Window")

        self.var3 = IntVar()
        Checkbutton(self.newWindow, text="UseOurData (default is RGB)", variable=self.var3, command=self.update_status).grid(row=0, sticky=W, pady=5)


        for i in range(self.PigNum):
            Label(self.newWindow, relief=SOLID, text="     ", bg="#%02x%02x%02x" % tuple(self.AllData.PD_vertices[i].round().astype(np.uint8))).grid(row=1, sticky=W, padx=135+55*(i+1))
        
        self.var_list={}
        for i in range(self.PigNum):
            self.var_list.setdefault("p-"+str(i), IntVar())
        
        Label(self.newWindow, text="Pigments-choice").grid(row=2,sticky=W, pady=5)
        self.var_for_all=IntVar()
        Checkbutton(self.newWindow, text="All", variable=self.var_for_all, command=self.update_status1).grid(row=2, sticky=W, padx=125, pady=5)

        for i in range(self.PigNum):
            Checkbutton(self.newWindow, text="p-"+str(i), variable=self.var_list["p-"+str(i)], command=self.update_status2).grid(row=2, sticky=W, padx=125+55*(i+1), pady=5)
        
        self.var_showing_our_data=IntVar()
        Checkbutton(self.newWindow, text="Showing_our_data", variable=self.var_showing_our_data, command=self.update_status3).grid(row=3, sticky=W, pady=5)

                
        self.var_use_intermediate_mask=IntVar()
        Checkbutton(self.newWindow, text="Use_intermediate_mask", variable=self.var_use_intermediate_mask, command=self.update_use_intermediate_mask).grid(row=3, sticky=W, padx=200, pady=5)
    

        self.var1 = IntVar()
        Checkbutton(self.newWindow, text="foreground", variable=self.var1, command=self.update_txt1).grid(row=4, sticky=W, rowspan=2, pady=10)
        self.var2 = IntVar()
        Checkbutton(self.newWindow, text="background", variable=self.var2, command=self.update_txt2).grid(row=4, sticky=W, rowspan=2, padx=100, pady=10)
        

        self.thickness=IntVar()
        Label(self.newWindow, text="Scribble_size").grid(row=4,sticky=W, rowspan=2, padx=240, pady=10)
        self.thickness=Scale(self.newWindow, from_=2, to=20, orient=HORIZONTAL)
        self.thickness.grid(row=4, sticky=W, padx=350, rowspan=2, pady=10)
        self.thickness.set(5)


        Button(self.newWindow, text='Execute (Can press multi times)', command=self.Execute).grid(row=7, sticky=W, pady=4)
        
        Button(self.newWindow, text='Reset', command=self.Reset).grid(row=7, sticky=W, padx=240, pady=4)

        Button(self.newWindow, text='Save', command=self.save_as).grid(row=7, sticky=W, padx=320, pady=4)

        Button(self.newWindow, text='Quit', command=self.Quit).grid(row=7, sticky=W, padx=400, pady=4)
    
    
    def update_txt1(self):
        if self.var1.get()==1:
            self.var2.set(0)
    def update_txt2(self):
        if self.var2.get()==1:
            self.var1.set(0)

    def update_status(self):
        if self.rect_or_mask==1:
            self.rect_or_mask=0


    def update_status1(self):
        if self.var_for_all.get()==1:
            for i in range(len(self.var_list)):
                self.var_list['p-'+str(i)].set(1)

        if self.var_for_all.get()==0:
            for i in range(len(self.var_list)):
                self.var_list['p-'+str(i)].set(0)

        self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
        self.user_select_indices_copy=self.user_select_indices.copy()
        

    def update_status2(self):
        self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
        

        if self.var_showing_our_data.get()==1:
            # print self.user_select_indices
            # print self.user_select_indices_copy
            diff=self.user_select_indices-self.user_select_indices_copy

            if len(diff[diff==1])!=0: ### only count from 0 becoming 1 status.
                self.current_Extracted_pigments_index=np.arange(self.PigNum)[diff==1][0]
                # print self.current_Extracted_pigments_index

            if self.var_for_all.get()==1 and len(diff[diff==-1])!=0:  ### click on pig when "all" button is checked.
                self.current_Extracted_pigments_index=np.arange(self.PigNum)[diff==-1][0]
                self.var_list['p-'+str(self.current_Extracted_pigments_index)].set(1)
                # print self.current_Extracted_pigments_index

            for i in range(self.PigNum):
                if i!=self.current_Extracted_pigments_index:
                    self.var_list['p-'+str(i)].set(0)

            self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
            self.user_select_indices_copy=self.user_select_indices.copy()
            
            ###show image.
            self.Show_image(self.master, Image.fromarray(self.data_imgs_copy[:,:,self.current_Extracted_pigments_index]), option=1)


        if (self.user_select_indices!=0).any() or (self.user_select_indices==0).all():
            self.var_for_all.set(0)
        if (self.user_select_indices!=0).all():
            self.var_for_all.set(1)
     

    def update_status3(self):
        if self.var_showing_our_data.get()==0:
            self.Show_image(self.master, self.im, option=1)
        if self.var_showing_our_data.get()==1:
            self.Show_image(self.master, Image.fromarray(self.data_imgs_copy[:,:,self.current_Extracted_pigments_index]), option=1)

    def update_use_intermediate_mask(self):
        if self.var_use_intermediate_mask.get()==1:
            self.rect_or_mask=1

            row, col, M=self.data_imgs_copy.shape
            if (M%3)!=0:
                N=M/3+1
            else:
                N=M/3

            self.mask_copy=self.finalmask.copy() #### important!
            self.mask_copy[self.finalmask==255]=1

            self.masklist = np.zeros((row,col,N), dtype=np.uint8)
            self.masklist[:,:,:]=self.mask_copy.reshape((self.mask_copy.shape[0],self.mask_copy.shape[1],1))
            self.finalmasklist=self.masklist.copy()

        else:
            self.rect_or_mask=0





    
    def save_as(self):
        hen = asksaveasfilename(defaultextension = '.png')
        Image.fromarray(self.finalmask).save(hen)


        # if self.new_canvas!=None:
        #     ps=self.new_canvas.postscript(colormode='color')
        #     hen = asksaveasfilename(defaultextension = '.png')
        #     im = Image.open(io.BytesIO(ps.encode('utf-8')))
        #     im.save(hen)

    def Reset(self):
        
        self.img2=np.asarray(self.im)
        self.mask=np.zeros(self.img2.shape[:2], dtype=np.uint8)
        self.x = self.y = 0
        self.rect = None
        self.start_x = 0
        self.start_y = 0
        self.rect_or_mask=0
        self.rectangle=(0,0,1,1)
        self.showing=0

        self.var1.set(0)
        self.var2.set(0)
        
        if self.var3.get()==1:
            self.var_use_intermediate_mask.set(0)
            self.var_showing_our_data.set(0)

            self.var_for_all.set(0)
            for i in range(self.PigNum):
                self.var_list["p-"+str(i)].set(0)

            self.user_select_indices=np.zeros(self.PigNum, dtype=np.uint8)
            self.user_select_indices_copy=np.zeros(self.PigNum, dtype=np.uint8)

        self.var3.set(0)
        self.current_Extracted_pigments_index=0


        row, col, M=self.data_imgs.shape
        if (M%3)!=0:
            N=M/3+1
        else:
            N=M/3
        
        # print self.data_imgs.shape
        # print self.mask.shape
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
            
            self.data_imgs=self.data_imgs_copy[:,:,np.nonzero(self.user_select_indices)[0]].reshape((self.data_imgs_copy.shape[0], self.data_imgs_copy.shape[1], -1))
            
            # print self.user_select_indices
            # print self.data_imgs.shape

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

            if N>1:
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

            cv2.circle(self.mask,(curX,curY),self.thickness.get(),1,-1)

            for i in range(self.masklist.shape[-1]):
                temp_mask=self.masklist[:,:,i].copy()
                cv2.circle(temp_mask,(curX,curY),self.thickness.get(),1,-1)
                self.masklist[:,:,i]=temp_mask

            self.canvas.create_oval(curX-self.thickness.get(), curY-self.thickness.get(), curX+self.thickness.get(), curY+self.thickness.get(), outline="white", fill = "white" )

        elif self.var2.get()==1:#### background

            cv2.circle(self.mask,(curX,curY),self.thickness.get(),0,-1)

            for i in range(self.masklist.shape[-1]):
                temp_mask=self.masklist[:,:,i].copy()
                cv2.circle(temp_mask,(curX,curY),self.thickness.get(),0,-1)
                self.masklist[:,:,i]=temp_mask

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

