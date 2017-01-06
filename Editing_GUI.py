import Tkinter as tk
from Tkinter import *
from tkFileDialog   import askopenfilename, askdirectory
from PIL import Image, ImageTk
import sys,os
import numpy as np 

from Grabcut_app_tkinter import *
from Copy_Paste_Insert_Delete_app_tkinter import *
from Recoloring_app_tkinter import *
from Local_Alpha_Matting_app_tkinter import *
from Thickness_Scattering_Weights_modification_app_tkinter import *




class Main_app:

    def __init__(self, master):
        self.master=master
        self.master.title('Inital Window')


        Welcome = """\nWelcome to PigMania!\n\nPlease click "File" Menu to start.\n"""
        self.msg = Message(self.master, text = Welcome, width=600)
        self.msg.config(bg='lightgreen', font=('times', 24, 'italic'))
        self.msg.pack()

        #### menu for different task
        menu = tk.Menu(master)
        self.master.config(menu=menu)
        filemenu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)

        filemenu.add_command(label="Open processed example image", command=self.OpenFile)

        filemenu.add_separator()
        filemenu.add_command(label="Grabcut", command=self.Grabcut)
        
        filemenu.add_separator()
        filemenu.add_command(label="Copy_Paste_Insert_Delete", command=self.Copy_Paste_Insert_Delete)
        
        filemenu.add_separator()
        filemenu.add_command(label="Recoloring", command=self.Recoloring)

        filemenu.add_separator()
        filemenu.add_command(label="Thickness, Scattering and Weights modification", command=self.Thickness_Scattering_Weights_modification)

        filemenu.add_separator()
        filemenu.add_command(label="Local_Alpha_Matting", command=self.Local_Alpha_Matting)

        filemenu.add_separator()
        filemenu.add_command(label="Open unprocessed example image and Extract pigments", command=self.OpenFile_version2)

        filemenu.add_separator()
        filemenu.add_command(label="Reset", command=self.Reset)

        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit)


        helpmenu = tk.Menu(menu)
        menu.add_cascade(label="Help", menu=helpmenu)
        helpmenu.add_command(label="About", command=self.About)


        self.mask=None
        self.control=0



    def Show_image(self, img, option=0):
        self.master.title('Main window')
        width,height=img.size
        # print width, height
        if option==0:
            self.canvas = tk.Canvas(self.master, width=width, height=height, cursor="cross")
            self.canvas.pack(side="top", fill="both", expand=True)
        else:
            self.canvas.config(width=width, height=height)
            self.canvas.pack(side="top", fill="both", expand=True)

        self.tk_im = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)


    def Reset(self):
        self.Show_image(self.im, option=1)
        self.control=0

    def About(self):
        print "####"


    class AllData():  ### to collect all datas.
        def __init__(self):
            self.KM_pigments=None
            self.KM_weights=None
            self.KM_thickness=None
            self.PD_vertices=None
            self.PD_weights=None
            self.PD_opacities=None
            self.Pig_order=None
            self.Existing_KS=None




    def OpenFile(self): #### open a processed image and all other files together.
        self.name = askopenfilename()
        self.directory=os.path.split(self.name)[0]
        print self.name
        print self.directory
        self.im=Image.open(self.name).convert('RGB')
        self.img2=np.asarray(self.im)

        self.msg.pack_forget() ###remove message box and show image.
        
        
        self.Show_image(self.im, option=self.control)
        self.control=1

        self.path_prefix=os.path.splitext(self.name)[0]
        
        
        self.OpenFile0()
        self.OpenFile1()
        self.OpenFile2()
        self.OpenFile3()
        self.OpenFile4()
        self.OpenFile5()
        self.OpenFile6()
        self.OpenFile7()

        print "Finish loading all data!"
        

    def OpenFile0(self):
        name=self.directory+"/order1.txt"
        self.AllData.Pig_order=np.loadtxt(name).astype(np.uint8)
        self.PigNum=len(self.AllData.Pig_order)
        print self.PigNum
        print self.AllData.Pig_order
        


    def OpenFile1(self):
        name=self.path_prefix+"-"+str(self.PigNum)+"-primary_pigments_KS.txt" 
        self.AllData.KM_pigments=np.loadtxt(name)
        print self.AllData.KM_pigments.shape
        self.AllData.KM_pigments=self.AllData.KM_pigments[self.AllData.Pig_order, :] ### reorder

        



    def OpenFile2(self):

        # ### read txt is slow
        # name=self.path_prefix+"-"+str(self.PigNum)+"-KM_mixing-weights.txt"
        # self.AllData.KM_weights=np.loadtxt(name)
        # self.AllData.KM_weights=self.AllData.KM_weights.reshape((self.img2.shape[0],self.img2.shape[1],self.AllData.KM_weights.shape[-1]))


        self.AllData.KM_weights=np.zeros((self.img2.shape[0],self.img2.shape[1],self.PigNum))
        for i in range(self.PigNum):
            name=self.path_prefix+"-"+str(self.PigNum)+"-KM_mixing-weights_map-%02d.png"%i
            self.AllData.KM_weights[:,:,i]=np.asarray(Image.open(name).convert('L'))/255.0
        
        print self.AllData.KM_weights.shape
        self.AllData.KM_weights=self.AllData.KM_weights[:,:,self.AllData.Pig_order] ### reorder 


    def OpenFile3(self):
        
        # ###read txt file is slow
        # name=self.path_prefix+"-"+str(self.PigNum)+"-KM_layers-order1-thickness.txt"
        # self.AllData.KM_thickness=np.loadtxt(name)
        # self.AllData.KM_thickness=self.AllData.KM_thickness.reshape((self.img2.shape[0],self.img2.shape[1],self.AllData.KM_thickness.shape[-1]))
        

        self.AllData.KM_thickness=np.zeros((self.img2.shape[0],self.img2.shape[1],self.PigNum))
        for i in range(self.PigNum):
            name=self.path_prefix+"-"+str(self.PigNum)+"-KM_layers-order1-thickness_map-%02d.png"%i
            self.AllData.KM_thickness[:,:,i]=np.asarray(Image.open(name).convert('L'))/255.0

        self.AllData.KM_thickness+=1e-11
        print self.AllData.KM_thickness.shape

    
    def OpenFile4(self):
        name=self.path_prefix+"-"+str(self.PigNum)+"-primary_pigments_RGB_color.js" 
        with open(name) as myfile:
            self.AllData.PD_vertices=np.asarray(json.load(myfile)['vs'])
        print self.AllData.PD_vertices.shape
        self.AllData.PD_vertices=self.AllData.PD_vertices[self.AllData.Pig_order, :] ### reorder 



    def OpenFile5(self):
        
        # #### reading js file is slow
        # name=self.path_prefix+"-"+str(self.PigNum)+"-PD_mixing-weights.js"
        # with open(name) as myfile:
        #     self.AllData.PD_weights=np.asarray(json.load(myfile)['weights'])
        # self.AllData.PD_weights=self.AllData.PD_weights.reshape((self.img2.shape[0],self.img2.shape[1],self.AllData.PD_weights.shape[-1]))
        

        
        self.AllData.PD_weights=np.zeros((self.img2.shape[0],self.img2.shape[1],self.PigNum))
        for i in range(self.PigNum):
            name=self.path_prefix+"-"+str(self.PigNum)+"-PD_mixing-weights_map-%02d.png"%i
            self.AllData.PD_weights[:,:,i]=np.asarray(Image.open(name).convert('L'))/255.0

        print self.AllData.PD_weights.shape
        self.AllData.PD_weights=self.AllData.PD_weights[:,:,self.AllData.Pig_order] ### reorder 



    def OpenFile6(self):
        
        # ### reading txt file is slow
        # name=self.path_prefix+"-"+str(self.PigNum)+"-PD_layers-order1-opacities.txt"
        # self.AllData.PD_opacities=np.loadtxt(name)
        # self.AllData.PD_opacities=self.AllData.PD_opacities.reshape((self.img2.shape[0],self.img2.shape[1],self.AllData.PD_opacities.shape[-1]))
        
        self.AllData.PD_opacities=np.zeros((self.img2.shape[0],self.img2.shape[1],self.PigNum))
        for i in range(self.PigNum):
            name=self.path_prefix+"-"+str(self.PigNum)+"-PD_layers-order1-opacities_map-%02d.png"%i
            self.AllData.PD_opacities[:,:,i]=np.asarray(Image.open(name).convert('L'))/255.0


        print self.AllData.PD_opacities.shape


    def OpenFile7(self):
        name=self.directory+"/Existing_KS_parameter_KS.txt"
        self.AllData.Existing_KS=np.loadtxt(name)
        print self.AllData.Existing_KS.shape 



  






    def Grabcut(self):
        global grabcut_app
        grabcut_app=Grabcut_app(self)
        


    def Copy_Paste_Insert_Delete(self):
        self.mask=grabcut_app.return_results()
        copy_paste_insert_delete_app=Copy_Paste_Insert_Delete_app(self)
        

    def Recoloring(self): ### can local or global
        recoloring_app=Recoloring_app(self)
    
    def Thickness_Scattering_Weights_modification(self):
        thickness_scattering_weights_modification_app=Thickness_Scattering_Weights_modification_app(self)

    def Local_Alpha_Matting(self):
        local_alpha_matte_app=Local_Alpha_Matting_app(self)

  
    def OpenFile_version2(self): ### open a unprocessed image only, for extracting palette from given image
        # import step1_ANLS_with_autograd
        ### only parameter is pignum (4-7) and other parameters are fixed for GUI.
        pass






if __name__ == "__main__":

    # base_dir = os.path.split( os.path.realpath(__file__) )[0]

    root=tk.Tk()
    app = Main_app(root)
    root.mainloop()



