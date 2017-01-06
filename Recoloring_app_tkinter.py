# -*- coding: utf-8 -*-
import Tkinter as tk
from Tkinter import *
from tkFileDialog   import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk
import sys,os,io
import numpy as np 
import cv2
from KS_helper import *
import warnings
import json
from Constant_Values import *

L=len(Illuminantnew[:,1])

def KM_pigments_to_RGB_color(K_raw_subarray, S_raw_subarray):
    M=K_raw_subarray.shape[0]
    gt_R=equations_in_RealPigments(K_raw_subarray, S_raw_subarray, r=1.0, h=1.0)
    gt_P=gt_R*Illuminantnew[:,1].reshape((1,-1))
    gt_R_xyz=(gt_P.reshape((M,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    gt_R_xyz/=Normalize ####xyz value shape is M*3
    gt_R_rgb=np.dot(xyztorgb,gt_R_xyz.transpose()).transpose() ###linear rgb value shape is M*3
    gt_R_rgb=Gamma_trans_img(gt_R_rgb.clip(0,1.0))
    return gt_R_rgb



def KM_mixing_rendering_global(img, Weights, KS):

    N=Weights.shape[0]
    L=KS.shape[1]/2
    K0=KS[:,:L]
    S0=KS[:,L:]

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, Weights, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction

    shape=img.shape
    output=R_rgb.reshape((shape[0],shape[1],3))
    return (output*255).round().astype(np.uint8)



def KM_mixing_rendering_with_mask(KM_reconstructed, Weights, KS, mask):

	N=Weights.shape[0]
	L=KS.shape[1]/2
	K0=KS[:,:L]
	S0=KS[:,L:]

	nonzeros_inds=np.nonzero(mask.reshape(-1))[0]

	### reconstruction of input
	R_vector=KM_mixing_multiplepigments(K0, S0, Weights[nonzeros_inds,:], r=1.0, h=1.0)
	### from R spectrum x wavelength spectrums to linear rgb colors 
	P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
	R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
	Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
	R_xyz/=Normalize ####xyz value shape is N*3
	R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
	R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction

	shape=KM_reconstructed.shape
	output=KM_reconstructed.reshape((-1,3))
	output[nonzeros_inds,:]=R_rgb
	output=output.reshape((shape[0],shape[1],3))
	return (output*255).round().astype(np.uint8)


def KM_mixing_rendering_with_mouse_position(KM_reconstructed, Weights, KS, mouse_position):
	pass





def PD_mixing_rendering_global(img, Weights, color_vertex_new):
	shape=img.shape
	output=np.dot(Weights.reshape((shape[0]*shape[1],-1)),color_vertex_new.reshape((-1,3)))
	return output.reshape(shape).round().astype(np.uint8)


def PD_mixing_rendering_with_mask(PD_reconstructed, Weights, color_vertex_new, mask):
	shape=PD_reconstructed.shape
	nonzeros_inds=np.nonzero(mask.reshape(-1))[0]
	output=PD_reconstructed.reshape((-1,3))
	output[nonzeros_inds,:]=np.dot(Weights.reshape((shape[0]*shape[1],-1))[nonzeros_inds,:], color_vertex_new.reshape((-1,3)))

	return output.reshape(shape).round().astype(np.uint8)



def PD_mixing_rendering_with_mouse_position(PD_reconstructed, Weights, color_vertex, color_vertex_new, mouse_position):
	pass






def KM_recoloring(img, KS, KS_gt, weights, KM_reconstructed, index1, index2, scale, global_flag=1, mask_flag=0, mask=None, mouse_flag=0, mouse_position=None):
    
    rows,cols=img.shape[:2]
    new_scale=2**(scale-5)
    KS_copy=KS.copy()
    KS_copy[index1,:]=KS_gt[index2,:].copy()*new_scale

    if global_flag==1:
    	output = KM_mixing_rendering_global(img, weights, KS_copy)

    elif mask_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS_copy, mask)

    elif mouse_flag==1:
    	output = KM_mixing_rendering_with_mouse_position(KM_reconstructed, weights, KS_copy, mouse_position)



    output_expand=np.ones((rows,cols+70,3),dtype=np.uint8)*255
    output_expand[:,:cols,:]=output.copy()

    R_rgb=KM_pigments_to_RGB_color(KS[index1,:L].reshape((1,-1)),KS[index1,L:].reshape((1,-1)))
    output_expand[rows/4:rows/4+50, cols+10:cols+60, :]=(R_rgb*255).round().astype(np.uint8)
    scaled_pigment_RGB=KM_pigments_to_RGB_color(KS_copy[index1,:L].reshape((1,-1)),KS_copy[index1,L:].reshape((1,-1)))
    output_expand[3*rows/4:3*rows/4+50, cols+10:cols+60, :]=(scaled_pigment_RGB*255).round().astype(np.uint8)
    return output_expand




def PD_recoloring(img, color_vertex, existing_rgb, weights, PD_reconstructed, index1, index2, global_flag=1, mask_flag=0, mask=None, mouse_flag=0, mouse_position=None):

	rows,cols=img.shape[:2]
	color_vertex_new=color_vertex.copy()

	color_vertex_new[index1]=existing_rgb[index2]

	if global_flag==1:
		output=PD_mixing_rendering_global(img, weights, color_vertex_new)
	elif mask_flag==1:
		output=PD_mixing_rendering_with_mask(PD_reconstructed, weights, color_vertex_new, mask)
	elif mouse_flag==1:
		output=PD_mixing_rendering_with_mouse_position(PD_reconstructed, weights, color_vertex_new, mouse_position)


	output_expand=np.ones((rows,cols+70,3),dtype=np.uint8)*255
	output_expand[:,:cols,:]=output.copy()
	output_expand[rows/4:rows/4+50, cols+10:cols+60, :]=color_vertex[index1].reshape((1,1,3)).round().astype(np.uint8)
	output_expand[3*rows/4:3*rows/4+50, cols+10:cols+60, :]=existing_rgb[index2].reshape((1,1,3)).round().astype(np.uint8)
	return output_expand




class Recoloring_app:

	def __init__(self, controller):
		self.master=controller.master
		self.canvas=controller.canvas
		self.mask=controller.mask

		if self.mask==None:
		    print "no input mask"
		else:
		    print self.mask.shape

		self.AllData=controller.AllData

		self.canvas.pack(side="top", fill="both", expand=True)
		self.canvas.bind("<ButtonPress-1>", self.on_button_press)
		self.canvas.bind("<B1-Motion>", self.on_move_press)
		self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

		self.im=controller.im
		self.img2=np.asarray(self.im)
		self.x = self.y = 0
		self.rect = None
		self.start_x = 0
		self.start_y = 0
		self.thickness=8
		self.showing=0

		self.new_master=None
		self.new_canvas=None



		self.newWindow=tk.Toplevel(self.master)

		self.newWindow.title("Color Transferring Window")
		self.newWindow.geometry("700x400")
		self.shift(self.newWindow)

		self.var3 = IntVar()
		Checkbutton(self.newWindow, text="Use PD version (default is KM version)", variable=self.var3, command=self.update_status).grid(row=0, sticky=W, pady=15)

		self.PigNum=self.AllData.KM_pigments.shape[0]


		for i in range(self.PigNum):
		    Label(self.newWindow, relief=SOLID, text="     ", bg="#%02x%02x%02x" % tuple(self.AllData.PD_vertices[i].round().astype(np.uint8))).grid(row=1, sticky=W, padx=135+55*(i+1))

		self.var_list={}
		for i in range(self.PigNum):
		    self.var_list.setdefault("p-"+str(i), IntVar())

		Label(self.newWindow, text="Pigments-choice").grid(row=2,sticky=W, pady=5)
		self.var_for_all=IntVar()
		Checkbutton(self.newWindow, text="Accumulate", variable=self.var_for_all, command=self.update_status3).grid(row=2, sticky=W, padx=125, pady=5)

		for i in range(self.PigNum):
		    Checkbutton(self.newWindow, text="p-"+str(i), variable=self.var_list["p-"+str(i)], command=self.update_status2).grid(row=2, sticky=W, padx=125+55*(i+1), pady=5)


		self.var0 = IntVar()
		Checkbutton(self.newWindow, text="Global", variable=self.var0, command=self.update_txt0).grid(row=4, sticky=W, pady=10)

		self.var1 = IntVar()
		Checkbutton(self.newWindow, text="Local (mask-based)", variable=self.var1, command=self.update_txt1).grid(row=4, sticky=W, padx=200, pady=10)

		self.var2 = IntVar()
		Checkbutton(self.newWindow, text="Local (mouse-based) ", variable=self.var2, command=self.update_txt2).grid(row=4, sticky=W, padx=400, pady=5)


		self.Existing_pigments_index=IntVar()
		Label(self.newWindow, text="Existing_pigments_index").grid(row=6,sticky=W, rowspan=2, pady=10)
		self.Existing_pigments_index=Scale(self.newWindow, from_=0, to=self.AllData.Existing_KS.shape[0]-1, orient=HORIZONTAL)
		self.Existing_pigments_index.grid(row=6, sticky=W, padx=180, rowspan=2)
		self.Existing_pigments_index.set(0)

		self.KS_scales=IntVar()
		Label(self.newWindow, text="KS_scales").grid(row=9,sticky=W, rowspan=2, pady=10)
		self.KS_scales=Scale(self.newWindow, from_=0, to=10, orient=HORIZONTAL)
		self.KS_scales.grid(row=9, sticky=W, padx=180, rowspan=2)
		self.KS_scales.set(5)



		Button(self.newWindow, text='Execute', command=self.Execute).grid(row=14, sticky=W, pady=15)

		Button(self.newWindow, text='Reset', command=self.Reset).grid(row=14, sticky=W, padx=120, pady=15)

		Button(self.newWindow, text='Save', command=self.save_as).grid(row=14, sticky=W, padx=240, pady=15)

		Button(self.newWindow, text='Quit', command=self.Quit).grid(row=14, sticky=W, padx=360, pady=15)

		self.KM_reconstructed = KM_mixing_rendering_global(self.img2, self.AllData.KM_weights.reshape((-1,self.PigNum)), self.AllData.KM_pigments)/255.0#### 0 to 255
		self.PD_reconstructed = PD_mixing_rendering_global(self.img2, self.AllData.PD_weights.reshape((-1,self.PigNum)), self.AllData.PD_vertices)#### 0 to 255

		self.Existing_rgb=KM_pigments_to_RGB_color(self.AllData.Existing_KS[:,:L],self.AllData.Existing_KS[:,L:]) *255.0



	def update_txt0(self):
		if self.var0.get()==1:
		    self.var1.set(0)
		    self.var2.set(0)

	def update_txt1(self):
		if self.var1.get()==1:
		    self.var0.set(0)
		    self.var2.set(0)


	def update_txt2(self):
		if self.var2.get()==1:
		    self.var0.set(0)
		    self.var1.set(0)





	def update_status(self):
	    pass


	def update_status2(self):

		for i in range(self.PigNum):
			if self.var_list['p-'+str(i)].get()==1:
				self.Extracted_pigments_index=i
				for j in range(self.PigNum):
					if j!=i:
						self.var_list['p-'+str(j)].set(0)

        	

	def update_status3(self):
		pass







	def Reset(self):
	    self.img2=np.asarray(self.im)
	    self.Show_image(self.master, self.im, option=1)
	    
	    self.var0.set(0)
	    self.var1.set(0)
	    self.var2.set(0)
	    self.var3.set(0)

	    self.var_for_all.set(0)

	    for i in range(self.PigNum):
	        self.var_list["p-"+str(i)].set(0)

	    self.showing=0

	    self.new_master.destroy() ##close results window



	def save_as(self):
	    if self.new_canvas!=None:
	        ps=self.new_canvas.postscript(colormode='color')
	        hen = asksaveasfilename(defaultextension = '.png')
	        im = Image.open(io.BytesIO(ps.encode('utf-8')))
	        im.save(hen)


	def Quit(self):
	    self.Reset()
	    self.newWindow.destroy()  ### close application window.



	def Show_image(self, master, img, option=0):
	    
	    width,height=img.size
	    # print width, height
	    if option==0:
	        self.canvas = tk.Canvas(master, width=width, height=height, cursor="cross")
	        self.canvas.pack(side="top", fill="both", expand=True)
	    self.tk_im = ImageTk.PhotoImage(img)
	    self.canvas.create_image(0,0,anchor="nw",image=self.tk_im)



	def Execute(self):

		print self.Extracted_pigments_index

		if self.var3.get()==0:  ### KM version

			output=KM_recoloring(self.img2, 
								 self.AllData.KM_pigments, 
								 self.AllData.Existing_KS, 
								 self.AllData.KM_weights.reshape((-1,self.PigNum)), 
								 self.KM_reconstructed, 
								 self.Extracted_pigments_index, 
								 self.Existing_pigments_index.get(), 
								 self.KS_scales.get(), 
								 global_flag=self.var0.get(), 
								 mask_flag=self.var1.get(), 
								 mask=self.mask, 
								 mouse_flag=self.var2.get(), 
								 mouse_position=(self.start_y, self.start_x)
								 )


		elif self.var3.get()==1: #### PD version
			
			output=PD_recoloring(self.img2, 
								 self.AllData.PD_vertices, 
								 self.Existing_rgb,
								 self.AllData.PD_weights.reshape((-1,self.PigNum)), 
								 self.PD_reconstructed, 
								 self.Extracted_pigments_index, 
								 self.Existing_pigments_index.get(), 
								 global_flag=self.var0.get(), 
								 mask_flag=self.var1.get(), 
								 mask=self.mask, 
								 mouse_flag=self.var2.get(), 
								 mouse_position=(self.start_y, self.start_x)
								 )


		self.Show_results(output)



	def Show_results(self, output):
	    # #### show results in other window.
	    img=Image.fromarray(output)
	    width,height=img.size

	    if self.showing==0:
	        self.new_master=tk.Toplevel()
	        self.new_master.title('Results')
	        self.new_canvas = tk.Canvas(self.new_master, width=width, height=height, cursor="cross")
	        self.new_canvas.pack(side="top", fill="both", expand=True)
	        self.showing=1

	    tk_im = ImageTk.PhotoImage(img)
	    self.new_canvas.create_image(0,0,anchor="nw",image=tk_im)
	    self.shift(self.new_master, position='center', scale=1.0)
	    self.new_master.mainloop()




	def on_button_press(self, event):
	    # save mouse drag start position
	    self.dst_x = event.x
	    self.dst_y = event.y


	def on_move_press(self, event):
	    pass

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







