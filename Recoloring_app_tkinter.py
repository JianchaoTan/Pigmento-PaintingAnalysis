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





def KM_mixing_rendering_global(img, Weights, KS, normalize_flag=1):

    N=Weights.shape[0]
    L=KS.shape[1]/2
    K0=KS[:,:L]
    S0=KS[:,L:]

    if normalize_flag==0:
    	thickness=Weights.sum(axis=1).reshape((-1,1))
    	print thickness.max()
    else:
    	thickness=1.0

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, Weights, r=1.0, h=thickness) ### there are nomalization inside this function.
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



def KM_mixing_rendering_with_mask(KM_reconstructed, Weights, KS, mask, normalize_flag=1):

	N=Weights.shape[0]
	L=KS.shape[1]/2
	K0=KS[:,:L]
	S0=KS[:,L:]


	


	nonzeros_inds=np.nonzero(mask.reshape(-1))[0]


	if normalize_flag==0:
		thickness=Weights[nonzeros_inds,:].sum(axis=1).reshape((-1,1))
	else:
		thickness=1.0

	### reconstruction of input
	R_vector=KM_mixing_multiplepigments(K0, S0, Weights[nonzeros_inds,:], r=1.0, h=thickness)
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







def KM_recoloring(img, KS, weights, KM_reconstructed, global_flag=1, mask_flag=0, mask=None, mouse_flag=0, mouse_mask=None):
    

    if global_flag==1:
    	output = KM_mixing_rendering_global(img, weights, KS)

    elif mask_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS, mask)

    elif mouse_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS, mouse_mask)

    return output




def PD_recoloring(img, color_vertex, weights, PD_reconstructed, global_flag=1, mask_flag=0, mask=None, mouse_flag=0, mouse_mask=None):



	if global_flag==1:
		output=PD_mixing_rendering_global(img, weights, color_vertex)
	elif mask_flag==1:
		output=PD_mixing_rendering_with_mask(PD_reconstructed, weights, color_vertex, mask)
	elif mouse_flag==1:
		output=PD_mixing_rendering_with_mask(PD_reconstructed, weights, color_vertex, mouse_mask)

	return output







def KM_change_weights(img, KS, weights, KM_reconstructed, global_flag=1, mask_flag=0, mask=None, mouse_flag=0, mouse_mask=None, normalize_flag=1):
    
    if global_flag==1:
    	output = KM_mixing_rendering_global(img, weights, KS, normalize_flag=normalize_flag)

    elif mask_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS, mask, normalize_flag=normalize_flag)

    elif mouse_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS, mouse_mask, normalize_flag=normalize_flag)

    return output


def KM_change_scattering(img, KS, weights, KM_reconstructed, global_flag=1, mask_flag=0, mask=None, mouse_flag=0, mouse_mask=None):

    if global_flag==1:
    	output = KM_mixing_rendering_global(img, weights, KS)

    elif mask_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS, mask)

    elif mouse_flag==1:
    	output = KM_mixing_rendering_with_mask(KM_reconstructed, weights, KS, mouse_mask)

    return output





class Recoloring_app:

	def __init__(self, controller):
		self.master=controller.master
		self.canvas=controller.canvas
		self.mask=controller.mask

		if self.mask is None:
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
		self.mouse_mask=np.zeros((self.img2.shape[:2]), dtype=np.uint8)
		self.x = self.y = 0
		self.rect = None
		self.start_x = 0
		self.start_y = 0
		self.thickness=8
		self.showing=0

		self.new_master=None
		self.new_canvas=None

		self.PigNum=self.AllData.KM_pigments.shape[0]

		self.KM_reconstructed = KM_mixing_rendering_global(self.img2, self.AllData.KM_weights.reshape((-1,self.PigNum)), self.AllData.KM_pigments)/255.0#### 0 to 255
		self.PD_reconstructed = PD_mixing_rendering_global(self.img2, self.AllData.PD_weights.reshape((-1,self.PigNum)), self.AllData.PD_vertices)#### 0 to 255
		self.KM_reconstructed_copy=self.KM_reconstructed.copy()
		self.PD_reconstructed_copy=self.PD_reconstructed.copy()

		self.Existing_rgb=KM_pigments_to_RGB_color(self.AllData.Existing_KS[:,:L],self.AllData.Existing_KS[:,L:]) *255.0

		#### should be same as self.AllData.PD_vertices.
		self.Extracted_rgb=(KM_pigments_to_RGB_color(self.AllData.KM_pigments[:,:L],self.AllData.KM_pigments[:,L:]) *255.0).round().astype(np.uint8)
		self.Extracted_rgb_copy=self.Extracted_rgb.copy()
		self.Extracted_KS=self.AllData.KM_pigments.copy()
		self.Extracted_KS_copy=self.Extracted_KS.copy()

		self.KM_weights=self.AllData.KM_weights.copy()
		self.PD_weights=self.AllData.PD_weights.copy()


		self.user_select_indices=np.zeros(self.PigNum, dtype=np.uint8)
		self.user_select_indices_copy=np.zeros(self.PigNum, dtype=np.uint8)
		self.current_Extracted_pigments_index=0




		self.newWindow=tk.Toplevel(self.master)

		self.newWindow.title("Color Transferring Window")
		self.newWindow.geometry("750x640")
		self.shift(self.newWindow)

		self.var3 = IntVar()
		Checkbutton(self.newWindow, text="Use PD version (default is KM version)", variable=self.var3, command=self.update_status).grid(row=0, sticky=W, pady=15)


		self.Labels_extracted_pigments_list=[]
		for i in range(self.PigNum):
		    l=Label(self.newWindow, relief=SOLID, text="     ", bg="#%02x%02x%02x" % tuple(self.Extracted_rgb[i].round().astype(np.uint8)))
		    l.grid(row=1, sticky=W, padx=135+55*(i+1))
		    self.Labels_extracted_pigments_list.append(l)


		self.var_list={}
		for i in range(self.PigNum):
		    self.var_list.setdefault("p-"+str(i), IntVar())

		Label(self.newWindow, text="Pigments-choice").grid(row=2,sticky=W, pady=5)

		self.var_for_all=IntVar()
		Checkbutton(self.newWindow, text="All", variable=self.var_for_all, command=self.update_status1).grid(row=2, sticky=W, padx=125, pady=5)
		for i in range(self.PigNum):
		    Checkbutton(self.newWindow, text="p-"+str(i), variable=self.var_list["p-"+str(i)], command=self.update_status2).grid(row=2, sticky=W, padx=125+55*(i+1), pady=5)


		self.var0 = IntVar()
		Checkbutton(self.newWindow, text="Global", variable=self.var0, command=self.update_txt0).grid(row=4, sticky=W, pady=10)

		self.var1 = IntVar()
		Checkbutton(self.newWindow, text="Local (mask-based)", variable=self.var1, command=self.update_txt1).grid(row=4, sticky=W, padx=100, pady=10)

		self.var2 = IntVar()
		Checkbutton(self.newWindow, text="Local (mouse-based) ", variable=self.var2, command=self.update_txt2).grid(row=4, sticky=W, padx=275, pady=10)

		self.scribble_size=IntVar()
		Label(self.newWindow, text="Scribble_size").grid(row=4,sticky=W, padx=480, pady=10)
		self.scribble_size=Scale(self.newWindow, from_=2, to=50, orient=HORIZONTAL)
		self.scribble_size.grid(row=4, sticky=W, padx=600, pady=10)
		self.scribble_size.set(50)

		self.var_change_pig = IntVar()
		Checkbutton(self.newWindow, text="1. Change Pigments", variable=self.var_change_pig, command=self.update_change_pigments).grid(row=5, sticky=W, pady=10)


		Label(self.newWindow, text="Existing_pigments_color:").grid(row=6,sticky=W, pady=10)
		self.Labels_existing_pigments_list=[]
		for i in range(len(self.Existing_rgb)):
		    l=Label(self.newWindow, text="%02d"%i, bg="#%02x%02x%02x" % tuple(self.Existing_rgb[i].round().astype(np.uint8)))
		    l.grid(row=6, sticky=W, padx=180+20*i, pady=10)
		    self.Labels_existing_pigments_list.append(l)

		self.Existing_pigments_index=IntVar()
		Label(self.newWindow, text="Existing_pigments_index:").grid(row=7,sticky=W, rowspan=2, pady=10)
		self.Existing_pigments_index=Scale(self.newWindow, length=20*26, from_=0, to=self.AllData.Existing_KS.shape[0]-1, orient=HORIZONTAL, command=self.update_chosed_existing_pig_index)
		self.Existing_pigments_index.grid(row=7, sticky=W, padx=180, rowspan=2)
		self.Existing_pigments_index.set(0)

		self.KS_scales=IntVar()
		Label(self.newWindow, text="KS_scales").grid(row=10,sticky=W, rowspan=2, pady=10)
		self.KS_scales=Scale(self.newWindow, from_=0, to=10, orient=HORIZONTAL)
		self.KS_scales.grid(row=10, sticky=W, padx=180, rowspan=2)
		self.KS_scales.set(5)



		self.var_change_weights = IntVar()
		Checkbutton(self.newWindow, text="2. Change Weights", variable=self.var_change_weights, command=self.update_change_weights).grid(row=13, sticky=W, pady=10)
		self.weights_scales=IntVar()
		Label(self.newWindow, text="Weights_scales").grid(row=14,sticky=W, rowspan=2, pady=10)
		self.weights_scales=Scale(self.newWindow, from_=0, to=10, orient=HORIZONTAL)
		self.weights_scales.grid(row=14, sticky=W, padx=180, rowspan=2)
		self.weights_scales.set(5)

		self.normalize_flag=IntVar()
		Checkbutton(self.newWindow, text="Normalize", variable=self.normalize_flag).grid(row=14, sticky=W, padx=380, pady=10)




		self.var_change_scattering = IntVar()
		Checkbutton(self.newWindow, text="3. Change Scattering", variable=self.var_change_scattering, command=self.update_change_scattering).grid(row=17, sticky=W, pady=10)
		self.scatter_scales=IntVar()
		Label(self.newWindow, text="Scatter_scales").grid(row=18,sticky=W, rowspan=2, pady=10)
		self.scatter_scales=Scale(self.newWindow, from_=0, to=20, orient=HORIZONTAL)
		self.scatter_scales.grid(row=18, sticky=W, padx=180, rowspan=2)
		self.scatter_scales.set(10)


		Button(self.newWindow, text='Execute', command=self.Execute).grid(row=21, sticky=W, pady=15)

		Button(self.newWindow, text='Reset', command=self.Reset).grid(row=21, sticky=W, padx=120, pady=15)

		Button(self.newWindow, text='Save', command=self.save_as).grid(row=21, sticky=W, padx=240, pady=15)

		Button(self.newWindow, text='Save palette', command=self.save_palette).grid(row=21, sticky=W, padx=360, pady=15)

		Button(self.newWindow, text='Quit', command=self.Quit).grid(row=21, sticky=W, padx=540, pady=15)



	def update_change_scattering(self):
		if self.var_change_scattering.get()==1:
			self.var_change_pig.set(0)
			self.var_change_weights.set(0)
			self.var3.set(0) ### can only used in KM version.

	def update_change_weights(self):


		if self.var_change_weights.get()==1:
			self.var_change_scattering.set(0)
			self.var_change_pig.set(0)
			self.var3.set(0) ### can only used in KM version.


	def update_change_pigments(self):
		if self.var_change_pig.get()==1:
			self.var_change_scattering.set(0)
			self.var_change_weights.set(0)



	def update_chosed_existing_pig_index(self, args):
		for i in range(len(self.Existing_rgb)):
			if i !=self.Existing_pigments_index.get():
				self.Labels_existing_pigments_list[i].config(relief=FLAT)
			else:
				self.Labels_existing_pigments_list[i].config(relief=SOLID)

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

	def save_palette(self):
	    hen = asksaveasfilename(defaultextension = '.png')
	    Image.fromarray(self.Extracted_rgb.reshape((1,-1,3))).save(hen)



	def update_status(self):
		pass

	def update_status1(self):


		if self.var_for_all.get()==1:
		    for i in range(len(self.var_list)):
		        self.var_list['p-'+str(i)].set(1)

		if self.var_for_all.get()==0:
		    for i in range(len(self.var_list)):
		        self.var_list['p-'+str(i)].set(0)

		self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
		self.user_select_indices_copy=self.user_select_indices.copy()
		# print self.user_select_indices
		# print self.user_select_indices_copy


	def update_status2(self):


		self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
		

		if self.var_change_pig.get()==1:  ## this is needed by "recolor by changing pigments".
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




		if (self.user_select_indices!=0).any() or (self.user_select_indices==0).all():
		    self.var_for_all.set(0)
		if (self.user_select_indices!=0).all():
		    self.var_for_all.set(1)


        









	def Reset(self):
		self.img2=np.asarray(self.im)
		# self.Show_image(self.master, self.im, option=1)

		self.var0.set(0)
		self.var1.set(0)
		self.var2.set(0)
		self.var3.set(0)
		self.var_for_all.set(0)
		self.var_change_weights.set(0)
		self.var_change_pig.set(0)
		self.var_change_scattering.set(0)
		self.normalize_flag.set(0)


		for i in range(self.PigNum):
		    self.var_list["p-"+str(i)].set(0)

		self.showing=0

		if self.new_master is not None:
			self.new_master.destroy() ##close results window

		### reshow origianl extracted primary color.
		for i in range(self.PigNum):
			self.Labels_extracted_pigments_list[i].config(bg="#%02x%02x%02x" % tuple(self.Extracted_rgb_copy[i].round().astype(np.uint8)))

		self.Existing_pigments_index.set(0)
		self.KS_scales.set(5)
		self.weights_scales.set(5)
		self.scatter_scales.set(10)

		self.Extracted_KS=self.AllData.KM_pigments.copy()
		self.Extracted_rgb=self.Extracted_rgb_copy.copy()

		self.user_select_indices=np.zeros(self.PigNum, dtype=np.uint8)
		self.user_select_indices_copy=np.zeros(self.PigNum, dtype=np.uint8)
		self.current_Extracted_pigments_index=0

		self.mouse_mask=np.zeros((self.img2.shape[:2]), dtype=np.uint8)
		self.KM_reconstructed=self.KM_reconstructed_copy.copy()
		self.PD_reconstructed=self.PD_reconstructed_copy.copy()

		self.KM_weights=self.AllData.KM_weights.copy()
		self.PD_weights=self.AllData.PD_weights.copy()

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

		if self.var_change_pig.get()==1: ### recoloring by changing pigments.

			# print self.current_Extracted_pigments_index
			new_scale=2**(self.KS_scales.get()-5)
			self.Extracted_KS[self.current_Extracted_pigments_index,:]=self.AllData.Existing_KS[self.Existing_pigments_index.get(),:].copy()*new_scale

			#### update color of primary pigments:
			self.Extracted_rgb[self.current_Extracted_pigments_index]=(KM_pigments_to_RGB_color(self.Extracted_KS[self.current_Extracted_pigments_index,:L].reshape((1,-1)), self.Extracted_KS[self.current_Extracted_pigments_index,L:].reshape((1,-1)))*255).round().astype(np.uint8)

			for i in range(self.PigNum):
				self.Labels_extracted_pigments_list[i].config(bg="#%02x%02x%02x" % tuple(self.Extracted_rgb[i].round().astype(np.uint8)))


			if self.var3.get()==0:  ### KM version

				output=KM_recoloring(self.img2, 
									 self.Extracted_KS,
									 self.KM_weights.reshape((-1,self.PigNum)), 
									 self.KM_reconstructed,
									 global_flag=self.var0.get(), 
									 mask_flag=self.var1.get(), 
									 mask=self.mask, 
									 mouse_flag=self.var2.get(), 
									 mouse_mask=self.mouse_mask,
									)


			elif self.var3.get()==1: #### PD version
				
				output=PD_recoloring(self.img2,  
									 self.Extracted_rgb,
									 self.PD_weights.reshape((-1,self.PigNum)), 
									 self.PD_reconstructed, 
									 global_flag=self.var0.get(), 
									 mask_flag=self.var1.get(), 
									 mask=self.mask, 
									 mouse_flag=self.var2.get(), 
									 mouse_mask=self.mouse_mask
									 )


			self.results_img=output.copy()

			if self.var2.get()==1:
				self.Show_results(output)
			else:
				self.Show_results_2(output)

		elif self.var_change_weights.get()==1: ### recoloring by changing weights.

			new_scale=2**(self.weights_scales.get()-5)
			print new_scale
			user_chosed_ind_list=np.nonzero(self.user_select_indices)[0]
			print user_chosed_ind_list
			print self.normalize_flag.get()

			self.KM_weights.reshape((-1,self.PigNum))[:,user_chosed_ind_list]=self.AllData.KM_weights.reshape((-1,self.PigNum))[:,user_chosed_ind_list]*new_scale

			if self.var3.get()==0:  ### KM version

				output=KM_change_weights(self.img2, 
									 self.Extracted_KS,
									 self.KM_weights.reshape((-1,self.PigNum)), 
									 self.KM_reconstructed,
									 global_flag=self.var0.get(), 
									 mask_flag=self.var1.get(), 
									 mask=self.mask,
								     mouse_flag=self.var2.get(), 
									 mouse_mask=self.mouse_mask,
									 normalize_flag=self.normalize_flag.get()
									)


			self.results_img=output.copy()

			if self.var2.get()==1:
				self.Show_results(output)
			else:
				self.Show_results_2(output)



		elif self.var_change_scattering.get()==1: ### recoloring by changing scattering.

			new_scale=2**(self.scatter_scales.get()-10)

			user_chosed_ind_list=np.nonzero(self.user_select_indices)[0]
			print self.user_select_indices
			print new_scale
			print user_chosed_ind_list

			self.Extracted_KS[user_chosed_ind_list,L:]=self.AllData.KM_pigments[user_chosed_ind_list,L:]*new_scale

			#### update color of primary pigments:
			self.Extracted_rgb[user_chosed_ind_list,:]=(KM_pigments_to_RGB_color(self.Extracted_KS[user_chosed_ind_list,:L].reshape((-1,L)), self.Extracted_KS[user_chosed_ind_list,L:].reshape((-1,L)))*255).round().astype(np.uint8)

			for i in range(self.PigNum):
				self.Labels_extracted_pigments_list[i].config(bg="#%02x%02x%02x" % tuple(self.Extracted_rgb[i].round().astype(np.uint8)))


			if self.var3.get()==0:  ### KM version

				output=KM_change_scattering(self.img2, 
									 self.Extracted_KS,
									 self.KM_weights.reshape((-1,self.PigNum)), 
									 self.KM_reconstructed,
									 global_flag=self.var0.get(), 
									 mask_flag=self.var1.get(), 
									 mask=self.mask, 
									 mouse_flag=self.var2.get(), 
									 mouse_mask=self.mouse_mask
									)

			self.results_img=output.copy()

			if self.var2.get()==1:
				self.Show_results(output)
			else:
				self.Show_results_2(output)




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

	    # self.new_master.mainloop()
	    self.new_master.update_idletasks()

	def Show_results_2(self, output):
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
	    self.start_x = event.x
	    self.start_y = event.y


	def on_move_press(self, event):
		self.start_x = event.x
		self.start_y = event.y
		#### mouse based recoloring

		cv2.circle(self.mouse_mask,(self.start_x,self.start_y),self.scribble_size.get(),1,-1)
		# print len(self.mouse_mask[self.mouse_mask==1])
		if self.var2.get()==1:
			self.Execute()


	def on_button_release(self, event):
		self.mouse_mask=np.zeros((self.img2.shape[:2]), dtype=np.uint8)
		self.Show_results_2(self.results_img)


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







