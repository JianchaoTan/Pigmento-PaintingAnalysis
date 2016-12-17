import numpy as np
import PIL.Image as Image
import sys,os

from Constant_Values import *
from KS_helper import *

def KM_mixing_rendering(H, W, img):
    shape=img.shape

    N=shape[0]*shape[1]
    M=H.shape[0]
    L=H.shape[1]/2

    Weights=W.reshape((N,M))
    Weights=Weights/Weights.sum(axis=1).reshape((-1,1))

    K0=H[:,:L]
    S0=H[:,L:]
    
    K=np.dot(Weights,K0) ### N*L shape,  per pixel K
    S=np.dot(Weights,S0)

    ### reconstruction of input
    R_vector=equations_in_RealPigments(K, S, r=np.ones((N,L)), h=np.ones((N,1))) ## r should be (N*L)shape. h should be (N*1)shape
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    R_rgb=R_rgb.reshape(shape) ### reshape to same shape as target img.
    return (R_rgb*255).round().astype(np.uint8)



###command: python KM_rendering.py  img.png  pigments.txt  mixing_weights.txt

if __name__=="__main__":

	img_name=sys.argv[1] #### only used to give image shape.
	pigments_KS_name=sys.argv[2] #### primary pigments KS.
	weights_name=sys.argv[3] #### mixing weights txt file.

	img=np.asarray(Image.open(img_name).convert('RGB'))
	H=np.loadtxt(pigments_KS_name)
	Weights=np.loadtxt(weights_name) 

	rendered_img=KM_mixing_rendering(H,Weights,img)  
	Image.fromarray(rendered_img).save(os.path.splitext(img_name)[0]+"-mixing_rendered_img.png")







