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



def Alpha_compositing(before, RGB_vertex, opacity):
    after=before*(1-opacity.reshape((-1,1)))+RGB_vertex.reshape((-1,3))*opacity.reshape((-1,1))
    return after


def object_insertion_RGB(img_pigment_RGB, img_opacity, imsize, img_weights, chosed_indices, mask, obj_opacity, central_position_x, central_position_y, insert_onto_layer_index, normalize_flag):
    M=len(img_pigment_RGB)
    img_weights=img_weights.reshape((-1,M))
    img_opacity=img_opacity.reshape((-1,M))
    obj_pixel_RGB=np.dot(img_weights[:,chosed_indices],img_pigment_RGB[chosed_indices,:]).reshape((imsize[0],imsize[1],-1))

    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
    final_mask=np.zeros((imsize[0],imsize[1]),dtype=np.uint8)

    nonzerosinds=np.nonzero(mask)
    x_coord=nonzerosinds[0]
    y_coord=nonzerosinds[1]
    x_center=x_coord.sum()/n
    y_center=y_coord.sum()/n

    x_coord_copy=x_coord.copy()
    y_coord_copy=y_coord.copy()

    x_shift=central_position_x-x_center
    y_shift=central_position_y-y_center

    x_coord+=x_shift
    y_coord+=y_shift


    final_obj_RGB=[]
    for i in range(n):
        x=x_coord[i]
        y=y_coord[i]
        if x>=0 and x<imsize[0] and y>=0 and y<imsize[1]:
            final_mask[x,y]=255
            final_obj_RGB.append(list(obj_pixel_RGB[x-x_shift,y-y_shift,:]))


    final_obj_RGB=np.asarray(final_obj_RGB)

    final_mask_reverse=np.ones(final_mask.shape,dtype=np.uint8)*255
    final_mask_reverse[final_mask==255]=0
    final_mask=final_mask.reshape(-1)
    final_mask_reverse=final_mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(final_mask)[0]
    final_nonzeros_inds2=np.nonzero(final_mask_reverse)[0]

    

    new_img_opacity=img_opacity.copy()
    final_obj_opacity=np.ones((len(final_nonzeros_inds1),1))*obj_opacity
    
    if normalize_flag==1:
        total=new_img_opacity[final_nonzeros_inds1,:].sum(axis=1)+final_obj_opacity.sum(axis=1)
        total=total.reshape((-1,1))

        new_img_opacity[final_nonzeros_inds1,:]=new_img_opacity[final_nonzeros_inds1,:]/total
        final_obj_opacity=final_obj_opacity/total




    output=np.ones((N,3))*255 #### can start from any background, because first layer will be opaque.

    for i in range(insert_onto_layer_index+1):
        output=Alpha_compositing(output, img_pigment_RGB[i:i+1,:], new_img_opacity[:,i:i+1])

    output[final_nonzeros_inds1,:]=Alpha_compositing(output[final_nonzeros_inds1,:], final_obj_RGB, final_obj_opacity)

    for i in range(insert_onto_layer_index+1,M):
        output=Alpha_compositing(output, img_pigment_RGB[i:i+1,:], new_img_opacity[:,i:i+1])

    output=output.reshape((imsize[0],imsize[1],3))
    return output.round().astype(np.uint8)







def object_insertion_KS(img_pigment_KS, img_thickness, imsize, img_weights, chosed_indices , mask, obj_thickness, central_position_x, central_position_y, insert_onto_layer_index, normalize_flag):
   
    M=len(img_pigment_KS)
    img_weights=img_weights.reshape((-1,M))
    img_thickness=img_thickness.reshape((-1,M))
   
    obj_pixel_KS=np.dot(img_weights[:,chosed_indices],img_pigment_KS[chosed_indices,:]).reshape((imsize[0],imsize[1],-1))
    

    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
   
    final_mask=np.zeros((imsize[0],imsize[1]),dtype=np.uint8)

    nonzerosinds=np.nonzero(mask)
    x_coord=nonzerosinds[0]
    y_coord=nonzerosinds[1]
    x_center=x_coord.sum()/n
    y_center=y_coord.sum()/n
    
    x_coord_copy=x_coord.copy()
    y_coord_copy=y_coord.copy()

    x_shift=central_position_x-x_center
    y_shift=central_position_y-y_center

    x_coord+=x_shift
    y_coord+=y_shift


    final_obj_KS=[]
    for i in range(n):
        x=x_coord[i]
        y=y_coord[i]
        if x>=0 and x<imsize[0] and y>=0 and y<imsize[1]:
            final_mask[x,y]=255
            final_obj_KS.append(list(obj_pixel_KS[x-x_shift,y-y_shift,:]))

    final_obj_KS=np.asarray(final_obj_KS)


    final_mask_reverse=np.ones(final_mask.shape,dtype=np.uint8)*255
    final_mask_reverse[final_mask==255]=0
    final_mask=final_mask.reshape(-1)
    final_mask_reverse=final_mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(final_mask)[0]
    final_nonzeros_inds2=np.nonzero(final_mask_reverse)[0]

    

    new_img_thickness=img_thickness.copy()
    final_obj_thickness=np.ones((len(final_nonzeros_inds1),1))*obj_thickness
    
    if normalize_flag==1:
        total=new_img_thickness[final_nonzeros_inds1,:].sum(axis=1)+final_obj_thickness.sum(axis=1)
        total=total.reshape((-1,1))

        new_img_thickness[final_nonzeros_inds1,:]=new_img_thickness[final_nonzeros_inds1,:]/total
        final_obj_thickness=final_obj_thickness/total



    L=img_pigment_KS.shape[1]/2
    K0=img_pigment_KS[:,:L]
    S0=img_pigment_KS[:,L:]

    R_vector=np.ones((N,L)) #### pure white background

    for i in range(insert_onto_layer_index+1):
        R_vector=equations_in_RealPigments(K0[i:i+1,:], S0[i:i+1,:], R_vector, new_img_thickness[:,i:i+1])

    R_vector[final_nonzeros_inds1,:]=equations_in_RealPigments(final_obj_KS[:,:L], final_obj_KS[:,L:], R_vector[final_nonzeros_inds1,:], final_obj_thickness)

    for i in range(insert_onto_layer_index+1,M):
        R_vector=equations_in_RealPigments(K0[i:i+1,:], S0[i:i+1,:], R_vector, new_img_thickness[:,i:i+1])


    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    output=R_rgb.reshape((imsize[0],imsize[1],3))
    return (output*255).round().astype(np.uint8)





def object_paste_RGB(img_pigment_RGB, img_weights, imsize, chosed_indices, mask, central_position_x, central_position_y, normalize_flag, scales):
    
    M=len(img_pigment_RGB)
    img_weights=img_weights.reshape((-1,M))

    obj_weights=np.zeros(img_weights.shape)
    obj_weights[:,chosed_indices]=img_weights[:,chosed_indices]

    obj_weights=obj_weights.reshape((imsize[0],imsize[1],-1))
    


    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])

    final_mask=np.zeros((imsize[0],imsize[1]),dtype=np.uint8)

    nonzerosinds=np.nonzero(mask)
    x_coord=nonzerosinds[0]
    y_coord=nonzerosinds[1]
    x_center=x_coord.sum()/n
    y_center=y_coord.sum()/n

    x_coord_copy=x_coord.copy()
    y_coord_copy=y_coord.copy()

    x_shift=central_position_x-x_center
    y_shift=central_position_y-y_center

    x_coord+=x_shift
    y_coord+=y_shift


    final_obj_weights=[]
    for i in range(n):
        x=x_coord[i]
        y=y_coord[i]
        if x>=0 and x<imsize[0] and y>=0 and y<imsize[1]:
            final_mask[x,y]=255
            final_obj_weights.append(list(obj_weights[x-x_shift,y-y_shift,:]))


    final_obj_weights=np.asarray(final_obj_weights)


    final_mask_reverse=np.ones(final_mask.shape,dtype=np.uint8)*255
    final_mask_reverse[final_mask==255]=0
    final_mask=final_mask.reshape(-1)
    final_mask_reverse=final_mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(final_mask)[0]
    final_nonzeros_inds2=np.nonzero(final_mask_reverse)[0]




    output=np.ones((N,3))*255 #### starting from white background.

    ### unmasked area is same as before
    output[final_nonzeros_inds2,:]=np.dot(img_weights[final_nonzeros_inds2,:],img_pigment_RGB)


    ### change masked area
    new_weights=img_weights[final_nonzeros_inds1,:].reshape((-1,M))+final_obj_weights * scales


    if normalize_flag==1:
        new_weights_sum=new_weights.sum(axis=1).reshape((-1,1))
        new_weights_sum[new_weights_sum==0]=1e-15
        new_weights=new_weights/new_weights_sum

    output[final_nonzeros_inds1,:]=np.dot(new_weights, img_pigment_RGB)



    output=output.reshape((imsize[0],imsize[1],3))
    return output.round().astype(np.uint8)







def object_paste_KS(img_pigment_KS, img_weights, imsize, chosed_indices, mask, central_position_x, central_position_y, normalize_flag, scales):
   
    M=len(img_pigment_KS)
    img_weights=img_weights.reshape((-1,M))
    
    num=len(chosed_indices)
    assert(num>0)

    obj_weights=np.zeros(img_weights.shape)
    obj_weights[:,chosed_indices]=img_weights[:,chosed_indices]
    
    obj_weights=obj_weights.reshape((imsize[0],imsize[1],-1))
    

    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
   
    final_mask=np.zeros((imsize[0],imsize[1]),dtype=np.uint8)

    nonzerosinds=np.nonzero(mask)
    x_coord=nonzerosinds[0]
    y_coord=nonzerosinds[1]
    x_center=x_coord.sum()/n
    y_center=y_coord.sum()/n
    
    x_coord_copy=x_coord.copy()
    y_coord_copy=y_coord.copy()

    x_shift=central_position_x-x_center
    y_shift=central_position_y-y_center

    x_coord+=x_shift
    y_coord+=y_shift


    final_obj_weights=[]
    for i in range(n):
        x=x_coord[i]
        y=y_coord[i]
        if x>=0 and x<imsize[0] and y>=0 and y<imsize[1]:
            final_mask[x,y]=255
            final_obj_weights.append(list(obj_weights[x-x_shift,y-y_shift,:]))

    final_obj_weights=np.asarray(final_obj_weights)


    final_mask_reverse=np.ones(final_mask.shape,dtype=np.uint8)*255
    final_mask_reverse[final_mask==255]=0
    final_mask=final_mask.reshape(-1)
    final_mask_reverse=final_mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(final_mask)[0]
    final_nonzeros_inds2=np.nonzero(final_mask_reverse)[0]
    


    L=img_pigment_KS.shape[1]/2
    K0=img_pigment_KS[:,:L]
    S0=img_pigment_KS[:,L:]


    
    R_vector=np.ones((N,L)) #### pure white background


    ### unmasked area is same as before
    R_vector[final_nonzeros_inds2,:]=equations_in_RealPigments(np.dot(img_weights[final_nonzeros_inds2,:],K0), np.dot(img_weights[final_nonzeros_inds2,:],S0), r=R_vector[final_nonzeros_inds2,:], h=1.0)


    ### change masked area
    new_weights=img_weights[final_nonzeros_inds1,:].reshape((-1,M))+final_obj_weights * scales

    if normalize_flag==1:
        new_weights_sum=new_weights.sum(axis=1).reshape((-1,1))
        new_weights_sum[new_weights_sum==0]=1e-15
        new_weights=new_weights/new_weights_sum

    R_vector[final_nonzeros_inds1,:]=equations_in_RealPigments(np.dot(new_weights,K0), np.dot(new_weights,S0), r=R_vector[final_nonzeros_inds1,:], h=1.0)




    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    output=R_rgb.reshape((imsize[0],imsize[1],3))
    return (output*255).round().astype(np.uint8)



def object_paste_KS_inpaint_based(img_pigment_KS, img_weights, imsize, mask, radius, central_position_x, central_position_y, normalize_flag, scales):
   
    M=img_weights.shape[-1]
    img_weights=img_weights.reshape((imsize[0], imsize[1], -1))
    new_img_weights=(img_weights.copy()*255).round().astype(np.uint8)
    for i in range(M):
        new_img_weights[:,:,i]=cv2.inpaint(new_img_weights[:,:,i], mask, radius, cv2.INPAINT_TELEA)
    

    
    obj_weights=img_weights-new_img_weights/255.0

    new_img_weights=new_img_weights.reshape((-1,M))/255.0
    img_weights=img_weights.reshape((-1,M))



    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
   
    final_mask=np.zeros((imsize[0],imsize[1]),dtype=np.uint8)

    nonzerosinds=np.nonzero(mask)
    x_coord=nonzerosinds[0]
    y_coord=nonzerosinds[1]
    x_center=x_coord.sum()/n
    y_center=y_coord.sum()/n
    
    x_coord_copy=x_coord.copy()
    y_coord_copy=y_coord.copy()

    x_shift=central_position_x-x_center
    y_shift=central_position_y-y_center

    x_coord+=x_shift
    y_coord+=y_shift


    final_obj_weights=[]
    for i in range(n):
        x=x_coord[i]
        y=y_coord[i]
        if x>=0 and x<imsize[0] and y>=0 and y<imsize[1]:
            final_mask[x,y]=255
            final_obj_weights.append(list(obj_weights[x-x_shift,y-y_shift,:]))

    final_obj_weights=np.asarray(final_obj_weights)


    final_mask_reverse=np.ones(final_mask.shape,dtype=np.uint8)*255
    final_mask_reverse[final_mask==255]=0
    final_mask=final_mask.reshape(-1)
    final_mask_reverse=final_mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(final_mask)[0]
    final_nonzeros_inds2=np.nonzero(final_mask_reverse)[0]
    


    L=img_pigment_KS.shape[1]/2
    K0=img_pigment_KS[:,:L]
    S0=img_pigment_KS[:,L:]


    
    R_vector=np.ones((N,L)) #### pure white background


    ### unmasked area is same as before
    R_vector[final_nonzeros_inds2,:]=equations_in_RealPigments(np.dot(img_weights[final_nonzeros_inds2,:],K0), np.dot(img_weights[final_nonzeros_inds2,:],S0), r=R_vector[final_nonzeros_inds2,:], h=1.0)


    ### change masked area
    new_weights=img_weights[final_nonzeros_inds1,:].reshape((-1,M))+final_obj_weights * scales

    if normalize_flag==1:
        new_weights_sum=new_weights.sum(axis=1).reshape((-1,1))
        new_weights_sum[new_weights_sum==0]=1e-15
        new_weights=new_weights/new_weights_sum

    R_vector[final_nonzeros_inds1,:]=equations_in_RealPigments(np.dot(new_weights,K0), np.dot(new_weights,S0), r=R_vector[final_nonzeros_inds1,:], h=1.0)




    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    output=R_rgb.reshape((imsize[0],imsize[1],3))
    return (output*255).round().astype(np.uint8)




def object_remove_KS_inpaint_based(img_pigment_KS, img_weights, imsize, mask, radius, normalize_flag):
    M=img_weights.shape[-1]
    img_weights=img_weights.reshape((imsize[0], imsize[1], -1))
    new_img_weights=(img_weights.copy()*255).round().astype(np.uint8)
    for i in range(M):
        new_img_weights[:,:,i]=cv2.inpaint(new_img_weights[:,:,i], mask, radius, cv2.INPAINT_TELEA)
    new_img_weights=new_img_weights.reshape((-1,M))/255.0
    img_weights=img_weights.reshape((-1,M))

    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
    mask_reverse=np.ones(mask.shape,dtype=np.uint8)*255
    mask_reverse[mask==255]=0
    mask=mask.reshape(-1)
    mask_reverse=mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(mask)[0]
    final_nonzeros_inds2=np.nonzero(mask_reverse)[0]

    L=img_pigment_KS.shape[1]/2
    K0=img_pigment_KS[:,:L]
    S0=img_pigment_KS[:,L:]

    R_vector=np.ones((N,L)) #### pure white background
    
    ###non-masked area is keep same
    R_vector[final_nonzeros_inds2,:]=equations_in_RealPigments(np.dot(img_weights[final_nonzeros_inds2,:],K0), np.dot(img_weights[final_nonzeros_inds2,:],S0), r=R_vector[final_nonzeros_inds2,:], h=1.0)
    

    ### change masked area
    new_weights=new_img_weights[final_nonzeros_inds1,:].reshape((-1,M))
    
    if normalize_flag==1:
        ## Normalize!
        new_weights_sum=new_weights.sum(axis=1).reshape((-1,1))
        new_weights_sum[new_weights_sum==0]=1e-15
        new_weights=new_weights/new_weights_sum

    R_vector[final_nonzeros_inds1,:]=equations_in_RealPigments(np.dot(new_weights,K0), np.dot(new_weights,S0), r=R_vector[final_nonzeros_inds1,:], h=1.0)



    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    output=R_rgb.reshape((imsize[0],imsize[1],3))
    return (output*255).round().astype(np.uint8)
    



def object_remove_KS(img_pigment_KS, img_weights, imsize, mask, chosed_indices, normalize_flag):
    
    M=len(img_pigment_KS)

    all_indices=np.arange(M)

    remain_list=list(set(list(all_indices))-set(list(chosed_indices)))
    
    num=len(remain_list)
    if num!=0:
        remain_indices=np.array(remain_list).reshape(-1) ### indices order will change, but will not affect anything, since weights will not have order.
    

    img_weights=img_weights.reshape((-1,M))
    
    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
    mask_reverse=np.ones(mask.shape,dtype=np.uint8)*255
    mask_reverse[mask==255]=0
    mask=mask.reshape(-1)
    mask_reverse=mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(mask)[0]
    final_nonzeros_inds2=np.nonzero(mask_reverse)[0]

    L=img_pigment_KS.shape[1]/2
    K0=img_pigment_KS[:,:L]
    S0=img_pigment_KS[:,L:]

    R_vector=np.ones((N,L)) #### pure white background
    
    ###non-masked area is keep same
    R_vector[final_nonzeros_inds2,:]=equations_in_RealPigments(np.dot(img_weights[final_nonzeros_inds2,:],K0), np.dot(img_weights[final_nonzeros_inds2,:],S0), r=R_vector[final_nonzeros_inds2,:], h=1.0)
    
    if num!=0:
        ### change masked area
        new_weights=img_weights[final_nonzeros_inds1,:].reshape((-1,M))[:,remain_indices].reshape((-1,num))
        
        if normalize_flag==1:
            ## Normalize!
            new_weights_sum=new_weights.sum(axis=1).reshape((-1,1))
            new_weights_sum[new_weights_sum==0]=1e-15
            new_weights=new_weights/new_weights_sum

        R_vector[final_nonzeros_inds1,:]=equations_in_RealPigments(np.dot(new_weights,K0[remain_indices,:].reshape((num,-1))), np.dot(new_weights,S0[remain_indices,:].reshape((num,-1))), r=R_vector[final_nonzeros_inds1,:], h=1.0)



    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    output=R_rgb.reshape((imsize[0],imsize[1],3))
    return (output*255).round().astype(np.uint8)



def object_remove_RGB(img_pigment_RGB, img_weights, imsize, mask, chosed_indices, normalize_flag):

    M=len(img_pigment_RGB)
    img_weights=img_weights.reshape((-1,M))
    
    all_indices=np.arange(M)
    remain_list=list(set(list(all_indices))-set(list(chosed_indices)))
    
    num=len(remain_list)
    if num!=0:
        remain_indices=np.array(remain_list) ### indices order will change, but will not affect anything, since weights will not have order.


    N=imsize[0]*imsize[1]
    n=len(mask[mask!=0])
    mask_reverse=np.ones(mask.shape,dtype=np.uint8)*255
    mask_reverse[mask==255]=0
    mask=mask.reshape(-1)
    mask_reverse=mask_reverse.reshape(-1)

    final_nonzeros_inds1=np.nonzero(mask)[0]
    final_nonzeros_inds2=np.nonzero(mask_reverse)[0]


    output=np.ones((N,3))*255


    ###non-masked area is keep same
    output[final_nonzeros_inds2,:]=np.dot(img_weights[final_nonzeros_inds2,:],img_pigment_RGB)
    

    if num!=0:

        ### change masked area
        new_weights=img_weights[final_nonzeros_inds1,:].reshape((-1,M))[:,remain_indices].reshape((-1,num))
       
        if normalize_flag==1:
            ## Normalize!
            new_weights_sum=new_weights.sum(axis=1).reshape((-1,1))
            new_weights_sum[new_weights_sum==0]=1e-15
            new_weights=new_weights/new_weights_sum

        output[final_nonzeros_inds1,:]=np.dot(new_weights,img_pigment_RGB[remain_indices,:].reshape((num,-1)))





    output=output.reshape((imsize[0],imsize[1],3))
    return output.round().astype(np.uint8)





class Copy_Paste_Insert_Delete_app:

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
        # self.canvas.bind("<B1-Motion>", self.on_move_press)
        # self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.im=controller.im
        self.img2=np.asarray(self.im)
        self.x = self.y = 0
        self.rect = None
        self.thickness=8
        self.showing=0

        self.new_master=None
        self.new_canvas=None

        

        self.newWindow=tk.Toplevel(self.master)

        self.newWindow.title("Copy_Paste_Insert_Delete Window")
        self.newWindow.geometry("700x500")
        self.shift(self.newWindow)

        self.var3 = IntVar()
        Checkbutton(self.newWindow, text="Use PD version (default is KM version)", variable=self.var3, command=self.update_status).grid(row=0, sticky=W, pady=15)
        

        
        self.PigNum=self.AllData.KM_pigments.shape[0]
        self.user_select_indices=np.zeros(self.PigNum, dtype=np.uint8)


        for i in range(self.PigNum):
            Label(self.newWindow, relief=SOLID, text="     ", bg="#%02x%02x%02x" % tuple(self.AllData.PD_vertices[i].round().astype(np.uint8))).grid(row=1, sticky=W, padx=135+55*(i+1))
        
        self.var_list={}
        for i in range(self.PigNum):
            self.var_list.setdefault("p-"+str(i), IntVar())
        
        Label(self.newWindow, text="Pigments-choice").grid(row=2,sticky=W, pady=5)
        self.var_for_all=IntVar()
        Checkbutton(self.newWindow, text="All", variable=self.var_for_all, command=self.update_status3).grid(row=2, sticky=W, padx=125, pady=5)

        for i in range(self.PigNum):
            Checkbutton(self.newWindow, text="p-"+str(i), variable=self.var_list["p-"+str(i)], command=self.update_status2).grid(row=2, sticky=W, padx=125+55*(i+1), pady=5)


        #### CAF-based:  can use cv2.inpaint, 
        self.var0_0 = IntVar()
        Checkbutton(self.newWindow, text="CopyPaste (CAF-based)", variable=self.var0_0, command=self.update_txt0_0).grid(row=3, sticky=W, pady=10)
        
        self.var0_1 = IntVar()
        Checkbutton(self.newWindow, text="Delete (CAF-based)", variable=self.var0_1, command=self.update_txt0_1).grid(row=3, sticky=W, padx=200, pady=10)
        

        self.radius=IntVar()
        Label(self.newWindow, text="Inpaint_radius").grid(row=3,sticky=W, rowspan=2, padx=400, pady=10)
        self.radius=Scale(self.newWindow, from_=3, to=200, orient=HORIZONTAL)
        self.radius.grid(row=3, sticky=W, padx=525, rowspan=2)
        self.radius.set(55)


        self.var1_0 = IntVar()
        Checkbutton(self.newWindow, text="CopyPaste (Select-based)", variable=self.var1_0, command=self.update_txt1_0).grid(row=4, sticky=W, pady=10)
        
        self.var1_1 = IntVar()
        Checkbutton(self.newWindow, text="Delete (Select-based)", variable=self.var1_1, command=self.update_txt1_1).grid(row=4, sticky=W, padx=200, pady=10)
        
        self.var2 = IntVar()
        Checkbutton(self.newWindow, text="Object Insertion ", variable=self.var2, command=self.update_txt2).grid(row=5, sticky=W, pady=10)
        
        self.normalize_flag = IntVar()
        Checkbutton(self.newWindow, text="Normalize", variable=self.normalize_flag).grid(row=5, sticky=W, padx=200, pady=10)


        self.obj_weights_scales=IntVar()
        Label(self.newWindow, text="Obj_weights_scale").grid(row=6,sticky=W, rowspan=2, pady=10)
        self.obj_weights_scales=Scale(self.newWindow, from_=1, to=100, orient=HORIZONTAL)
        self.obj_weights_scales.grid(row=6, sticky=W, padx=180, rowspan=2)
        self.obj_weights_scales.set(50)



        self.obj_thickness=IntVar()
        Label(self.newWindow, text="Object_thickness").grid(row=8,sticky=W, rowspan=2, pady=10)
        self.obj_thickness=Scale(self.newWindow, from_=1, to=100, orient=HORIZONTAL)
        self.obj_thickness.grid(row=8, sticky=W, padx=180, rowspan=2)
        self.obj_thickness.set(10)
        
        self.obj_opacity=IntVar()
        Label(self.newWindow, text="Object_opacity").grid(row=11,sticky=W, rowspan=2, pady=10)
        self.obj_opacity=Scale(self.newWindow, from_=1, to=10, orient=HORIZONTAL)
        self.obj_opacity.grid(row=11, sticky=W, padx=180, rowspan=2)
        self.obj_opacity.set(10)
        
        self.insert_onto_layer_index=IntVar()
        Label(self.newWindow, text="Insert_onto_layer_index").grid(row=14,sticky=W, rowspan=2, pady=10)
        self.insert_onto_layer_index=Scale(self.newWindow, from_=0, to=self.PigNum-1, orient=HORIZONTAL)
        self.insert_onto_layer_index.grid(row=14, sticky=W, padx=180, rowspan=2)
        self.insert_onto_layer_index.set(self.PigNum-1)


        


        Button(self.newWindow, text='Execute', command=self.Execute).grid(row=17, sticky=W, pady=15)
        
        Button(self.newWindow, text='Reset', command=self.Reset).grid(row=17, sticky=W, padx=120, pady=15)

        Button(self.newWindow, text='Save', command=self.save_as).grid(row=17, sticky=W, padx=240, pady=15)

        Button(self.newWindow, text='Quit', command=self.Quit).grid(row=17, sticky=W, padx=360, pady=15)
    
    
    def update_txt0_0(self):
        if self.var0_0.get()==1:
            self.var0_1.set(0)
            self.var1_0.set(0)
            self.var1_1.set(0)
            self.var2.set(0)

    def update_txt0_1(self):
        if self.var0_1.get()==1:
            self.var0_0.set(0)
            self.var1_0.set(0)
            self.var1_1.set(0)
            self.var2.set(0)

    def update_txt1_0(self):
        if self.var1_0.get()==1:
            self.var0_0.set(0)
            self.var0_1.set(0)
            self.var1_1.set(0)
            self.var2.set(0)

    def update_txt1_1(self):
        if self.var1_1.get()==1:
            self.var0_0.set(0)
            self.var0_1.set(0)
            self.var1_0.set(0)
            self.var2.set(0)


    def update_txt2(self):
        if self.var2.get()==1:
            self.var0_0.set(0)
            self.var0_1.set(0)
            self.var1_0.set(0)
            self.var1_1.set(0)





    def update_status(self):
        pass
        # if self.var3.get()==1:
        #     self.normalize_flag.set(1) ### PD version need normalize.
    
    def update_status2(self):

        self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
        if (self.user_select_indices!=0).any() or (self.user_select_indices==0).all():
            self.var_for_all.set(0)
        if (self.user_select_indices!=0).all():
            self.var_for_all.set(1)


    def update_status3(self):

        if self.var_for_all.get()==1:
            for i in range(len(self.var_list)):
                self.var_list['p-'+str(i)].set(1)

        if self.var_for_all.get()==0:
            for i in range(len(self.var_list)):
                self.var_list['p-'+str(i)].set(0)

        self.user_select_indices=np.asarray([self.var_list['p-'+str(i)].get() for i in range(self.PigNum)])
        # print self.user_select_indices







    def Reset(self):
        self.img2=np.asarray(self.im)
        self.Show_image(self.master, self.im, option=1)
        
        self.var0_0.set(0)
        self.var0_1.set(0)
        self.var1_0.set(0)
        self.var1_1.set(0)
        self.var2.set(0)
        self.var3.set(0)
        self.var_for_all.set(0)
        for i in range(self.PigNum):
            self.var_list["p-"+str(i)].set(0)

        self.showing=0
        
        if self.new_master is not None:
            self.new_master.destroy() ##close results window
        
        self.user_select_indices=np.zeros(self.PigNum, dtype=np.uint8)



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


        chosed_indices=np.nonzero(self.user_select_indices)[0]
        # print chosed_indices
        imsize=self.img2.shape[:-1]

        ### KM version
        if self.var3.get()==0:
            # print "KM version"


            ### object insertion
            if self.var2.get()==1:
                

                output=object_insertion_KS(self.AllData.KM_pigments,
                                        self.AllData.KM_thickness,
                                        imsize,
                                        self.AllData.KM_weights,
                                        chosed_indices,
                                        self.mask, 
                                        self.obj_thickness.get()/10.0, 
                                        self.dst_y, ### x, y from mouse click is reverse to params in this function
                                        self.dst_x, 
                                        self.insert_onto_layer_index.get(),
                                        self.normalize_flag.get()
                                        )


            ### copypaste (select-based)
            if self.var1_0.get()==1:
                output=object_paste_KS(self.AllData.KM_pigments, 
                                     self.AllData.KM_weights, 
                                     imsize, 
                                     chosed_indices, 
                                     self.mask, 
                                     self.dst_y, 
                                     self.dst_x,
                                     self.normalize_flag.get(),
                                     self.obj_weights_scales.get()/10.0
                                     )

            ### delete (select-based)
            if self.var1_1.get()==1:

                output=object_remove_KS(self.AllData.KM_pigments, 
                                         self.AllData.KM_weights,
                                         imsize, 
                                         self.mask,
                                         chosed_indices,
                                         self.normalize_flag.get()
                                         )
            
            if self.var0_0.get()==1: ### copy paste (inpaint based)
                output=object_paste_KS_inpaint_based(self.AllData.KM_pigments, 
                                                     self.AllData.KM_weights, 
                                                     imsize, 
                                                     self.mask, 
                                                     self.radius.get(), 
                                                     self.dst_y, 
                                                     self.dst_x,
                                                     self.normalize_flag.get(),
                                                     self.obj_weights_scales.get()/10.0
                                                     )

            if self.var0_1.get()==1: ### remove (inpaint based)
                output=object_remove_KS_inpaint_based(self.AllData.KM_pigments, 
                                                     self.AllData.KM_weights, 
                                                     imsize, 
                                                     self.mask, 
                                                     self.radius.get(),  
                                                     self.normalize_flag.get()
                                                     )


 
        ###PD version
        elif self.var3.get()==1: 
            # print "PD version"

            ### object insertion
            if self.var2.get()==1: 

                output=object_insertion_RGB(self.AllData.PD_vertices,
                                        self.AllData.PD_opacities,
                                        imsize,
                                        self.AllData.PD_weights,
                                        chosed_indices,
                                        self.mask, 
                                        self.obj_opacity.get()/10.0, 
                                        self.dst_y, ### x, y from mouse click is reverse to params in this function
                                        self.dst_x, 
                                        self.insert_onto_layer_index.get(),
                                        self.normalize_flag.get()
                                        )

            ### copypaste (select-based)    
            if self.var1_0.get()==1: 

                output=object_paste_RGB(self.AllData.PD_vertices, 
                                     self.AllData.PD_weights, 
                                     imsize, 
                                     chosed_indices, 
                                     self.mask, 
                                     self.dst_y, 
                                     self.dst_x,
                                     self.normalize_flag.get(),
                                     self.obj_weights_scales.get()/10.0
                                     )


            ### delete (select-based)
            if self.var1_1.get()==1: 
                output=object_remove_RGB(self.AllData.PD_vertices, 
                                         self.AllData.PD_weights,
                                         imsize, 
                                         self.mask,
                                         chosed_indices,
                                         self.normalize_flag.get()
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



