# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import scipy.optimize as sopt
from KS_helper_autograd import *
import time
import warnings
import json
import time
from numba import jit
from Cross_Bilateral_Filter import *
from Constant_Values import *
import PIL.Image as Image

if use_autograd==True:
    from autograd.numpy import *
    import autograd.numpy as np
    from autograd import elementwise_grad, jacobian
    def jit(f):
        # print "###"
        return f
else:
    from numpy import *
    import numpy as np
    

def choose_gama_function(use_autograd):
    if use_autograd==True:
        Gamma_trans_img=Gamma_trans_img3  ### gamma correction
        # Gamma_trans_img=Gamma_trans_img2  ### no gamma correction
    else:
        Gamma_trans_img=Gamma_trans_img1
    return Gamma_trans_img

Gamma_trans_img=choose_gama_function(use_autograd)



import fast_energy_laplacian
import scipy.sparse


def cross_bilateral(Weights_map, img, w, sigma_d, sigma_r):
    # Pre-compute Gaussian distance weights.
    X,Y = np.mgrid[-w:w+1,-w:w+1]
    G = np.exp(-(X**2+Y**2)/(2*(sigma_d**2)))
    M=Weights_map.shape[-1]

    ## Apply bilateral filter.
    shape=Weights_map.shape
    new_Weights_map = np.zeros(shape)
    count=0
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):

            # if count%200000==0:
            #     print count
            # count+=1

            # Extract local region.
            iMin = max(np.array([i-w,0]))
            iMax = min(np.array([i+w+1,shape[0]]))
            jMin = max(np.array([j-w,0]))
            jMax = min(np.array([j+w+1,shape[1]]))
            I = Weights_map[iMin:iMax,jMin:jMax]

            # To compute weights from the color image
            J = img[iMin:iMax,jMin:jMax]

            # Compute Gaussian intensity weights according to the color image
            H = np.exp(-np.square(J-img[i,j]).sum(axis=-1)/(2*(sigma_r**2)))

            # Calculate bilateral filter response.
            F = H*G[iMin-i+w:iMax-i+w,jMin-j+w:jMax-j+w]

            ### F shape is (2*win+1, 2*win+1). I shape is (2*win+1, 2*win+1, M) for M is 1 or any values.
            new_Weights_map[i,j] = (F[...,np.newaxis]*I).reshape((-1,M)).sum(axis=0)/F.sum()
                   

    return new_Weights_map






@jit 
def objective_func_vector_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0,W_neighbors=0.0, neighbors=None):


    M=H.shape[0]
    L=H.shape[1]/2
    N=len(x0)/M
    W=x0.reshape((N,M))

    # W_sum=W.sum(axis=1).reshape((-1,1))
    # W_sum=np.maximum(W_sum, 1e-15) #### to fit for autograd.
    # W=np.divide(W,W_sum)


    K0=H[:,:L]
    S0=H[:,L:]

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, W, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    
    # if R_rgb.shape[0] != arr.shape[0]*arr.shape[1]:
    #     print R_rgb.shape
    #     print arr.shape

    obj=R_rgb.reshape(-1)-arr.reshape(-1)
  

    if W_w!=0.0:
        W_sum=(W.sum(axis=1)-1)*sqrt(W_w)
        obj=np.concatenate((obj,W_sum))

    if W_sparse!=0.0:
        Sparse_term=np.sqrt(np.maximum((1.0-np.square(W-1.0)).sum(axis=1),eps))*sqrt(W_sparse/M)
        obj=np.concatenate((obj,Sparse_term.reshape(-1)))

    if W_spatial!=0.0:
        x=W.reshape((arr.shape[0],arr.shape[1],-1))
        gx,gy,gz=np.gradient(x) ### gz is not meaningful here.
        gradient=np.sqrt(np.square(gx).sum(axis=2)+np.square(gy).sum(axis=2))
        Spatial_term=gradient*sqrt(W_spatial/M)
        obj=np.concatenate((obj,Spatial_term.reshape(-1)))

    if W_neighbors!=0.0 and neighbors!=None: #### this is for per pixel solving, w_spaital should be 0 and x0 length is M.
        neighbor_term=W.reshape((1,1,-1))-neighbors
        neighbor_term=np.sqrt(np.maximum(np.square(neighbor_term).sum(axis=2),eps))*sqrt(W_neighbors*N/(1.0*M*neighbors.shape[0]*neighbors.shape[1]))
        obj=np.concatenate((obj,neighbor_term.reshape(-1)))



    # print np.square(obj).sum()

    return obj



def jacobian_objective_func_vector_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0,W_neighbors=0.0, neighbors=None):
    Jac=jacobian(objective_func_vector_fixed_KS,0)
    return Jac(x0, arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial, W_neighbors, neighbors)



@jit
def objective_func_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0, W_neighbors=0.0, neighbors=None):
    
    # ##### normalize x0
    # M=H.shape[0]
    # N=len(x0)/M
    # W=x0.reshape((N,M))
    # W_sum=W.sum(axis=1).reshape((-1,1))
    # W_sum=np.maximum(W_sum, 1e-15) #### to fit for autograd.
    # W=W/W_sum
    # x0=W.flatten()




    # obj=objective_func_vector_fixed_KS(x0, arr, H, None, W_w, W_sparse, W_spatial, W_neighbors, neighbors)
    # return np.square(obj).sum()



    obj=objective_func_vector_fixed_KS(x0, arr, H, None, W_w, W_sparse, 0.0, W_neighbors, neighbors)
    
    spatial_term=0.0

    if W_spatial!=0.0:

        #### this is ok, but not supported by autograd library to compute gradient.
        M=H.shape[0]
        spatial_term=np.dot(x0,Smooth_Matrix.dot(x0))*W_spatial/M

        return np.square(obj).sum()+spatial_term
    else:
        return np.square(obj).sum()






# ##### if use Lap, then it is not suitable, because autograd do not support sparse Lap matrix. 
# def gradient_objective_func_fixed_KS(x0, arr, H, Lap, W_w=2.0, W_sparse=0.01,W_spatial=0.0, W_neighbors=0.0, neighbors=None):
#     grad=elementwise_grad(objective_func_fixed_KS,0)
#     return grad(x0, arr, H, Lap, W_w, W_sparse, W_spatial, W_neighbors, neighbors)





def gradient_objective_func_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0, W_neighbors=0.0, neighbors=None):
    
    # ##### normalize x0
    # M=H.shape[0]
    # N=len(x0)/M
    # W=x0.reshape((N,M))
    # W_sum=W.sum(axis=1).reshape((-1,1))
    # W_sum=np.maximum(W_sum, 1e-15) #### to fit for autograd.
    # W=W/W_sum
    # x0=W.flatten()
    


    def temp(x0, arr, H, W_w, W_sparse, W_neighbors, neighbors):
        obj=objective_func_vector_fixed_KS(x0, arr, H, None, W_w, W_sparse, 0.0, W_neighbors, neighbors)
        return np.square(obj).sum()

    grad=elementwise_grad(temp,0)
    g=grad(x0, arr, H, W_w, W_sparse, W_neighbors, neighbors)

    if W_spatial!=0.0:
        M=H.shape[0]
        g2=2*Smooth_Matrix.dot(x0)*W_spatial/M
        g=g+g2

    return g






@jit
def objective_func_vector_fixed_Weights(x0, arr, Weights, W_sm_K, W_sm_S, W_sm_KS):
    
    M=Weights.shape[1]
    L=len(x0)/(2*M)

    N=Weights.shape[0]
    W_sum=Weights.sum(axis=1).reshape((-1,1))
    W_sum=np.maximum(W_sum,1e-15) 
    # W_sum[W_sum==0.0]=1e-15
    W=Weights/W_sum      #### this will disable the weights sum to be 1 regularization term.

    x_H=x0.reshape((M,-1))
    K0=x_H[:,:L]
    S0=x_H[:,L:]

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, W, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction

    obj=R_rgb-arr.reshape((-1,3))

    #### is it reasonable to do this?
    # obj=np.sqrt(np.square(obj).sum(aixs=0))

    obj=obj.reshape(-1)
    common_w=sqrt(1.0*N/(M*L))

    if W_sm_K!=0.0:
        dx,dy=np.gradient(K0)
        g_K=dy*sqrt(W_sm_K)*common_w
        obj=np.concatenate((obj,g_K.reshape(-1)))

    if W_sm_S!=0.0:
        dx,dy=np.gradient(S0)
        g_S=dy*sqrt(W_sm_S)*common_w
        obj=np.concatenate((obj,g_S.reshape(-1)))

    if W_sm_KS!=0.0:
        KS0=K0/S0
        dx,dy=np.gradient(KS0)
        g_KS=dy*sqrt(W_sm_KS)*common_w
        obj=np.concatenate((obj,g_KS.reshape(-1)))

    return obj


def jacobian_objective_func_vector_fixed_Weights(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS):
    Jac=jacobian(objective_func_vector_fixed_Weights,0)
    return Jac(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS)


@jit
def objective_func_fixed_Weights(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS):
    obj=objective_func_vector_fixed_Weights(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS)
    return np.square(obj).sum()


def gradient_objective_func_fixed_Weights(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS):
    grad=elementwise_grad(objective_func_fixed_Weights,0)
    return grad(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS)




def optimize(arr, x0, H, Smooth_Matrix, saver=None, W_w=2.0, W_sparse=0.1, W_spatial=0.0, method='L-BFGS-B', W_neighbors=0.0, neighbors=None):
    # print type(x0)
    arr_shape=arr.shape
    N=arr_shape[0]*arr_shape[1]
    M=len(x0)/N
    L=H.shape[1]/2
    lb=1e-15
    ub=1.0
    #### bounds0 are for least_squares function parameters.
    bounds0=(lb, ub)
    bounds3=[]
    for i in xrange(len(x0)):
        bounds3.append((lb,ub))
    
    # print x0.max()
    # print x0.min()
    # x0[x0<1e-15]=2e-15
    # x0[x0>=1.0]=1.0-1e-15

    # print x0.max()
    # print x0.min()


    start_time=time.clock()
    if method=='trf':
        if use_autograd==False:
            res=sopt.least_squares(objective_func_vector_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial, W_neighbors, neighbors), bounds=bounds0, jac='2-point', method='trf', xtol=1e-6)   
        else:
            res=sopt.least_squares(objective_func_vector_fixed_KS, x0, args=(arr,H, Smooth_Matrix, W_w, W_sparse, W_spatial, W_neighbors, neighbors), bounds=bounds0, jac=jacobian_objective_func_vector_fixed_KS, method='trf',xtol=1e-6)   

    else:
        if use_autograd==False:
            res=sopt.minimize(objective_func_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial, W_neighbors, neighbors), bounds=bounds3, method=method, callback=saver)
        else:
            # print 'this'
            res=sopt.minimize(objective_func_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial, W_neighbors, neighbors), bounds=bounds3, method=method, jac=gradient_objective_func_fixed_KS, callback=saver,options={'gtol':1e-4, 'ftol': 1e-4})

    end_time=time.clock()
    # print 'Optimize variables of size ', x0.shape, ' took ', (end_time-start_time), ' seconds.'
    x=res["x"]



    # W=x.reshape((N,M))
    # W_sum=W.sum(axis=1).reshape((-1,1))
    # W_sum=np.maximum(W_sum, 1e-15) #### to fit for autograd.
    # W=W/W_sum
    # x=W.flatten()


    return x



def optimize_fixed_Weights(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS, method='L-BFGS-B'):

    arr_shape=arr.shape
    N=arr_shape[0]*arr_shape[1]
    M=W.shape[1]
    L=len(x0)/(2*M)

    eps=1e-15

    #### bounds0 and bounds1 are for least_squares function parameters.
    bounds0=(eps, np.inf)

    bounds3=[]
    for i in xrange(len(x0)):
        bounds3.append((eps,None))


    start_time=time.clock()
    if method=='trf':
        if use_autograd==False:
            res=sopt.least_squares(objective_func_vector_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS), bounds=bounds0,jac='2-point', method='trf',xtol=1e-6)   
        else:
            res=sopt.least_squares(objective_func_vector_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS), bounds=bounds0,jac=jacobian_objective_func_vector_fixed_Weights, method='trf',xtol=1e-6)
    else:
        if use_autograd==False:
            res=sopt.minimize(objective_func_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS),bounds=bounds3, method=method)
        else:
            res=sopt.minimize(objective_func_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS),bounds=bounds3, method=method, jac=gradient_objective_func_fixed_Weights)

    end_time=time.clock()

    x=res["x"]
    print res["message"]

    return x



def save_results(x0, arr, H, output_prefix):
    shape=arr.shape
    original_shape=shape
    img_size=shape[:2]
    N=shape[0]*shape[1]
    M=H.shape[0]
    L=H.shape[1]/2

    Weights=x0.reshape((N,M))
    K0=H[:,:L]
    S0=H[:,L:]
    print Weights.sum(axis=1).min()
    print Weights.sum(axis=1).max()

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, Weights, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    
    print 'RGB RMSE: ', np.sqrt(np.square(255*(arr.reshape((-1,3))-R_rgb)).sum()/N)
    
    filename=output_prefix+"-fixed_KS-reconstructed.png"
    plt.imsave(filename,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))
    np.savetxt(output_prefix+"-mixing_weights.txt", Weights)


    #### save for applications
    filename=save_for_application_path_prefix+os.path.splitext(img_file)[0]+"-"+str(M)+"-KM_mixing-reconstructed.png"
    plt.imsave(filename,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))


    ### compute sparsity
    sparsity_thres_array=np.array([0.000001, 0.00001, 0.0001,0.001,0.01,0.1])
    Weights_sparsity_list=np.ones(len(sparsity_thres_array))
    for thres_ind in xrange(len(sparsity_thres_array)):
        Weights_sparsity_list[thres_ind]=len(Weights[Weights<=sparsity_thres_array[thres_ind]])*1.0/(N*M)
    
    print "Weights_sparsity_list: ", Weights_sparsity_list
    np.savetxt(output_prefix+"-mixing_weights-Sparsity.txt", Weights_sparsity_list)



    # normalized_Weights=Weights/Weights.sum(axis=1).reshape((-1,1))
    # for i in xrange(M):
    #     #### save normalized_weights_map for each pigment.
    #     normalized_weights_map_name=output_prefix+"-normalized_mixing_weights_map-%02d.png" % i
    #     normalized_Weights_map=normalized_Weights[:,i].reshape(img_size).copy()
    #     Image.fromarray((normalized_Weights_map*255.0).clip(0,255).round().astype(np.uint8)).save(normalized_weights_map_name)
        

    Weights_sum_map=Weights.sum(axis=1).reshape(img_size)
    W_min=Weights_sum_map.min()
    W_max=Weights_sum_map.max()
    Weights_sum_map=Weights_sum_map/W_max

    Image.fromarray((Weights_sum_map*255.0).round().astype(np.uint8)).save(output_prefix+"-mixing_weights_sum_map-min-"+str(W_min)+"-max-"+str(W_max)+".png")
        



    for i in xrange(M):
        #### save weights_map for each pigment.
        weights_map_name=output_prefix+"-mixing_weights_map-%02d.png" % i
        Weights_map=Weights[:,i].reshape(img_size).copy()
        Image.fromarray((Weights_map*255.0).clip(0,255).round().astype(np.uint8)).save(weights_map_name)
        
        #### save for applications
        weights_map_name=save_for_application_path_prefix+os.path.splitext(img_file)[0]+"-"+str(M)+"-KM_mixing-weights_map-%02d.png" % i
        Image.fromarray((Weights_map*255.0).clip(0,255).round().astype(np.uint8)).save(weights_map_name)







        # #### use weights as thickness, to show each pigments on white background.
        # pigments_map_name=output_prefix+"-pigments_visualize_map-%02d.png" % i
        # #####this assign is not good
        # # Weights_map[Weights_map==0]=1e-8
        # #####this assign is not good
        # Weights_map+=1e-4
        # if Weights_map.max()>1.0:
        #     Weights_map=Weights_map/Weights_map.max()

        
        # R_vector=equations_in_RealPigments(K0[i:i+1,:], S0[i:i+1,:], r=1.0, h=Weights_map.reshape((-1,1)))
        # ### from R spectrum x wavelength spectrums to linear rgb colors 
        # P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
        # R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
        # Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
        # R_xyz/=Normalize ####xyz value shape is N*3
        # R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
        # R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
        # plt.imsave(pigments_map_name,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))



def save_results_2(x0, arr_shape, H, output_prefix):
    shape=arr_shape
    original_shape=shape
    img_size=shape[:2]
    N=shape[0]*shape[1]
    M=H.shape[0]
    L=H.shape[1]/2

    Weights=x0.reshape((N,M))
    K0=H[:,:L]
    S0=H[:,L:]
    print Weights.sum(axis=1).min()
    print Weights.sum(axis=1).max()

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, Weights, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    
    # print 'RGB RMSE: ', np.sqrt(np.square(255*(arr.reshape((-1,3))-R_rgb)).sum()/N)
    
    filename=output_prefix+"-fixed_KS-reconstructed.png"
    plt.imsave(filename,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))
    # np.savetxt(output_prefix+"-mixing_weights.txt", Weights)


    ## compute sparsity
    # sparsity_thres_array=np.array([0.000001, 0.00001, 0.0001,0.001,0.01,0.1])
    # Weights_sparsity_list=np.ones(len(sparsity_thres_array))
    # for thres_ind in xrange(len(sparsity_thres_array)):
    #     Weights_sparsity_list[thres_ind]=len(Weights[Weights<=sparsity_thres_array[thres_ind]])*1.0/(N*M)
    
    # print "Weights_sparsity_list: ", Weights_sparsity_list
    # np.savetxt(output_prefix+"-mixing_weights-Sparsity.txt", Weights_sparsity_list)



    # normalized_Weights=Weights/Weights.sum(axis=1).reshape((-1,1))
    # for i in xrange(M):
    #     #### save normalized_weights_map for each pigment.
    #     normalized_weights_map_name=output_prefix+"-normalized_mixing_weights_map-%02d.png" % i
    #     normalized_Weights_map=normalized_Weights[:,i].reshape(img_size).copy()
    #     cv2.imwrite(normalized_weights_map_name, (normalized_Weights_map*255.0).clip(0,255).round().astype(np.uint8))
        

    # Weights_sum_map=Weights.sum(axis=1).reshape(img_size)
    # W_min=Weights_sum_map.min()
    # W_max=Weights_sum_map.max()
    # Weights_sum_map=Weights_sum_map/W_max

    # cv2.imwrite(output_prefix+"-mixing_weights_sum_map-min-"+str(W_min)+"-max-"+str(W_max)+".png", (Weights_sum_map*255.0).round().astype(np.uint8))
        

    for i in xrange(M):
        #### save weights_map for each pigment.
        weights_map_name=output_prefix+"-mixing_weights_map-%02d.png" % i
        Weights_map=Weights[:,i].reshape(img_size).copy()
        Image.fromarray((Weights_map*255.0).clip(0,255).round().astype(np.uint8)).save(weights_map_name)





def save_pigments(x_H, M, output_dir):

    L=len(x_H)/(2*M)
    x_H=x_H.reshape((M,-1))

    np.savetxt(output_dir+"-primary_pigments.txt", x_H)

    K0=x_H[:,:L]
    S0=x_H[:,L:]

    R_vector=equations_in_RealPigments(K0, S0, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
    R_xyz=(P_vector.reshape((M,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is M*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction


    C=3 ## channel number
    w=h=50
    pigments=np.ones((w,h*M,C))
    for i in xrange(M):
        pigments[:,i*h:i*h+h]=R_rgb[i]
    plt.imsave(output_dir+"-primary_pigments_color.png", (pigments*255.0).round().astype(np.uint8))

    from scipy.spatial import ConvexHull
    hull=ConvexHull(R_rgb*255.0)
    faces=hull.points[hull.simplices]
    with open(output_dir+"-primary_pigments_color_vertex.js", "w") as fd:
        json.dump({'vs': (R_rgb*255.0).tolist(),'faces': faces.tolist()}, fd)


    xaxis=np.arange(L)
    for i in xrange(M):
        fig=plt.figure()
        plt.plot(xaxis, K0[i], 'b-')
        fig.savefig(output_dir+"-primary_pigments_K_curve-"+str(i)+".png")
        fig=plt.figure()
        plt.plot(xaxis, S0[i], 'b-')
        fig.savefig(output_dir+"-primary_pigments_S_curve-"+str(i)+".png")
        fig=plt.figure()
        plt.plot(xaxis, R_vector[i], 'b-')
        fig.savefig(output_dir+"-primary_pigments_R_curve-"+str(i)+".png")
        fig=plt.figure()
        plt.plot(xaxis, K0[i]/S0[i], 'b-')
        fig.savefig(output_dir+"-primary_pigments_KS_curve-"+str(i)+".png")
        plt.close('all')







def Unique_colors_pixel_mapping(img_data): ### shape is row*col*channel
    new_data=img_data.copy()
    rows=new_data.shape[0]
    columns=new_data.shape[1]
    dims=new_data.shape[2]
    # print new_data.shape
    new_data=new_data.reshape((-1,3))
    # print new_data.dtype
    # print new_data.shape

    ### colors2count dict and colors2xy dict
    # colors2count ={}
    colors2xy ={}

    unique_new_data=list(set(list(tuple(element) for element in new_data))) #### order will be random


    for element in unique_new_data:
        # colors2count[tuple(element)]=0
        colors2xy.setdefault(tuple(element),[])
        
    for index in xrange(len(new_data)):
        element=new_data[index]
        # colors2count[tuple(element)]+=1
        colors2xy[tuple(element)].append(index)

    # print len(unique_new_data)
    return colors2xy





def create_laplacian(arr_shape, M):
    Lap = fast_energy_laplacian.gen_grid_laplacian( arr_shape[0], arr_shape[1] )
    ## Now repeat Lap #pigments times.
    ## Because the layer values are the innermost dimension,
    ## every entry (i,j, val) in Lap should be repeated
    ## (i*#pigments + k, j*#pigments + k, val) for k in range(#pigments).
    Lap = Lap.tocoo()
    ## Store the shape. It's a good habit, because there may not be a nonzero
    ## element in the last row and column.
    shape = Lap.shape
            
    ## Fastest
    ks = arange( M )
    rows = ( repeat( asarray( Lap.row ).reshape( Lap.nnz, 1 ) * M, M, 1 ) + ks ).ravel()
    cols = ( repeat( asarray( Lap.col ).reshape( Lap.nnz, 1 ) * M, M, 1 ) + ks ).ravel()
    vals = ( repeat( asarray( Lap.data ).reshape( Lap.nnz, 1 ), M, 1 ) ).ravel()
    
    Lap = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( shape[0]*M, shape[1]*M ) ).tocsr()
    return Lap





def create_cross_bilateral(arr, M):
    Blf=generate_Bilateral_Matrix_hat(arr)

    Blf = Blf.tocoo()
    ## Store the shape. It's a good habit, because there may not be a nonzero
    ## element in the last row and column.
    shape = Blf.shape
            
    ## Fastest
    ks = arange( M )
    rows = ( repeat( asarray( Blf.row ).reshape( Blf.nnz, 1 ) * M, M, 1 ) + ks ).ravel()
    cols = ( repeat( asarray( Blf.col ).reshape( Blf.nnz, 1 ) * M, M, 1 ) + ks ).ravel()
    vals = ( repeat( asarray( Blf.data ).reshape( Blf.nnz, 1 ), M, 1 ) ).ravel()
    
    newBlf = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( shape[0]*M, shape[1]*M ) ).tocsr()
    return newBlf




if __name__=="__main__":
    global img_file
    img_file=sys.argv[1]
    KS_file_name=sys.argv[2]
    Weights_file_name=sys.argv[3]
    output_prefix=sys.argv[4]
    W_w=np.float(sys.argv[5])
    W_sparse=np.float(sys.argv[6])
    solve_choice=np.int(sys.argv[7])
    W_spatial=np.float(sys.argv[8])
    print 'W_spatial',W_spatial

    global save_for_application_path_prefix
    save_for_application_path_prefix="./Application_Files/"


    W_neighbors=0.0
    if solve_choice==3 or solve_choice==6 : #### solve per pixel with neighborhood info constraints
        W_neighbors=np.float(sys.argv[9])

    print 'W_neighbors',W_neighbors
    

    START=time.time()
    img=np.asarray(Image.open(img_file).convert('RGB'))

    arr=img.copy()/255.0
    H=np.loadtxt(KS_file_name)
    print H.shape
    eps=1e-15

    # ##### is it OK? for real KS value, all are non-zero? 
    # H[H==0]=eps

    original_shape=img.shape
    img_size=img.shape[:2]

    N=arr.shape[0]*arr.shape[1]
    M=H.shape[0]
    L=H.shape[1]/2
        
    if Weights_file_name=="None":

        # W=np.random.random_sample((arr.shape[0]*arr.shape[1],M))
        # W=W/W.sum(axis=1).reshape((-1,1))

        W=np.ones((arr.shape[0],arr.shape[1],M))/M

        prior_mixing_weights=None

    elif Weights_file_name=="primary_pigments_mixing_weights.txt":
        print Weights_file_name
        prior_mixing_weights=np.loadtxt(Weights_file_name)
        print prior_mixing_weights.shape
        W=np.random.random_sample((arr.shape[0]*arr.shape[1],H.shape[0]))
        W=W/W.sum(axis=1).reshape((-1,1))
    else:
        extention=os.path.splitext(Weights_file_name)[1]
        print extention
        if extention==".js":
            with open(Weights_file_name) as data_file:    
                W_js = json.load(data_file)
            W=np.array(W_js['weights'])
        if extention==".txt":
            W=np.loadtxt(Weights_file_name)
            W=W.reshape((arr.shape[0],arr.shape[1],M))
        
        print W.shape
        

        # for i in xrange(W.shape[-1]):
        #     weights_map_name=output_prefix+"-initial-weights_map-%02d.png" % i
        #     Weights_map=W[:,:,i]
        #     cv2.imwrite(weights_map_name, (Weights_map*255.0).clip(0,255).round().astype(np.uint8))

        W=W.reshape((-1,W.shape[-1]))
        N=W.shape[0]
        M=W.shape[1]
        L=H.shape[1]/2
        W_normalize=W/W.sum(axis=1).reshape((-1,1))


        initial_error= objective_func_vector_fixed_KS(W.reshape(-1), arr, H, None, W_w=0.0, W_sparse=0.0,W_spatial=0.0,W_neighbors=0.0, neighbors=None)
        print 'initial_error: ', np.sqrt(np.square(initial_error*255.0).sum()/N)
        initial_recover=initial_error.reshape((-1,3))+arr.reshape((-1,3))
        plt.imsave(output_prefix+"-initial_recover.png",(initial_recover.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))
        prior_mixing_weights=None



    x0=W.reshape(-1)



    kSaveEverySeconds = 50
    ## [ number of iterations, time of last save, arr.shape ]
    last_save = [ None, None, None ]
    def reset_saver( arr_shape ):
        last_save[0] = 0
        last_save[1] = time.clock()
        last_save[2] = arr_shape
    def saver( xk):
        arr_shape = last_save[2]
        last_save[0] += 1
        now = time.clock()
        ## Save every 10 seconds!
        if now - last_save[1] > kSaveEverySeconds:
            print 'Iteration', last_save[0]
            save_results_2(xk, arr_shape, H, output_prefix)
            ## Get the time again instead of using 'now', because that doesn't take into
            ## account the time to actually save the images, which is a lot for large images.
            last_save[1] = time.clock()

    import scipy.ndimage
    import skimage.transform
    def optimize_smaller_whole(large_arr, large_Y0, level, smooth_choice):

        solve_smaller_factor=2
        too_small=40
        ## Terminate recursion if the image is too small.
        if large_arr.shape[0]/solve_smaller_factor < too_small or large_arr.shape[1]/solve_smaller_factor < too_small:
            return large_Y0, level
        
        ## small_arr = downsample( large_arr )
        # small_arr = large_arr[::solve_smaller_factor,::solve_smaller_factor]
        # small_arr = scipy.ndimage.zoom(large_arr,[1.0/solve_smaller_factor,1.0/solve_smaller_factor,1])
        
        small_arr = skimage.transform.pyramid_reduce(large_arr, downscale=solve_smaller_factor, order=3)


        ## small_Y0 = downsample( large_Y0 )
        small_Y0 = large_Y0.reshape( large_arr.shape[0], large_arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
        # large_Y0=large_Y0.reshape( large_arr.shape[0], large_arr.shape[1], -1 )
        # small_Y0 = scipy.ndimage.zoom(large_Y0,[1.0/solve_smaller_factor,1.0/solve_smaller_factor,1])
        # small_Y0=small_Y0.ravel()

        ## get an improved Y by recursively shrinking
        small_Y1, level = optimize_smaller_whole(small_arr, small_Y0, level, smooth_choice)
        

        ## solve on the downsampled problem
        print '==> Optimizing on a smaller image:', small_arr.shape, 'instead of', large_arr.shape
        
        arr_shape=(small_arr.shape[0],small_arr.shape[1])

        
        time0=time.time()


        if smooth_choice=='lap' or smooth_choice=='lap_blf':
            Lap=create_laplacian(arr_shape,M)
            Smooth_Matrix=Lap

        elif smooth_choice=='blf':
            Blf=create_cross_bilateral(small_arr*255.0, M)
            Smooth_Matrix=Blf

        else:
            print "Error! No such choice!"



        time1=time.time()
        print "compute smooth matrix time: ", time1-time0

        reset_saver(small_arr.shape)
        small_Y = optimize(small_arr, small_Y1, H, Smooth_Matrix, saver=saver, W_w=W_w, W_sparse=W_sparse, W_spatial=W_spatial, method='L-BFGS-B')
        saver(small_Y.reshape(-1))
        time2=time.time()
        print 'this level use time: ', time2-time1
        # save_layers(small_Y.reshape(-1), small_arr, H, output_prefix+"-recursivelevel-"+str(level))

        level+=1
        ## large_Y1 = upsample( small_Y )
        ### 1 Make a copy
        large_Y1 = array( large_Y0 ).reshape( large_arr.shape[0], large_arr.shape[1], -1 )
        ### 2 Fill in as much as will fit using numpy.repeat()
        small_Y = small_Y.reshape( small_arr.shape[0], small_arr.shape[1], -1 )
        
        # small_Y_upsampled = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
        # small_Y_upsampled = scipy.ndimage.zoom(small_Y,[solve_smaller_factor,solve_smaller_factor,1])
        small_Y_upsampled = skimage.transform.pyramid_expand(small_Y, upscale=solve_smaller_factor, order=3)

        large_Y1[:,:] = small_Y_upsampled[ :large_Y1.shape[0], :large_Y1.shape[1] ]

        return large_Y1.ravel(), level




    from joblib import Parallel, delayed
    import multiprocessing

    kDisableParallel = False
    if kDisableParallel:
        def Parallel( iter, n_jobs = None ):
            while iter.next(): pass
        def delayed( f ): return f

    
    def per_pixel_solve(i, pixel, z0,initial, H, method='trf'):
        if i%10000==0:
            print "pixel: ", i
        pixel=pixel.reshape((1,1,-1))
        y0=initial.copy()
        y0/=y0.sum()
        y0 = optimize(pixel, y0, H, None, saver=None, W_w=W_w, W_sparse=W_sparse,W_spatial=0.0,method=method)
        z0[i,:]=np.array(y0)
   
    
    def per_patch_solve(arr,col_step, ind, x1, x0, H, p_size, offset, Lap, method='L-BFGS-B'):
        # print method
        i=ind/col_step
        j=ind%col_step
        if ind%100==0:
            print ind
        patch = np.array(arr[offset+p_size*i:offset+p_size*(i+1), offset+p_size*j:offset+p_size*(j+1), :])
        y0 = x0[offset+p_size*i:offset+p_size*(i+1), offset+p_size*j:offset+p_size*(j+1), :].flatten()
        y0 = optimize(patch, y0, H, Lap, saver=None, W_w=W_w, W_sparse=W_sparse,W_spatial=W_spatial, method=method)
        

        # Y_sum=y0.reshape((p_size,p_size,-1)).sum(axis=-1)
        # Y_sum_max=Y_sum.max()
        # Y_sum_min=Y_sum.min()
        # if (abs(Y_sum_max-1.0)>=0.2).any() or (abs(Y_sum_min-1.0)>=0.2).any():
        #     print ind,"abnormal"
        #     y0 = np.random.random_sample((p_size,p_size,M))
        #     y0 = y0/y0.sum(axis=2).reshape((p_size,p_size,1))
        #     y0 = y0.flatten()
        #     y0 = optimize(patch, y0, H, Lap, saver=None, W_w=W_w, W_sparse=W_sparse,W_spatial=W_spatial)

        x1[offset+p_size*i:offset+p_size*(i+1), offset+p_size*j:offset+p_size*(j+1), :]=y0.reshape((p_size,p_size,M))

    
    def From_wegihts_for_compositedPigments_to_weights_for_uniqueColors(UNIQ_foat, H, prior_mixing_weights):
        N1=len(UNIQ_foat)
        M=H.shape[0]
        L=H.shape[1]/2

        if prior_mixing_weights==None:
            Final_weights=np.random.random_sample((N1,M))
        else:
            K0=H[:,:L]
            S0=H[:,L:]
            N=prior_mixing_weights.shape[0]
            assert(prior_mixing_weights.shape[1]==M)
            ### reconstruction of input
            R_vector=KM_mixing_multiplepigments(K0, S0, prior_mixing_weights, r=1.0, h=1.0)
            ### from R spectrum x wavelength spectrums to linear rgb colors 
            P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
            R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
            Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
            R_xyz/=Normalize ####xyz value shape is N*3
            R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
            R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
            
            #### R_rgb is N RGB cluster center, each cluster center has some member of uniq colors(UNIQ_float) 
            #### N1 >> N
            Final_weights=np.ones((N1,M))

            diff=UNIQ_foat.reshape((N1,1,3))-R_rgb.reshape((1,N,3)) #### shape is N1*N*3
            diff=np.square(diff).sum(axis=2) ### shape is (N1,N)
            min_indices=np.argmin(diff,axis=1)#### shape is (N1,)
            
            for i in xrange(N1):
                Final_weights[i]=prior_mixing_weights[min_indices[i]]

   
        return Final_weights
    
    


    # def optimize_smaller_parallel_per_patch(large_arr, H, p_size, offset, Lap, level, method='L-BFGS-B'):

    #     solve_smaller_factor=2
    #     too_small=50

    #     ## Terminate recursion if the image is too small.
    #     if large_arr.shape[0]/solve_smaller_factor < too_small or large_arr.shape[1]/solve_smaller_factor < too_small:
            
    #         #### for smallest image, just return random weights.
    #         large_Y1=np.random.random_sample((large_arr.shape[0],large_arr.shape[1],M))
    #         large_Y1=large_Y1/large_Y1.sum(axis=2).reshape((large_arr.shape[0],large_arr.shape[1],1))
            
    #         return large_Y1,offset,level
        

    #     ## small_arr = downsample( large_arr )
    #     small_arr = large_arr[::solve_smaller_factor,::solve_smaller_factor]

    #     ## get an improved Y by recursively shrinking
    #     small_Y1,offset,level = optimize_smaller_parallel_per_patch(small_arr, H, p_size, offset, Lap, level)
    #     ## solve on the downsampled problem
    #     print '==> Optimizing on a smaller image:', small_arr.shape, 'instead of', large_arr.shape

    #     if level%2==0:
    #         offset=0
    #         row_step=small_arr.shape[0]/p_size #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
    #         col_step=small_arr.shape[1]/p_size
    #     if level%2==1:
    #         offset=p_size/2
    #         row_step=small_arr.shape[0]/p_size-1 #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
    #         col_step=small_arr.shape[1]/p_size-1
       

    #     print small_arr.shape
    #     print 'level ', level
    #     print 'p_size ', p_size
    #     print 'offset ', offset
    #     print row_step, col_step
    #     print method


    #     small_Y1=small_Y1.reshape((small_arr.shape[0],small_arr.shape[1],M))


    #     #### spatial smoothing
    #     # small_Y1=cross_bilateral(small_Y1, small_arr, 3, 2.0, 0.1)



    #     ##### per patch solve
    #     small_Y = np.memmap('results',shape=(small_arr.shape[0],small_arr.shape[1],M), mode='w+', dtype=np.float64)
    #     small_Y[:,:,:] = small_Y1[:,:,:]
    #     num_cores = multiprocessing.cpu_count()
    #     Parallel(n_jobs=num_cores)(delayed(per_patch_solve)(small_arr, col_step, i, small_Y, small_Y1, H, p_size,offset,Lap, method=method) for i in xrange(row_step*col_step))


    #     # ##### per pixel solve
    #     # small_Y1=small_Y1.reshape((-1,M))
    #     # R=small_arr.reshape((-1,3))
    #     # small_Y = np.memmap('results_choice0',shape=(small_Y1.shape[0],M), mode='w+', dtype=np.float64)
    #     # num_cores = multiprocessing.cpu_count()
    #     # Parallel(n_jobs=num_cores)(delayed(per_pixel_solve)(i, R[i],small_Y,small_Y1[i], H) for i in xrange(small_Y.shape[0]))




    #     # save_results(small_Y.reshape(-1), small_arr, H, output_prefix+"-"+str(p_size)+"-recursivelevel-"+str(level))
    #     level+=1

    #     ### 1 Make a copy
    #     large_Y1 = np.ones((large_arr.shape[0], large_arr.shape[1], M ))
    #     ### 2 Fill in as much as will fit using numpy.repeat()
    #     small_Y = small_Y.reshape( small_arr.shape[0], small_arr.shape[1], -1 )
    #     small_Y_upsampled = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
    #     large_Y1[:,:] = small_Y_upsampled[ :large_Y1.shape[0], :large_Y1.shape[1] ]

    #     return large_Y1.ravel(),offset,level

    def get_neighbors(i,x0,shape):
        rows=shape[0]
        cols=shape[1]
        row_ind=i/cols
        col_ind=i%cols
        imin=max(np.array([row_ind-half_win,0]))   ###### if from autograd.numpy import *, then use default python min(a,b) will cause error.
        imax=min(np.array([row_ind+half_win+1,rows]))
        jmin=max(np.array([col_ind-half_win,0]))
        jmax=min(np.array([col_ind+half_win+1,cols]))
        neighbors=x0[imin:imax, jmin:jmax, :].copy()
        return neighbors



    # def per_pixel_solve_with_spatial_info(i, pixel, shape, x1, y0, H, neighbors, half_win=1, method='trf'):
    #     rows=shape[0]
    #     cols=shape[1]
    #     row_ind=i/cols
    #     col_ind=i%cols

    #     if i%10000==0:
    #         print "pixel: ", i
    #         print time.time()
   
    #     y = optimize(pixel, y0, H, None, saver=None, W_w=W_w, W_sparse=W_sparse, W_spatial=0.0, method=method, W_neighbors=W_neighbors, neighbors=neighbors)
    #     x1[row_ind,col_ind,:]=y
   
    
    #####old version
    def per_pixel_solve_with_spatial_info(i, arr, x1, x0, H, half_win=1, method='trf'):
        rows=arr.shape[0]
        cols=arr.shape[1]
        row_ind=i/cols
        col_ind=i%cols
        pixel=arr[row_ind,col_ind].reshape((1,1,3)).copy()

        if i%10000==0:
            print "pixel: ", i
            # print time.time()

        imin=max(np.array([row_ind-half_win,0]))   ###### if from autograd.numpy import *, then use default python min(a,b) will cause error.
        imax=min(np.array([row_ind+half_win+1,rows]))
        jmin=max(np.array([col_ind-half_win,0]))
        jmax=min(np.array([col_ind+half_win+1,cols]))

        neighbors=x0[imin:imax, jmin:jmax, :].copy()
        
        y0=x0[row_ind,col_ind].copy()
        y = optimize(pixel, y0, H, None, saver=None, W_w=W_w, W_sparse=W_sparse, W_spatial=0.0, method=method, W_neighbors=W_neighbors, neighbors=neighbors)
        x1[row_ind,col_ind,:]=y




    def optimize_smaller(large_arr, H, p_size, Lap, level):

        solve_smaller_factor=2
        too_small=40

        ## Terminate recursion if the image is too small.
        if large_arr.shape[0]/solve_smaller_factor < too_small or large_arr.shape[1]/solve_smaller_factor < too_small:
            
            #### for smallest image, just return random weights.
            large_Y1=np.random.random_sample((large_arr.shape[0],large_arr.shape[1],M))
            large_Y1=large_Y1/large_Y1.sum(axis=2).reshape((large_arr.shape[0],large_arr.shape[1],1))
            large_Y=large_Y1.copy()


            # large_Y = np.memmap('results_choice0',shape=(large_Y1.shape[0],large_Y1.shape[1],M), mode='w+', dtype=np.float64)
            # large_Y[:,:,:] = large_Y1[:,:,:]
            # num_cores = multiprocessing.cpu_count()
            # row_step=large_arr.shape[0]/p_size #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
            # col_step=large_arr.shape[1]/p_size
            # Parallel(n_jobs=num_cores)(delayed(per_patch_solve)(large_arr, col_step, i, large_Y, large_Y1, H, p_size ,0, Lap, method='L-BFGS-B') for i in xrange(row_step*col_step))
        

            return large_Y,level
        

        ## small_arr = downsample( large_arr )
        small_arr = large_arr[::solve_smaller_factor,::solve_smaller_factor]

        ## get an improved Y by recursively shrinking
        small_Y1,level = optimize_smaller(small_arr, H, p_size, Lap, level)
        ## solve on the downsampled problem
        print '==> Optimizing on a smaller image:', small_arr.shape, 'instead of', large_arr.shape
       
        
        if level%2==0:
            offset=0
            row_step=small_arr.shape[0]/p_size #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
            col_step=small_arr.shape[1]/p_size
        if level%2==1:
            offset=p_size/2
            row_step=small_arr.shape[0]/p_size-1 #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
            col_step=small_arr.shape[1]/p_size-1
       

        print small_arr.shape
        print 'level ', level
        print 'p_size ', p_size


        small_Y1=small_Y1.reshape((small_arr.shape[0],small_arr.shape[1],M))


        #### to make value nonzero, for autograd.
        small_Y1[small_Y1<eps]=eps




        small_Y = np.memmap('result-recursive-'+str(level),shape=(small_arr.shape[0],small_arr.shape[1],M), mode='w+', dtype=np.float64)
        small_Y[:,:,:] = small_Y1[:,:,:]
        start_t=time.time()
         
        LOOP=row_step*col_step

        #### parallel verion
        num_cores = multiprocessing.cpu_count()


        Parallel(n_jobs=num_cores)(delayed(per_patch_solve)(small_arr, col_step, i, small_Y, small_Y1, H, p_size,offset,Lap, method='L-BFGS-B') for i in xrange(LOOP))
        
        # #### for loop version.
        # # LOOP=1
        # for i in xrange(LOOP):
        #     per_patch_solve(small_arr, col_step, i, small_Y, small_Y1, H, p_size, offset, Lap, method='L-BFGS-B')


        end_t=time.time()

        print "current level time: ", end_t-start_t

        # save_results(small_Y.reshape(-1), small_arr, H, output_prefix+"-p_size-"+str(p_size)+"-recursivelevel-"+str(level))
        
        level+=1


        ### 1 Make a copy
        large_Y1 = np.ones((large_arr.shape[0], large_arr.shape[1], M ))
        ### 2 Fill in as much as will fit using numpy.repeat()
        small_Y = small_Y.reshape( small_arr.shape[0], small_arr.shape[1], -1 )
        small_Y_upsampled = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
        large_Y1[:,:] = small_Y_upsampled[ :large_Y1.shape[0], :large_Y1.shape[1] ]

        return large_Y1.ravel(),level





    if solve_choice==0: #### downsampled to smallest one to solve and recursively upsample to original size.
                        #### if do not consider spatial info, this choice is not useful. 
        print 'choice: ', solve_choice
        
        smooth_choice=sys.argv[10]
        recursive_choice=sys.argv[11]

        print 'smooth_choice: ',smooth_choice
        print 'recursive_choice: ',recursive_choice



        if recursive_choice=='Yes':
            level=0
            x0, level=optimize_smaller_whole(arr, x0.reshape(-1), level, smooth_choice)
       


        reset_saver( arr.shape )
        arr_shape=(arr.shape[0],arr.shape[1])
        
        time0=time.time()

        if smooth_choice=='lap':
            Lap=create_laplacian(arr_shape,M)
            Smooth_Matrix=Lap

        elif smooth_choice=='blf':
            Blf=create_cross_bilateral(arr*255.0, M)
            Smooth_Matrix=Blf

        elif smooth_choice=='lap_blf' and recursive_choice=='Yes':
            Blf=create_cross_bilateral(arr*255.0, M)
            Smooth_Matrix=Blf
        else:
            print "Error! No such choice combination!"


        time1=time.time()
        print "compute smooth matrix time: ", time1-time0

        x0 = optimize( arr, x0.reshape(-1), H, Smooth_Matrix, saver=saver, W_w=W_w, W_sparse=W_sparse, W_spatial=W_spatial, method='L-BFGS-B')
        save_results(x0, arr, H, output_prefix+"-final_recursivelevel-")
        time2=time.time()
        print "final level use time: ", time2-time1
        




    elif solve_choice==1: 
    #### per pixel solve (still slow for large size image, weights map is not smooth)
        
        R=arr.reshape((-1,3))
        if Weights_file_name=="None":
            ##### using random weights as input
            Initial=np.random.random_sample((N,M))
            Initial=Initial/Initial.sum(axis=2).reshape((-1,1))
        else:
            Initial=x0.reshape((N,M))

        x1 = np.memmap('results_choice2',shape=(N,M), mode='w+', dtype=np.float64)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(per_pixel_solve)(i, R[i],x1,Initial[i], H) for i in xrange(N))
        
        x0=x1.copy()
        x0=x0.reshape(-1)
        save_results(x0, arr, H, output_prefix)


    elif solve_choice==2:  #### per unique color pixel solve (much faster than choice 1, weights map is not smooth)
        
        method='trf'
        UNIQ_map=Unique_colors_pixel_mapping(img) ### data type is uint8, (0,255)
        UNIQ=np.array(UNIQ_map.keys()) #### should be N1*3
        print "unique colors: ", UNIQ.shape
        N1=len(UNIQ)
        plt.imsave(output_prefix+"-unique_colors.png", UNIQ.reshape((N1,1,3)))
        UNIQ_foat=UNIQ/255.0 ### do not foget        

        Initial=From_wegihts_for_compositedPigments_to_weights_for_uniqueColors(UNIQ_foat, H, prior_mixing_weights)

        if prior_mixing_weights!=None:
            Initial_x0=np.ones((N,M))
            ### map unique color pixel's weights back to orignal pixel postion.
            for i in xrange(N1):
                index_list=UNIQ_map[tuple(UNIQ[i])]
                Initial_x0[index_list,:]=Initial[i:i+1,:]

            Initial_x0=Initial_x0.reshape(-1)
            save_results(Initial_x0, arr, H, output_prefix+"-initial_recover")



        z0 = np.memmap('results_choice2',shape=(N1,M), mode='w+', dtype=np.float64)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(per_pixel_solve)(i, UNIQ_foat[i],z0,Initial[i], H) for i in xrange(N1))


        print z0.sum(axis=1).max()
        print z0.sum(axis=1).min()



        x0=np.ones((N,M))
        ### map unique color pixel's weights back to orignal pixel postion.
        for i in xrange(N1):
            index_list=UNIQ_map[tuple(UNIQ[i])]
            x0[index_list,:]=z0[i:i+1,:]

        x0=x0.reshape(-1)
        save_results(x0, arr, H, output_prefix)



    elif solve_choice==3:   

        print 'choice: ', solve_choice
        p_size=10
        patch_shape=(p_size,p_size)
        Lap=create_laplacian(patch_shape,M)


        x0,level=optimize_smaller(arr, H, p_size, Lap, 0)
 
        x0=x0.reshape((arr.shape[0],arr.shape[1],M))

        save_results(x0.reshape(-1), arr, H, output_prefix+"-p_size-"+str(p_size)+"-final_recursivelevel-initial-")


        x0[x0<eps]=eps #### to make value nonzero，for autograd.

        

        ### per pixel with spatial info.
        rows=arr.shape[0]
        cols=arr.shape[1]
        N=rows*cols
        num_cores = multiprocessing.cpu_count()
        max_loop=1
        half_win=3
        

        use_autograd=False ##### not use autograd for per pixel solve.
        ###### choose which gamma correction to use.
        Gamma_trans_img=choose_gama_function(use_autograd)

        # method='L-BFGS-B'
        method='trf'



        for loop in xrange(max_loop):
            print half_win
            print W_neighbors
            # x1=np.random.random_sample((arr.shape[0],arr.shape[1],M))
            x1 = np.memmap('results-perpixel-solve', shape=(rows,cols, M), mode='w+', dtype=np.float64)
            x1[:,:,:]=x0[:,:,:]


            # ### for loop version
            # for i in xrange(N): 
            #     row_ind=i/rows
            #     col_ind=i%cols
            #     per_pixel_solve_with_spatial_info(i, arr[row_ind,col_ind,:].reshape((1,1,3)), arr.shape, x1, x0[row_ind,col_ind,:], H, get_neighbors(i,x0,arr.shape), half_win=half_win)
           

            # ### old for loop version
            # for i in xrange(N):
            #     per_pixel_solve_with_spatial_info(i, arr, x1, x0, H, half_win=half_win)


                 
            # ### parallel version
            # Parallel(n_jobs=num_cores)(delayed(per_pixel_solve_with_spatial_info)(i, arr[i/rows,i%cols,:].reshape((1,1,3)), arr.shape, x1, x0[i/rows,i%cols,:].copy(), H, get_neighbors(i,x0,arr.shape), half_win=half_win) for i in xrange(N))
            

            ### old parallel version
            Parallel(n_jobs=num_cores)(delayed(per_pixel_solve_with_spatial_info)(i, arr, x1, x0, H, half_win=half_win, method=method) for i in xrange(N))


            x0=x1.copy()
            save_results(x0.reshape(-1), arr, H, output_prefix+"-p_size-"+str(p_size)+"-final_recursivelevel-loop-"+str(loop))
            half_win+=1
            W_neighbors+=2.0






    elif solve_choice==4:  #### per original patch solve #### can add spatial term to solver. (slower than choice 2 but may get smooth weights map)
        print "choice: ", solve_choice


        offset=0
        p_size=10
        row_step=arr.shape[0]/p_size #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
        col_step=arr.shape[1]/p_size
        patch_shape=(p_size,p_size)
        Lap=create_laplacian(patch_shape,M)

        if Weights_file_name=="None":
            ##### using random weights as input
            x0=np.random.random_sample((arr.shape[0],arr.shape[1],M))
            x0=x0/x0.sum(axis=2).reshape((arr.shape[0],arr.shape[1],1))

        elif Weights_file_name=="primary_pigments_mixing_weights.txt": #### use prior composited pigments's mixing weights as initial value.
                        
            UNIQ_map=Unique_colors_pixel_mapping(img) ### data type is uint8, (0,255)
            UNIQ=np.array(UNIQ_map.keys()) #### should be N1*3
            print "unique colors: ", UNIQ.shape
            N1=len(UNIQ)
            plt.imsave(output_prefix+"-unique_colors.png", UNIQ.reshape((N1,1,3)))
            UNIQ_foat=UNIQ/255.0 ### do not foget  
            
            Initial=From_wegihts_for_compositedPigments_to_weights_for_uniqueColors(UNIQ_foat, H, prior_mixing_weights)
            Initial_x0=np.ones((N,M))
            ### map unique color pixel's weights back to orignal pixel postion.
            for i in xrange(N1):
                index_list=UNIQ_map[tuple(UNIQ[i])]
                Initial_x0[index_list,:]=Initial[i:i+1,:]

            Initial_x0=Initial_x0.reshape(-1)
            save_results(Initial_x0, arr, H, output_prefix+"-initial_recover")
            x0=Initial_x0.copy()
            x0=x0.reshape((arr.shape[0],arr.shape[1],M))
            x0=x0/x0.sum(axis=2).reshape((arr.shape[0],arr.shape[1],1))

        else:
            ##### using existing wieghts file as input. x0 is already same shape as origianl image.
            x0=x0.reshape((arr.shape[0],arr.shape[1],M))
            x0=x0/x0.sum(axis=2).reshape((arr.shape[0],arr.shape[1],1))



        ##### parallel version
        x1 = np.memmap('results_choice4',shape=(arr.shape[0],arr.shape[1],M), mode='w+', dtype=np.float64)
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(per_patch_solve)(arr,col_step, i, x1, x0, H, p_size,offset,Lap) for i in xrange(row_step*col_step))
        
        x0=x1.copy()
        x0=x0.reshape(-1)
        save_results(x0, arr, H, output_prefix)
        


    elif solve_choice==5: #### use other choice weights results as input (so sys.argv[3] should not be None, solve whole image directly again. 
        print "choice: ", solve_choice
        p_size=10

        ##### this is suitable for using choice 4 weights results as input
        offset=p_size/2
        row_step=arr.shape[0]/p_size-1 #### here is naive situation: the original image can be divided by p_size. Need modify this to robustly fit for other examples. 
        col_step=arr.shape[1]/p_size-1


        patch_shape=(p_size,p_size)
        Lap=create_laplacian(patch_shape,M)
        
        ##### using existing wieghts file as input.
        x0=x0.reshape((arr.shape[0],arr.shape[1],M))
        x0=x0/x0.sum(axis=2).reshape((arr.shape[0],arr.shape[1],1))


        ###### parallel version
        x1 = np.memmap('results_choice5',shape=(arr.shape[0],arr.shape[1],M), mode='w+', dtype=np.float64)
        x1[:,:,:] = x0[:,:,:]
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(per_patch_solve)(arr,col_step, i, x1, x0,H, p_size,offset,Lap) for i in xrange(row_step*col_step))

        x0=x1.copy()
        x0=x0.reshape(-1)
        save_results(x0, arr, H, output_prefix)
       
    
    
    elif solve_choice==6:

        print 'solve_choice', solve_choice
        W_sm_K=np.float(sys.argv[10])
        W_sm_S=np.float(sys.argv[11])
        W_sm_KS=np.float(sys.argv[12])

        # W_sm_K, W_sm_S, W_sm_KS=np.array([0.005,0.05,1e-6])
        # # W_sm_K, W_sm_S, W_sm_KS=np.array([100.0,100.0,1.0])
        # # W_sm_K, W_sm_S, W_sm_KS=np.array([0.0,0.0,0.0])
        print W_sm_K, W_sm_S, W_sm_KS

        output_prefix=output_prefix+"-"+str(W_sm_K)+"-"+str(W_sm_S)+"-"+str(W_sm_KS)
        # output_prefix=output_prefix+"-W_sm_K_"+str(W_sm_K)+"-W_sm_S_"+str(W_sm_S)+"-W_sm_KS_"+str(W_sm_KS)
        
        Max_loop=5

        p_size=10

        output_prefix=output_prefix+"-Psize-"+str(p_size)

        patch_shape=(p_size,p_size)
        Lap=create_laplacian(patch_shape,M)

        x0,level=optimize_smaller(arr, H, p_size, Lap, 0)
        x0=x0.reshape((arr.shape[0],arr.shape[1],M))

        rows=arr.shape[0]
        cols=arr.shape[1]
        N=rows*cols
        num_cores = multiprocessing.cpu_count()

        half_win=2

        output_prefix=output_prefix+"-Win_"+str(2*half_win+1)

        for loop in xrange(Max_loop):

            #### do per pixel with neighbor info first to get weights
            x1 = np.memmap('results_choice6',shape=(rows,cols, M), mode='w+', dtype=np.float64)
            x1[:,:,:]=x0[:,:,:]
            Parallel(n_jobs=num_cores)(delayed(per_pixel_solve_with_spatial_info)(i, arr, x1, x0, H, half_win=half_win) for i in xrange(N))
            x0=x1.copy()

            save_results(x0.reshape(-1), arr, H, output_prefix+"-Weights"+"-alternate_loop-"+str(loop))

            if loop<Max_loop-1:
                #### fixed Weights, resolve KS, and using last loop KS as initial
                Weights=x0.reshape((-1,M))
                x_initial=H.reshape(-1)
                start1=time.time()
                x_H=optimize_fixed_Weights(x_initial, arr, Weights, W_sm_K, W_sm_S, W_sm_KS, method='trf')
                end1=time.time()
                print 'fixed weights,solve KS time: ', (end1-start1)
                save_pigments(x_H, M, output_prefix+"-Pigments"+"-alternate_loop-"+str(loop+1))
                H=x_H.reshape((M,-1))
                print "pigments KS diff", x_H-x_initial



    END=time.time()
    print 'total time: ', (END-START)
    

    
    