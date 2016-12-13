# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import scipy.optimize as sopt
from KS_helper_autograd import *
import time
import warnings
import json
import time
import PIL.Image as Image
from numba import jit
from Cross_Bilateral_Filter import *
from Constant_Values import *
from SILD_convexhull_simplification import *
import fast_energy_laplacian
import scipy.sparse


from autograd.numpy import *
import autograd.numpy as np
from autograd import elementwise_grad, jacobian
def jit(f):
    # print "###"
    return f

Gamma_trans_img=Gamma_trans_img3






@jit 
def objective_func_vector_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0,W_neighbors=0.0, neighbors=None):
    
    eps=1e-15
    M=H.shape[0]
    L=H.shape[1]/2
    N=len(x0)/M
    W=x0.reshape((N,M))


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

    return obj



def jacobian_objective_func_vector_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0):
    Jac=jacobian(objective_func_vector_fixed_KS,0)
    return Jac(x0, arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial)



@jit
def objective_func_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0):
    
    obj=objective_func_vector_fixed_KS(x0, arr, H, None, W_w, W_sparse, 0.0)
    
    spatial_term=0.0

    if W_spatial!=0.0:

        #### this is ok, but not supported by autograd library to compute gradient.
        M=H.shape[0]
        spatial_term=np.dot(x0,Smooth_Matrix.dot(x0))*W_spatial/M

        return np.square(obj).sum()+spatial_term
    else:
        return np.square(obj).sum()





def gradient_objective_func_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0):

    def temp(x0, arr, H, W_w, W_sparse):
        obj=objective_func_vector_fixed_KS(x0, arr, H, None, W_w, W_sparse, 0.0)
        return np.square(obj).sum()

    grad=elementwise_grad(temp,0)
    g=grad(x0, arr, H, W_w, W_sparse)

    if W_spatial!=0.0:
        M=H.shape[0]
        g2=2*Smooth_Matrix.dot(x0)*W_spatial/M
        g=g+g2

    return g


####suitable for autograd, since autograd is not supporting a[a>=100]=val such operation.
def Large_KS_value_penalty(KS_list):
    eps=1e-50
    thres=100
    out1=np.minimum(KS_list, thres)-thres
    out2=np.maximum(KS_list, thres)-thres

    temp1=0.002*KS_list-0.1
    temp2=0.001*KS_list
    
    out=np.divide((temp1*out1),(out1+eps))+np.divide((temp2*out2),(out2+eps))
    return out


@jit
def objective_func_vector_fixed_Weights(x0, arr, Weights, W_sm_K, W_sm_S, W_sm_KS):
    
    M=Weights.shape[1]
    L=len(x0)/(2*M)

    N=Weights.shape[0]

    x_H=x0.reshape((M,-1))
    K0=x_H[:,:L]
    S0=x_H[:,L:]

    ### reconstruction of input
    R_vector=KM_mixing_multiplepigments(K0, S0, Weights, r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction

    obj=R_rgb-arr.reshape((-1,3))
    obj=obj.reshape(-1)

    common_values=sqrt((1.0*N)/(M*(L-1)))

    obj2=np.ones(0)

    if W_sm_K!=0.0:
        K_gradient=K0[:,:-1]-K0[:,1:] 
        o1=K_gradient.reshape(M*(L-1)) * sqrt(W_sm_K)*common_values
        obj2=np.concatenate((obj2,o1))

    if W_sm_S!=0.0:
        S_gradient=S0[:,:-1]-S0[:,1:]
        o2=S_gradient.reshape(M*(L-1)) * sqrt(W_sm_S)*common_values
        obj2=np.concatenate((obj2,o2))
    
    if W_sm_KS!=0.0:
        KS_vector=np.divide(K0,S0)
        KS_gradient=KS_vector[:,:-1]-KS_vector[:,1:]
        o3=KS_gradient.reshape(M*(L-1))*sqrt(W_sm_KS)*common_values
        obj2=np.concatenate((obj2,o3))


        ###### large KS value penalty.
        # o4=Large_KS_value_penalty(KS_vector)
        # W_large_KS=0.0000001*common_values
        # obj2=np.concatenate((obj2,W_large_KS*o4))
    

    obj=np.concatenate((obj,obj2))


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




def optimize(arr, x0, H, Smooth_Matrix, saver=None, W_w=2.0, W_sparse=0.1, W_spatial=0.0, method='L-BFGS-B'):
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

    x0[x0<lb]=lb


    start_time=time.clock()
    if method=='trf':
        if use_autograd==False:
            res=sopt.least_squares(objective_func_vector_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial), bounds=bounds0, jac='2-point', method='trf')   
        else:
            res=sopt.least_squares(objective_func_vector_fixed_KS, x0, args=(arr,H, Smooth_Matrix, W_w, W_sparse, W_spatial), bounds=bounds0, jac=jacobian_objective_func_vector_fixed_KS, method='trf')   

    else:
        if use_autograd==False:
            res=sopt.minimize(objective_func_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial), bounds=bounds3, method=method, callback=saver)
        else:
            # res=sopt.minimize(objective_func_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial), bounds=bounds3, method=method, jac=gradient_objective_func_fixed_KS, callback=saver)
            res=sopt.minimize(objective_func_fixed_KS, x0, args=(arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial), bounds=bounds3, method=method, jac=gradient_objective_func_fixed_KS, callback=saver,options={'gtol':1e-4, 'ftol': 1e-4})

    end_time=time.clock()
    # print 'Optimize variables of size ', x0.shape, ' took ', (end_time-start_time), ' seconds.'
    x=res["x"]


    return x



def optimize_fixed_Weights(x0, arr, W, W_sm_K, W_sm_S, W_sm_KS, method='L-BFGS-B'):

    arr_shape=arr.shape
    N=arr_shape[0]*arr_shape[1]
    M=W.shape[1]
    L=len(x0)/(2*M)

    eps=1e-8
    x0[x0<eps]=eps

    #### bounds0 and bounds1 are for least_squares function parameters.
    bounds0=(eps, np.inf)

    bounds3=[]
    for i in xrange(len(x0)):
        bounds3.append((eps,None))


    start_time=time.clock()
    if method=='trf':
        if use_autograd==False:
            res=sopt.least_squares(objective_func_vector_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS), bounds=bounds0,jac='2-point', method='trf')   
        else:
            res=sopt.least_squares(objective_func_vector_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS), bounds=bounds0,jac=jacobian_objective_func_vector_fixed_Weights, method='trf')
    else:
        if use_autograd==False:
            res=sopt.minimize(objective_func_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS),bounds=bounds3, method=method)
        else:
            # res=sopt.minimize(objective_func_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS),bounds=bounds3, method=method, jac=gradient_objective_func_fixed_Weights)
            res=sopt.minimize(objective_func_fixed_Weights, x0, args=(arr, W, W_sm_K, W_sm_S, W_sm_KS),bounds=bounds3, method=method, jac=gradient_objective_func_fixed_Weights, options={'gtol':1e-4, 'ftol': 1e-4})



    end_time=time.clock()

    x=res["x"]
    # print res["message"]

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
    
    filename=output_prefix+"fixed_KS-reconstructed.png"
    plt.imsave(filename,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))
    np.savetxt(output_prefix+"mixing_weights.txt", Weights)


    ### compute sparsity
    sparsity_thres_array=np.array([0.000001, 0.00001, 0.0001,0.001,0.01,0.1])
    Weights_sparsity_list=np.ones(len(sparsity_thres_array))
    for thres_ind in xrange(len(sparsity_thres_array)):
        Weights_sparsity_list[thres_ind]=len(Weights[Weights<=sparsity_thres_array[thres_ind]])*1.0/(N*M)
    
    print "Weights_sparsity_list: ", Weights_sparsity_list
    np.savetxt(output_prefix+"mixing_weights-Sparsity.txt", Weights_sparsity_list)



    # normalized_Weights=Weights/Weights.sum(axis=1).reshape((-1,1))
    # for i in xrange(M):
    #     #### save normalized_weights_map for each pigment.
    #     normalized_weights_map_name=output_prefix+"normalized_mixing_weights_map-%02d.png" % i
    #     normalized_Weights_map=normalized_Weights[:,i].reshape(img_size).copy()
    #     cv2.imwrite(normalized_weights_map_name, (normalized_Weights_map*255.0).clip(0,255).round().astype(np.uint8))
        

    # Weights_sum_map=Weights.sum(axis=1).reshape(img_size)
    # W_min=Weights_sum_map.min()
    # W_max=Weights_sum_map.max()
    # Weights_sum_map=Weights_sum_map/W_max

    # cv2.imwrite(output_prefix+"mixing_weights_sum_map-min-"+str(W_min)+"-max-"+str(W_max)+".png", (Weights_sum_map*255.0).round().astype(np.uint8))
        



    for i in xrange(M):
        #### save weights_map for each pigment.
        weights_map_name=output_prefix+"mixing_weights_map-%02d.png" % i
        Weights_map=Weights[:,i].reshape(img_size).copy()
        Image.fromarray((Weights_map*255.0).clip(0,255).round().astype(np.uint8)).save(weights_map_name)


        # #### use weights as thickness, to show each pigments on white background.
        # pigments_map_name=output_prefix+"pigments_visualize_map-%02d.png" % i
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
    # shape=arr_shape
    # original_shape=shape
    # img_size=shape[:2]
    # N=shape[0]*shape[1]
    # M=H.shape[0]
    # L=H.shape[1]/2

    # Weights=x0.reshape((N,M))
    # K0=H[:,:L]
    # S0=H[:,L:]
    # print Weights.sum(axis=1).min()
    # print Weights.sum(axis=1).max()

    # ### reconstruction of input
    # R_vector=KM_mixing_multiplepigments(K0, S0, Weights, r=1.0, h=1.0)
    # ### from R spectrum x wavelength spectrums to linear rgb colors 
    # P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    # R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    # Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    # R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    # R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    # R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    
    # # print 'RGB RMSE: ', np.sqrt(np.square(255*(arr.reshape((-1,3))-R_rgb)).sum()/N)
    
    # filename=output_prefix+"-fixed_KS-reconstructed.png"
    # plt.imsave(filename,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))
    # # np.savetxt(output_prefix+"-mixing_weights.txt", Weights)


    # ## compute sparsity
    # # sparsity_thres_array=np.array([0.000001, 0.00001, 0.0001,0.001,0.01,0.1])
    # # Weights_sparsity_list=np.ones(len(sparsity_thres_array))
    # # for thres_ind in xrange(len(sparsity_thres_array)):
    # #     Weights_sparsity_list[thres_ind]=len(Weights[Weights<=sparsity_thres_array[thres_ind]])*1.0/(N*M)
    
    # # print "Weights_sparsity_list: ", Weights_sparsity_list
    # # np.savetxt(output_prefix+"-mixing_weights-Sparsity.txt", Weights_sparsity_list)



    # # normalized_Weights=Weights/Weights.sum(axis=1).reshape((-1,1))
    # # for i in xrange(M):
    # #     #### save normalized_weights_map for each pigment.
    # #     normalized_weights_map_name=output_prefix+"-normalized_mixing_weights_map-%02d.png" % i
    # #     normalized_Weights_map=normalized_Weights[:,i].reshape(img_size).copy()
    # #     cv2.imwrite(normalized_weights_map_name, (normalized_Weights_map*255.0).clip(0,255).round().astype(np.uint8))
        

    # # Weights_sum_map=Weights.sum(axis=1).reshape(img_size)
    # # W_min=Weights_sum_map.min()
    # # W_max=Weights_sum_map.max()
    # # Weights_sum_map=Weights_sum_map/W_max

    # # cv2.imwrite(output_prefix+"-mixing_weights_sum_map-min-"+str(W_min)+"-max-"+str(W_max)+".png", (Weights_sum_map*255.0).round().astype(np.uint8))
        

    # for i in xrange(M):
    #     #### save weights_map for each pigment.
    #     weights_map_name=output_prefix+"-mixing_weights_map-%02d.png" % i
    #     Weights_map=Weights[:,i].reshape(img_size).copy()
    #     cv2.imwrite(weights_map_name, (Weights_map*255.0).clip(0,255).round().astype(np.uint8))

    pass





def save_pigments(x_H, M, output_dir):

    L=len(x_H)/(2*M)
    x_H=x_H.reshape((M,-1))

    np.savetxt(output_dir+"primary_pigments_KS-"+str(M)+".txt", x_H)

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
    plt.imsave(output_dir+"primary_pigments_color-"+str(M)+".png", (pigments*255.0).round().astype(np.uint8))

    # from scipy.spatial import ConvexHull
    # hull=ConvexHull(R_rgb*255.0)
    # faces=hull.points[hull.simplices]
    with open(output_dir+"primary_pigments_color_vertex-"+str(M)+".js", "w") as fd:
        json.dump({'vs': (R_rgb*255.0).tolist()}, fd)

  
    # xaxis=np.arange(L)
    # for i in xrange(M):
    #     fig=plt.figure()
    #     plt.plot(xaxis, K0[i], 'b-')
    #     fig.savefig(output_dir+"primary_pigments_K_curve-"+str(i)+".png")
    #     fig=plt.figure()
    #     plt.plot(xaxis, S0[i], 'b-')
    #     fig.savefig(output_dir+"primary_pigments_S_curve-"+str(i)+".png")
    #     fig=plt.figure()
    #     plt.plot(xaxis, R_vector[i], 'b-')
    #     fig.savefig(output_dir+"primary_pigments_R_curve-"+str(i)+".png")
    #     fig=plt.figure()
    #     plt.plot(xaxis, K0[i]/S0[i], 'b-')
    #     fig.savefig(output_dir+"primary_pigments_KS_curve-"+str(i)+".png")
    #     plt.close('all')







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


kSaveEverySeconds = 10
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


def per_pixel_solve(i, pixel, z0, initial, H, W_w, W_sparse, W_spatial, method='trf'):
    # if i%10000==0:
    #     print "pixel: ", i
    pixel=pixel.reshape((1,1,-1))
    y0=initial
    y0/=y0.sum()
    y0 = optimize(pixel, y0, H, None, saver=None, W_w=W_w, W_sparse=W_sparse, W_spatial=0.0, method=method)
    z0[i,:]=np.array(y0)
   

def KM_solve_ANLS(arr, W, H, output_prefix, max_loop, W_w, W_sparse, W_spatial, W_sm_K, W_sm_S, W_sm_KS):
    arr_shape=arr.shape[:2]
    method1='L-BFGS-B'
    method2='trf'
    M=H.shape[0]
    N=len(W.reshape(-1))/M
    R=arr.reshape((-1,3))
    L=H.shape[-1]/2
    
    final_loop=0
    x0=W.reshape((-1,M))
    num_cores = multiprocessing.cpu_count()
    thres=0.1

    for loop in xrange(max_loop):

        if W_spatial==0.0:
            # x1 = np.memmap(output_prefix+"-results",shape=(N,M), mode='w+', dtype=np.float64)
            # x1[:,:]=x0[:,:]
            # Parallel(n_jobs=num_cores)(delayed(per_pixel_solve)(i, R[i], x1, x0[i], H, W_w, W_sparse, W_spatial, method=method1) for i in xrange(N))
            # x0=x1.copy()
            Smooth_Matrix=None

        else:
            Blf=create_cross_bilateral(arr*255.0, M)
            Smooth_Matrix=Blf

        x0 = optimize(arr, x0.reshape(-1), H, Smooth_Matrix, saver=None, W_w=W_w, W_sparse=W_sparse, W_spatial=W_spatial, method=method1)




        # save_results(x0.reshape(-1), arr, H, output_prefix+"-Weights"+"-alternate_loop-"+str(loop)+"-")

        if loop<max_loop-1:

            #### fixed Weights, resolve KS, and using last loop KS as initial
            Weights=x0.reshape((-1,M))
            x_initial=H.reshape(-1)


            start1=time.time()
            x_H=optimize_fixed_Weights(x_initial, arr, Weights, W_sm_K, W_sm_S, W_sm_KS, method=method1)
            end1=time.time()
            # print 'fixed weights,solve KS time: ', (end1-start1)
            
            # save_pigments(x_H, M, output_prefix+"-Pigments"+"-alternate_loop-"+str(loop+1)+"-")

            H=x_H.reshape((M,-1))
            # print "pigments KS diff: \n", x_H-x_initial

            diff=x_H-x_initial

            max_diff_ratio=(abs(diff)/np.minimum(x_H,x_initial)).max()
            print max_diff_ratio


            final_loop=loop
            
            if max_diff_ratio<=thres:
                print "H diff abs max value is smaller than "+str(thres)+" after "+str(loop)+" iterations"
                thres=thres/10

            if max_diff_ratio<=1e-3 or loop==(max_loop-2):
                print max_diff_ratio
                abs_diff=abs(diff).reshape((M,-1))
                print abs_diff
                print loop
                # K_diff=abs_diff[:,:L]
                # print K_diff.max()
                # S_diff=abs_diff[:,L:]
                # print S_diff.max()

                break

            


    return x0, H, final_loop+1






def choose_good_initial_H_from_existing_H(arr, Existing_H, M, representative_color_choice=0, choose_corresponding_existing_KS_RGB_color_choice=0, output_prefix=None):
    L=Existing_H.shape[-1]/2
    H=np.ones((M,2*L))
    RGB_colors=np.ones((M,1,3))
    data=arr.reshape((-1,3))
    from scipy.spatial import ConvexHull
    from sklearn.cluster import KMeans
    output_rawhull_obj_file=output_prefix+"/SILD_mesh_objfile.obj"


##### generate representative RGB colors for arr.

    if representative_color_choice==0:  #### default option
    #### use simplification hull vertices as representative colors
        
        hull=ConvexHull(data)
        write_convexhull_into_obj_file(hull, output_rawhull_obj_file)
        N=500
        mesh=TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
        for i in range(N):
            old_num=len(mesh.vs)
            mesh=TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
            mesh=remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh,option=2)
            newhull=ConvexHull(mesh.vs)
            write_convexhull_into_obj_file(newhull, output_rawhull_obj_file)

            if len(mesh.vs)==M:
                Final_hull=newhull
                break

        Hull_vertices=Final_hull.points[Final_hull.vertices].clip(0,1)



    elif representative_color_choice==1: 

        Hull=ConvexHull(data)
        Hull_vertices=Hull.points[Hull.vertices]
        k_means = KMeans(n_clusters=M, n_init=4)
        k_means.fit(Hull_vertices)
        values = k_means.cluster_centers_.squeeze()
        Hull_vertices=values.copy()



    elif representative_color_choice==2: 
        k_means = KMeans(n_clusters=M, n_init=4)
        k_means.fit(data)
        values = k_means.cluster_centers_.squeeze()
        Hull_vertices=values.copy()



    def Get_closest_matching_color_and_Remove_duplicate_primary_color(diff):
        ### diff shape is N1*M 
        min_indices=np.argmin(diff,axis=0)#### shape is (M,)

        # return min_indices

        new_min_indices=min_indices.copy()
        min_diff=np.ones(min_indices.shape)

        # print diff[:,0]
        # print min_indices[0]

        for i in range(M):
            min_diff[i]=diff[min_indices[i],i]

        print min_indices
        
        for i in range(M-1):
            for j in range(i+1,M):
                if min_indices[i]==min_indices[j]:
                    if min_diff[i]<=min_diff[j]:
                        temp_diff=diff[:,j]
                        temp_ind=np.argsort(temp_diff)
                        ind=temp_ind[1] #### temp_ind[1] is second smallest indices
                        new_min_indices[j]=ind
                        min_diff[j]=temp_diff[ind]
                    else:
                        temp_diff=diff[:,i]
                        temp_ind=np.argsort(temp_diff)
                        ind=temp_ind[1] #### temp_ind[1] is second smallest indices
                        new_min_indices[i]=ind
                        min_diff[i]=temp_diff[ind]

        return new_min_indices



    if choose_corresponding_existing_KS_RGB_color_choice==0:  ####default option

        ####compute existing KS pigments RGB colors
        R_vector=equations_in_RealPigments(Existing_H[:,:L], Existing_H[:,L:], r=1.0, h=2.0)
        ### from R spectrum x wavelength spectrums to linear rgb colors 
        P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
        R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
        Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
        R_xyz/=Normalize ####xyz value shape is M*3
        R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
        R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction

         
        ##### get closet corresponding RGB colors indices.
        diff=R_rgb.reshape((-1,1,3))-Hull_vertices.reshape((1,M,3)) #### shape is N1*M*3
        diff=np.square(diff).sum(axis=2) ### shape is (N1,M)


        # min_indices=np.argmin(diff,axis=0)#### shape is (M,)
        
        min_indices=Get_closest_matching_color_and_Remove_duplicate_primary_color(diff)


        for i in range(M):
            H[i]=Existing_H[min_indices[i]]
            RGB_colors[i,:,:]=R_rgb[min_indices[i]]


    elif choose_corresponding_existing_KS_RGB_color_choice==1: #### use interpolated Existing KS to do computation
        

        Num=len(Existing_H)
        Existing_H_expand=np.zeros((Num+Num*(Num-1)/2, Existing_H.shape[1]))
        Existing_H_expand[:Num,:]=Existing_H
        count=0
        for i in range(Num-1):
            for j in range(i+1,Num):
                Existing_H_expand[Num+count,:]=(Existing_H[i,:]+Existing_H[j,:])/2.0
                count+=1

        Existing_H=Existing_H_expand.copy()

        np.savetxt(output_prefix+"/Existing_H_expand_KS.txt", Existing_H_expand)


        ####compute existing KS pigments RGB colors
        R_vector=equations_in_RealPigments(Existing_H[:,:L], Existing_H[:,L:], r=1.0, h=1.0)
        ### from R spectrum x wavelength spectrums to linear rgb colors 
        P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
        R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
        Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
        R_xyz/=Normalize ####xyz value shape is M*3
        R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
        R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction


        with open (output_prefix+"/Existing_H_expand_RGB_colors.js","w") as myfile:
            json.dump({'vs': (R_rgb*255).tolist()}, myfile)

        Image.fromarray((R_rgb*255).round().astype(np.uint8).reshape((Num/2, -1, 3))).save(output_prefix+"/Existing_H_expand_RGB_colors.png")


         
        ##### get closet corresponding RGB colors indices.
        diff=R_rgb.reshape((-1,1,3))-Hull_vertices.reshape((1,M,3)) #### shape is N1*M*3
        diff=np.square(diff).sum(axis=2) ### shape is (N1,M)


        # min_indices=np.argmin(diff,axis=0)#### shape is (M,) ### sometimes two different vertices will match same okumura RGB color.
        
        min_indices=Get_closest_matching_color_and_Remove_duplicate_primary_color(diff)



        for i in range(M):
            H[i]=Existing_H[min_indices[i]]
            RGB_colors[i,:,:]=R_rgb[min_indices[i]]

        



    elif choose_corresponding_existing_KS_RGB_color_choice==2: ### solve optimization

        cova=np.cov(Existing_H.transpose())
        mean=np.average(Existing_H, axis=0)

        def obj_func_vector(x0, Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS):
            M=Hull_vertices.shape[0]
            L=len(x0)/(2*M)
            x_H=x0.reshape((M,2*L))

            K0=x_H[:,:L]
            S0=x_H[:,L:]

            R_vector=equations_in_RealPigments(K0, S0, r=1.0, h=1.0)
            P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
            R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
            Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
            R_xyz/=Normalize ####xyz value shape is M*3
            R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
            R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction

            obj=R_rgb-Hull_vertices
            obj=obj.reshape(-1)

            obj2=np.ones(0)
            if W_sm_K!=0.0:
                K_gradient=K0[:,:-1]-K0[:,1:]
                o1=K_gradient.reshape(M*(L-1)) * sqrt(W_sm_K)
                obj2=np.concatenate((obj2,o1))
            if W_sm_S!=0.0:
                S_gradient=S0[:,:-1]-S0[:,1:]
                o2=S_gradient.reshape(M*(L-1)) * sqrt(W_sm_S)
                obj2=np.concatenate((obj2,o2))
            
            if W_sm_KS!=0.0:
                KS_vector=np.divide(K0,S0)
                KS_gradient=KS_vector[:,:-1]-KS_vector[:,1:]
                o3=KS_gradient.reshape(M*(L-1))*sqrt(W_sm_KS)
                obj2=np.concatenate((obj2,o3))

            obj2=np.sqrt((1.0*N)/(M*(L-1)))*obj2
            obj=np.concatenate((obj,obj2))

            if W_Mahalanobis!=0.0:
                val_list=[]
                for i in range(M):
                    P=x_H[i:i+1,:]-mean.reshape((1,-1))
                    val_list.append(sqrt(np.dot(np.dot(P, np.linalg.inv(cova)),P.transpose())[0,0]))

                obj=np.concatenate((obj,np.array(val_list)*sqrt((N*1.0)/M)))

            return obj


        def obj_func(x0, Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS):
            obj=obj_func_vector(x0, Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS)
            return np.square(obj).sum()

        def gradient_obj_func(x0, Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS):
            grad=elementwise_grad(obj_func,0)
            return grad(x0, Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS)



        W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS=[1e-9,0.001,0.001,1e-6]

        #####initial x0
        x0=np.ones((M,2*L))
        ####compute existing KS pigments RGB colors
        R_vector=equations_in_RealPigments(Existing_H[:,:L], Existing_H[:,L:], r=1.0, h=1.0)
        ### from R spectrum x wavelength spectrums to linear rgb colors 
        P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
        R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
        Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
        R_xyz/=Normalize ####xyz value shape is M*3
        R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
        R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
        ##### get closet corresponding RGB colors indices.
        diff=R_rgb.reshape((-1,1,3))-Hull_vertices.reshape((1,M,3)) #### shape is N1*M*3
        diff=np.square(diff).sum(axis=2) ### shape is (N1,M)
        min_indices=np.argmin(diff,axis=0)#### shape is (M,)
        for i in range(M):
            x0[i]=Existing_H[min_indices[i]]


        lb=1e-3
        ub=10.0

        bounds0=(lb, ub)
        bounds3=[]
        for i in range(2*M*L):
            bounds3.append((lb,ub))

        x0[x0<lb]=lb
        res=sopt.least_squares(obj_func_vector, x0.reshape(-1), args=(Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS), bounds=bounds0,jac='2-point', method='trf')   
        # res=sopt.minimize(obj_func, x0.reshape(-1), args=(Hull_vertices, cova, W_Mahalanobis, W_sm_K, W_sm_S, W_sm_KS), bounds=bounds3,jac=gradient_obj_func, method='L-BFGS-B')   

        x0=res["x"]
        H=x0.reshape((M,2*L))

        R_vector=equations_in_RealPigments(H[:,:L], H[:,L:], r=1.0, h=1.0)
        P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
        R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
        Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
        R_xyz/=Normalize ####xyz value shape is M*3
        RGB_colors=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
        RGB_colors=Gamma_trans_img(RGB_colors.clip(0,1)) ##clip and gamma correction




    return H, RGB_colors*255.0, Hull_vertices*255.0




def choose_good_initial_H_from_existing_H_for_patch_with_2_pigments_model(arr, Existing_H, M=2):
    #### svd to get primary dirction, and project patch color points on to that axis.
    L=Existing_H.shape[-1]/2
    arr=arr.reshape((-1,3))
    arr_mean=arr.mean(axis=0)

    uu,ss,vv=np.linalg.svd(arr-arr_mean.reshape((-1,3)))
    a=(vv[0].reshape((1,-1))*(arr-arr_mean)).sum(axis=1)
    arr_projected_max=arr[a==a.max()][0]
    arr_projected_min=arr[a==a.min()][0]
    arr_projected=np.ones((2,3))
    arr_projected[0]=arr_projected_max
    arr_projected[1]=arr_projected_min
    

    Num=len(Existing_H)
    Existing_H_expand=np.zeros((Num+Num*(Num-1)/2, Existing_H.shape[1]))
    Existing_H_expand[:Num,:]=Existing_H[:,:]
    count=0
    for i in range(Num-1):
        for j in range(i+1,Num):
            Existing_H_expand[Num+count,:]=(Existing_H[i,:]+Existing_H[j,:])/2.0
            count+=1

    Existing_H=Existing_H_expand.copy()

    # np.savetxt(output_prefix+"/Existing_H_expand_KS.txt", Existing_H_expand)


    ####compute existing KS pigments RGB colors
    R_vector=equations_in_RealPigments(Existing_H[:,:L], Existing_H[:,L:], r=1.0, h=1.0)
    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is M*L
    R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape M*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is M*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is M*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction


    # with open (output_prefix+"/Existing_H_expand_RGB_colors.js","w") as myfile:
    #     json.dump({'vs': (R_rgb*255).tolist()}, myfile)

    # Image.fromarray((R_rgb*255).round().astype(np.uint8).reshape((Num/2, Num+1, -1))).save(output_prefix+"/Existing_H_expand_RGB_colors.png")


     
    ##### get closet corresponding RGB colors indices.
    diff=R_rgb.reshape((-1,1,3))-arr_projected.reshape((1,M,3)) #### shape is N1*M*3
    diff=np.square(diff).sum(axis=2) ### shape is (N1,M)
    min_indices=np.argmin(diff,axis=0)#### shape is (M,)

    H=np.ones((M,2*L))
    RGB_colors=np.ones((M,1,3))

    for i in range(M):
        H[i]=Existing_H[min_indices[i]]
        RGB_colors[i,:,:]=R_rgb[min_indices[i]]


    return H, RGB_colors*255.0, arr_projected*255.0



def downsample(img_file_name):
    import numpy as np
    import skimage.transform
    import PIL.Image as Image
    import os
    img=Image.open(img_file_name).convert('RGB')
    img=np.asfarray(img)
    # print img.dtype()
    img_small=skimage.transform.pyramid_reduce(img,downscale=8,order=3)
    # print img_small.dtype()
    Image.fromarray(img_small.astype(np.uint8)).save(os.path.splitext(img_file_name)[0]+'-downsample.png')




####### sample pixels from image and put them together.
def sample_RGBcolors(RGB_colors, sample_num, bin_num=16):

    bin2count ={}
    bin2xy ={}

    import itertools
    bin_list=itertools.product(np.arange(bin_num), repeat=3)
    for element in bin_list:
        bin2count[element]=0
        bin2xy.setdefault(element,[])

    step=256/bin_num
    for index in range(len(RGB_colors)):
        element=RGB_colors[index].copy()
        element/=step
        bin2count[tuple(element)]+=1
        bin2xy[tuple(element)].append(index)

    count_colors=np.array(bin2count.values())
    bin_list_num=len(count_colors)
    
    # print count_colors.min()
    # print count_colors.max()
    NonEmptyNum=bin_list_num-len(count_colors[count_colors==0]) #### number of bins that contain at least one color of input image.
    # print NonEmptyNum
        
    i=0
    sample_RGB_colors=np.zeros((sample_num,1,3),dtype=np.uint8)
    while True:
        ind1=(int)(np.random.random_sample()*bin_list_num)
        x=ind1/(bin_num*bin_num)
        y=(ind1-x*bin_num*bin_num)/bin_num
        z=ind1-x*bin_num*bin_num-y*bin_num
        index_list=bin2xy[tuple(np.array([x,y,z]))]
        if len(index_list)!=0:
            ind2=(int)(np.random.random_sample()*len(index_list))
            index=index_list[ind2]
            sample_RGB_colors[i,0,:]=RGB_colors[index,:]
            i+=1
        if i==sample_num:
            break
            
    return sample_RGB_colors



def sample_RGBcolors_new(RGB_colors, sample_num, bin_num=16):
    data=RGB_colors.reshape((-1,3))
    hull=ConvexHull(data)
    vertices=hull.points[hull.vertices]
    vertices_num=len(vertices)
    print vertices_num

    return vertices.reshape((-1,1,3)).astype(np.uint8) ####  directly use hull vertices to be sampled pixels

    if vertices_num>sample_num:
        print "Already increase sample pixel number to 900"
        sample_num=900
        
    sample_pixels=np.ones((sample_num,1, 3), dtype=np.uint8)

    sample_pixels[:vertices_num,0,:]=vertices
    
    if vertices_num<sample_num:
        sampled_RGB_colors=sample_RGBcolors(RGB_colors, sample_num-vertices_num)
        sample_pixels[vertices_num:, :, :]=sampled_RGB_colors
    
    return sample_pixels


### Four usage examples:
### cd /sampled_pixels-flower-small-groundtruthTest/sampled_pixels-400
### python ../../step1_ANLS_with_autograd.py sampled_pixels-400.png None 0 None flower-small-groundtruth-sampled_pixels-400-ANLS-with_random_KS  0 5 10.0 0.1 0.0 0.005 0.05 1e-6
### python ../../step1_ANLS_with_autograd.py sampled_pixels-400.png Existing_KS_parameter_KS_chosed_for_flower-5.txt 1 None flower-small-groundtruth-sampled_pixels-400-ANLS-with_known_KS  0 5 10.0 0.1 0.0 0.005 0.05 1e-6
### python ../../step1_ANLS_with_autograd.py sampled_pixels-400.png Existing_KS_parameter_KS.txt 2 None flower-small-groundtruth-sampled_pixels-400-ANLS-with_exisiting_26_KS  0 5 10.0 0.1 0.0 0.005 0.05 1e-6

### python ../../step1_ANLS_with_autograd.py downsampled.png None 0 None flower-small-groundtruth-downsampled-ANLS  0 5 10.0 0.1 1.0 0.005 0.05 1e-6




if __name__=="__main__":

    img_file=sys.argv[1]

    KS_file_name=sys.argv[2]
    KS_choice=np.float(sys.argv[3]) #### 0 means input KS is not speficied. 1 means input is some choosed KS. 2 means input is 26 existing KS, so need choose M KS later.
    Weights_file_name=sys.argv[4]
    output_prefix=sys.argv[5]
    solve_choice=np.int(sys.argv[6])
    M=np.int(sys.argv[7])

    W_w=np.float(sys.argv[8])
    W_sparse=np.float(sys.argv[9])
    W_spatial=np.float(sys.argv[10])
    W_sm_K=np.float(sys.argv[11])
    W_sm_S=np.float(sys.argv[12])
    W_sm_KS=np.float(sys.argv[13])

    foldername=sys.argv[14]
    gt_H_name=sys.argv[15]
    representative_color_choice=np.int(sys.argv[16])
    choose_corresponding_existing_KS_RGB_color_choice=np.int(sys.argv[17])
    max_loop=np.int(sys.argv[18])
    sample_num=np.int(sys.argv[19])




    # W_sm_K, W_sm_S, W_sm_KS=np.array([0.005,0.05,1e-6])
    # # W_sm_K, W_sm_S, W_sm_KS=np.array([100.0,100.0,1.0])
    # # W_sm_K, W_sm_S, W_sm_KS=np.array([0.0,0.0,0.0])
    print 'pigment num ', M
    print 'W_w ', W_w
    print 'W_sparse ',W_sparse
    print 'W_spatial ',W_spatial
    print W_sm_K, W_sm_S, W_sm_KS


    output_prefix=output_prefix+"-KS_choice-"+str(KS_choice)+"-solve_choice-"+str(solve_choice)+"-M-"+str(M)+"-representative_color_choice-"+str(representative_color_choice)+"-choose_corresponding_existing_KS_RGB_color_choice-"+str(choose_corresponding_existing_KS_RGB_color_choice)
    output_prefix=output_prefix+"-W_w-"+str(W_w)+"-W_sparse-"+str(W_sparse)+"-W_spatial-"+str(W_spatial)+"-W_sm_K-"+str(W_sm_K)+"-W_sm_S-"+str(W_sm_S)+"-W_sm_KS-"+str(W_sm_KS)+"-max_loop-"+str(max_loop)

    base_dir="/Users/jianchao/Documents/Research/Adobe_Jianchao/Brushstroke_Project/Adobe_inside/CODE/pigment-parameters-newVersion/new_pipeline_executable"
    base_dir=base_dir+foldername+"/"
    output_prefix_copy=base_dir+output_prefix
    make_sure_path_exists(output_prefix_copy)
    output_prefix=output_prefix_copy+"/ANLS"
    
    img=np.asarray(Image.open(base_dir+img_file).convert('RGB'))
    print img.shape


    # arr=sample_RGBcolors(img.reshape((-1,3)), sample_num) #### sample pixels from image. sample_num is set to be square number. like 400, 625, 900, 1600 and so on.
    arr=sample_RGBcolors_new(img.reshape((-1,3)), sample_num)

    # arr=arr.reshape((np.int(sqrt(sample_num)),np.int(sqrt(sample_num)),3))
    arr=arr.reshape((-1,1,3))

    Image.fromarray(arr).save(base_dir+"/sampled_pixels-"+str(sample_num)+".png")
    arr=arr/255.0

    L=len(cie1931new)
    eps=1e-15


    if KS_choice==0 and KS_file_name=="None":

        H=np.random.random_sample((M,2*L))

        # H=np.ones((M,2*L))
        # H[:,:]=np.random.random_sample((M,1))

        H[:,:L]*=10
        H[H<eps]=eps

    elif KS_choice==1 and KS_file_name!="None":
        H=np.loadtxt(base_dir+KS_file_name)
        print H.shape

    elif KS_choice==2 and KS_file_name!="None":
        Existing_H=np.loadtxt(base_dir+KS_file_name)

        H, RGB_Colors,Hull_vertices=choose_good_initial_H_from_existing_H(arr, Existing_H, M, representative_color_choice, choose_corresponding_existing_KS_RGB_color_choice, output_prefix_copy)
        # H, RGB_Colors,Hull_vertices=choose_good_initial_H_from_existing_H(img.reshape((-1,3))/255.0, Existing_H, M, representative_color_choice, choose_corresponding_existing_KS_RGB_color_choice, output_prefix_copy)

        print H.shape


        prefix = output_prefix+"-representative_color_for_input_img"
        with open( prefix + ( '-%02d.js' % len(Hull_vertices )), 'w') as myfile:
            json.dump({'vs': Hull_vertices.reshape((-1,3)).tolist()}, myfile)
        Image.fromarray( Hull_vertices.round().astype(np.uint8).reshape((1,-1,3)) ).save(prefix + ( '-%02d.png' % len(Hull_vertices)))

        np.savetxt(output_prefix+"-choosed_good_initial_H_from_Existing_H.txt", H)

        with open(output_prefix+"-choosed_good_initial_H_from_Existing_H-pigment_colors.js", 'w') as myfile2:
            json.dump({'vs': RGB_Colors.reshape((-1,3)).tolist()}, myfile2)
        Image.fromarray( RGB_Colors.round().astype(np.uint8).reshape((1,-1,3)) ).save(output_prefix+"-choosed_good_initial_H_from_Existing_H-pigment_colors.png")

    else:
        print "wrong KS choice!"


    original_shape=img.shape
    img_size=img.shape[:2]

        
    if Weights_file_name=="None":

        # W=np.random.random_sample((arr.shape[0]*arr.shape[1],M))
        # W=W/W.sum(axis=1).reshape((-1,1))

        W=np.ones((arr.shape[0]*arr.shape[1],M))/M

        x0=W.reshape(-1)

    else:
        extention=os.path.splitext(Weights_file_name)[1]
        print extention
        if extention==".js":
            with open(base_dir+Weights_file_name) as data_file:    
                W_js = json.load(data_file)
            W=np.array(W_js['weights'])
        if extention==".txt":
            W=np.loadtxt(base_dir+"/"+Weights_file_name)
            W=W.reshape((arr.shape[0],arr.shape[1],M))


        for i in xrange(W.shape[-1]):
            weights_map_name=output_prefix+"-initial-weights_map-%02d.png" % i
            Weights_map=W[:,:,i]
            Image.fromarray((Weights_map*255.0).clip(0,255).round().astype(np.uint8)).save(weights_map_name)

        W=W.reshape((-1,W.shape[2]))
        N=W.shape[0]
        M=W.shape[1]
        L=H.shape[1]/2

        x0=W.reshape(-1)
 
    


    START=time.time()

    if solve_choice==0:
        recover_W, recover_H, final_loop = KM_solve_ANLS(arr, x0, H, output_prefix, max_loop, W_w, W_sparse, W_spatial, W_sm_K, W_sm_S, W_sm_KS)
        
        print recover_H.shape
        save_pigments(recover_H.reshape(-1), M, output_prefix+"-Pigments"+"-alternate_loop-"+str(final_loop)+"-")
        save_pigments(recover_H.reshape(-1), M, base_dir+"/")
        save_results(recover_W.reshape(-1), arr, recover_H, output_prefix+"-Weights"+"-alternate_loop-"+str(final_loop)+"-")

        #### measure groundtruth and recovered H. 
        if gt_H_name!="None":
            gt_H=np.loadtxt(base_dir+gt_H_name)
            from Compare_Recovered_KS_with_GT_KS import *
            Visualize_Recovered_KS_with_GT_KS(gt_H, recover_H, output_prefix_copy+"/pigments-display")


    END=time.time()
    print 'total time: ', (END-START)
    

    
    