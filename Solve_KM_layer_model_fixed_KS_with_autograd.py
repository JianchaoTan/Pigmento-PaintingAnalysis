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





@jit 
def objective_func_vector_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0,W_neighbors=0.0, neighbors=None):

    M=H.shape[0]
    L=H.shape[1]/2
    N=len(x0)/M
    Thickness=x0.reshape((N,M))
    K0=H[:,:L]
    S0=H[:,L:]



    R_vector=np.ones((N,L)) #### pure white background
    for i in range(M):
        R_vector=equations_in_RealPigments(K0[i:i+1,:], S0[i:i+1,:], R_vector, Thickness[:,i:i+1])

    ### from R spectrum x wavelength spectrums to linear rgb colors 
    P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
    R_xyz=(P_vector.reshape((N,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz=R_xyz/Normalize ####xyz value shape is N*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
    R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
    

    obj=R_rgb.reshape(-1)-arr.reshape(-1)
  

    if W_w!=0.0:
        Thickness_sum=(Thickness.sum(axis=1)-1)*sqrt(W_w)
        obj=np.concatenate((obj,Thickness_sum))

    if W_sparse!=0.0:
        Sparse_term=np.sqrt(np.maximum((1.0-np.square(Thickness-1.0)).sum(axis=1),eps))*sqrt(W_sparse/M)
        obj=np.concatenate((obj,Sparse_term.reshape(-1)))

    if W_spatial!=0.0:
        x=Thickness.reshape((arr.shape[0],arr.shape[1],-1))
        gx,gy,gz=np.gradient(x) ### gz is not meaningful here.
        gradient=np.sqrt(np.square(gx).sum(axis=2)+np.square(gy).sum(axis=2))
        Spatial_term=gradient*sqrt(W_spatial/M)
        obj=np.concatenate((obj,Spatial_term.reshape(-1)))

    if W_neighbors!=0.0 and neighbors!=None: #### this is for per pixel solving, w_spaital should be 0 and x0 length is M.
        neighbor_term=Thickness.reshape((1,1,-1))-neighbors
        neighbor_term=np.sqrt(np.maximum(np.square(neighbor_term).sum(axis=2),eps))*sqrt(W_neighbors*N/(1.0*M*neighbors.shape[0]*neighbors.shape[1]))
        obj=np.concatenate((obj,neighbor_term.reshape(-1)))


    return obj



def jacobian_objective_func_vector_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0,W_neighbors=0.0, neighbors=None):
    Jac=jacobian(objective_func_vector_fixed_KS,0)
    return Jac(x0, arr, H, Smooth_Matrix, W_w, W_sparse, W_spatial, W_neighbors, neighbors)



@jit
def objective_func_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0, W_neighbors=0.0, neighbors=None):
    

    obj=objective_func_vector_fixed_KS(x0, arr, H, None, W_w, W_sparse, 0.0, W_neighbors, neighbors)
    
    spatial_term=0.0

    if W_spatial!=0.0:

        #### this is ok, but not supported by autograd library to compute gradient.
        M=H.shape[0]
        spatial_term=np.dot(x0,Smooth_Matrix.dot(x0))*W_spatial/M

        return np.square(obj).sum()+spatial_term
    else:
        return np.square(obj).sum()






def gradient_objective_func_fixed_KS(x0, arr, H, Smooth_Matrix, W_w=2.0, W_sparse=0.01,W_spatial=0.0, W_neighbors=0.0, neighbors=None):
    

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







def optimize(arr, x0, H, Smooth_Matrix, saver=None, W_w=2.0, W_sparse=0.1, W_spatial=0.0, method='L-BFGS-B', W_neighbors=0.0, neighbors=None):
    # print type(x0)
    arr_shape=arr.shape
    N=arr_shape[0]*arr_shape[1]
    M=len(x0)/N
    L=H.shape[1]/2
    lb=1e-10
    ub=1.0
    #### bounds0 are for least_squares function parameters.
    bounds0=(lb, ub)
    bounds3=[]
    for i in xrange(len(x0)):
        bounds3.append((lb,ub))
    

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


    return x




def save_layers(x0, arr, H, output_prefix):
    shape=arr.shape
    original_shape=shape
    img_size=shape[:2]
    N=shape[0]*shape[1]
    M=H.shape[0]
    L=H.shape[1]/2

    Thickness=x0.reshape((N,M))
    K0=H[:,:L]
    S0=H[:,L:]
    print Thickness.sum(axis=1).min()
    print Thickness.sum(axis=1).max()

   
    R_vector=np.ones((N,L))
    for i in range(M):
        R_vector=equations_in_RealPigments(K0[i:i+1,:], S0[i:i+1,:], r=R_vector, h=Thickness[:,i:i+1])

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
    np.savetxt(output_prefix+"-thickness.txt", Thickness)

    
    #### save for applications
    filename=save_for_application_path_prefix+os.path.splitext(img_file)[0]+"-"+str(M)+"-KM_layers-"+os.path.splitext(order_file_name)[0]+"-reconstructed.png"
    plt.imsave(filename,(R_rgb.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))




    ### compute sparsity
    sparsity_thres_array=np.array([0.000001, 0.00001, 0.0001,0.001,0.01,0.1])
    Thickness_sparsity_list=np.ones(len(sparsity_thres_array))
    for thres_ind in xrange(len(sparsity_thres_array)):
        Thickness_sparsity_list[thres_ind]=len(Thickness[Thickness<=sparsity_thres_array[thres_ind]])*1.0/(N*M)
    
    print "Thickness_sparsity_list: ", Thickness_sparsity_list
    np.savetxt(output_prefix+"-Thickness-Sparsity.txt", Thickness_sparsity_list)



    # normalized_Thickness=Thickness/Thickness.sum(axis=1).reshape((-1,1))

    # for i in xrange(M):
    #     #### save normalized_weights_map for each pigment.
    #     normalized_thickness_map_name=output_prefix+"-normalized_thickness_map-%02d.png" % i
    #     normalized_thickness_map=normalized_Thickness[:,i].reshape(img_size).copy()
    #     Image.fromarray((normalized_thickness_map*255.0).clip(0,255).round().astype(np.uint8)).save(normalized_thickness_map_name)
    

    Thickness_sum_map=Thickness.sum(axis=1).reshape(img_size)
    T_min=Thickness_sum_map.min()
    T_max=Thickness_sum_map.max()
    Thickness_sum_map=Thickness_sum_map/T_max

    Image.fromarray((Thickness_sum_map*255.0).round().astype(np.uint8)).save(output_prefix+"-thickness_sum_map-min-"+str(T_min)+"-max-"+str(T_max)+".png")
    

    for i in xrange(M):
        thickness_map_name=output_prefix+"-layer_thickness_map-%02d.png" % i
        Thickness_map=Thickness[:,i].reshape(img_size).copy()
        Large_than_one=len(Thickness_map[Thickness_map>1.0])
        if Large_than_one>0:
            print "Number of Thickness value that is larger than 1.0 is : ", Large_than_one
        Image.fromarray((Thickness_map*255.0).clip(0,255).round().astype(np.uint8)).save(thickness_map_name)
        
        ####save for application
        thickness_map_name=save_for_application_path_prefix+os.path.splitext(img_file)[0]+"-"+str(M)+"-KM_layers-"+os.path.splitext(order_file_name)[0]+"-thickness_map-%02d.png" % i
        Image.fromarray((Thickness_map*255.0).clip(0,255).round().astype(np.uint8)).save(thickness_map_name)

 

def save_layers_2(x0, arr_shape, H, output_prefix):
    shape=arr_shape
    original_shape=shape
    img_size=shape[:2]
    N=shape[0]*shape[1]
    M=H.shape[0]
    L=H.shape[1]/2

    Thickness=x0.reshape((N,M))
    K0=H[:,:L]
    S0=H[:,L:]
    print Thickness.sum(axis=1).min()
    print Thickness.sum(axis=1).max()

    R_vector=np.ones((N,L))
    for i in range(M):
        R_vector=equations_in_RealPigments(K0[i:i+1,:], S0[i:i+1,:], r=R_vector, h=Thickness[:,i:i+1])

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

    for i in xrange(M):
        thickness_map_name=output_prefix+"-layer_thickness_map-%02d.png" % i
        Thickness_map=Thickness[:,i].reshape(img_size).copy()
        Large_than_one=len(Thickness_map[Thickness_map>1.0])
        if Large_than_one>0:
            print "Number of Thickness value that is larger than 1.0 is : ", Large_than_one
        Image.fromarray((Thickness_map*255.0).clip(0,255).round().astype(np.uint8)).save(thickness_map_name)






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
    Thickness_file_name=sys.argv[3]
    output_prefix=sys.argv[4]
    W_w=np.float(sys.argv[5])
    W_sparse=np.float(sys.argv[6])
    print 'W_sparse',W_sparse
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


    original_shape=img.shape
    img_size=img.shape[:2]

    N=arr.shape[0]*arr.shape[1]
    M=H.shape[0]
    L=H.shape[1]/2
        
    if Thickness_file_name=="None":
        Thickness=np.ones((arr.shape[0],arr.shape[1],M))/M
    else:
        Thickness=np.loadtxt(Thickness_file_name)
        Thickness=Thickness.reshape((arr.shape[0],arr.shape[1],M))
        print Thickness.shape

        initial_error= objective_func_vector_fixed_KS(Thickness.reshape(-1), arr, H, None, W_w=0.0, W_sparse=0.0,W_spatial=0.0,W_neighbors=0.0, neighbors=None)
        print 'initial_error: ', np.sqrt(np.square(initial_error*255.0).sum()/N)
        initial_recover=initial_error.reshape((-1,3))+arr.reshape((-1,3))
        plt.imsave(output_prefix+"-initial_recover.png",(initial_recover.reshape(original_shape)*255.0).clip(0,255).round().astype(np.uint8))



    x0=Thickness.reshape(-1)


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
            save_layers_2(xk, arr_shape, H, output_prefix)
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





    if solve_choice==0: 
                        
        print 'choice: ', solve_choice
        
        smooth_choice=sys.argv[10]
        recursive_choice=sys.argv[11]

        global order_file_name
        order_file_name=sys.argv[12]
        
        if order_file_name!=None:
            order=np.loadtxt(order_file_name)
            order=order.astype(np.uint8)
            print 'order', order
            
            ### save for application
            np.savetxt(save_for_application_path_prefix+order_file_name, order)


            #### reorder the primary pigments
            H_ordered=H[order,:]
            H=H_ordered.copy()


            ### save reordered primary pigments.
            np.savetxt(os.path.splitext(KS_file_name)[0]+"-"+os.path.splitext(order_file_name)[0]+".txt", H)
            R_vector=equations_in_RealPigments(H[:,:L], H[:,L:], r=1.0, h=1.0)
            P_vector=R_vector*Illuminantnew[:,1].reshape((1,-1)) ### shape is N*L
            R_xyz=(P_vector.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)   ###shape N*3*L to shape N*3 
            Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
            R_xyz=R_xyz/Normalize ####xyz value shape is N*3
            R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value, shape is N*3
            R_rgb=Gamma_trans_img(R_rgb.clip(0,1)) ##clip and gamma correction
            R_rgb=(R_rgb*255.0).round()
            filename="primary_pigments_color-"+str(M)+"-"+os.path.splitext(order_file_name)[0]+".png"
            Image.fromarray(R_rgb.reshape((1,-1,3)).astype(np.uint8)).save(filename)

            with open("primary_pigments_color_vertex-"+str(M)+"-"+os.path.splitext(order_file_name)[0]+".js", 'w') as myfile:
                json.dump({"vs": R_rgb.tolist()}, myfile )




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
        save_layers(x0, arr, H, output_prefix+"-final_recursivelevel-")
        time2=time.time()
        print "final level use time: ", time2-time1
        


    END=time.time()
    print 'total time: ', (END-START)
    

    
    