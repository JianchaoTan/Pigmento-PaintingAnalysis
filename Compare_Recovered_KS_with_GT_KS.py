import numpy as np
from Constant_Values import *
import sys,os
import PIL.Image as Image
from KS_helper import *
import matplotlib.pyplot as plt


##### This is for testing if the recovered KS is on the line of GT KS in KS space or not.
def solve_interpolation(gt_h, h):

    def objective_func_vector(alpha, gt_h, h):
        obj=gt_h[0]*alpha+gt_h[1]*(1-alpha)-h
        return obj.reshape(-1)

    def objective_func(alpha, gt_h, h):
        obj=objective_func_vector(alpha, gt_h, h)
        return np.square(obj).sum()


    import scipy.optimize as sopt
    alpha=0.5
    res=sopt.least_squares(objective_func_vector, alpha, args=(gt_h, h), method='trf')
    # print res["message"]
    x0=res["x"]

    error=np.sqrt(objective_func(x0, gt_h, h))

    return error, x0




def Test_Recovered_KS_with_GT_KS(gt_H, H, output_prefix, patch_reconstruct_RGB_RMSE_list):
    M=H.shape[0]
    L=H.shape[1]/2
    print L
    errors=np.ones(M)
    interpolations=np.ones(M)
    output_prefix2=output_prefix+"-error_histogram-display"
    make_sure_path_exists(output_prefix2)


    # RGB_RMSE_thres=0.5
    RGB_RMSE_thres=patch_reconstruct_RGB_RMSE_list.max()
    # RGB_RMSE_thres=np.median(patch_reconstruct_RGB_RMSE_list)
    print RGB_RMSE_thres

    myfile = open(output_prefix2+'/display-pigments-whose_patch_RGB_RMSE_smaller_than-'+str(RGB_RMSE_thres)+'.html', 'w')
    myfile.write("<!DOCTYPE html>\n <html><body>\n")


    myfile.write("<h2>Red line is groundtruth pigments, blue line is recovered ones, green line is its projected ones. Black line is error between recovered ones and projectd ones. Left two figure is for K curve, and Right two figure is for S curve.</h2> \n")



    for i in range(M/2):
        gt_h=gt_H[2*i:2*i+2]

        if patch_reconstruct_RGB_RMSE_list[i]<=RGB_RMSE_thres:

            for j in range(0,2):

                h=H[2*i+j:2*i+j+1]
                error, interpolation=solve_interpolation(gt_h, h)

                errors[2*i+j]=error
                interpolations[2*i+j]=interpolation

                if interpolation <0 or interpolation>1:
                    print i
                    print error
                    print interpolation

                projected_h=gt_h[0]*interpolation+gt_h[1]*(1-interpolation)
                error_array=abs(h-projected_h)
                error_ratio_array=error_array/projected_h
                



                myfile.write("<h3>Patch-"+str(i)+" recovered pigment "+str(j)+" its corresponding linear interpolation value: "+str(interpolation)+" and total euclidian distance between it and its projected pigments is : "+str(error)+" . And the corresponding patch RGB reconstruct RMSE is "+str(patch_reconstruct_RGB_RMSE_list[i])+"  </h3> \n")
                xaxis=np.arange(L)
                name_list=["K-curve", "error-K-curve", "S-curve", "error-S-curve"]
                
                for ind in range(len(name_list)):
                    name=name_list[ind]
                    fig=plt.figure()
                    # print gt_h[0,:L].shape
                    # print xaxis.shape
                    # print h[0,:L].shape
                    # print error_array[0,:L].shape


                    if ind==0:
                        plt.plot(xaxis, gt_h[0,:L], 'ro-', xaxis, gt_h[1,:L], 'rs-', xaxis, h[0,:L], 'b-', xaxis, projected_h[:L],'gx-')

                    if ind==1:
                        plt.plot(xaxis, projected_h[:L], 'gx-', xaxis, error_array[0,:L], 'k-')
                    
                    if ind==2:
                        plt.plot(xaxis, gt_h[0,L:], 'ro-', xaxis, gt_h[1,L:], 'rs-', xaxis, h[0,L:], 'b-', xaxis, projected_h[L:], 'gx-')

                    if ind==3:
                        plt.plot(xaxis, projected_h[L:], 'gx-', xaxis, error_array[0,L:], 'k-')


                    filename="Patch-"+str(i)+"-pigment-"+str(j)+"-"+name+".png"
                    fig.savefig(output_prefix2+"/"+filename)
                    plt.close(fig)
                    results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
                    myfile.write(results_items)




    myfile.write("</body> </html> \n")
    myfile.close()

    np.savetxt(output_prefix+"-distance-errors.txt", errors)
    np.savetxt(output_prefix+"-interpolation-values.txt", interpolations)
    print errors.min()
    print errors.max()



def Test_Recovered_KS_with_GT_KS_one_pigment_per_patch(gt_H, H, output_prefix, patch_reconstruct_RGB_RMSE_list, gt_thickness_map, recover_thickness_map):
    M=H.shape[0]
    L=H.shape[1]/2

    output_prefix2=output_prefix+"-error_histogram-display"
    make_sure_path_exists(output_prefix2)
    
    Num=len(gt_thickness_map.reshape(-1))/M #### pixel num in each patch
    print gt_thickness_map.shape
    print recover_thickness_map.shape

    # RGB_RMSE_thres=1.5
    RGB_RMSE_thres=patch_reconstruct_RGB_RMSE_list.max()
    # RGB_RMSE_thres=np.median(patch_reconstruct_RGB_RMSE_list)
    print RGB_RMSE_thres


    myfile = open(output_prefix2+'/display-pigments-whose_patch_RGB_RMSE_smaller_than-'+str(RGB_RMSE_thres)+'.html', 'w')
    myfile.write("<!DOCTYPE html>\n <html><body>\n")


    myfile.write("<h2>Red line is groundtruth pigments, blue line is recovered ones. First is for K curve, and Second figure is for S curve, third is groundtruth thickness, fourth is recovered thickness.</h2> \n")



    for i in range(M):

        if patch_reconstruct_RGB_RMSE_list[i]<=RGB_RMSE_thres:

            gt_h=gt_H[i:i+1]
            h=H[i:i+1]
            size=np.int(sqrt(Num))
            gt_thick=gt_thickness_map[i*Num:i*Num+Num].reshape((size,size)) #### default patch shape is nbyn, Num=n^2.
            recover_thick=recover_thickness_map[i*Num:i*Num+Num].reshape((size,size))
        
                
            myfile.write("<h3>Patch-"+str(i)+" recovered pigment "+str(i)+". And the corresponding patch RGB reconstruct RMSE is "+str(patch_reconstruct_RGB_RMSE_list[i])+". groundtruth thickness range is ("+str(gt_thick.min())+","+str(gt_thick.max())+"), and recovered thickness range is ("+str(recover_thick.min())+", "+str(recover_thick.max())+")  </h3> \n")
            xaxis=np.arange(L)
            name_list=["K-curve", "S-curve"]
            
            for ind in range(len(name_list)):
                name=name_list[ind]
                fig=plt.figure()

                if ind==0:
                    plt.plot(xaxis, gt_h[0,:L], 'ro-', xaxis, h[0,:L], 'b-')
                
                if ind==1:
                    plt.plot(xaxis, gt_h[0,L:], 'ro-', xaxis, h[0,L:], 'b-')


                filename="Patch-"+str(i)+"-pigment-"+str(i)+"-"+name+".png"
                fig.savefig(output_prefix2+"/"+filename)
                plt.close(fig)
                results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
                myfile.write(results_items)


            filename="Patch-"+str(i)+"-pigment-"+str(i)+"-groundtruth_thickness.png"
            Image.fromarray((gt_thick*255).round().clip(0,255).astype(np.uint8), mode='L').save(output_prefix2+"/"+filename)
            results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
            myfile.write(results_items)

            filename="Patch-"+str(i)+"-pigment-"+str(i)+"-recovered_thickness.png"
            Image.fromarray((recover_thick*255).round().clip(0,255).astype(np.uint8), mode='L').save(output_prefix2+"/"+filename)
            results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
            myfile.write(results_items)


    myfile.write("</body> </html> \n")
    myfile.close()





def Test_Recovered_KS_with_GT_KS_one_variable_version(gt_H, H, output_prefix, patch_reconstruct_RGB_RMSE_list):
    M=H.shape[0]
    L=H.shape[1]
    gt_H_copy=gt_H[:,:L]/gt_H[:,L:]
    gt_H=gt_H_copy.copy()

    errors=np.ones(M)
    interpolations=np.ones(M)
    output_prefix2=output_prefix+"-error_histogram-display"
    make_sure_path_exists(output_prefix2)

    
    # RGB_RMSE_thres=0.5
    # RGB_RMSE_thres=1.0
    RGB_RMSE_thres=patch_reconstruct_RGB_RMSE_list.max()
    # RGB_RMSE_thres=np.median(patch_reconstruct_RGB_RMSE_list)
    print RGB_RMSE_thres


    myfile = open(output_prefix2+'/display-pigments-whose_patch_RGB_RMSE_smaller_than-'+str(RGB_RMSE_thres)+'.html', 'w')
    myfile.write("<!DOCTYPE html>\n <html><body>\n")


 


    myfile.write("<h1>Red line is groundtruth pigments, blue line is recovered pigments, green line is its projected pigments. Black line is error between recovered pigments and projectd pigments</h1> \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    
    for i in range(M/2):
        gt_h=gt_H[2*i:2*i+2]

        if patch_reconstruct_RGB_RMSE_list[i]<=RGB_RMSE_thres:

            for j in range(0,2):

                h=H[2*i+j:2*i+j+1]
                error, interpolation=solve_interpolation(gt_h, h)

                errors[2*i+j]=error
                interpolations[2*i+j]=interpolation

                if interpolation <0 or interpolation>1:
                    print i
                    print error
                    print interpolation

                projected_h=gt_h[0]*interpolation+gt_h[1]*(1-interpolation)
                error_array=abs(h-projected_h)
                error_ratio_array=error_array/projected_h
                


                
                myfile.write("<h3>Patch-"+str(i)+" recovered pigment "+str(j)+" its corresponding linear interpolation value: "+str(interpolation)+" and total euclidian distance between it and its projected pigments is : "+str(error)+" . And the corresponding patch RGB reconstruct RMSE is "+str(patch_reconstruct_RGB_RMSE_list[i])+"  </h3> \n")
                xaxis=np.arange(L)
                name_list=["KS-curve", "error-KS-curve"]
                
                for ind in range(len(name_list)):
                    name=name_list[ind]
                    fig=plt.figure()

                    if ind==0:
                        plt.plot(xaxis, gt_h[0], 'ro-', xaxis, gt_h[1], 'rs-', xaxis, h[0], 'b-', xaxis, projected_h,'gx-')

                    if ind==1:
                        plt.plot(xaxis, projected_h, 'gx-', xaxis, error_array[0], 'k-')


                    filename="Patch-"+str(i)+"-pigment-"+str(j)+"-"+name+".png"
                    fig.savefig(output_prefix2+"/"+filename)
                    plt.close(fig)
                    results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
                    myfile.write(results_items)




    myfile.write("</body> </html> \n")
    myfile.close()

    np.savetxt(output_prefix+"-distance-errors.txt", errors)
    np.savetxt(output_prefix+"-interpolation-values.txt", interpolations)
    print errors.min()
    print errors.max()






##### This is for visualizing the recovered KS and gt KS in same figure per pigments. 
def Visualize_Recovered_KS_with_GT_KS(gt_H, H, path):
    M=H.shape[0]
    L=H.shape[1]/2
    K0=H[:,:L]
    S0=H[:,L:]
    K_vector=gt_H[:,:L]
    S_vector=gt_H[:,L:]

    make_sure_path_exists(path)


    #### write later images into html files:
    myfile = open(path+'/display.html', 'w')
    myfile.write("<!DOCTYPE html>\n <html><body>\n")





#### rendering recovered pigments
    print "model: ", MODEL
    R0=equations_in_RealPigments(K0, S0, r=1.0, h=1.0, model=MODEL)
    P0=R0*Illuminantnew[:,1].reshape((1,-1))
    R_xyz=(P0.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is M*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value shape is M*3
    R_rgb=R_rgb.clip(0,1)
    R_rgb=Gamma_trans_img(R_rgb)
    


#### rendering groundtruth pigments
    gt_R=equations_in_RealPigments(K_vector, S_vector, r=1.0, h=1.0)
    gt_P=gt_R*Illuminantnew[:,1].reshape((1,-1))
    R_xyz=(gt_P.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is M*3
    gt_R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value shape is M*3
    gt_R_rgb=gt_R_rgb.clip(0,1)
    gt_R_rgb=Gamma_trans_img(gt_R_rgb)
    gt_Pigment=(gt_R_rgb.reshape((-1,1,3))*255.0).round().astype(np.uint8)




    
    ###### this is not accurate if the recovered pigments set is not similar to gt pigments set.
    ##### and it is slow if M is large/
    indices=np.arange(M)
    import itertools
    color_diff=[]
    permuts=list(itertools.permutations(indices))
    for permut in permuts:
        color_diff.append(np.square(gt_R_rgb-R_rgb[np.array(permut)]).sum())

    index=np.argmin(color_diff)
    final_permut=np.array(permuts[index])
    K0=K0[final_permut]
    S0=S0[final_permut]
    R0=R0[final_permut]
    R_rgb=R_rgb[final_permut]
    
    print "final_permut: ", final_permut

    Pigment=(R_rgb.reshape((-1,1,3))*255.0).round().astype(np.uint8)


##### write into png and html. 
    myfile.write("<h3>Groundtruth Pigments</h3> \n")
    for i in range(M):
        n=50
        Pigment_expand=np.ones((n,n,3), dtype=np.uint8)
        Pigment_expand[:,:,:]=gt_Pigment[i:i+1,:,:]
        filename="gt_Pigment-"+str(i)+".png"
        Image.fromarray(Pigment_expand).save(path+"/"+filename)
        ##save filename into html file.
        results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
        myfile.write(results_items)


    myfile.write("<h3>Recovered Pigments</h3> \n")
    for i in range(M):
        n=50
        Pigment_expand=np.ones((n,n,3), dtype=np.uint8)
        Pigment_expand[:,:,:]=Pigment[i:i+1,:,:]
        filename="recovered_Pigment-"+str(i)+".png"
        Image.fromarray(Pigment_expand).save(path+"/"+filename)
        ##save filename into html file.
        results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
        myfile.write(results_items)


    myfile.write("<h2>Red dot is groundtruth, blue line is recovered.\nFrom left to right, the figure is: K, S, R and K divide by S</h2> \n")

    xaxis=np.arange(L)
    name_list=["K-curve", "S-curve", "onWhite-R-curve", "KS-curve"]
    for j in range(M):
        myfile.write("<h3>Pigment "+str(j)+" Curve</h3> \n")
        for i in range(4):
            name=name_list[i]
            fig=plt.figure()
            if i==0:
                plt.plot(xaxis, K_vector[j], 'ro', xaxis, K0[j], 'b-')
                plt.ylim((0,10))
            if i==1:
                plt.plot(xaxis, S_vector[j], 'ro', xaxis, S0[j], 'b-')
                plt.ylim((0,1))
            if i==2:
                plt.plot(xaxis, gt_R[j], 'ro', xaxis, R0[j], 'b-')
                plt.ylim((0,1))
            if i==3:
                plt.plot(xaxis, K_vector[j]/S_vector[j], 'ro', xaxis, K0[j]/S0[j], 'b-')
                plt.ylim(ymin=0)

            filename="Pigment-"+str(j)+"-"+name+".png"
            fig.savefig(path+"/"+filename)
            ##save filename into html file.
            results_items = """<img src=\"""" + filename + """"\t style="width:300px;height:300px;">\n"""
            myfile.write(results_items)

            plt.close('all')

    
    myfile.write("</body> </html> \n")
    myfile.close()





def Visualize_Recovered_KS_with_GT_KS_one_variable_version(gt_H, H, path):

    M=H.shape[0]
    L=H.shape[1]
    K_vector=gt_H[:,:L]
    S_vector=gt_H[:,L:]
    KS_vector=gt_H[:,:L]/gt_H[:,L:]


    KS0=H


    make_sure_path_exists(path)

    #### write later images into html files:
    myfile = open(path+'/display.html', 'w')
    myfile.write("<!DOCTYPE html>\n <html><body>\n")


#### rendering recovered pigments
    print "model: ", MODEL
    R0=equations_in_RealPigments(KS0, np.ones((M,L))*S_Constant, r=1.0, h=1.0, model=MODEL)
    P0=R0*Illuminantnew[:,1].reshape((1,-1))
    R_xyz=(P0.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is M*3
    R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value shape is M*3
    R_rgb=R_rgb.clip(0,1)
    R_rgb=Gamma_trans_img(R_rgb)
    


#### rendering groundtruth pigments
    gt_R=equations_in_RealPigments(K_vector, S_vector, r=1.0, h=1.0)
    gt_P=gt_R*Illuminantnew[:,1].reshape((1,-1))
    R_xyz=(gt_P.reshape((-1,1,L))*R_xyzcoeff.reshape((1,3,L))).sum(axis=2)
    Normalize=(Illuminantnew[:,1]*R_xyzcoeff[1,:]).sum() ### scalar value.
    R_xyz/=Normalize ####xyz value shape is M*3
    gt_R_rgb=np.dot(xyztorgb,R_xyz.transpose()).transpose() ###linear rgb value shape is M*3
    gt_R_rgb=gt_R_rgb.clip(0,1)
    gt_R_rgb=Gamma_trans_img(gt_R_rgb)
    gt_Pigment=(gt_R_rgb.reshape((-1,1,3))*255.0).round().astype(np.uint8)




    
    ###### this is not accurate if the recovered pigments set is not similar to gt pigments set.
    ##### and it is slow if M is large/
    indices=np.arange(M)
    import itertools
    color_diff=[]
    permuts=list(itertools.permutations(indices))
    for permut in permuts:
        color_diff.append(np.square(gt_R_rgb-R_rgb[np.array(permut)]).sum())

    index=np.argmin(color_diff)
    final_permut=np.array(permuts[index])
    KS0=KS0[final_permut]
    R0=R0[final_permut]
    R_rgb=R_rgb[final_permut]


    Pigment=(R_rgb.reshape((-1,1,3))*255.0).round().astype(np.uint8)


##### write into png and html. 
    myfile.write("<h3>Groundtruth Pigments</h3> \n")
    for i in range(M):
        n=50
        Pigment_expand=np.ones((n,n,3), dtype=np.uint8)
        Pigment_expand[:,:,:]=gt_Pigment[i:i+1,:,:]
        filename="gt_Pigment-"+str(i)+".png"
        Image.fromarray(Pigment_expand).save(path+"/"+filename)
        ##save filename into html file.
        results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
        myfile.write(results_items)


    myfile.write("<h3>Recovered Pigments</h3> \n")
    for i in range(M):
        n=50
        Pigment_expand=np.ones((n,n,3), dtype=np.uint8)
        Pigment_expand[:,:,:]=Pigment[i:i+1,:,:]
        filename="recovered_Pigment-"+str(i)+".png"
        Image.fromarray(Pigment_expand).save(path+"/"+filename)
        ##save filename into html file.
        results_items = """<img src=\"""" + filename + """"\t style="width:220px;height:220px;">\n"""
        myfile.write(results_items)


    myfile.write("<h2>Red dot is groundtruth, blue line is recovered.\nFrom left to right, the figure is: K divide by S curve, and R curve</h2> \n")

    xaxis=np.arange(L)
    name_list=["KS-curve","onWhite-R-curve"]
    for j in range(M):
        myfile.write("<h3>Pigment "+str(j)+" Curve</h3> \n")
        for i in range(len(name_list)):
            name=name_list[i]
            fig=plt.figure()
            if i==0:
                plt.plot(xaxis, KS_vector[j], 'ro', xaxis, KS0[j], 'b-')
                plt.ylim(ymin=0)
            if i==2:
                plt.plot(xaxis, gt_R[j], 'ro', xaxis, R0[j], 'b-')
                plt.ylim((0,1))


            filename="Pigment-"+str(j)+"-"+name+".png"
            fig.savefig(path+"/"+filename)
            ##save filename into html file.
            results_items = """<img src=\"""" + filename + """"\t style="width:300px;height:300px;">\n"""
            myfile.write(results_items)

            plt.close('all')

    
    myfile.write("</body> </html> \n")
    myfile.close()






if __name__=="__main__":
    gt_H_name=sys.argv[1]
    recovered_H_name=sys.argv[2]
    output_prefix=sys.argv[3]
    patch_reconstruct_RGB_RMSE_list_file_name=sys.argv[4]
    task_choice=np.int(sys.argv[5])


    gt_H=np.loadtxt(gt_H_name)
    print gt_H.shape
    recovered_H=np.loadtxt(recovered_H_name)
    print recovered_H.shape
    
    patch_reconstruct_RGB_RMSE_list=None
    if patch_reconstruct_RGB_RMSE_list_file_name!="None":
        patch_reconstruct_RGB_RMSE_list=np.loadtxt(patch_reconstruct_RGB_RMSE_list_file_name).reshape(-1)
        print patch_reconstruct_RGB_RMSE_list.max()
        print patch_reconstruct_RGB_RMSE_list.min()
        print np.median(patch_reconstruct_RGB_RMSE_list)


    if task_choice==1:
        Visualize_Recovered_KS_with_GT_KS(gt_H, recovered_H, output_prefix)

    if task_choice==2:
        Test_Recovered_KS_with_GT_KS(gt_H, recovered_H, output_prefix, patch_reconstruct_RGB_RMSE_list)

    if task_choice==3:
        Test_Recovered_KS_with_GT_KS_one_variable_version(gt_H, recovered_H, output_prefix, patch_reconstruct_RGB_RMSE_list)

    if task_choice==4:
        Visualize_Recovered_KS_with_GT_KS_one_variable_version(gt_H, recovered_H, output_prefix)

    if task_choice==5:
        gt_thickness_map=np.loadtxt(sys.argv[6])
        recover_thickness_map=np.loadtxt(sys.argv[7])
        Test_Recovered_KS_with_GT_KS_one_pigment_per_patch(gt_H, recovered_H, output_prefix, patch_reconstruct_RGB_RMSE_list, gt_thickness_map, recover_thickness_map)






