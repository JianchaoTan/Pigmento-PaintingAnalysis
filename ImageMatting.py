import numpy as np
import scipy as sp
import cv2
import sys, os


def Create_Laplacian_Matrix(Img, Mask, _WinSize=5, Lambda=1e-7, Options="Linear", s=1.0):

    WinSize=np.array([_WinSize,_WinSize])
    HalfWinSize=(WinSize-1)/2
    ImgSize=Img.shape[0:2]
    AllPixelInds=np.arange(0,ImgSize[0]*ImgSize[1]).reshape(ImgSize)
    TotalPixelNumInWindow=WinSize[0]*WinSize[1]
    
    Temp=Mask[HalfWinSize[0]:-HalfWinSize[0],HalfWinSize[1]:-HalfWinSize[1]]
    NumTrainingPixels=len(Temp[Temp==0.5])
    NumofNoneZeroEntries=NumTrainingPixels*TotalPixelNumInWindow**2

    Row_indices=np.zeros(NumofNoneZeroEntries)
    Col_indices=Row_indices.copy()
    Coeff_values=Row_indices.copy()

    from numpy import matlib
    from scipy.spatial.distance import pdist, squareform
    Index=0
    for i in range(HalfWinSize[0],ImgSize[0]-HalfWinSize[0]):
        for j in range(HalfWinSize[1],ImgSize[1]-HalfWinSize[1]):

            if Mask[i,j]==0.5: ### pixel that has unknown alphas

                LocalWin=Img[i-HalfWinSize[0]:i+HalfWinSize[0]+1,j-HalfWinSize[1]:j+HalfWinSize[1]+1,:].reshape((-1,Img.shape[2]))
                
                Xi=np.ones((LocalWin.shape[0],LocalWin.shape[1]+1))
                Xi[:,:-1]=LocalWin
                I=np.eye(Xi.shape[0])

                VAL=None
                if Options=="Linear":
                    VAL=np.dot(Xi,Xi.transpose())
                elif Options=="Gaussiankernel":
                    pairwise_dists = squareform(pdist(Xi, 'euclidean'))
                    VAL = sp.exp((-pairwise_dists**2)/(2.0*(s**2)))

                F=sp.linalg.solve((VAL+Lambda*I).transpose(), VAL.transpose())
                I_F=np.eye(F.shape[0])-F
                Local_Lap_Coeff=np.dot(I_F, I_F.transpose())

                LocalWindowInds=AllPixelInds[i-HalfWinSize[0]:i+HalfWinSize[0]+1,j-HalfWinSize[1]:j+HalfWinSize[1]+1]
                Row_indices[Index:TotalPixelNumInWindow**2+Index]=matlib.repmat(LocalWindowInds.reshape((-1,1)),1,TotalPixelNumInWindow).reshape(TotalPixelNumInWindow**2)
                Col_indices[Index:TotalPixelNumInWindow**2+Index]=matlib.repmat(LocalWindowInds.reshape((-1,1)).transpose(),TotalPixelNumInWindow,1).reshape(TotalPixelNumInWindow**2)
                Coeff_values[Index:TotalPixelNumInWindow**2+Index]=Local_Lap_Coeff.reshape(-1)
                Index=Index+TotalPixelNumInWindow**2
    
    Lap = sp.sparse.csr_matrix((Coeff_values, (Row_indices, Col_indices)), shape=(ImgSize[0]*ImgSize[1], ImgSize[0]*ImgSize[1]))
    
    return Lap


def Solve_for_all_Alphas_from_Known_Alphas(Lap,C,Alpha_known,Gamma=1e-7):
    
    Originshape=Alpha_known.shape
    Alpha_known=Alpha_known.reshape((-1,1))

    I=sp.sparse.eye(Lap.shape[0], Lap.shape[1]).tocsr()
    B=C.dot(Alpha_known)
    A=Lap+C+Gamma*I

    from scipy.sparse.linalg import spsolve
    Alpha=spsolve(A, B).reshape(Originshape) #Alpha=A.inv()*b
    
    return Alpha.clip(0,1)


def Image_Matting_By_Learning(Img, Trimap_mask, WinSize=5, c=1000.0, Lambda=1e-7, Options="Linear", s=1.0):
    
    Mask=np.ones(Trimap_mask.shape)*0.5
    Mask[Trimap_mask==255]=1
    Mask[Trimap_mask==0]=0
    
    KnownVsUnknown_mask=np.zeros(Mask.shape)
    KnownVsUnknown_mask[Mask!=0.5]=1

    Lap=Create_Laplacian_Matrix(Img, Mask, WinSize, Lambda, Options, s)

    TotalPixelNum=Mask.shape[0]*Mask.shape[1]
    C=c*sp.sparse.spdiags(KnownVsUnknown_mask.reshape(TotalPixelNum),0,TotalPixelNum,TotalPixelNum).tocsr()

    Alpha_known=Mask.copy()
    Alpha=Solve_for_all_Alphas_from_Known_Alphas(Lap,C,Alpha_known)

    return Alpha


## change color image:_Img to different feature data:Img. 
def Get_newformat_data(_Img, Choice):
    Img_rgb=cv2.cvtColor(_Img, cv2.COLOR_BGR2RGB)/255.0
    Img_lab=cv2.cvtColor(_Img, cv2.COLOR_BGR2LAB)/255.0
    Img_gray=cv2.cvtColor(_Img, cv2.COLOR_BGR2GRAY)/255.0
    Img=None
    if Choice=="rgblab":
        Img=np.zeros((_Img.shape[0],_Img.shape[1],6))
        Img[:,:,:3]=Img_rgb
        Img[:,:,3:]=Img_lab
    if Choice=="lab":
        Img=Img_lab
    if Choice=="rgb":
        Img=Img_rgb
    if Choice=="rgblabgradient":
        Sobelx = cv2.Sobel(Img_gray,cv2.CV_64F,1,0,ksize=3)
        Sobely = cv2.Sobel(Img_gray,cv2.CV_64F,0,1,ksize=3)
        mag=np.sqrt(np.square(Sobelx)+np.square(Sobely))
        Sobelx=Sobelx/(mag+1.0)
        Sobely=Sobely/(mag+1.0)
        Img=np.zeros((_Img.shape[0],_Img.shape[1],9))
        Img[:,:,:3]=Img_rgb
        Img[:,:,3:6]=Img_lab
        Img[:,:,6]=(mag+1.0)/(mag+1.0).max()
        Img[:,:,7]=Sobelx
        Img[:,:,8]=Sobely
    return Img


##### command line: python ImageMatting.py GT01.png Trimap1 Linear rgb 3
##### or: python ImageMatting.py GT01.png Trimap1 Gaussiankernel rgblab 3 1.0
if __name__=="__main__":
	
    ### Samples input: 

    # Trimap_name='./data/Trimap/Trimap1/GT01.png'
    # Img_name='./data/Img/GT01.png'
    # Groundtruth_name='./data/Groundtruth/GT01.png'
    # Saveresults_name ='./data/Results/GT01-Trimap1-Alphas-'

    # # Option="Linear"
    # Option="Gaussiankernel"

    # Choice="rgblab"
    # # Choice="lab"
    # # Choice="rgb"
    # # Choice="rgblabgradient"
    
    Img_name='./data/Img/'+sys.argv[1]
    Trimap_name="./data/Trimap/"+sys.argv[2]+"/"+sys.argv[1]
    Groundtruth_name='./data/Groundtruth/'+sys.argv[1]
    Saveresults_name='./data/Results/'+ os.path.splitext(sys.argv[1])[0]+'-'+sys.argv[2]+'-Alphas-'
    Option=sys.argv[3]
    Choice=sys.argv[4]
    WinSize=np.int(sys.argv[5])
    Std=1.0
    if Option=="Gaussiankernel":
        Std=np.float(sys.argv[6])

    print "Current Trimap used: ", Trimap_name
    print "Options: ", Option
    print "Choices: ", Choice

    _Img=cv2.imread(Img_name)
    Total_pixels=_Img.shape[0]*_Img.shape[1]

    Img=Get_newformat_data(_Img, Choice)

    Trimap=cv2.imread(Trimap_name,0)
    Alpha_gt=cv2.imread(Groundtruth_name,0)
    
    if Option=="Linear":
        Alpha=Image_Matting_By_Learning(Img, Trimap, WinSize=WinSize, Options=Option)
        Alpha=(Alpha*255).astype(np.uint8)
        cv2.imwrite(Saveresults_name+Option+"-"+Choice+".png", Alpha)
        Diff=Alpha-Alpha_gt
        print "RMSE:", np.sqrt(np.square(Diff).sum()/Total_pixels)

    if Option=="Gaussiankernel":
        Alpha=Image_Matting_By_Learning(Img, Trimap, WinSize=WinSize, Options=Option, s=Std)
        Alpha=(Alpha*255).astype(np.uint8)
        cv2.imwrite(Saveresults_name+Option+"-Std-"+str(Std)+"-"+Choice+".png", Alpha)
        Diff=Alpha-Alpha_gt
        print "Std:", Std, "\tRMSE:", np.sqrt(np.square(Diff).sum()/Total_pixels)



