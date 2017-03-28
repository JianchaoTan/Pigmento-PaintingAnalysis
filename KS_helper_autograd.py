# from numpy import *
# import numpy as np
from numba import jit
#from sympy import mpmath


global use_autograd
# use_autograd=False
use_autograd=True



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



##### according to https://en.wikipedia.org/wiki/SRGB
###for single value:
@jit
def Gamma_trans(C_linear):
    if C_linear<=0.0031308:
        return 12.92*C_linear
    else:
        a=0.055
        return (1+a)*(C_linear**(1.0/2.4))-a



@jit
def Gamma_trans_img1(RGB_linear_img):
    # print "#1"
    thres=0.0031308
    out=np.ones(RGB_linear_img.shape)
    out[RGB_linear_img<=thres]=12.92* RGB_linear_img[RGB_linear_img<=thres]
    a=0.055
    out[RGB_linear_img>thres]=(1+a)*(RGB_linear_img[RGB_linear_img>thres]**(1.0/2.4))-a
    return out



@jit
def Gamma_trans_img2(RGB_linear_img):
    # print "#2"
    return RGB_linear_img #### for comparing autograd and without autograd.

## new version, very slow!
# @jit
# def Gamma_trans_img(RGB_linear_img):
#     RGB_linear_img_flatten=RGB_linear_img.flatten()
#     out=np.array([Gamma_trans(item) for item in RGB_linear_img_flatten]).reshape(RGB_linear_img.shape)
#     return out


# #### new version, a little error here.
@jit
def Gamma_trans_img3(RGB_linear_img):
    # print "#3"
    eps=1e-50
    RGB_linear_img=RGB_linear_img.clip(eps,1.0)
    thres=0.0031308
    a=0.055

    ### what if some value of RGB_lienar_img is equal to thres? then some error will happen, but probability is very small
    out1=np.minimum(RGB_linear_img, thres)-thres
    out2=np.maximum(RGB_linear_img, thres)-thres

    temp1=12.92*RGB_linear_img
    temp2=(1+a)*(RGB_linear_img**(1.0/2.4))-a
    
    out=np.divide((temp1*out1),(out1+eps))+np.divide((temp2*out2),(out2+eps))
    return out




@jit
def mycoth(x):
    # print x.shape
    # print type(x)
    ex = np.exp(2*x)
    return (ex+1.0)/(ex-1.0)

### use coth instead of 1/tanh(x)
@jit
def equations_in_RealPigments(K,S,r,h, eps=1e-8, model='normal'): ## r is substrate reflectance, h is layer thicness, all of parameters can either be array or scalar values
    
    K=np.maximum(K,eps)
    S=np.maximum(S,eps)

    a=1+K/S
    # b=sqrt(a**2-1.0)
    b=(a**2-1.0)**(1/2.0)

    if model=='normal':
        d=mycoth(b*S*h)
        numerator=1-r*(a-b*d)
        denumerator=a-r+b*d
        R=numerator/denumerator
    elif model=='infinite':
        R=a-b
    else:
        print 'wrong option!'
    return R


# def equations_in_RealPigments(K,S,r,h,eps=1e-8, model='normal'): ## r is substrate reflectance, h is layer thicness, all of parameters can either be array or scalar values

#     K=np.maximum(K,eps)
#     S=np.maximum(S,eps)

#     a=1+K/S
#     b=(a**2-1.0)**(1/2.0)
#     c=np.tanh(b*S*h)
#     numerator=1-r*(a-b/c)
#     denumerator=a-r+b/c
#     R=numerator/denumerator
#     return R


### vectorized pixels, for any number of pigments
@jit
def KM_mixing_multiplepigments(K_vector, S_vector, weights, r=1.0, h=1.0, model='normal'): ### here the weights should be normalized.

    ###### Normalize weights!!!
    W_sum=weights.sum(axis=1).reshape((-1,1))
    W_sum=np.maximum(W_sum, 1e-15) #### to fit for autograd.
    weights_normalized=np.divide(weights,W_sum)


    # weights_normalized=weights/weights.sum(axis=1).reshape((-1,1))


    nominator=np.dot(weights_normalized,K_vector)
    denominator=np.dot(weights_normalized,S_vector)
    
    ### default is on white background,r=1.0 and thickness=0.5
    r_array=np.ones(nominator.shape)*r
    R_vector=equations_in_RealPigments(nominator,denominator, r_array, h, model=model)
    
    return R_vector #### shape is N*L




# #@jit
# def equations_in_RealPigments_use_KS_S(KS,S,r,h,eps=1e-8): ## r is substrate reflectance, h is layer thicness, all of parameters can either be array or scalar values
#     KS[KS<eps]=eps
#     S[S<eps]=eps
#     a=1+KS
#     b=sqrt(a**2-1.0)
#     c=np.tanh(b*S*h)
#     numerator=1-r*(a-b/c)
#     denumerator=a-r+b/c
#     R=numerator/denumerator
#     return R 


# ### vectorized pixels, for any number of pigments
# #@jit
# def KM_mixing_multiplepigments_use_KS_S(KS_vector, S_vector, weights, r=1.0, h=1.0): ### here the weights should be normalized.
#     #### KS_vector is K_vector/S_vector
#     N=len(weights) ### weights shape is N*M
#     M=len(KS_vector) #### KS_vector shape is M*L 
#     L=KS_vector.shape[1]
#     nominator=np.zeros((N,L))
#     denominator=nominator.copy()
#     ###### Normalize weights!!!
#     weights_normalized=weights.copy()
#     weights_normalized=weights_normalized/weights_normalized.sum(axis=1).reshape((-1,1))
#     for i in range(0,M):
#         nominator+=weights_normalized[:,i:i+1]*KS_vector[i:i+1,:]*S_vector[i:i+1,:]
#         denominator+=weights_normalized[:,i:i+1]*S_vector[i:i+1,:]
    
#     ### default is on white background,r=1.0 and thickness=0.5
#     r_array=np.ones(nominator.shape)*r
#     R_vector=equations_in_RealPigments(nominator,denominator, r_array, h)
#     return R_vector #### shape is N*L






# ### for vectorized pixels.
# #@jit
# def KM_layering_multiplepigments(K,S,thickness,r=1.0): ## K[0], S[0] is for first pigment, K[1], S[1] is for second pigment.... thickness[0] is second pigment's thickness, thickness[1] is third pigments thickness.
#     N=thickness.shape[0]
#     M=K.shape[0]
#     L=K.shape[1]

#     K=K.reshape((M,1,L))*np.ones((1,N,1))
#     S=S.reshape((M,1,L))*np.ones((1,N,1))
     
#     r_array=np.ones((N,L))*r
#     for i in range(0,M):
#         r_array=equations_in_RealPigments(K[i],S[i],r_array,thickness[:,i:i+1])
#     return r_array ### shape is N*L
        
    
# ### assume input is multiple wavelength
# ##@jit
# #def PigmentOnWhite(K,S,thickness,Illuminantnew, Normalize, R_rgbcoeff):
# #    r=np.ones(K.shape)## white reflectance is 1.
# #    R=equations_in_RealPigments(K,S,r,thickness)
    
# #    ## project to 3 channels
# #    P=R*Illuminantnew[:,1]
# #    R_rgb=(P.reshape((1,-1))*R_rgbcoeff).sum(axis=1)
# #    R_rgb/=Normalize
# #    return R_rgb
    
# ##@jit
# #def PigmentOnWhite_show(K,S,Illuminantnew,Normalize,R_rgbcoeff, thickness=1.0, start=-8, end=1.9, SCALE=1):
 
# #    R_layering=PigmentOnWhite(K,S,thickness,Illuminantnew,Normalize, R_rgbcoeff)
# #    R_layering=(R_layering*SCALE).clip(0,1) ## scale and clip
# #    img_layering=np.ones((500,500,3))*R_layering.reshape((1,1,3))
# #    results=Gamma_trans_img(img_layering) ### gamma correction from linear RGB to be sRGB
# #    return (results*255).clip(0,255).astype(np.uint8)
     


# ##@jit
# # def PigmentOnWhite_show(K,S,Illuminantnew,Normalize,R_rgbcoeff, start=-8, end=1.9, SCALE=1):
# #     R_layering=[]
# #     Num=5000

# #     for thickness in np.logspace(start, end, num=Num, base=10):
# #         R_layering.append(PigmentOnWhite(K,S,thickness,Illuminantnew,Normalize,R_rgbcoeff))
# #     print R_layering[0], R_layering[-1]

# #     R_layering=(R_layering*SCALE).clip(0,1) ## scale exposure and clip
# #     img_layering=np.ones((50,1,3))*np.array(R_layering).reshape((1,Num,3))
# #     results=Gamma_trans_img(img_layering) ### gamma correction from linear RGB to be sRGB
# #     return (results*255).clip(0,255).astype(np.uint8)
     


# # 

if __name__ == '__main__':
    # def f(x):
    #     K = x[:800].reshape(100,8)
    #     S = x[800:].reshape(100,8)
    #     R = equations_in_RealPigments( K, S, 1.0, 1.0 )
    #     return R.sum()
    # from autograd import grad
    # gradf = grad(f)
    # g = gradf( np.linspace( 1,2, 1600 ) )
    # print(g)

    from autograd import grad, jacobian
    
    # initial=np.zeros(5)
    initial= np.random.random_sample(5)
    print initial


    Jac1 = jacobian(Gamma_trans_img)
    j1 = Jac1(initial)
    print j1

    Jac2 = jacobian(Gamma_trans_img2)
    j2 = Jac2(initial)
    
    print j2
    print abs(j1-j2).sum()

