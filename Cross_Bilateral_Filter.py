import numpy as np
import time
##### copy code from paper "The Fast Bilateral Solver: Jonathan T. Barron et al." and modify a little.  

MAX_VAL = 255.0
from scipy.sparse import csr_matrix

def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx] 
    return valid_idx, locs


class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=32, sigma_luma=8, sigma_chroma=8):
        # im_yuv = rgb2yuv(im)
        im_yuv=im.copy()
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] /sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**np.arange(self.dim))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)
        
    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True) 
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # print self.S.shape
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in xrange(self.dim):
            blur = 0.0
            for offset in (-1,0,1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                          (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)
            
        
        
    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)
    
    def slice(self, y):
        return self.S.T.dot(y)
    
    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(np.ones_like(x))))




from scipy.sparse import diags,identity,linalg


def bistochastize(grid, maxiter=50):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in xrange(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)

    Dm_inv=diags(1.0/m,0)  ##### so m will not have 0 value, right? 
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    return Dn, Dm, Dm_inv



#### not doubly stocastic matrix.
def generate_Bilateral_matrix(img, sigma_spatial=8, sigma_luma=4, sigma_chroma=4):
    grid = BilateralGrid(img, sigma_spatial, sigma_luma, sigma_chroma)
    S=grid.S
    return S.T.dot(grid.blur(S))


#### slower.
def generate_Bilateral_Matrix_hat_1(img, sigma_spatial=8, sigma_luma=4, sigma_chroma=4):
    W=generate_Bilateral_matrix(img, sigma_spatial, sigma_luma, sigma_chroma)
    iters=1000
    N=img.shape[0]*img.shape[1]
    r=np.ones((N,1))
    for i in range(iters):
        c=1.0/(W.T.dot(r))
        # print c.shape
        r=1.0/(W.dot(c))
        # print r.shape

    C=diags(c.flatten(),0)
    R=diags(r.flatten(),0)
    W_hat=R.dot(W.dot(C))
    return identity(N)-W_hat



###### much faster than generate_Bilateral_Matrix_hat_1
def generate_Bilateral_Matrix_hat_2(img, sigma_spatial=8, sigma_luma=4, sigma_chroma=4):
    grid = BilateralGrid(img, sigma_spatial, sigma_luma, sigma_chroma)
    Dn,Dm,Dm_inv=bistochastize(grid)
    S=grid.S
    A=S.T.dot(Dm_inv)
    B=Dn.dot(grid.blur(Dn))
    C=Dm_inv.dot(S)
    W_hat=A.dot(B.dot(C))
    N=img.shape[0]*img.shape[1]
    return identity(N)-W_hat



generate_Bilateral_Matrix_hat=generate_Bilateral_Matrix_hat_2











if __name__=="__main__":
    import sys,os
    image_name=sys.argv[1]
    from skimage.io import imread, imsave
    img=imread(image_name).astype(np.float64)
    
    img=img[::2,::2]
    im_shape=img.shape[:2]
    print im_shape


    grid_params = {
        'sigma_luma' : 4, # Brightness bandwidth
        'sigma_chroma': 4, # Color bandwidth
        'sigma_spatial': 8 # Spatial bandwidth
    }

    import time
    t1=time.time()
    grid = BilateralGrid(img, **grid_params)
    t2=time.time()
    print t2-t1

    Bilateral_matrix=generate_Bilateral_matrix(img, **grid_params)
    print Bilateral_matrix.shape

    Bilateral_matrix_hat=generate_Bilateral_Matrix_hat_2(img, **grid_params)
    print Bilateral_matrix_hat.shape
   
    print Bilateral_matrix
    print abs(Bilateral_matrix.T-Bilateral_matrix).sum()
    print "###"
    print Bilateral_matrix_hat
    diff=Bilateral_matrix_hat.T-Bilateral_matrix_hat
    print abs(diff).sum()
    print abs(diff).max()
    print abs(diff).min()
    N=img.shape[0]*img.shape[1]
    print abs(Bilateral_matrix_hat.dot(np.ones(N))).sum()
    print abs(Bilateral_matrix_hat.T.dot(np.ones(N))).sum()
    t3=time.time()
    print t3-t2




