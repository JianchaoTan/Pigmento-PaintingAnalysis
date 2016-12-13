# README
* The input image should be a good size, like 600*600. If it is large image, it will be slow for weights map exraction or layer map extraction.


#### Dependencies:
 * Python 2.7
 * Numpy
 * Scipy
 * Pillow
 * numba
 * matplotlib
 * autograd
 * joblib
 * scikit-image
 * scikit-learn
 * cvxopt (with 'glpk' solver option)


#### Three main code files:
* step1_ANLS_with_autograd.py
* Solve_KM_mixing_model_fixed_KS_with_autograd.py
* Solve_KM_layer_model_fixed_KS_with_autograd.py


#### Commands (run 1 first, then run 2 or 3): 

##### 1. Extract primary pigments: 
User can give number of pigmetns, for example, "6" in the command line below.
```sh
	$ cd /new_pipeline_exectuable

	$ python step1_ANLS_with_autograd.py wheatfield-crop.png
	Existing_KS_parameter_KS.txt 2 None wheatfield-sampled_pixels-400 
	0 6 10.0 0.0 0.0 0.001 0.001 1e-6 /wheatfield-crop None 0 1 1000 
	400
```


##### 2. Extract mixing weights:
You can use default parameter values in command line directly, only need change example name.

```sh
	$ cd /new_pipeline_exectuable/wheatfield-crop

	$ python ../Solve_KM_mixing_model_fixed_KS_with_autograd.py
	wheatfield-crop.png  primary_pigments_KS-6.txt  None
	wheatfield-crop-primary_pigments_color_vertex-6-KM_weights-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes 10.0 0.1 0 1.0 0.0 blf Yes
```



##### 3. Extract layers: 
You need create a layer order file manually, for example: "wheatfield-pigments-order1.txt", which contains order: 0 1 2 3 4 5 or their permutations.
```sh
	$ cd /new_pipeline_exectuable/wheatfield-crop

	$ python ../Solve_KM_layer_model_fixed_KS_with_autograd.py 
	wheatfield-crop.png  primary_pigments_KS-6.txt  None
	wheatfield-crop-primary_pigments_color_vertex-6-KM_layers-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes-order1 10.0 0.1 0 1.0 0.0 blf Yes wheatfield-pigments-order1.txt
```
