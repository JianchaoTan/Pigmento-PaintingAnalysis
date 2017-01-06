# README
* The input image should be a good size, like 600*600. If it is large image, it will be slow for weights map extraction or layer map extraction.


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
 * opencv3 (brew install opencv3)


#### Main code files:
* step1_ANLS_with_autograd.py
* Solve_KM_mixing_model_fixed_KS_with_autograd.py
* Solve_KM_layer_model_fixed_KS_with_autograd.py
* fast_energy_RGB_lap_adjusted_weights.py
* Editing_GUI.py



#### Commands (run 1 first, then run 2 or 3): 

##### 1. Extract KM primary pigments: 
User can give number of pigmetns, for example, "6" in the command line below.
```sh
	$ cd /new_pipeline_executable

	$ python step1_ANLS_with_autograd.py wheatfield-crop.png Existing_KS_parameter_KS.txt 2 None wheatfield-sampled_pixels-400 0 6 10.0 0.0 0.0 0.001 0.001 1e-6 /wheatfield-crop None 0 1 10000 400 1 0
```


##### 2. Extract KM mixing weights:
You can use default parameter values in command line directly, only need change example name.

```sh
	$ cd /new_pipeline_executable/wheatfield-crop

	$ python ../Solve_KM_mixing_model_fixed_KS_with_autograd.py wheatfield-crop.png  primary_pigments_KS-6.txt  None wheatfield-crop-primary_pigments_color_vertex-6-KM_weights-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes 10.0 0.1 0 1.0 0.0 blf Yes
```



##### 3. Extract KM layers: 
You need create a layer order file manually: "order1.txt" and put it in /wheatfield-crop folder, which can be order like: 0 1 2 3 4 5 or their permutations. Then you can run below command.
```sh
	$ cd /new_pipeline_executable/wheatfield-crop

	$ python ../Solve_KM_layer_model_fixed_KS_with_autograd.py wheatfield-crop.png  primary_pigments_KS-6.txt  None wheatfield-crop-primary_pigments_color_vertex-6-KM_layers-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes-order1 10.0 0.1 0 1.0 0.0 blf Yes order1.txt
```



##### 4. Extract PD layers and weights (Tan 2016) using KM pigments's RGB colors as primary color. It will use same order as KM layers. 
```sh
	$ cd /new_pipeline_executable

	$ python fast_energy_RGB_lap_adjusted_weights.py  /wheatfield-crop wheatfield-crop.png order1.txt primary_pigments_color_vertex-6.js  --weights weights-poly3-opaque400-dynamic40000.js  --solve-smaller-factor 2 --save-every 50
```



##### 5. GUI code 
```sh
	$ cd /new_pipeline_executable

	$ python Editing_GUI.py
```
