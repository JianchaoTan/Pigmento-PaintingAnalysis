import numpy as np 
from SILD_convexhull_simplification import *
from scipy.spatial import ConvexHull
import os, sys
import PIL.Image as Image
import errno


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__=="__main__":

    img_file=sys.argv[1]
    M=np.int(sys.argv[2])
    foldername=sys.argv[3]

    current_folder="."+foldername+"/" 

    output_prefix=current_folder+"Tan2016_PD_results/"
    make_sure_path_exists(output_prefix)

    output_prefix=output_prefix+os.path.splitext(img_file)[0]+"-"+str(M)
    
    output_rawhull_obj_file=output_prefix+"-temp_mesh_objfile.obj"
   
    img=Image.open(current_folder+img_file).convert("RGB")
    img.save(current_folder+"Tan2016_PD_results/"+img_file)
    data=np.asfarray(img).reshape((-1,3))

    hull=ConvexHull(data)
    write_convexhull_into_obj_file(hull, output_rawhull_obj_file)
    N=5000
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

    Hull_vertices=Final_hull.points[Final_hull.vertices].clip(0,255)
    Hull_vertices=Hull_vertices.round().astype(np.uint8)

    print Hull_vertices

    import json
    with open (output_prefix+"-PD_palettes.js", "w") as myfile:
    	json.dump({"vs":Hull_vertices.reshape((-1,3)).tolist()}, myfile)

    Hull_vertices=Hull_vertices.reshape((1, M, 3))
    Image.fromarray(Hull_vertices).save(output_prefix+"-PD_palettes.png")















