import numpy as np
import geometry_xyz as gxyz
from sage.plot.plot3d.shapes import *


def draw_structure(vectors, atoms):

    S = Sphere(0.1, color='yellow')
    S = S + arrow3d((0,0,0), tuple(vectors[0]), width=3)
    S = S + arrow3d((0,0,0), tuple(vectors[1]), width=3)
    S = S + arrow3d((0,0,0), tuple(vectors[2]), width=3)
    for i in range(len(atoms)):
        
        a = atoms[i]
        a_r = None
        a_c = None
        
        if a.t == "Ga":
            a_r = gxyz.ga_r
            a_c = gxyz.ga_c
        elif a.t == "In":
            a_r = gxyz.in_r
            a_c = gxyz.in_c
        elif a.t == "Al":
            a_r = gxyz.al_r
            a_c = gxyz.al_c  
        elif a.t == "O":
            a_r = gxyz.o_r
            a_c = gxyz.o_c
        else:
            pass
    
        S = S + Sphere(a_r, color=a_c).translate(a.x ,a.y , a.z)
    
    S.show(aspect_ratio=1)
    
    #save(S,'temp.png', axes=False, aspect_ratio=True) 
    #os.system('display temp.png')

def draw_structure_around(vectors, atoms, origin, r=5):

    o_x = origin.x
    o_y = origin.y
    o_z = origin.z

    S = Sphere(0.1, color='yellow')
    S = S + arrow3d((0,0,0), tuple(vectors[0]), width=3)
    S = S + arrow3d((0,0,0), tuple(vectors[1]), width=3)
    S = S + arrow3d((0,0,0), tuple(vectors[2]), width=3)
    for i in range(len(atoms)):
        
        a = atoms[i]
        a_r = None
        a_c = None
        
        dx = a.x - o_x
        dy = a.y - o_y
        dz = a.z - o_z
        
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        if d > r:
            continue
        
        print("d: {0}".format(d))
        
        if a.t == "Ga":
            a_r = gxyz.ga_r
            a_c = gxyz.ga_c
        elif a.t == "In":
            a_r = gxyz.in_r
            a_c = gxyz.in_c
        elif a.t == "Al":
            a_r = gxyz.al_r
            a_c = gxyz.al_c  
        elif a.t == "O":
            a_r = gxyz.o_r
            a_c = gxyz.o_c
        else:
            pass
    
        S = S + Sphere(a_r, color=a_c).translate(dx, dy, dz)
    
    S.show(aspect_ratio=1)

vectors, uc_atoms = gxyz.read_geometry_file("/home/tadek/Coding/Kaggle/Nomad2018/train/2/geometry.xyz")
atoms = gxyz.build_structure(vectors, 
                             uc_atoms,
                             n_x=0,
                             n_y=0,
                             n_z=0)
gxyz.check_for_duplicates(atoms)
atoms = gxyz.cut_ball_from_structure(atoms, radious=25)
draw_structure(vectors, atoms)

index = 31
print("uc_atoms[index]: {0}".format(uc_atoms[index].t))
draw_structure_around(vectors, atoms, uc_atoms[index], r=3)
print(vectors)

#parse_all_structures("/home/tadek/Coding/Kaggle/Nomad2018/test/test/*")
