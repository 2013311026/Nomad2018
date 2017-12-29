import os
import glob
import logging
from sage.plot.plot3d.shapes import *
from support_classes import Atom

logger = logging.getLogger('SageMath - structure_visualization')
logger.setLevel(logging.DEBUG)

logger.debug("Working in Sage...")



ga_mass = 31
al_mass = 13
in_mass = 49
o_mass = 8

ga_r = ga_mass/in_mass
al_r = al_mass/in_mass
in_r = in_mass/in_mass
o_r = o_mass/in_mass

in_c = 'red'
ga_c = 'blue'
al_c = 'green'
o_c = 'black' 

atom_types = {"Ga": 0, 
              "In": 0,
              "O":  0,
              "Al": 0}


def read_geometry_file(path_to_file):
    with open(path_to_file) as f:
        lines = f.readlines()
		
    vec_x = lines[3].split()
    vec_y = lines[4].split()
    vec_z = lines[5].split()

    vec_x = [float(vec_x[i]) for i in range(1, len(vec_x))]
    vec_y = [float(vec_y[i]) for i in range(1, len(vec_y))]
    vec_z = [float(vec_z[i]) for i in range(1, len(vec_z))]

    vectors = [vec_x, vec_y, vec_z]
    uc_atoms = []
    for i in range(6, len(lines)):
        sl = lines[i].split()
        x = float(sl[1])
        y = float(sl[2])
        z = float(sl[3])
        t = sl[4]
        
        atom_types[t] = atom_types[t] + 1
        
        a = Atom(x, y, z, t)
        uc_atoms.append(a)
    return vectors, uc_atoms

def parse_all_structures(path_to_structures):
    
    folders = glob.glob(path_to_structures)
    
    for i in range(len(folders)):
        logger.debug("Reading file - i: {0}, {1}".format(i, folders[i]))
        vectors, uc_atoms = read_geometry_file(folders[i] + "/geometry.xyz")
    
        atoms = build_structure(vectors, 
                                uc_atoms,
                                n_x=4,
                                n_y=4,
                                n_z=4)
                                
        check_for_duplicates(atoms)
        
    logger.debug("Parsed all structures.")

def build_structure(vectors, 
                    uc_atoms,
                    n_x=0,
                    n_y=0,
                    n_z=0):
                        
    logger.debug("Building structure...")

    atoms = []
    
    for a in range(len(uc_atoms)):
        a_x = uc_atoms[a].x
        a_y = uc_atoms[a].y
        a_z = uc_atoms[a].z
        a_t = uc_atoms[a].t
             
        for i in range(-1*n_x, n_x + 1, 1):
            for j in range(-1*n_y, n_y + 1, 1):
                for k in range(-1*n_z, n_z + 1, 1):
                    n_a_x = a_x + i*vectors[0][0] + j*vectors[1][0] + k*vectors[2][0]
                    n_a_y = a_y + i*vectors[0][1] + j*vectors[1][1] + k*vectors[2][1]
                    n_a_z = a_z + i*vectors[0][2] + j*vectors[1][2] + k*vectors[2][2]
                    n_a_t = a_t
                    
                    atoms.append(Atom(n_a_x, n_a_y, n_a_z, a_t))
                    
    logger.debug("Structure built")
    return atoms


def check_for_duplicates(atoms):
    
    logger.debug("Checking for duplicates...")
    set_of_atoms = set()
    for i in range(len(atoms)):
        set_of_atoms.add(atoms[i])
        
    assert len(set_of_atoms) == len(atoms), "There are positions with duplicate atoms!"
    logger.debug("No duplicates found")

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
            a_r = ga_r
            a_c = ga_c
        elif a.t == "In":
            a_r = in_r
            a_c = in_c
        elif a.t == "Al":
            a_r = al_r
            a_c = al_c  
        elif a.t == "O":
            a_r = o_r
            a_c = o_c
        else:
            pass
            
    
        S = S + Sphere(a_r, color=a_c).translate(a.x ,a.y , a.z)
    
    S.show(aspect_ratio=1)
    
    #save(S,'temp.png', axes=False, aspect_ratio=True) 
    #os.system('display temp.png')
    

# vectors, uc_atoms = read_geometry_file("/home/tadek/Coding/Kaggle/Nomad2018/train/1233/geometry.xyz")
# atoms = build_structure(vectors, 
                        # uc_atoms,
                        # n_x=1,
                        # n_y=1,
                        # n_z=1)
# check_for_duplicates(atoms)
# draw_structure(vectors, atoms)
# print(vectors)

parse_all_structures("/home/tadek/Coding/Kaggle/Nomad2018/test/test/*")
print(atom_types)

