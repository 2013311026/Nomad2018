import os
import glob
from sage.plot.plot3d.shapes import *
from support_classes import Atom


print("Working in Sage...")

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

atom_types = {}


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
    atoms = []
    for i in range(6, len(lines)):
        sl = lines[i].split()
        x = float(sl[1])
        y = float(sl[2])
        z = float(sl[3])
        t = sl[4]
        
        if t in atom_types:
            atom_types[t] = atom_types[t] + 1
        else:
            atom_types[t] = 1
        
        a = Atom(x, y, z, t)
        atoms.append(a)
    return vectors, atoms

def parse_all_structures(path_to_structures):
    
    folders = glob.glob(path_to_structures)
    
    for i in range(len(folders)):
        print("i: {0}, {1}".format(i, folders[i]))
        read_geometry_file(folders[i] + "/geometry.xyz")
    


def draw_structure(atoms):
    
    S = Sphere(0.1, color='yellow')
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
    
    S.show()
    
    save(S,'temp.png', axes=False, aspect_ratio=True) 
    os.system('display temp.png')
    

vectors, atoms = read_geometry_file("/home/tadek/Coding/Kaggle/Nomad2018/train/1/geometry.xyz")

parse_all_structures("/home/tadek/Coding/Kaggle/Nomad2018/train/*")

draw_structure(atoms)
print(atom_types)

