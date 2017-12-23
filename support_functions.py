from support_classes import Atom

def read_geometry_file(file_path):

    f = open(file_path, "r")
    lines = f.readlines()

    # We ignore the first 3 lines in the file.
    # Start reading with index 3.
    vec_x = lines[3].split()
    vec_y = lines[4].split()
    vec_z = lines[5].split()

    vec_x = [float(vec_x[i]) for i in range(1, len(vec_x))]
    vec_y = [float(vec_y[i]) for i in range(1, len(vec_y))]
    vec_z = [float(vec_z[i]) for i in range(1, len(vec_z))]

    vectors = [vec_x, vec_y, vec_z]
    atoms = []
    # Read the atoms.
    for i in range(6, len(lines)):

        ls = lines[i].split()
        x = float(ls[1])
        y = float(ls[2])
        z = float(ls[3])
        t = ls[4]

        a = Atom(x, y, z, t)
        atoms.append(a)

    return vectors, atoms