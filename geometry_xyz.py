# This module contains functions that we use to
# work with the geometry file. These include processing
# of the xyz coordinates as well as feature extraction files.

import time
import os
import glob
import logging
import numpy as np
from support_classes import Atom
import global_flags as gf

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gf.LOGGING_LEVEL)



ga_mass = 31.0
al_mass = 13.0
in_mass = 49.0
o_mass = 8.0

ga_r = ga_mass / in_mass
al_r = al_mass / in_mass
in_r = in_mass / in_mass
o_r = o_mass / in_mass

in_c = 'red'
ga_c = 'blue'
al_c = 'green'
o_c = 'black'

global_atom_types = {"Ga": 0,
                     "In": 0,
                     "O": 0,
                     "Al": 0}


def read_geometry_file(path_to_file):
    """
    Read geometry file and save the contants into
    a list of vectors and a list of Atoms.

    :param path_to_file:
    :return:
    """
    logger.info("Reading geometry file.")
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

        global_atom_types[t] = global_atom_types[t] + 1

        a = Atom(x, y, z, t)
        uc_atoms.append(a)
    logger.info("Geomtery file read.")
    return vectors, uc_atoms


def parse_all_structures(path_to_structures):
    """
    Read all the geometry files in path_to_structures,
    build small crystals from the unit cell atoms and
    check that there are no duplicate atoms.

    :param path_to_structures:
    :return:
    """

    folders = glob.glob(path_to_structures)

    for i in range(len(folders)):
        logger.info("Reading file - i: {0}, {1}".format(i, folders[i]))
        vectors, uc_atoms = read_geometry_file(folders[i] + "/geometry.xyz")

        atoms = build_structure(vectors,
                                uc_atoms,
                                n_x=4,
                                n_y=4,
                                n_z=4)

        check_for_duplicates(atoms)

    logger.info("Parsed all structures.")


def build_structure(vectors,
                    uc_atoms,
                    n_x=0,
                    n_y=0,
                    n_z=0):
    """
    Copy the unit cell atoms (uc_atoms) n times in
    directions x, y, z so that a bigger crystal is created.

    :param vectors:
    :param uc_atoms:
    :param n_x:
    :param n_y:
    :param n_z:
    :return:
    """

    logger.info("Building structure...")

    atoms = []

    for a in range(len(uc_atoms)):
        a_x = uc_atoms[a].x
        a_y = uc_atoms[a].y
        a_z = uc_atoms[a].z
        a_t = uc_atoms[a].t

        for i in range(-1 * n_x, n_x + 1, 1):
            for j in range(-1 * n_y, n_y + 1, 1):
                for k in range(-1 * n_z, n_z + 1, 1):
                    n_a_x = a_x + i * vectors[0][0] + j * vectors[1][0] + k * vectors[2][0]
                    n_a_y = a_y + i * vectors[0][1] + j * vectors[1][1] + k * vectors[2][1]
                    n_a_z = a_z + i * vectors[0][2] + j * vectors[1][2] + k * vectors[2][2]
                    n_a_t = a_t

                    atoms.append(Atom(n_a_x, n_a_y, n_a_z, a_t))

    logger.info("Structure built")
    return atoms


def cut_ball_from_structure(atoms, radious=50):

    logger.info("Cutting a ball.")
    new_atoms = []
    for i in range(len(atoms)):
        a_x = atoms[i].x
        a_y = atoms[i].y
        a_z = atoms[i].z

        r = np.sqrt(a_x*a_x + a_y*a_y + a_z*a_z)
        if r < radious:
            new_atoms.append(atoms[i])

    logger.info("Ball with radious {0} cut.".format(radious))
    return new_atoms


def check_for_duplicates(atoms):
    """
    After the coping of the unit cell atoms
    there might be positions within the crystal with
    duplicate atoms. Here we check that such positions
    do not exist. If the do the assert will fail.

    :param atoms:
    :return:
    """

    logger.info("Checking for duplicates...")
    set_of_atoms = set()
    for i in range(len(atoms)):
        set_of_atoms.add(atoms[i])

    assert len(set_of_atoms) == len(atoms), "There are positions with duplicate atoms!"
    logger.info("No duplicates found")


def unite_cell_volume(vectors):

    return np.dot(vectors[0], np.cross(vectors[1], vectors[2]))


def atom_density_per_A3(atoms, radious):

    volume = (4/3)*np.pi*radious*radious*radious

    local_atom_count = {"Ga": 0,
                        "In": 0,
                        "O": 0,
                        "Al": 0}

    for i in range(len(atoms)):

        a_t = atoms[i].t
        local_atom_count[a_t] = local_atom_count[a_t] + 1

    atom_density = {"rho_Ga": local_atom_count["Ga"]/volume,
                    "rho_In": local_atom_count["In"]/volume,
                    "rho_O": local_atom_count["O"]/volume,
                    "rho_Al": local_atom_count["Al"]/volume}

    return atom_density

def vector_length(vec):
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])

def ef_density(id,
               data_type="train",
               n_x=5,
               n_y=5,
               n_z=5,
               r=-1):
    """
    Extract features: atom density.

    :param id:
    :param type:
    :return:
    """

    logger.info("Creating feature: atom density.")
    vectors, uc_atoms = read_geometry_file(os.getcwd() + "/" + data_type + "/" + str(id) + "/geometry.xyz")
    atoms = build_structure(vectors,
                            uc_atoms,
                            n_x=n_x,
                            n_y=n_y,
                            n_z=n_z)

    # If the radious is not given.
    if r == -1:
        r0 = vector_length(vectors[0])
        r1 = vector_length(vectors[1])
        r2 = vector_length(vectors[2])

        v_avg = np.mean([r0, r1, r2])
        r = v_avg*(np.mean([n_x, n_y, n_z])/3)
        logger.info("r0: {0}, r1: {1}, r2: {2}, v_avg: {3}, r: {4}".format(r0, r1, r2, v_avg, r))

    check_for_duplicates(atoms)

    logger.info("atoms before cut: " + str(len(atoms)))
    atoms = cut_ball_from_structure(atoms, radious=r)
    logger.info("atoms before cut: {0}, r: {1}".format(len(atoms), r))
    atom_density = atom_density_per_A3(atoms, radious=r)

    return atom_density


if __name__ == "__main__":
    # logger.info(os.getcwd())
    #
    # id = 1
    # vectors, uc_atoms = read_geometry_file(os.getcwd() + "/train/" + str(id) + "/geometry.xyz")
    # atoms = build_structure(vectors,
    #                         uc_atoms,
    #                         n_x=5,
    #                         n_y=5,
    #                         n_z=5)
    # check_for_duplicates(atoms)
    #
    # r = 30
    # atoms = cut_ball_from_structure(atoms, radious=r)
    # atom_density = atom_density(atoms, radious=r)
    #
    # logger.info("Number of atoms: " + str(len(atoms)))
    # logger.info("vectors: " + str(vectors))
    # uc_vol = unite_cell_volume(vectors)
    # logger.info("uc_vol: " + str(uc_vol))
    # logger.info("Cluster r: " + str(r))

    file_name = "train.csv"
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)
    logger.info("data.shape: " + str(data.shape))

    n, m = data.shape
    ids = data[:, 0].reshape(-1, 1)
    x = data[:, 1:(m-2)]
    y_fe = data[:, m-2].reshape(-1, 1)
    y_bg = data[:, m-1].reshape(-1, 1)

    rho_data = np.zeros((n, 4))


    for i in range(1, n):
        start = time.time()
        logger.info("===========================")
        logger.info("n: {0}, i: {1}".format(n, i))
        id = int(ids[i])
        atom_density = ef_density(id=id,
                   data_type="train",
                   n_x=5,
                   n_y=5,
                   n_z=5,
                   r=-1)
        rho_data[i][0] = atom_density["rho_Ga"]
        rho_data[i][1] = atom_density["rho_Al"]
        rho_data[i][2] = atom_density["rho_In"]
        rho_data[i][3] = atom_density["rho_O"]

        stop = time.time()

        logger.info("rho_Ga: {0:.9f}, rho_Al: {1:.9f}, rho_In: {2:.9f}, rho_O: {3:.9f}".format(atom_density["rho_Ga"],
                                                                                         atom_density["rho_Al"],
                                                                                         atom_density["rho_In"],
                                                                                         atom_density["rho_O"]))
        logger.info("time: " + str(stop - start))

    logger.info(ids.shape)
    logger.info(x.shape)
    logger.info(rho_data.shape)
    logger.info(y_fe.shape)
    logger.info(y_bg.shape)

    new_data = np.hstack((ids, x, rho_data, y_fe, y_bg))
    rho_data = np.hstack((ids, rho_data))

    logger.info("new_data.shape: " + str(new_data.shape))
    np.savetxt("train_mod.csv", new_data, delimiter=",")
    np.savetxt("rho_data.csv", rho_data, delimiter=",")