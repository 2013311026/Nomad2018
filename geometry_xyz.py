# This module contains functions that we use to
# work with the geometry file. These include processing
# of the xyz coordinates as well as feature extraction files.

import time
import os
import glob
import logging
import numpy as np
import pymatgen
from pymatgen.analysis import ewald
from support_classes import Atom
from support_classes import UCAtoms
import global_flags_constanst as gfc
import matplotlib.pyplot as plt

from support_functions import split_data_into_id_x_y

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gfc.LOGGING_LEVEL)

# Sphere radious in which nearest
# neighbours should be searched.
SEARCH_NEIGHBOUR_RADIOUS = 7.0
EPSILON = 1e-12

# Source: https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)#dagger
ga_radii_empirical = 1.3
al_radii_empirical = 1.25
in_radii_empirical = 1.55
o_radii_empirical = 0.6

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


def hist_data(data, text=""):

    plt.figure()
    plt.hist(data, bins=20)
    plt.title(text)
    plt.show()

def build_angle_triangles():

    """
    We want to calculate the angles between nearest
    neighbour atoms.

    For example the angle between Al-Ga-In

        Al
      /
    Ga  <- angle to calculate
      \
       In

    This function build a list of all possibilities.
    Each possibility is treated as a triangle.

    We will also calculate the distance between Al and In atoms.

    :param uc_atoms:
    :param atoms:
    :return:
    """

    logger.info("Building angle triangles")

    trangle_point_A = ["Ga", "Al", "In", "O"]
    trangle_point_B_C = [["Ga", "Al"],
                         ["Ga", "In"],
                         ["Ga",  "O"],
                         ["Al", "In"],
                         [ "O", "In"],
                         [ "O", "Al"]]

    triangles = []
    for i in range(len(trangle_point_A)):
        for j in range(len(trangle_point_B_C)):

            three_point = [trangle_point_A[i], trangle_point_B_C[j]]

            triangles.append(three_point)

    for k in range(len(triangles)):
        logger.debug(triangles[k])

    logger.info("Finished building angle triangles")
    return triangles


# Constant table of triangles.
triangles_list = build_angle_triangles()


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

        if sl[4] == "Ga":
            c = ga_mass
        elif sl[4] == "Al":
            c = al_mass
        elif sl[4] == "In":
            c = in_mass
        elif sl[4] == "O":
            c = o_mass

        global_atom_types[t] = global_atom_types[t] + 1

        a = Atom(x, y, z, t, c)
        uc_atoms.append(a)
    logger.info("Geomtery file read.")
    # uc_atoms = UCAtoms(uc_atoms)

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


def atom_density_per_a3(atoms, radious):
    """
    Atom density per cubic Angstrom.

    :param atoms:
    :param radious:
    :return:
    """

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


def extract_features(id,
                     data_type="train",
                     n_x=5,
                     n_y=5,
                     n_z=5,
                     r=-1):
    """
    Main function to extract features from the geometry file.

    :param id:
    :param data_type:
    :param n_x:
    :param n_y:
    :param n_z:
    :param r:
    :return:
    """

    logger.info("Creating feature: atom density.")
    vectors, uc_atoms = read_geometry_file(os.getcwd() + "/" + data_type + "/" + str(id) + "/geometry.xyz")

    unit_cell_params = unit_cell_dimensions(vectors)
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
        logger.debug("r0: {0}, r1: {1}, r2: {2}, v_avg: {3}, r: {4}".format(r0, r1, r2, v_avg, r))

    check_for_duplicates(atoms)

    logger.debug("atoms before cut: " + str(len(atoms)))
    atoms = cut_ball_from_structure(atoms, radious=r)

    # build_angle_triangles()
    angles_and_rs = calculate_angles_of_nn_atoms(uc_atoms,
                                                 atoms)

    nn_bond_properties = nearest_neighbour_bond_parameters(uc_atoms,
                                                           atoms)

    logger.debug("atoms before cut: {0}, r: {1}".format(len(atoms), r))
    atom_density = atom_density_per_a3(atoms, radious=r)
    percentage_of_atoms = calculate_atom_percentages(atoms)

    new_features = {}
    new_features["atom_density"] = atom_density
    new_features['percentage_of_atoms'] = percentage_of_atoms
    new_features["unit_cell_params"] = unit_cell_params
    new_features["nn_bond_properties"] = nn_bond_properties
    new_features["angles_and_rs"] = angles_and_rs

    return new_features


def calculate_atom_percentages(atoms):

    logger.info("Calculating atom percentages.")

    n_ga = 0
    n_al = 0
    n_in = 0
    n_o = 0

    tot = len(atoms)
    for i in range(tot):
        a = atoms[i]
        if a.t == "Ga":
            n_ga = n_ga + 1
        elif a.t == "In":
            n_in = n_in + 1
        elif a.t == "Al":
            n_al = n_al + 1
        elif a.t == "O":
            n_o = n_o + 1
        else:
            assert False, "Atom type not recognized!"

    percentage_of_atoms = {}

    p_ga = n_ga / tot
    p_al = n_al / tot
    p_in = n_in / tot
    p_o = n_o / tot

    percentage_of_atoms["percentage_of_ga"] = p_ga
    percentage_of_atoms["percentage_of_al"] = p_al
    percentage_of_atoms["percentage_of_in"] = p_in
    percentage_of_atoms["percentage_of_o"] = p_o

    logger.debug("p_ga: {0:.9f}; p_al: {1:.9f}; p_in: {2:.9f}; p_o: {3:.9f}".format(p_ga, p_al, p_in, p_o))

    return percentage_of_atoms


def nearest_neighbour_bond_parameters(uc_atoms,
                                      atoms):
    """
    For every possible nearest neighbour bond calculates the average
    length, its standard deviation, is maximal and minimal values.

    :param uc_atoms:
    :param atoms:
    :return:
    """

    logger.info("Calculating the average nearest neighbour bond parameters.")

    bond_pair = [["Ga", "Ga"],
                 ["Ga", "Al"],
                 ["Ga", "In"],
                 ["Ga",  "O"],

                 ["Al", "Ga"],
                 ["Al", "Al"],
                 ["Al", "In"],
                 ["Al",  "O"],

                 ["In", "Ga"],
                 ["In", "Al"],
                 ["In", "In"],
                 ["In",  "O"],

                 ["O",  "Ga"],
                 ["O",  "Al"],
                 ["O",  "In"],
                 ["O",   "O"]]

    nn_bond_properties = {}

    for i in range(len(bond_pair)):
        origin_atom_type = bond_pair[i][0]
        destination_atom = bond_pair[i][1]

        origin_exists = check_if_atom_exists_in_structure(uc_atoms, origin_atom_type)
        destination_exists = check_if_atom_exists_in_structure(uc_atoms, destination_atom)

        avg_bond_str = "avg_" + origin_atom_type + "-" + destination_atom
        std_bond_str = "std_" + origin_atom_type + "-" + destination_atom
        max_bond_str = "max_" + origin_atom_type + "-" + destination_atom
        min_bond_str = "min_" + origin_atom_type + "-" + destination_atom

        if (origin_exists == False) or (destination_exists == False):
            nn_bond_properties[avg_bond_str] = -1
            nn_bond_properties[std_bond_str] = -1
            nn_bond_properties[max_bond_str] = -1
            nn_bond_properties[min_bond_str] = -1
            continue

        nn_b_p = nn_bond_parameters_between_two_specific_atoms(uc_atoms,
                                                               atoms,
                                                               origin_atom_type=origin_atom_type,
                                                               destination_atom=destination_atom)

        nn_bond_properties[avg_bond_str] = nn_b_p["avg_nn_bond_length"]
        nn_bond_properties[std_bond_str] = nn_b_p["std_nn_bond_length"]
        nn_bond_properties[max_bond_str] = nn_b_p["max_nn_bond"]
        nn_bond_properties[min_bond_str] = nn_b_p["min_nn_bond"]


    for key, val in sorted(nn_bond_properties.items()):
        logger.debug("bond props {0}: {1}".format(key, val))

    return nn_bond_properties


def check_if_atom_exists_in_structure(uc_atoms,
                                      atom_type):

    for i in range(len(uc_atoms)):
        a = uc_atoms[i]
        if a.t == atom_type:
            return True

    return False


def nn_bond_parameters_between_two_specific_atoms(uc_atoms,
                                                  atoms,
                                                  origin_atom_type="Al",
                                                  destination_atom="Al"):

    """
    For every atom of type (origin_atom_type) in UC calculates: the average bond length,
    the standard deviation of this length, the maximal bond length and
    the minimal bond length to the nearest atom of another type (destination_atom).

    :param uc_atoms:
    :param atoms:
    :param origin_atom_type:
    :param destination_atom:
    :return:
    """

    logger.info("Getting avg nearest neighbour bond beteewn {0} - {1}".format(origin_atom_type,
                                                                              destination_atom))

    nn = {}

    nearest_neighbour_bonds_list = []
    n_atoms_of_type = 0
    for i in range(len(uc_atoms)):
        logger.debug("Scanning UC for nn, i: {0}".format(i))
        uc_a = uc_atoms[i]
        if uc_a.t != origin_atom_type:
            continue

        n_atoms_of_type = n_atoms_of_type + 1
        minimal_distance = np.inf

        for j in range(len(atoms)):
            a = atoms[j]

            if a.t != destination_atom:
                continue

            dx = a.x - uc_a.x
            dy = a.y - uc_a.y
            dz = a.z - uc_a.z

            d = np.sqrt(dx*dx + dy*dy + dz*dz)

            # if d > SEARCH_NEIGHBOUR_RADIOUS:
            #     continue
            # else:
            #     atoms_to_far = False

            d = round(d, 6)
            # logger.info("d: {0}".format(d))

            if d < EPSILON:
                continue

            if minimal_distance > d:
                minimal_distance = d

        nearest_neighbour_bonds_list.append(minimal_distance)

    avg_nn_bond_length = np.mean(nearest_neighbour_bonds_list)
    std_nn_bond_length = np.std(nearest_neighbour_bonds_list)

    max_nn_bond = np.max(nearest_neighbour_bonds_list)
    min_nn_bond = np.min(nearest_neighbour_bonds_list)

    logger.debug("avg_nn_bond_length: {0}".format(avg_nn_bond_length))
    logger.debug("std_nn_bond_length: {0}".format(std_nn_bond_length))
    logger.debug("max_nn_bond: {0}".format(max_nn_bond))
    logger.debug("min_nn_bond: {0}".format(min_nn_bond))

    # logger.debug("Number of atoms with type {0}: {1}".format(atom_type, n_atoms_of_type))
    # for key, val in nn.items():
    #     logger.debug("nn distance: {0}, nn val: {1}".format(key, val))

    nn_bond_properties = {}
    nn_bond_properties["avg_nn_bond_length"] = avg_nn_bond_length
    nn_bond_properties["std_nn_bond_length"] = std_nn_bond_length
    nn_bond_properties["max_nn_bond"] = max_nn_bond
    nn_bond_properties["min_nn_bond"] = min_nn_bond

    return nn_bond_properties


def find_closest_atom(origin_atom,
                      destination_atom_type,
                      atoms):

    """
    Finds an atom of type destination_atom_type that is
    closest to the origin_atom.

    :param origin_atom:
    :param destination_atom_type:
    :param atoms:
    :return:
    """

    minimal_distance = np.inf
    closest_a_x = None
    closest_a_y = None
    closest_a_z = None
    closest_a_t = None

    for i in range(len(atoms)):
        a = atoms[i]

        if a.t != destination_atom_type:
            continue

        dx = a.x - origin_atom.x
        dy = a.y - origin_atom.y
        dz = a.z - origin_atom.z

        d = np.sqrt(dx * dx + dy * dy + dz * dz)

        d = round(d, 6)
        # logger.info("d: {0}".format(d))

        if d < EPSILON:
            continue

        if minimal_distance > d:
            minimal_distance = d
            closest_a_x = a.x
            closest_a_y = a.y
            closest_a_z = a.z
            closest_a_t = a.t

    closest_atom = Atom(closest_a_x,
                        closest_a_y,
                        closest_a_z,
                        closest_a_t)

    return closest_atom


def calculate_angles_of_nn_atoms(uc_atoms,
                                 atoms):

    logger.info("Calculating angles for given structure.")
    nt = len(triangles_list)

    angles_and_rs = {}
    # Loop over possible triangles
    for i in range(nt):
        angle_list = []
        r_list = []

        origin_atom_type = triangles_list[i][0] # Triangle point A
        bond_b_atom_type = triangles_list[i][1][0] # Triangle point B
        bond_c_atom_type = triangles_list[i][1][1] # Triangle point C

        avg_angle_str = "avg_angle_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                                     bond_b_atom_type,
                                                                     bond_c_atom_type)

        std_angle_str = "std_angle_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                                     bond_b_atom_type,
                                                                     bond_c_atom_type)

        max_angle_str = "max_angle_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                                     bond_b_atom_type,
                                                                     bond_c_atom_type)

        min_angle_str = "min_angle_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                                     bond_b_atom_type,
                                                                     bond_c_atom_type)

        avg_r_str = "avg_r_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                             bond_b_atom_type,
                                                             bond_c_atom_type)

        std_r_str = "std_r_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                             bond_b_atom_type,
                                                             bond_c_atom_type)

        max_r_str = "max_r_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                             bond_b_atom_type,
                                                             bond_c_atom_type)

        min_r_str = "min_r_between_nn_bc_{0}_{1}_{2}".format(origin_atom_type,
                                                             bond_b_atom_type,
                                                             bond_c_atom_type)

        logger.debug("origin_type: {0}, bond_B: {1}, bond_C: {2}".format(origin_atom_type,
                                                                         bond_b_atom_type,
                                                                         bond_c_atom_type))

        origin_exists = check_if_atom_exists_in_structure(uc_atoms, origin_atom_type)
        b_exists = check_if_atom_exists_in_structure(uc_atoms, bond_b_atom_type)
        c_exists = check_if_atom_exists_in_structure(uc_atoms, bond_c_atom_type)

        logger.debug("origin_exists: {0}; b_exists: {1}; c_exists: {2}".format(origin_exists,
                                                                              b_exists,
                                                                              c_exists))

        if ((origin_exists is False) or
            (b_exists is False) or
            (c_exists is False)):

            logger.debug("Trinagle not valid!")

            angles_and_rs[avg_angle_str] = -1
            angles_and_rs[std_angle_str] = -1
            angles_and_rs[max_angle_str] = -1
            angles_and_rs[min_angle_str] = -1

            angles_and_rs[avg_r_str] = -1
            angles_and_rs[std_r_str] = -1
            angles_and_rs[max_r_str] = -1
            angles_and_rs[min_r_str] = -1

            continue

        logger.debug("Triangle exists! Lets find angles!")

        for j in range(len(uc_atoms)):

            if uc_atoms[j].t != origin_atom_type:
                continue

            origin_atom = uc_atoms[j]
            b_atom = find_closest_atom(origin_atom,
                                       bond_b_atom_type,
                                       atoms)

            c_atom = find_closest_atom(origin_atom,
                                       bond_c_atom_type,
                                       atoms)

            logger.debug("origin_atom: {0}".format(origin_atom))
            logger.debug("b_atom: {0}".format(b_atom))
            logger.debug("c_atom: {0}".format(c_atom))


            vec_ab = [b_atom.x - origin_atom.x,
                      b_atom.y - origin_atom.y,
                      b_atom.z - origin_atom.z]

            vec_ac = [c_atom.x - origin_atom.x,
                      c_atom.y - origin_atom.y,
                      c_atom.z - origin_atom.z]

            vec_ab = np.array(vec_ab)
            vec_ac = np.array(vec_ac)

            dot12 = np.dot(vec_ab, vec_ac)
            norm1 = vector_length(vec_ab)
            norm2 = vector_length(vec_ac)

            alpha = np.arccos(dot12 / (norm1 * norm2))

            vec_cb = vec_ab - vec_ac

            r = vector_length(vec_cb)

            logger.debug("alpha: {0}; r: {1}".format(alpha, r))

            angle_list.append(alpha)
            r_list.append(r)

        avg_angle_between_nn_bc = np.mean(angle_list)
        std_angle_between_nn_bc = np.std(angle_list)
        max_angle_between_nn_bc = np.max(angle_list)
        min_angle_between_nn_bc = np.min(angle_list)

        avg_r_between_nn_bc = np.mean(r_list)
        std_r_between_nn_bc = np.std(r_list)
        max_r_between_nn_bc = np.max(r_list)
        min_r_between_nn_bc = np.min(r_list)

        angles_and_rs[avg_angle_str] = avg_angle_between_nn_bc
        angles_and_rs[std_angle_str] = std_angle_between_nn_bc
        angles_and_rs[max_angle_str] = max_angle_between_nn_bc
        angles_and_rs[min_angle_str] = min_angle_between_nn_bc

        angles_and_rs[avg_r_str] = avg_r_between_nn_bc
        angles_and_rs[std_r_str] = std_r_between_nn_bc
        angles_and_rs[max_r_str] = max_r_between_nn_bc
        angles_and_rs[min_r_str] = min_r_between_nn_bc

    for key, val in sorted(angles_and_rs.items()):
        logger.debug("angle or r: {0}, val: {1}".format(key, val))

    return angles_and_rs


def unit_cell_dimensions(vectors):
    """
    Calculates the basic unit cell dimensions and their variants.


    :param vectors:
    :return:
    """

    vec_a = vectors[0]
    vec_b = vectors[1]
    vec_c = vectors[2]

    a = vector_length(vec_a)
    b = vector_length(vec_b)
    c = vector_length(vec_c)

    a2 = a*a
    b2 = b*b
    c2 = c*c

    vec_a_star = 2.0*np.pi*np.cross(vec_b, vec_c)/np.dot(vec_a, np.cross(vec_b, vec_c))
    vec_b_star = 2.0*np.pi*np.cross(vec_c, vec_a)/np.dot(vec_b, np.cross(vec_c, vec_a))
    vec_c_star = 2.0*np.pi*np.cross(vec_a, vec_b)/np.dot(vec_c, np.cross(vec_a, vec_b))

    a_star = vector_length(vec_a_star)
    b_star = vector_length(vec_b_star)
    c_star = vector_length(vec_c_star)

    a_star2 = a_star*a_star
    b_star2 = b_star*b_star
    c_star2 = c_star*c_star

    unit_cell_params = {}
    unit_cell_params["a"] = a
    unit_cell_params["a2"] = a2
    unit_cell_params["one_over_a"] = 1.0/a
    unit_cell_params["one_over_a2"] = 1.0/a2
    unit_cell_params["a_star"] = a_star
    unit_cell_params["a_star2"] = a_star2
    unit_cell_params["one_over_a_star"] = 1.0/a_star
    unit_cell_params["one_over_a_star2"] = 1.0/a_star2

    unit_cell_params["b"] = b
    unit_cell_params["b2"] = b2
    unit_cell_params["one_over_b"] = 1.0/b
    unit_cell_params["one_over_b2"] = 1.0/b2
    unit_cell_params["b_star"] = b_star
    unit_cell_params["b_star2"] = b_star2
    unit_cell_params["one_over_b_star"] = 1.0/b_star
    unit_cell_params["one_over_b_star2"] = 1.0/b_star2

    unit_cell_params["c"] = c
    unit_cell_params["c2"] = c2
    unit_cell_params["one_over_c"] = 1.0/c
    unit_cell_params["one_over_c2"] = 1.0/c2
    unit_cell_params["c_star"] = c_star
    unit_cell_params["c_star2"] = c_star2
    unit_cell_params["one_over_c_star"] = 1.0/c_star
    unit_cell_params["one_over_c_star2"] = 1.0/c_star2

    unit_cell_params["ga_radii_div_a"] = ga_radii_empirical / a
    unit_cell_params["ga_radii_div_b"] = ga_radii_empirical / b
    unit_cell_params["ga_radii_div_c"] = ga_radii_empirical / c

    unit_cell_params["al_radii_div_a"] = al_radii_empirical / a
    unit_cell_params["al_radii_div_b"] = al_radii_empirical / b
    unit_cell_params["al_radii_div_c"] = al_radii_empirical / c

    unit_cell_params["in_radii_div_a"] = in_radii_empirical / a
    unit_cell_params["in_radii_div_b"] = in_radii_empirical / b
    unit_cell_params["in_radii_div_c"] = in_radii_empirical / c

    unit_cell_params["o_radii_div_a"] = o_radii_empirical / a
    unit_cell_params["o_radii_div_b"] = o_radii_empirical / b
    unit_cell_params["o_radii_div_c"] = o_radii_empirical / c

    return unit_cell_params


def scan_through_geometry_files_and_extrac_features(data,
                                                    data_type="train",
                                                    file_name_type=""):

    logger.info("data.shape: " + str(data.shape))

    n, m = data.shape
    ids = data[:, 0].reshape(-1, 1)
    x = data[:, 1:(m - 2)]
    y_fe = data[:, m - 2].reshape(-1, 1)
    y_bg = data[:, m - 1].reshape(-1, 1)

    rho_data = np.zeros((n, 4))
    percentage_atom_data = np.zeros((n, 4))
    unit_cell_data = np.zeros((n, 36))
    nn_bond_parameters_data = np.zeros((n, 64))
    angles_and_rs_data = np.zeros((n, 24 * 8))

    for i in range(0, n):
        start = time.time()
        logger.info("===========================")
        logger.info("n: {0}, i: {1}".format(n, i))
        id = int(ids[i])
        new_features = extract_features(id=id,
                                        data_type=data_type,
                                        n_x=4,
                                        n_y=4,
                                        n_z=4,
                                        r=-1)

        atom_density = new_features["atom_density"]
        percentage_of_atoms = new_features["percentage_of_atoms"]
        unit_cell_params = new_features["unit_cell_params"]
        nn_bond_properties = new_features["nn_bond_properties"]
        angles_and_rs = new_features["angles_and_rs"]

        # New features

        index = 0
        # The dictionary needs to be sorted by keys so the features are
        # always in the same order.
        for key, val in sorted(angles_and_rs.items(), key=lambda t: t[0]):
            logger.info("Writing {0} to array, value {1}".format(key, val))
            angles_and_rs_data[i][index] = val
            index = index + 1

        index = 0
        for key, val in sorted(atom_density.items(), key=lambda t: t[0]):
            logger.info("Writing {0} to array, value {1}".format(key, val))
            rho_data[i][index] = val
            index = index + 1

        index = 0
        for key, val in sorted(percentage_of_atoms.items(), key=lambda t: t[0]):
            logger.info("Writing {0} to array, value {1}".format(key, val))
            percentage_atom_data[i][index] = val
            index = index + 1

        index = 0
        for key, val in sorted(unit_cell_params.items(), key=lambda t: t[0]):
            logger.info("Writing {0} to array, value {1}".format(key, val))
            unit_cell_data[i][index] = val
            index = index + 1

        index = 0
        # The dictionary needs to be sorted by keys so the features are
        # always in the same order.
        for key, val in sorted(nn_bond_properties.items(), key=lambda t: t[0]):
            logger.info("Writing {0} to array, value {1}".format(key, val))
            nn_bond_parameters_data[i][index] = val
            index = index + 1

        stop = time.time()
        logger.info("rho_Ga: {0:.9f}, rho_Al: {1:.9f}, rho_In: {2:.9f}, rho_O: {3:.9f}".format(atom_density["rho_Ga"],
                                                                                               atom_density["rho_Al"],
                                                                                               atom_density["rho_In"],
                                                                                               atom_density["rho_O"]))
        logger.info("time: " + str(stop - start))

    logger.info("ids.shape: " +str(ids.shape))
    logger.info("x.shape: " + str(x.shape))
    logger.info("rho_data.shape: " + str(rho_data.shape))
    logger.info("y_fe.shape: " + str(y_fe.shape))
    logger.info("y_bg.shape: " + str(y_bg.shape))

    new_data = np.hstack((ids, x, rho_data, y_fe, y_bg))
    rho_data = np.hstack((ids, rho_data))
    percentage_atom_data = np.hstack((ids, percentage_atom_data))
    unit_cell_data = np.hstack((ids, unit_cell_data))
    nn_bond_parameters_data = np.hstack((ids, nn_bond_parameters_data))
    angles_and_rs_data = np.hstack((ids, angles_and_rs_data))

    logger.info("new_data.shape: " + str(new_data.shape))

    np.savetxt(file_name_type + "train_mod.csv", new_data, delimiter=",")
    np.savetxt(file_name_type + "_rho_data.csv", rho_data, delimiter=",")
    np.savetxt(file_name_type + "_percentage_atom_data.csv", percentage_atom_data, delimiter=",")
    np.savetxt(file_name_type + "_unit_cell_data.csv", unit_cell_data, delimiter=",")
    np.savetxt(file_name_type + "_nn_bond_parameters_data.csv", nn_bond_parameters_data, delimiter=",")
    np.savetxt(file_name_type + "_angles_and_rs_data.csv", angles_and_rs_data, delimiter=",")

    np.save(file_name_type + "train_mod.npy", new_data)
    np.save(file_name_type + "_rho_data.npy", rho_data)
    np.save(file_name_type + "_percentage_atom_data.npy", percentage_atom_data)
    np.save(file_name_type + "_unit_cell_data.npy", unit_cell_data)
    np.save(file_name_type + "_nn_bond_parameters_data.npy", nn_bond_parameters_data)
    np.save(file_name_type + "_angles_and_rs_data.npy", angles_and_rs_data)


def convert_uc_atoms_to_input_for_pymatgen(uc_atoms):

    n = len(uc_atoms)
    atom_coords = []
    atom_labels = []
    charge_list = []
    for i in range(n):
        x = uc_atoms[i].x
        y = uc_atoms[i].y
        z = uc_atoms[i].z
        t = uc_atoms[i].t
        c = uc_atoms[i].c

        vec = [x, y, z]

        atom_coords.append(vec)
        atom_labels.append(t)
        charge_list.append(c)
    site_properties = {"charge": charge_list}

    return atom_coords, atom_labels, site_properties


def ewald_matrix_features(data,
                          noa,
                          data_type="train",
                          file_name_type=""):

    # noa - number of atoms in unit cell
    logger.info("Extracting Ewald matrix features. data_type {0}".format(data_type))

    logger.info("data.shape: " + str(data.shape))
    ids, x, y_fe, y_bg = split_data_into_id_x_y(data, data_type=data_type)

    logger.info("x.shape: " + str(x.shape))

    n, m = ids.shape
    ewald_sum_data = np.zeros((n, 11))

    if noa != -1:
        ewald_sum_real_energy_matrix = np.zeros((n, noa*noa))
        ewald_sum_reciprocal_energy_matrix = np.zeros((n, noa*noa))
        ewald_sum_total_energy_matrix = np.zeros((n, noa*noa))
        ewald_sum_point_energy_matrix = np.zeros((n, noa))

    for i in range(n):
        id = int(ids[i, 0])
        print("id: {0}".format(id))

        vectors, uc_atoms = read_geometry_file(data_type + "/" + str(id) + "/geometry.xyz")
        atom_coords, atom_labels, site_properties = convert_uc_atoms_to_input_for_pymatgen(uc_atoms)

        lv1 = x[i, 5]
        lv2 = x[i, 6]
        lv3 = x[i, 7]

        lv1_c = vector_length(vectors[0])
        lv2_c = vector_length(vectors[1])
        lv3_c = vector_length(vectors[2])

        alpha = x[i, 8]
        beta = x[i, 9]
        gamma = x[i, 10]

        logger.info("lv1: {0}, lv2: {1}, lv3: {2}".format(lv1, lv2, lv3))
        logger.info("lv1: {0}, lv2: {1}, lv3: {2}".format(lv1_c, lv2_c, lv3_c))
        logger.info("alpha: {0}, beta: {1}, gamma: {2}".format(alpha, beta, gamma))

        lattice = pymatgen.Lattice.from_parameters(a=lv1,
                                                   b=lv2,
                                                   c=lv3,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   gamma=gamma)

        structure = pymatgen.Structure(lattice, atom_labels, atom_coords, site_properties=site_properties)

        ewald_sum = ewald.EwaldSummation(structure)

        logger.info("ewald_sum: \n{0}".format(ewald_sum))

        logger.info("Real space energy: {0}".format(ewald_sum.real_space_energy))
        logger.info("Reciprocal energy: {0}".format(ewald_sum.reciprocal_space_energy))
        logger.info("Point energy: {0}".format(ewald_sum.point_energy))
        logger.info("Total energy: {0}".format(ewald_sum.total_energy) )

        ewald_sum_data[i][0] = ewald_sum.real_space_energy/len(uc_atoms)
        ewald_sum_data[i][1] = ewald_sum.reciprocal_space_energy/len(uc_atoms)
        ewald_sum_data[i][2] = ewald_sum.point_energy/len(uc_atoms)
        ewald_sum_data[i][3] = ewald_sum.total_energy/len(uc_atoms)

        ewald_sum_data[i][4] = np.trace(ewald_sum.real_space_energy_matrix)
        ewald_sum_data[i][5] = np.trace(ewald_sum.reciprocal_space_energy_matrix)
        ewald_sum_data[i][6] = np.trace(ewald_sum.total_energy_matrix)
        ewald_sum_data[i][7] = np.sum(ewald_sum.point_energy_matrix)

        ewald_sum_data[i][8] = np.trace(np.fliplr(ewald_sum.real_space_energy_matrix))
        ewald_sum_data[i][9] = np.trace(np.fliplr(ewald_sum.reciprocal_space_energy_matrix))
        ewald_sum_data[i][10] = np.trace(np.fliplr(ewald_sum.total_energy_matrix))

        if noa != -1:
            ewald_sum_real_energy_matrix[i, :] = ewald_sum.real_space_energy_matrix.reshape(-1, )
            ewald_sum_reciprocal_energy_matrix[i, :] = ewald_sum.reciprocal_space_energy_matrix.reshape(-1, )
            ewald_sum_total_energy_matrix[i, :] = ewald_sum.total_energy_matrix.reshape(-1, )
            ewald_sum_point_energy_matrix[i, :] = ewald_sum.point_energy_matrix.reshape(-1, )

        logger.info("real_space_energy_matrix trace: " + str(ewald_sum_data[i][4]))
        logger.info("reciprocal_space_energy_matrix trace: " + str(ewald_sum_data[i][5]))
        logger.info("total_energy_matrix trace: " + str(ewald_sum_data[i][6]))

    ewald_sum_data = np.hstack((ids, ewald_sum_data))

    if noa != -1:
        ewald_sum_real_energy_matrix = np.hstack((ids, ewald_sum_real_energy_matrix))
        ewald_sum_reciprocal_energy_matrix = np.hstack((ids, ewald_sum_reciprocal_energy_matrix))
        ewald_sum_total_energy_matrix = np.hstack((ids, ewald_sum_total_energy_matrix))
        ewald_sum_point_energy_matrix = np.hstack((ids, ewald_sum_point_energy_matrix))

    np.savetxt(file_name_type + "_ewald_sum_data.csv", ewald_sum_data, delimiter=",")
    np.save(file_name_type + "_ewald_sum_data.npy", ewald_sum_data)

    if noa != -1:
        np.save(file_name_type + "_ewald_sum_real_energy_matrix.npy", ewald_sum_real_energy_matrix)
        np.save(file_name_type + "_ewald_sum_reciprocal_energy_matrix.npy", ewald_sum_reciprocal_energy_matrix)
        np.save(file_name_type + "_ewald_sum_total_energy_matrix.npy", ewald_sum_total_energy_matrix)
        np.save(file_name_type + "_ewald_sum_point_energy_matrix.npy", ewald_sum_point_energy_matrix)


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

    data_type = "train"
    file_name = "train.csv"
    train_data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    file_name = "test.csv"
    test_data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    train_total_number_of_atoms = np.unique(train_data[:, gfc.LABELS["number_of_total_atoms"]])
    test_total_number_of_atoms = np.unique(test_data[:, gfc.LABELS["number_of_total_atoms"]])

    logger.info("train_total_number_of_atoms: {0}".format(train_total_number_of_atoms))
    logger.info("test_total_number_of_atoms: {0}".format(test_total_number_of_atoms))

    assert_error_text = "The train and test data have examples with " + \
                        "a different amount of atoms! You won't be able " + \
                        "to make predictions!"

    assert np.array_equal(train_total_number_of_atoms, test_total_number_of_atoms), assert_error_text


    #scan_through_geometry_files_and_extrac_features(train_data, data_type="train", file_name_type="train_")
    ewald_matrix_features(train_data, -1, data_type="train", file_name_type="train_")

    scan_through_geometry_files_and_extrac_features(test_data, data_type="test", file_name_type="test_")
    ewald_matrix_features(test_data, -1, data_type="test", file_name_type="test_")

    for i in range(len(train_total_number_of_atoms)):

        noa = int(train_total_number_of_atoms[i])

        # ------------------
        # --- Train data ---
        # ------------------

        condition = train_data[:, gfc.LABELS["number_of_total_atoms"]] == noa
        conditioned_data = train_data[ condition ]
        logger.info("number of atoms {0}; data.shape: {1}".format(noa, conditioned_data.shape))

        # hist_data(data[:, -1], text=str(noa))
        file_name_type = "train_" + str(noa)
        local_data_type = "train"
        scan_through_geometry_files_and_extrac_features(conditioned_data,
                                                        data_type=local_data_type,
                                                        file_name_type=file_name_type)

        ewald_matrix_features(conditioned_data,
                              noa,
                              data_type=local_data_type,
                              file_name_type=file_name_type)

        # -----------------
        # --- Test data ---
        # -----------------

        condition = test_data[:, gfc.LABELS["number_of_total_atoms"]] == noa
        conditioned_data = test_data[ condition ]
        logger.info("number of atoms {0}; data.shape: {1}".format(noa, conditioned_data.shape))

        file_name_type = "test_" + str(noa)
        local_data_type = "test"
        scan_through_geometry_files_and_extrac_features(conditioned_data,
                                                        data_type=data_type,
                                                        file_name_type=file_name_type)

        ewald_matrix_features(conditioned_data,
                              noa,
                              data_type=local_data_type,
                              file_name_type=file_name_type)