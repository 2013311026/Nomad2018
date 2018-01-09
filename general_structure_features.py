import logging
import pymatgen
from pymatgen.analysis import ewald
import numpy as np

import global_flags_constanst as gfc

from geometry_xyz import read_geometry_file
from geometry_xyz import vector_length

from support_functions import split_data_into_id_x_y


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gfc.LOGGING_LEVEL)

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


def ewald_matrix_features(data, data_type="train"):

    ids, x, y_fe, y_bg = split_data_into_id_x_y(data)

    n, m = ids.shape
    ewald_sum_data = np.zeros((n, 4))
    for i in range(n):
        id = int(ids[i, 0])
        print("id: {0}".format(id))

        vectors, uc_atoms = read_geometry_file(data_type + "/" + str(id) + "/geometry.xyz")
        atom_coords, atom_labels, site_properties = convert_uc_atoms_to_input_for_pymatgen(uc_atoms)

        lv1 = x[id - 1, 5]
        lv2 = x[id - 1, 6]
        lv3 = x[id - 1, 7]

        lv1_c = vector_length(vectors[0])
        lv2_c = vector_length(vectors[1])
        lv3_c = vector_length(vectors[2])

        alpha = x[id - 1, 8]
        beta = x[id - 1, 9]
        gamma = x[id - 1, 10]

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

        ewald_sum_data[i][0] = ewald_sum.real_space_energy
        ewald_sum_data[i][1] = ewald_sum.reciprocal_space_energy
        ewald_sum_data[i][2] = ewald_sum.point_energy
        ewald_sum_data[i][3] = ewald_sum.total_energy

    ewald_sum_data = np.hstack((ids, ewald_sum_data))
    np.savetxt(data_type + "_ewald_sum_data.csv", ewald_sum_data, delimiter=",")

data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
ewald_matrix_features(data, data_type="train")

# id = 4
# data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
# vectors, uc_atoms = read_geometry_file("train/" + str(id) + "/geometry.xyz")
# print("Number of atoms in UC: " + str(len(uc_atoms)))
#
# atom_coords, atom_labels, site_properties = convert_uc_atoms_to_input_for_pymatgen(uc_atoms)
#
# lv1 = data[id - 1, 6]
# lv2 = data[id - 1, 7]
# lv3 = data[id - 1, 8]
#
# lv1_c = vector_length(vectors[0])
# lv2_c = vector_length(vectors[1])
# lv3_c = vector_length(vectors[2])
#
# alpha = data[id - 1, 9]
# beta = data[id - 1, 10]
# gamma = data[id - 1, 11]
#
#
# print("lv1: {0}, lv2: {1}, lv3: {2}".format(lv1, lv2, lv3))
# print("lv1: {0}, lv2: {1}, lv3: {2}".format(lv1_c, lv2_c, lv3_c))
# print("alpha: {0}, beta: {1}, gamma: {2}".format(alpha, beta, gamma))
# #structure = pymatgen.Structure.from_file("geometry.xyz")
#
# lattice = pymatgen.Lattice.from_parameters(a=lv1,
#                                            b=lv2,
#                                            c=lv3,
#                                            alpha=alpha,
#                                            beta=beta,
#                                            gamma=gamma)
#
# structure = pymatgen.Structure(lattice, atom_labels, atom_coords, site_properties=site_properties)
#
# ewald_sum = ewald.EwaldSummation(structure)
#
# #print(structure)
# print(ewald_sum)
# print(ewald_sum.real_space_energy_matrix.shape)
# print(ewald_sum.get_site_energy(1))