import logging
import os

import numpy as np
import pymatgen
from pymatgen.analysis import ewald



logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Atom:

    # Equality epsilon
    ee = 1e-6

    def __init__(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 t="",
                 c=0):

        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.c = c

    def __eq__(self, other):
        if (abs(self.x - other.x) < Atom.ee and
            abs(self.y - other.y) < Atom.ee and
            abs(self.z - other.z) < Atom.ee and
            self.t == other.t):

            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


    def __hash__(self):
        s = str(self.x) + str(self.y) + str(self.z) + str(self.t)
        return hash(s)


    def __str__(self):

        s = "x: " + str(self.x) + \
            " y: " + str(self.y) + \
            " z: " + str(self.z) + \
            " t: " + str(self.t)

        return s

    def __repr__(self):
        return self.__str__()


def split_data_into_id_x_y(data, data_type="train"):

    if data_type == "train":
        n, m = data.shape
        ids = data[:, 0].reshape(-1, 1)
        x = data[:, 1:(m-2)]
        y_fe = data[:, m-2].reshape(-1, 1)
        y_bg = data[:, m-1].reshape(-1, 1)
    else:
        ids = data[:, 0].reshape(-1, 1)
        x = data[:, 1:]
        y_fe = np.array([])
        y_bg = np.array([])

    return ids, x, y_fe, y_bg


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


def vector_length(vec):
    return np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


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

    ga_mass = 31.0
    al_mass = 13.0
    in_mass = 49.0
    o_mass = 8.0

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

        a = Atom(x, y, z, t, c)
        uc_atoms.append(a)
    logger.info("Geomtery file read.")
    # uc_atoms = UCAtoms(uc_atoms)

    return vectors, uc_atoms


def ewald_matrix_features(data,
                          data_type="train",
                          file_name=""):

    # noa - number of atoms in unit cell
    ids, x, y_fe, y_bg = split_data_into_id_x_y(data)

    n, m = ids.shape
    ewald_sum_data = np.zeros((n, 4))
    for i in range(n):
        c_id = int(ids[i, 0])
        logger.info("c_id: {0}".format(c_id))

        vectors, uc_atoms = read_geometry_file(data_type + "/" + str(c_id) + "/geometry.xyz")
        atom_coords, atom_labels, site_properties = convert_uc_atoms_to_input_for_pymatgen(uc_atoms)

        lv1 = x[c_id - 1, 5]
        lv2 = x[c_id - 1, 6]
        lv3 = x[c_id - 1, 7]

        lv1_c = vector_length(vectors[0])
        lv2_c = vector_length(vectors[1])
        lv3_c = vector_length(vectors[2])

        alpha = x[c_id - 1, 8]
        beta = x[c_id - 1, 9]
        gamma = x[c_id - 1, 10]

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

        ewald_sum_data[i][0] = np.trace(ewald_sum.real_space_energy_matrix)
        ewald_sum_data[i][1] = np.trace(ewald_sum.reciprocal_space_energy_matrix)
        ewald_sum_data[i][2] = np.trace(ewald_sum.total_energy_matrix)
        ewald_sum_data[i][3] = np.sum(ewald_sum.point_energy_matrix)

    # Take only space group and number of total atoms from x.
    ewald_sum_data = np.hstack((x[:, 0:2], ewald_sum_data, y_bg))
    # np.savetxt(file_name_type + "_ewald_sum_data.csv", ewald_sum_data, delimiter=",")
    np.save(file_name, ewald_sum_data)


def extract_feature_by_index_and_value(features, index, value):

    condition = features[:, index] == value
    f = features[condition]

    return f


if __name__ == "__main__":

    data = np.loadtxt("train.csv", delimiter=",", skiprows=1)

    data_type="train"
    file_name = "ewald_sum_data.py"

    features = None
    if os.path.isfile("")
        ewald_matrix_features(data,
                              data_type=data_type,
                              file_name=file_name)
    else:
        features = np.load(file_name)

    # nota - number of total atoms
    nota = [10, 20, 30, 40, 60, 80]
    nota_index = 1
    for i in range(len(nota)):
        f = extract_feature_by_index_and_value(features, nota_index, nota[i])

