import numpy as np
from matplotlib import pyplot as plt

import support_functions as sf

labels = {}
labels["id"] = 0
labels["spacegroup"] = 1
labels["number_of_total_atoms"] = 2
labels["percent_atom_al"] = 3
labels["percent_atom_ga"] = 4
labels["percent_atom_in"] = 5
labels["lattice_vector_1_ang"] = 6
labels["lattice_vector_2_ang"] = 7
labels["lattice_vector_3_ang"] = 8
labels["lattice_angle_alpha_degree"] = 9
labels["lattice_angle_beta_degree"] = 10
labels["lattice_angle_gamma_degree"] = 11
labels["formation_energy_ev_natom"] = 12
labels["bandgap_energy_ev"] = 13

def plot_two_features(data, x_label, y_label):

    x = data[:, labels[x_label]]
    y = data[:, labels[y_label]]

    plt.scatter(x, y, c="g", alpha=0.5, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

if __name__ == "__main__":

    file_name = "train.csv"
    data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    # for key, _ in labels.items():
    #     plot_two_features(data, key, "formation_energy_ev_natom")
    #     plot_two_features(data, key, "bandgap_energy_ev")

    vectors, atoms = sf.read_geometry_file("/home/tadek/Coding/Kaggle/Nomad2018/train/1/geometry.xyz")
    print(vectors)
    print(atoms)

    print("n atoms: " + str(len(atoms)))
