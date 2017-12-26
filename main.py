import numpy as np
from matplotlib import pyplot as plt

import support_functions as sf

from models import BaseModel
from models import GBRModel

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

    #vectors, atoms = sf.read_geometry_file("/home/tadek/Coding/Kaggle/Nomad2018/train/1/geometry.xyz")
    #print(vectors)
    #print(atoms)

    #print("n atoms: " + str(len(atoms)))

    ids = data[:, 0]
    x = data[:, 1:12]
    y_fe = data[:, 12]
    y_bg = data[:, 13]

    _, n_features = x.shape
    #bm = BaseModel(n_features=n_features)

    # Band gap model
    bgm = GBRModel(n_features=n_features,
                   verbose=1)

    # Formation energy model
    fem = GBRModel(n_features=n_features,
                   verbose=1)

    fem.fit(x, y_fe)
    bgm.fit(x, y_bg)
    y_fe_pred = fem.predict(x)
    y_bg_pred = bgm.predict(x)

    print("y_fe_pred.shape: {0}".format(y_fe_pred.shape))
    print("y_bg_pred.shape: {0}".format(y_bg_pred.shape))

    rmsle_fe = fem.evaluate(x, y_fe)
    rmsle_bg = bgm.evaluate(x, y_bg)

    rmsle = np.mean(rmsle_bg + rmsle_fe)
    print("rmsle_fe: " + str(rmsle_fe))
    print("rmsle_bg: " + str(rmsle_bg))

    #sf.pipeline_flow(x, bm, "temp")

    test_data = np.loadtxt("test.csv", delimiter=",", skiprows=1)
    test_ids = test_data[:, 0]
    test_x = test_data[:, 1:12]

    sf.pipeline_flow(ids,
                     test_x,
                     fem,
                     bgm,
                     "temp")