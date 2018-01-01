import logging
import numpy as np
import random
from matplotlib import pyplot as plt

import global_flags as gf
import support_functions as sf

from models import BaseModel
from models import GBRModel

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gf.LOGGING_LEVEL)


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

    data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    rho_data = np.loadtxt("rho_data.csv", delimiter=",", skiprows=0)
    percentage_atom_data = np.loadtxt("percentage_atom_data.csv", delimiter=",", skiprows=0)
    unit_cell_data = np.loadtxt("unit_cell_data.csv", delimiter=",", skiprows=0)

    logger.info("rho_data.shape: {0}".format(rho_data.shape))
    logger.info("percentage_atom_data.shape: {0}".format(percentage_atom_data.shape))
    logger.info("unit_cell_data.shape: {0}".format(unit_cell_data.shape))

    # for key, _ in labels.items():
    #     plot_two_features(data, key, "formation_energy_ev_natom")
    #     plot_two_features(data, key, "bandgap_energy_ev")

    #vectors, atoms = sf.read_geometry_file("/home/tadek/Coding/Kaggle/Nomad2018/train/1/geometry.xyz")
    #print(vectors)
    #print(atoms)

    #print("n atoms: " + str(len(atoms)))

    n, m = data.shape
    # ids = data[:, 0]
    # x = data[:, 1:12]
    # y_fe = data[:, 12]
    # y_bg = data[:, 13]

    ids = data[:, 0]
    x = data[:, 1:(m-2)]

    # Create additional non geometry features.
    # percent_atom_o = sf.get_percentage_of_o_atoms(data[:, labels["percent_atom_al"]],
    #                                               data[:, labels["percent_atom_ga"]],
    #                                               data[:, labels["percent_atom_in"]])
    # for i in range(10):
    #     logger.info("{0:.6f}, {1:.6f}, {2:.6f}, {3:.6f}".format(percent_atom_o[i],
    #                                                             data[i, labels["percent_atom_al"]],
    #                                                             data[i, labels["percent_atom_ga"]],
    #                                                             data[i, labels["percent_atom_in"]]))
    #     logger.info("sum: {0}".format(percent_atom_o[i] +
    #                                   data[i, labels["percent_atom_al"]] +
    #                                   data[i, labels["percent_atom_ga"]] +
    #                                   data[i, labels["percent_atom_in"]]))

    # = np.hstack((x, rho_data[:, 1:]))
    #x = np.hstack((x, rho_data[:, 1:], percentage_atom_data[:, 1:]))
    #x = np.hstack((x, rho_data[:, 1:], percentage_atom_data[:, 1:], unit_cell_data[:, 1:]))
    #x = np.hstack((x, unit_cell_data))
    y_fe = data[:, m-2].reshape((-1, 1))
    y_bg = data[:, m-1].reshape((-1, 1))

    _, n_features = x.shape

    logger.info("x: {0}".format(x.shape))
    logger.debug("y_fe: {0}".format(y_fe.shape))
    logger.debug("y_bg: {0}".format(y_bg.shape))

    gbrmodel_parameters = {"n_estimators": 100,
                           "learning_rate": 0.1,
                           "max_depth": 4,
                           "random_state": random.randint(1, 2**32 - 1),
                           "verbose": 0,
                           "max_features": "sqrt",
                           "n_features": n_features}

    sf.cross_validate(x,
                      y_bg,
                      GBRModel,
                      model_parameters=gbrmodel_parameters,
                      fraction=0.1)

    # print("ids.shape: " + str(ids.shape))
    # print("x.shape: " + str(x.shape))
    # print("y_fe.shape: " + str(y_fe.shape))
    # print("y_bg.shape: " + str(y_bg.shape))
    #
    # _, n_features = x.shape
    # #bm = BaseModel(n_features=n_features)
    #
    # # Band gap model
    # bgm = GBRModel(n_features=n_features,
    #                verbose=0)
    #
    # # Formation energy model
    # fem = GBRModel(n_features=n_features,
    #                verbose=0)
    #
    # fem.fit(x, y_fe)
    # bgm.fit(x, y_bg)
    # y_fe_pred = fem.predict(x)
    # y_bg_pred = bgm.predict(x)
    #
    # print("y_fe_pred.shape: {0}".format(y_fe_pred.shape))
    # print("y_bg_pred.shape: {0}".format(y_bg_pred.shape))
    #
    # rmsle_fe = fem.evaluate(x, y_fe)
    # rmsle_bg = bgm.evaluate(x, y_bg)
    #
    # for i in range(20):
    #     print()
    #     print("y_fe: {0:.9f} y_fe_pred: {1:.9f}".format(y_fe[i], y_fe_pred[i][0]))
    #     print("y_bg: {0:.9f} y_bg_pred: {1:.9f}".format(y_bg[i], y_bg_pred[i][0]))
    #
    # rmsle = np.mean(rmsle_bg + rmsle_fe)
    # print("rmsle_fe: " + str(rmsle_fe))
    # print("rmsle_bg: " + str(rmsle_bg))
    #
    # #sf.pipeline_flow(x, bm, "temp")
    #
    # test_data = np.loadtxt("test.csv", delimiter=",", skiprows=1)
    # test_ids = test_data[:, 0]
    # test_x = test_data[:, 1:12]

    # sf.pipeline_flow(ids,
    #                  test_x,
    #                  fem,
    #                  bgm,
    #                  "temp")