import logging
import sys
import numpy as np
import random
from matplotlib import pyplot as plt

import global_flags_constanst as gfc
import support_functions as sf

from models import BaseModel
from models import GBRModel
from models import XGBRegressorModel
from models import FeedForwardNeuralNetworkModel

from keras import backend as K

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gfc.LOGGING_LEVEL)




def objective(y_true, y_pred):
    pass


def plot_two_features(data, x_label, y_label):

    x = data[:, gfc.LABELS[x_label]]
    y = data[:, gfc.LABELS[y_label]]

    plt.scatter(x, y, c="g", alpha=0.5, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def recombine_data_shuffle_and_split(ids, x, y_fe, y_bg):
    """
    Once x has been filled with additional features
    we recombine the features with the labes and shuffle them.
    Once they are shuffled we split them back to their
    original form.

    :param ids:
    :param x:
    :param y_fe:
    :param y_bg:
    :return:
    """
    data = np.hstack((ids, x, y_fe, y_bg))
    np.random.shuffle(data)

    _, m = data.shape

    ids = data[:, 0]
    x = data[:, 1:(m-2)]
    y_fe = data[:, m-2].reshape((-1, 1))
    y_bg = data[:, m-1].reshape((-1, 1))

    return ids, x, y_fe, y_bg


if __name__ == "__main__":

    data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    rho_data = np.loadtxt("train_rho_data.csv", delimiter=",", skiprows=0)
    percentage_atom_data = np.loadtxt("train_percentage_atom_data.csv", delimiter=",", skiprows=0)
    unit_cell_data = np.loadtxt("train_unit_cell_data.csv", delimiter=",", skiprows=0)
    nn_bond_parameters_data = np.loadtxt("train_nn_bond_parameters_data.csv", delimiter=",", skiprows=0)
    symmetries_data = np.loadtxt("train_symmetries_data.csv", delimiter=",", skiprows=0)
    angles_and_rs_data = np.loadtxt("train_angles_and_rs_data.csv", delimiter=",", skiprows=0)
    ewald_sum_data = np.loadtxt("train_ewald_sum_data.csv", delimiter=",", skiprows=0)

    logger.info("train_rho_data.shape: {0}".format(rho_data.shape))
    logger.info("train_percentage_atom_data.shape: {0}".format(percentage_atom_data.shape))
    logger.info("train_unit_cell_data.shape: {0}".format(unit_cell_data.shape))
    logger.info("train_nn_bond_parameters_data.shape: {0}".format(nn_bond_parameters_data.shape))
    logger.info("train_symmetries_data.shape: {0}".format(symmetries_data.shape))
    logger.info("train_angles_and_rs_data.shape: {0}".format(angles_and_rs_data.shape))

    test_data = np.loadtxt("test.csv", delimiter=",", skiprows=1)
    test_rho_data = np.loadtxt("test_rho_data.csv", delimiter=",", skiprows=0)
    test_percentage_atom_data = np.loadtxt("test_percentage_atom_data.csv", delimiter=",", skiprows=0)
    test_unit_cell_data = np.loadtxt("test_unit_cell_data.csv", delimiter=",", skiprows=0)
    test_nn_bond_parameters_data = np.loadtxt("test_nn_bond_parameters_data.csv", delimiter=",", skiprows=0)
    test_symmetries_data = np.loadtxt("test_symmetries_data.csv", delimiter=",", skiprows=0)
    test_angles_and_rs_data = np.loadtxt("test_angles_and_rs_data.csv", delimiter=",", skiprows=0)

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

    ids = data[:, 0].reshape((-1, 1))
    x = data[:, 1:(m-2)]

    test_n, test_m = test_data.shape
    test_ids = test_data[:, 0].reshape((-1, 1))
    test_x = test_data[:, 1:]

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

    features = sys.argv[1]

    if features == "rho_data":
        logger.info("Adding rho_data")
        x = np.hstack((x, rho_data[:, 1:]))

        test_x = np.hstack((test_x, test_rho_data[:, 1:]))

    elif features == "rho_percentage_atom_data":
        logger.info("Adding rho_percentage_atom_data")
        x = np.hstack((x, rho_data[:, 1:], percentage_atom_data[:, 1:]))

        test_x = np.hstack((test_x, test_rho_data[:, 1:], test_percentage_atom_data[:, 1:]))

    elif features == "rho_percentage_atom_unit_cell_data":
        logger.info("Adding rho_percentage_atom_unit_cell_data")
        x = np.hstack((x,
                       rho_data[:, 1:],
                       percentage_atom_data[:, 1:],
                       unit_cell_data[:, 1:]))

        test_x = np.hstack((test_x,
                            test_rho_data[:, 1:],
                            test_percentage_atom_data[:, 1:],
                            test_unit_cell_data[:, 1:]))

    elif features == "rho_percentage_atom_unit_cell_nn_bond_parameters_data":
        logger.info("Adding rho_percentage_atom_unit_cell_nn_bond_parameters_data")
        x = np.hstack((x,
                       rho_data[:, 1:],
                       percentage_atom_data[:, 1:],
                       unit_cell_data[:, 1:],
                       nn_bond_parameters_data[:, 1:]))

        test_x = np.hstack((test_x,
                            test_rho_data[:, 1:],
                            test_percentage_atom_data[:, 1:],
                            test_unit_cell_data[:, 1:],
                            test_nn_bond_parameters_data[:, 1:]))

    elif features == "rho_percentage_atom_unit_cell_nn_bond_parameters_symmetries_data":
        logger.info("Adding rho_percentage_atom_unit_cell_nn_bond_parameters_symmetries_data")
        x = np.hstack((x,
                       rho_data[:, 1:],
                       percentage_atom_data[:, 1:],
                       unit_cell_data[:, 1:],
                       nn_bond_parameters_data[:, 1:],
                       symmetries_data[:, 1:]))

        test_x = np.hstack((test_x,
                            test_rho_data[:, 1:],
                            test_percentage_atom_data[:, 1:],
                            test_unit_cell_data[:, 1:],
                            test_nn_bond_parameters_data[:, 1:],
                            test_symmetries_data[:, 1:]))

    elif features == "unit_cell_data":
        logger.info("Adding unit_cell_data")
        x = np.hstack((x, unit_cell_data[:, 1:]))

        test_x = np.hstack((test_x, test_unit_cell_data[:, 1:]))

    elif features == "unit_cell_nn_bond_parameters_data":
        logger.info("Adding unit_cell_nn_bond_parameters_data")
        x = np.hstack((x, unit_cell_data[:, 1:], nn_bond_parameters_data[:, 1:]))

        test_x = np.hstack((test_x, test_unit_cell_data[:, 1:], test_nn_bond_parameters_data[:, 1:]))

    elif features == "unit_cell_nn_bond_parameters_angles_and_rs_data":
        logger.info("Adding unit_cell_nn_bond_parameters_angles_and_rs_data")
        x = np.hstack((x, unit_cell_data[:, 1:], nn_bond_parameters_data[:, 1:], angles_and_rs_data[:, 1:]))

        test_x = np.hstack((test_x,
                            test_unit_cell_data[:, 1:],
                            test_nn_bond_parameters_data[:, 1:],
                            test_angles_and_rs_data[:, 1:]))

    elif features == "unit_cell_nn_bond_parameters_symmetries_data":
        logger.info("Adding unit_cell_nn_bond_parameters_symmetries_data")
        x = np.hstack((x, unit_cell_data[:, 1:],
                       nn_bond_parameters_data[:, 1:],
                       symmetries_data[:, 1:]))

        test_x = np.hstack((test_x,
                            test_unit_cell_data[:, 1:],
                            test_nn_bond_parameters_data[:, 1:],
                            test_symmetries_data[:, 1:]))

    elif features == "nn_bond_parameters_data":
        logger.info("Adding nn_bond_parameters_data")
        x = np.hstack((x, nn_bond_parameters_data[:, 1:]))

        test_x = np.hstack((test_x, test_nn_bond_parameters_data[:, 1:]))

    elif features == "nn_bond_parameters_angles_and_rs_data":
        logger.info("Adding nn_bond_parameters_angles_and_rs_data")
        x = np.hstack((x, nn_bond_parameters_data[:, 1:], angles_and_rs_data[:, 1:]))

        test_x = np.hstack((test_x, test_nn_bond_parameters_data[:, 1:], test_angles_and_rs_data[:, 1:]))

    elif features == "nn_bond_parameters_symmetries_data":
        logger.info("Adding nn_bond_parameters_symmetries_data")
        x = np.hstack((x, nn_bond_parameters_data[:, 1:], symmetries_data[:, 1:]))

        test_x = np.hstack((test_x, test_nn_bond_parameters_data[:, 1:], test_symmetries_data[:, 1:]))

    elif features == "ewald_sum_data":
        logger.info("Adding ewald_sum_data")
        x = ewald_sum_data

    elif features == "unit_cell_nn_bond_parameters_symmetries_ewald_sum_data":
        logger.info("Adding ewald_sum_data")
        x = np.hstack((x, unit_cell_data[:, 1:],
                       nn_bond_parameters_data[:, 1:],
                       symmetries_data[:, 1:],
                       ewald_sum_data[:, 1:]))

        test_x = np.hstack((test_x,
                            test_unit_cell_data[:, 1:],
                            test_nn_bond_parameters_data[:, 1:],
                            test_symmetries_data[:, 1:]))

    elif features == "standard":
        pass
    else:
        sys.exit("features parameter not valid!")

    y_fe = data[:, m-2].reshape((-1, 1))
    y_bg = data[:, m-1].reshape((-1, 1))

    ids, x, y_fe, y_bg = recombine_data_shuffle_and_split(ids, x, y_fe, y_bg)


    _, n_features = x.shape

    logger.info("x: {0}".format(x.shape))
    logger.info("y_fe: {0}".format(y_fe.shape))
    logger.info("y_bg: {0}".format(y_bg.shape))

    y = np.hstack((y_fe, y_bg))
    #y = y_bg

    _, n_output = y.shape

    logger.info("y: {0}".format(y.shape))

    gbrmodel_parameters = {"n_estimators": 100,
                           "learning_rate": 0.1,
                           "max_depth": 4,
                           "random_state": random.randint(1, 2**32 - 1),
                           "verbose": 0,
                           "max_features": "sqrt",
                           "n_features": n_features}

    seed = int(random.randint(1, 2**16 - 1))
    colsample_bytree = random.random()
    subsample = random.random()
    xgb_regressor_model_parameters = {"max_depth": 6,
                                      "learning_rate": 0.1,
                                      "n_estimators": 300,
                                      "silent": True,
                                      "objective": 'reg:linear',
                                      "booster": 'gbtree',
                                      "n_jobs": 1,
                                      "nthread": None,
                                      "gamma": 0.0,
                                      "min_child_weight": 5,
                                      "max_delta_step": 0,
                                      "subsample": subsample,
                                      "colsample_bytree": colsample_bytree,
                                      "colsample_bylevel": 1,
                                      "reg_alpha": 0,
                                      "reg_lambda": 1,
                                      "scale_pos_weight": 1,
                                      "base_score": 0.5,
                                      "random_state": seed + 1,
                                      "seed": seed,
                                      "missing": None,
                                      "n_features": n_features}


    sf.cross_validate(x,
                      y_bg,
                      XGBRegressorModel,
                      model_parameters=xgb_regressor_model_parameters,
                      fraction=0.25)

    # nn_model_parameters = {"n_features": n_features,
    #                        "n_hidden_layers": 2,
    #                        "n_output": n_output,
    #                        "layer_dim": 100,
    #                        "dropout_rate": 0.9,
    #                        "alpha": 0.01,
    #                        "learning_rate": 0.01,
    #                        "loss": "mean_squared_logarithmic_error"}
    #
    # K.get_session()
    # sf.cross_validate(x,
    #                   y,
    #                   FeedForwardNeuralNetworkModel,
    #                   model_parameters=nn_model_parameters,
    #                   fraction=0.25)
    #
    # K.clear_session()

    sys.exit()

    # Band gap model
    bgm = XGBRegressorModel(**xgb_regressor_model_parameters)

    # Formation energy model
    fem = XGBRegressorModel(**xgb_regressor_model_parameters)
    #
    fem.fit(x, y_fe)
    bgm.fit(x, y_bg)

    rmsle_fe = fem.evaluate(x, y_fe)
    rmsle_bg = bgm.evaluate(x, y_bg)

    rmsle = np.mean(rmsle_bg + rmsle_fe)
    print("rmsle_fe: " + str(rmsle_fe))
    print("rmsle_bg: " + str(rmsle_bg))
    print("rmsle: {0}".format(rmsle))

    sf.pipeline_flow(test_ids,
                     test_x,
                     fem,
                     bgm,
                     "temp")