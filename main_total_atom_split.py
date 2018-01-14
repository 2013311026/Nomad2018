import logging
import glob
import numpy as np
import random

import matplotlib.pyplot as plt

import global_flags_constanst as gfc
import support_functions as sf

from models import BaseModel
from models import PolynomialModel
from models import XGBRegressorModel

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gfc.LOGGING_LEVEL)


def prepare_data_for_matrix_trace_based_model(noa,
                                              data_type="train",
                                              matrix_type="real_energy",
                                              y_type="band_gap"):

    # Load and prepare features
    data = np.loadtxt(data_type + ".csv", delimiter=",", skiprows=1)

    condition = data[:, gfc.NUMBER_OF_TOTAL_ATOMS] == noa
    noa_data = data[condition]
    noa_data = noa_data[noa_data[:, 0].argsort()]

    matrix_files = glob.glob(data_type + "_" + str(noa) + "*" + str(matrix_type) + "*matrix*npy")
    file_name = matrix_files[0]
    matrix_data = np.load(file_name)

    assert np.array_equal(noa_data[:, 0], matrix_data[:, 0]), "Ids do not agree!"

    noa_matrix = matrix_data[:, 1:]
    ids, x, y_fe, y_bg = sf.split_data_into_id_x_y(noa_data, data_type=data_type)

    n, m = noa_matrix.shape

    matrix_traces = np.zeros((n, 1))
    for i in range(n):
        matrix_traces[i] = np.trace(noa_matrix[i, :].reshape(noa, noa))

    # Features ready for training
    x = matrix_traces

    if y_type == "band_gap":
        y = y_bg
    elif y_type == "formation_energy":
        y = y_fe
    else:
        # If you reached this point then something is wrong.
        # Most probably the provided y_type does not match
        # "band_gap" nor does it match "formation_energy".
        assert False, "y cannot be None!"

    return x, y, ids

def get_matrix_trace_based_model_for_noa(noa,
                                         model_class,
                                         model_parameters,
                                         plot_model=False,
                                         y_type="band_gap",
                                         matrix_type="real_energy"):
    logger.info("Get matrix trace based model for NOA = {0}".format(noa))

    x, y, _ = prepare_data_for_matrix_trace_based_model(noa,
                                                        matrix_type=matrix_type,
                                                        y_type=y_type)
    _, n_features = x.shape

    model_parameters["n_features"] = n_features
    sf.one_left_cross_validation(x,
                                 y,
                                 model_class=model_class,
                                 model_parameters=model_parameters)

    trained_model = model_class(**model_parameters)
    trained_model.fit(x, y)

    xp = np.linspace(np.min(x), np.max(x), 1000)

    if plot_model == True:
        plt.figure()
        plt.plot(x.ravel(), y.ravel(),'.')
        plt.plot(xp, trained_model.predict(xp), '--')
        plt.title("noa: {0}, {1}".format(noa, matrix_type))
        #plt.savefig("noa: {0}, {1}.eps".format(noa, file_name.replace(".npy","")))
        plt.show()

    return trained_model


def prepare_data_for_model(noa,
                           additional_feature_list,
                           data_type="train",
                           y_type="band_gap"):

    # Prepare data for non matrix trace based models.
    data = np.loadtxt(data_type + ".csv", delimiter=",", skiprows=1)

    # If noa == -1 ignore the noa split.
    noa_data = None
    if noa == -1:
        noa_data = data
    else:
        condition = data[:, gfc.NUMBER_OF_TOTAL_ATOMS] == noa
        noa_data = data[condition]
        noa_data = noa_data[noa_data[:, 0].argsort()]

    logger.info("noa_data.shape {0}".format(noa_data.shape))

    ids, x, y_fe, y_bg = sf.split_data_into_id_x_y(noa_data, data_type=data_type)

    logger.info("x.shape: {0}".format(x.shape))
    logger.info("Adding additional features to data.")
    naf = len(additional_feature_list)
    for i in range(naf):
        logger.info("Adding {0} features...".format(additional_feature_list[i]))

        file_name = None
        if noa == -1:
            file_name = data_type + "_" + additional_feature_list[i] + ".npy"
        else:
            file_name = data_type + "_" + str(noa) + "_" + additional_feature_list[i] + ".npy"

        logger.info("Aditional features file: {0}".format(file_name))
        additional_feature = np.load(file_name)
        logger.info("additional_feature.shape: {0}".format(additional_feature.shape))

        x = np.hstack((x, additional_feature[:, 1:]))

    logger.info("x.shape: {0}".format(x.shape))

    y = None
    if y_type == "band_gap":
        y = y_bg
    elif y_type == "formation_energy":
        y = y_fe
    else:
        # If you reached this point then something is wrong.
        # Most probably the provided y_type does not match
        # "band_gap" nor does it match "formation_energy".
        assert False, "y cannot be None!"

    return x, y, ids


def get_model_for_noa(noa,
                      additional_feature_list,
                      model_class,
                      model_parameters,
                      y_type="band_gap"):

    logger.info("Get model for NOA = {0}".format(noa))

    x, y, _ = prepare_data_for_model(noa,
                                     additional_feature_list,
                                     data_type="train",
                                     y_type=y_type)

    _, n_features = x.shape
    model_parameters["n_features"] = n_features
    sf.cross_validate(x,
                      y,
                      model_class,
                      model_parameters=model_parameters,
                      fraction=0.25)

    trained_model = model_class(**model_parameters)
    trained_model.fit(x, y)

    return trained_model


if __name__ == "__main__":

    # Separate models will be fit for the
    # specimen with the following number of atoms.
    # number_of_total_atoms: rank
    # noa = 40 and noa = 80 not included
    # Simple models do not work for them.

    result = np.zeros((601, 3))


    noa_ranks = {10: [2, "real_energy"],
                 20: [2, "reciprocal_energy"],
                 30: [3, "total_energy"],
                 60: [2, "total_energy"]}
    noa_bg_matrix_trace_models = {}

    model = PolynomialModel
    for noa, rank_matrix_type in sorted(noa_ranks.items(), key=lambda t: t[0]):
        trained_model = get_matrix_trace_based_model_for_noa(noa,
                                                             model,
                                                             model_parameters={"rank": rank_matrix_type[0]},
                                                             plot_model=True,
                                                             matrix_type=rank_matrix_type[1])

        noa_bg_matrix_trace_models[noa] = trained_model

        x, y, ids  = prepare_data_for_matrix_trace_based_model(noa,
                                                               data_type="test",
                                                               matrix_type=rank_matrix_type[1])

        n, m = x.shape
        for i in range(n):
            id = int(ids[i])
            result[id][0] = id
            result[id][2] = trained_model.predict(x[i, 0])
            #print("f: {0}".format(result[id]))


    print(noa_bg_matrix_trace_models)

    additional_feature_list = [#"rho_data",
                               #"percentage_atom_data",
                               #"unit_cell_data",
                               #"nn_bond_parameters_data",
                               #"angles_and_rs_data",
                               "ewald_sum_data"]

    seed = int(random.randint(1, 2**16 - 1))
    colsample_bytree = random.random()
    subsample = random.random()
    xgb_regressor_model_parameters = {"max_depth": 5,
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
                                      "missing": None}

    noa_bg_list_for_general_models = [40, 80]
    noa_bg_general_models = {}


    for noa in noa_bg_list_for_general_models:
        trained_model = get_model_for_noa(noa,
                                          additional_feature_list,
                                          model_class=XGBRegressorModel,
                                          model_parameters=xgb_regressor_model_parameters,
                                          y_type="band_gap")

        x, y, ids = prepare_data_for_model(noa,
                                           additional_feature_list,
                                           data_type="test",
                                           y_type="band_gap")

        n, m = x.shape
        for i in range(n):
            id = int(ids[i])
            result[id][0] = id
            result[id][2] = trained_model.predict(x[i][:].reshape(1, -1))
            #print("f: {0}".format(result[id]))





    fe_general_model = get_model_for_noa(-1,
                                         additional_feature_list,
                                         model_class=XGBRegressorModel,
                                         model_parameters=xgb_regressor_model_parameters,
                                         y_type="formation_energy")

    x, y, ids = prepare_data_for_model(-1,
                                       additional_feature_list,
                                       data_type="test",
                                       y_type="formation_energy")

    n, m = x.shape
    for i in range(n):
        id = int(ids[i])
        result[id][0] = id
        result[id][1] = fe_general_model.predict(x[i][:].reshape(1, -1))
        #print("f: {0}".format(result[id]))


    file = open("temp", "w")
    file.write("id,formation_energy_ev_natom,bandgap_energy_ev\n")

    for i in range(1, len(result)):
        id = int(result[i][0])
        fe = result[i][1]
        bg = result[i][2]

        if fe < 0.0:
            fe = 0.0

        if bg < 0.0:
            bg = 0.0

        file.write("{0},{1},{2}\n".format(id, fe, bg))

    file.close()


    # noa = 80
    #
    # # Load and prepare features
    # train_data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    #
    # condition = train_data[:, gfc.NUMBER_OF_TOTAL_ATOMS] == noa
    # train_noa_data = train_data[condition]
    # train_noa_data = train_noa_data[train_noa_data[:,0].argsort()]
    #
    # print(train_noa_data[:, 0])
    #
    # matrix_files = glob.glob("train_" + str(noa) + "*matrix*npy")
    #
    # matrix_data = np.load(matrix_files[0])
    #
    # logger.info("matrix_data.shape: {0}".format(matrix_data.shape))
    # logger.info(matrix_data[:, 0])
    #
    # assert np.array_equal(train_noa_data[:, 0], matrix_data[:, 0]), "Ids do not agree!"
    #
    # noa_matrix = matrix_data[:, 1:]
    # ids, x, y_fe, y_bg = sf.split_data_into_id_x_y(train_noa_data, data_type="train")
    #
    # n, m = noa_matrix.shape
    #
    # # for i in range(n):
    # #     fig = plt.figure()
    # #     text=str(y_bg[i]) + " eV"
    # #     #sf.plot_matrix(noa_matrix[i, :].reshape(noa, noa), text=text)
    # #
    # #     ax = fig.add_subplot(111)
    # #     cax = ax.matshow(noa_matrix[i, :].reshape(noa, noa))
    # #     fig.colorbar(cax)
    # #     plt.title(text)
    # #     plt.draw()
    # # plt.show()
    #
    # matrix_traces = np.zeros((n, 1))
    # for i in range(n):
    #     matrix_traces[i] = np.trace(noa_matrix[i, :].reshape(noa, noa))
    #
    # # logger.info("matrix_traces: {0}".format(matrix_traces.ravel()))
    # # logger.info("y_bg: {0}".format(y_bg.ravel()))
    # #
    # plt.figure()
    #
    # plt.scatter(matrix_traces.ravel(), y_bg.ravel())
    #
    # plt.show()
    #
    # z = np.polyfit(matrix_traces.ravel(), y_bg.ravel(), 2)
    # p = np.poly1d(z)
    #
    # xp = np.linspace(np.min(matrix_traces), np.max(matrix_traces), 1000)
    #
    # plt.figure()
    #
    # plt.plot(matrix_traces.ravel(), y_bg.ravel(),'.')
    # plt.plot(xp, p(xp), '--')
    #
    # plt.show()
    # #
    #
    # # Features ready for training
    # #x = noa_matrix
    # x = matrix_traces
    # y = y_bg
    #
    # _, n_features = x.shape
    #
    # # seed = int(random.randint(1, 2**16 - 1))
    # # colsample_bytree = random.random()
    # # subsample = random.random()
    # # xgb_regressor_model_parameters = {"max_depth": 30,
    # #                                   "learning_rate": 0.1,
    # #                                   "n_estimators": 600,
    # #                                   "silent": True,
    # #                                   "objective": 'reg:linear',
    # #                                   "booster": 'gbtree',
    # #                                   "n_jobs": 1,
    # #                                   "nthread": None,
    # #                                   "gamma": 1.0,
    # #                                   "min_child_weight": 5,
    # #                                   "max_delta_step": 0,
    # #                                   "subsample": subsample,
    # #                                   "colsample_bytree": colsample_bytree,
    # #                                   "colsample_bylevel": 1,
    # #                                   "reg_alpha": 0,
    # #                                   "reg_lambda": 1,
    # #                                   "scale_pos_weight": 1,
    # #                                   "base_score": 0.5,
    # #                                   "random_state": seed + 1,
    # #                                   "seed": seed,
    # #                                   "missing": None,
    # #                                   "n_features": n_features}
    # #
    # # sf.one_left_cross_validation(x,
    # #                              y,
    # #                              XGBRegressorModel,
    # #                              model_parameters=xgb_regressor_model_parameters,
    # #                              fraction=0.1)
    #
    #
    # # sf.one_left_cross_validation(x,
    # #                              y,
    # #                              BaseModel,
    # #                              model_parameters={"n_features": n_features},
    # #                              fraction=0.1)
    #
    # sf.one_left_cross_validation(x,
    #                              y,
    #                              PolynomialModel,
    #                              model_parameters={"rank": 2,
    #                                                "n_features": n_features},
    #                              fraction=0.1)