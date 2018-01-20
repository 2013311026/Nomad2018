import logging
import glob
import numpy as np
import random
import math
import time

import matplotlib.pyplot as plt

import global_flags_constanst as gfc
import support_functions as sf

from models import BaseModel
from models import PolynomialModel
from models import XGBRegressorModel
from models import RidgeRegressionModel
from models import KernelRidgeRegressionModel

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

    print(matrix_files)
    print(noa_data[:, 0])
    print(matrix_data[:, 0])

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

    logger.debug("x.shape: {0}".format(x.shape))
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
    valid_avg = sf.cross_validate(x,
                                  y,
                                  model_class,
                                  model_parameters=model_parameters,
                                  fraction=0.25)

    trained_model = model_class(**model_parameters)
    if valid_avg != math.inf:
        trained_model.fit(x, y)

    return trained_model, valid_avg


if __name__ == "__main__":

    # Separate models will be fit for the
    # specimen with the following number of atoms.
    # number_of_total_atoms: rank
    # noa = 40 and noa = 80 not included
    # Simple models do not work for them.

    additional_feature_list = ["rho_data",
                               #"percentage_atom_data",
                               "unit_cell_data",
                               "nn_bond_parameters_data",
                               "angles_and_rs_data",
                               "ewald_sum_data",
                               "preliminary_predictions_data"]

    seed = int(random.randint(1, 2**16 - 1))
    colsample_bytree = random.random()
    subsample = random.random()
    model_parameters = {"max_depth": 8,
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

    # model_parameters = {"alpha": 0.5,
    #                     "kernel": "chi2",
    #                       "gamma": 0.1,
    #                       "degree": 10,
    #                       "coef0": 1,
    #                       "n_features": None,
    #                       "max_features": None,
    #                       "validation_data": None}

    bg_general_model, _ = get_model_for_noa(-1,
                                            additional_feature_list,
                                            model_class=XGBRegressorModel,
                                            model_parameters=model_parameters,
                                            y_type="band_gap")


    # fe_general_model = get_model_for_noa(-1,
    #                                      additional_feature_list,
    #                                      model_class=XGBRegressorModel,
    #                                      model_parameters=xgb_regressor_model_parameters,
    #                                      y_type="formation_energy")


    logger.info("-------------------------------")
    logger.info("---------Hyper tunning---------")
    logger.info("-------------------------------")

    minimal_avg = math.inf
    minimal_params = {}
    for i in range(1000):
        start = time.time()

        max_depth = random.randint(2, 40)

        lr = np.array([0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.5])
        learning_rate = np.random.choice(lr, 1)[0]

        n_estimators = random.randint(10, 1000)

        b = np.array(["gbtree", "gblinear", "dart"])
        booster = np.random.choice(b, 1)[0]

        gamma = random.uniform(0, 2)

        min_child_weight = random.randint(1, 10)
        max_delta_step = random.randint(0, 10)

        reg_alpha = random.uniform(0, 2)
        reg_lambda = random.uniform(0, 2)

        scale_pos_weight = random.uniform(0, 2)

        random_state = int(random.randint(1, 2 ** 16 - 1))
        seed = int(random.randint(1, 2 ** 16 - 1))

        model_parameters = {"max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                            "silent": True,
                            "objective": 'reg:linear',
                            "booster": booster,
                            "n_jobs": 1,
                            "nthread": None,
                            "gamma": gamma,
                            "min_child_weight": min_child_weight,
                            "max_delta_step": max_delta_step,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                            "colsample_bylevel": 1,
                            "reg_alpha": reg_alpha,
                            "reg_lambda": reg_lambda,
                            "scale_pos_weight": scale_pos_weight,
                            "base_score": 0.5,
                            "random_state": random_state,
                            "seed": seed,
                            "missing": None}

        logger.info("--- Parameters used for model ---")

        for key, val in sorted(model_parameters.items(), key=lambda t: t[0]):
            logger.info("{0}: {1}".format(key, val))

        logger.info("--- Model selection ---")
        bg_general_model, valid_avg = get_model_for_noa(-1,
                                                        additional_feature_list,
                                                        model_class=XGBRegressorModel,
                                                        model_parameters=model_parameters,
                                                        y_type="band_gap")

        if valid_avg < minimal_avg:
            minimal_avg = valid_avg
            minimal_params = model_parameters

        stop = time.time()

        logger.info("Times for one iteration: {0}".format(stop - start))

    logger.info(minimal_avg)
    logger.info(minimal_params)