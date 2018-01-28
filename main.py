import logging
import glob
import numpy as np
import random
import math
import time
import operator

import matplotlib.pyplot as plt

import global_flags_constanst as gfc
import support_functions as sf

from models import BaseModel
from models import PolynomialModel
from models import XGBRegressorModel
from models import RidgeRegressionModel
from models import KernelRidgeRegressionModel


logger = logging.getLogger(__name__)

file_handler = logging.FileHandler("logs.log")
file_handler.setLevel(gfc.LOGGING_LEVEL)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(handler)
logger.addHandler(file_handler)
logger.setLevel(gfc.LOGGING_LEVEL)



if __name__ == "__main__":

    # Separate models will be fit for the
    # specimen with the following number of atoms.
    # number_of_total_atoms: rank
    # noa = 40 and noa = 80 not included
    # Simple models do not work for them.

    additional_feature_list_bg = ["rho_data",
                               #"percentage_atom_data",
                               "unit_cell_data",
                               "nn_bond_parameters_data",
                               # "angles_and_rs_data",
                               "ewald_sum_data",
                               "preliminary_predictions_data"]

    seed = int(random.randint(1, 2**16 - 1))
    colsample_bytree = random.random()
    subsample = random.random()
    model_parameters = {"max_depth": 20,
                        "learning_rate": 0.1,
                        "n_estimators": 1000,
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

    model_parameters = {'reg_alpha': 0.5318365901304094, 'silent': True, 'base_score': 0.5, 'n_estimators': 1279,
     'objective': 'reg:linear', 'colsample_bylevel': 1, 'max_depth': 5, 'n_jobs': 1, 'missing': None,
     'learning_rate': 0.1, 'seed': seed + 1, 'booster': 'dart', 'max_delta_step': 9, 'min_child_weight': 7,
     'random_state': seed, 'reg_lambda': 0.23895903323721313, 'nthread': None,
     'scale_pos_weight': 0.9752036096475583, 'n_features': 127, 'gamma': 0.4666787074687966,
     'colsample_bytree': 0.5976718791124155, 'subsample': 0.7172703652129258}

    # model_parameters = {"alpha": 0.5,
    #                     "kernel": "chi2",
    #                       "gamma": 0.1,
    #                       "degree": 10,
    #                       "coef0": 1,
    #                       "n_features": None,
    #                       "max_features": None,
    #                       "validation_data": None}

    bg_general_model, _ = sf.get_model_for_noa(-1,
                                               additional_feature_list_bg,
                                               model_class=XGBRegressorModel,
                                               model_parameters=model_parameters,
                                               y_type="band_gap")

    # x, y, ids = sf.prepare_data_for_model(-1,
    #                                       additional_feature_list,
    #                                       data_type="train",
    #                                       y_type="band_gap")


    # x, y = sf.feature_split(x,
    #                         y,
    #                         feature_index=1,
    #                         feature_value=20,
    #                         op=operator.eq)

    # x, y = sf.feature_split(x,
    #                         y,
    #                         feature_index=0,
    #                         feature_value=33,
    #                         op=operator.eq)

    # y_pred = bg_general_model.predict(x)
    #
    #
    # plt.figure()
    # plt.scatter(x[:, -5], y,
    #             label="True")
    # plt.scatter(x[:, -5], y_pred,
    #             label="Pred")
    # plt.show()
    #
    #
    # x_45 = np.linspace(0, 5, 1000)
    # plt.figure()
    # plt.scatter(y, y_pred,
    #             label="True")
    # plt.plot(x_45, x_45, "--")
    # plt.show()

    additional_feature_list_fe = [#"rho_data",
                               #"percentage_atom_data",
                               #"unit_cell_data",
                               #"nn_bond_parameters_data",
                               #"angles_and_rs_data",
                               "ewald_sum_data"]

    # model_parameters = {'colsample_bylevel': 1, 'booster': 'gblinear', 'gamma': 0.30673961761670365,
    #  'silent': True, 'learning_rate': 0.1, 'n_features': 127, 'random_state': seed + 2,
    #  'nthread': None, 'colsample_bytree': 0.7532310254653647, 'base_score': 0.5,
    #  'subsample': 0.017351618873915342, 'reg_lambda': 0.7612174834705407, 'missing': None,
    #  'n_jobs': 1, 'min_child_weight': 1, 'max_delta_step': 0, 'seed': seed + 23,
    #  'scale_pos_weight': 0.769943913351496, 'objective': 'reg:linear', 'n_estimators': 1845,
    #  'reg_alpha': 0.4682791398199968, 'max_depth': 7}

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
                                      "random_state": seed + 12,
                                      "seed": seed + 21,
                                      "missing": None}

    fe_general_model, _ = sf.get_model_for_noa(-1,
                                               additional_feature_list_fe,
                                               model_class=XGBRegressorModel,
                                               model_parameters=xgb_regressor_model_parameters,
                                               y_type="formation_energy")

    x_fe, _, ids = sf.prepare_data_for_model(-1,
                                             additional_feature_list_fe,
                                             data_type="test",
                                             y_type="band_gap")

    x_bg, _, ids = sf.prepare_data_for_model(-1,
                                             additional_feature_list_bg,
                                             data_type="test",
                                             y_type="band_gap")

    sf.pipeline_flow_split(ids,
                           x_fe,
                           x_bg,
                           fe_general_model,
                           bg_general_model,
                           "temp2")


    # logger.info("-------------------------------")
    # logger.info("---------Hyper tunning---------")
    # logger.info("-------------------------------")
    #
    #
    # additional_feature_list = [#"rho_data",
    #                            #"percentage_atom_data",
    #                            #"unit_cell_data",
    #                            #"nn_bond_parameters_data",
    #                            # "angles_and_rs_data",
    #                            "ewald_sum_data"]
    #                            #"preliminary_predictions_data"]
    #
    #
    # minimal_avg = math.inf
    # minimal_params = {}
    # for i in range(100):
    #     start = time.time()
    #     logger.info("Hyper tunning i: {0}".format(i))
    #
    #     max_depth = random.randint(2, 40)
    #
    #     lr = np.array([0.001, 0.005, 0.01, 0.03, 0.05, 0.1])
    #     learning_rate = np.random.choice(lr, 1)[0]
    #
    #     n_estimators = random.randint(10, 2000)
    #
    #     b = np.array(["gbtree", "gblinear", "dart"])
    #     booster = np.random.choice(b, 1)[0]
    #
    #     gamma = random.uniform(0.0, 2.0)
    #
    #     min_child_weight = random.randint(1, 10)
    #     max_delta_step = random.randint(0, 10)
    #
    #     reg_alpha = random.uniform(0, 2)
    #     reg_lambda = random.uniform(0, 2)
    #
    #     scale_pos_weight = random.uniform(0, 2)
    #
    #     random_state = int(random.randint(1, 2 ** 16 - 1))
    #     seed = int(random.randint(1, 2 ** 16 - 1))
    #
    #     base_score = random.uniform(0.0, 1.0)
    #
    #     model_parameters = {"max_depth": max_depth,
    #                         "learning_rate": learning_rate,
    #                         "n_estimators": n_estimators,
    #                         "silent": True,
    #                         "objective": 'reg:linear',
    #                         "booster": booster,
    #                         "n_jobs": 1,
    #                         "nthread": None,
    #                         "gamma": gamma,
    #                         "min_child_weight": min_child_weight,
    #                         "max_delta_step": max_delta_step,
    #                         "subsample": subsample,
    #                         "colsample_bytree": colsample_bytree,
    #                         "colsample_bylevel": 1,
    #                         "reg_alpha": reg_alpha,
    #                         "reg_lambda": reg_lambda,
    #                         "scale_pos_weight": scale_pos_weight,
    #                         "base_score": 0.5,
    #                         "random_state": random_state,
    #                         "seed": seed,
    #                         "missing": None}
    #
    #     logger.info("--- Parameters used for model ---")
    #
    #     for key, val in sorted(model_parameters.items(), key=lambda t: t[0]):
    #         logger.info("{0}: {1}".format(key, val))
    #
    #     logger.info("--- Model selection ---")
    #     bg_general_model, valid_avg = sf.get_model_for_noa(-1,
    #                                                        additional_feature_list,
    #                                                        model_class=XGBRegressorModel,
    #                                                        model_parameters=model_parameters,
    #                                                        y_type="formation_energy")
    #
    #     if valid_avg < minimal_avg:
    #         minimal_avg = valid_avg
    #         minimal_params = model_parameters
    #
    #     stop = time.time()
    #
    #     logger.info("Times for one iteration: {0}".format(stop - start))
    #
    # logger.info(minimal_avg)
    # logger.info(minimal_params)



