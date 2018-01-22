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

    additional_feature_list = ["rho_data",
                               #"percentage_atom_data",
                               "unit_cell_data",
                               "nn_bond_parameters_data",
                               # "angles_and_rs_data",
                               "ewald_sum_data",
                               "preliminary_predictions_data"]

    seed = int(random.randint(1, 2**16 - 1))
    colsample_bytree = random.random()
    subsample = random.random()
    model_parameters = {"max_depth": 6,
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

    bg_general_model, _ = sf.get_model_for_noa(40,
                                               additional_feature_list,
                                               model_class=XGBRegressorModel,
                                               model_parameters=model_parameters,
                                               y_type="band_gap")


    # fe_general_model = get_model_for_noa(-1,
    #                                      additional_feature_list,
    #                                      model_class=XGBRegressorModel,
    #                                      model_parameters=xgb_regressor_model_parameters,
    #                                      y_type="formation_energy")

