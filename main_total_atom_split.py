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
from models import RidgeRegressionModel
from models import KernelRidgeRegressionModel

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gfc.LOGGING_LEVEL)



if __name__ == "__main__":

    # Separate models will be fit for the
    # specimen with the following number of atoms.
    # number_of_total_atoms: rank
    # noa = 40 and noa = 80 not included
    # Simple models do not work for them.

    result = np.zeros((601, 3))
    preliminary_predictions = np.zeros((2401, 2))

    for i in range(len(preliminary_predictions)):
        preliminary_predictions[i, 0] = i


    noa_ranks = {10: [2, "real_energy"],
                 20: [2, "reciprocal_energy"],
                 30: [3, "total_energy"],
                 40: [2, "point_energy"],
                 60: [2, "total_energy"],
                 80: [2, "point_energy"]}
    noa_bg_matrix_trace_models = {}

    model = PolynomialModel
    model_parameters = {"alpha": 0.5,
                        "kernel": "chi2",
                        "gamma": 0.1,
                        "degree": 3,
                        "coef0": 1,
                        "n_features": None,
                        "max_features": None,
                        "validation_data": None}

    for noa, rank_matrix_type in sorted(noa_ranks.items(), key=lambda t: t[0]):
        trained_model = sf.get_matrix_trace_based_model_for_noa(noa,
                                                                model,
                                                                model_parameters={"rank": rank_matrix_type[0]},
                                                                #model_parameters={"alpha": 0.5},
                                                                #model_parameters=model_parameters,
                                                                plot_model=True,
                                                                matrix_type=rank_matrix_type[1])

        train_x, train_y, train_ids = sf.prepare_data_for_matrix_trace_based_model(noa,
                                                                                   data_type="train",
                                                                                   matrix_type=rank_matrix_type[1])

        n, m = train_x.shape
        for i in range(n):
            id = int(train_ids[i])
            preliminary_predictions[id][0] = id

            y_prediction = trained_model.predict(train_x[i, 0])
            preliminary_predictions[id][1] = y_prediction

        noa_bg_matrix_trace_models[noa] = trained_model

        x, y, ids = sf.prepare_data_for_matrix_trace_based_model(noa,
                                                                 data_type="test",
                                                                 matrix_type=rank_matrix_type[1])


        y = np.zeros(x.shape)
        n, m = x.shape
        for i in range(n):
            id = int(ids[i])
            result[id][0] = id

            y_prediction = trained_model.predict(x[i, 0])
            y[i] = y_prediction
            result[id][2] = y_prediction
            #print("f: {0}".format(result[id]))


        np.save("for_plot_test.npy", np.hstack((x, y)))
        # input("Press Enter to continue...")

    np.savetxt("train_preliminary_predictions_data.csv", preliminary_predictions[1:, :], delimiter=",")
    np.save("train_preliminary_predictions_data.npy", preliminary_predictions[1:, :])


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
        trained_model, _ = sf.get_model_for_noa(noa,
                                                additional_feature_list,
                                                model_class=XGBRegressorModel,
                                                model_parameters=xgb_regressor_model_parameters,
                                                y_type="band_gap")

        x, y, ids = sf.prepare_data_for_model(noa,
                                              additional_feature_list,
                                              data_type="test",
                                              y_type="band_gap")

        n, m = x.shape
        for i in range(n):
            id = int(ids[i])
            result[id][0] = id
            result[id][2] = trained_model.predict(x[i][:].reshape(1, -1))
            #print("f: {0}".format(result[id]))


    fe_general_model, _ = sf.get_model_for_noa(-1,
                                               additional_feature_list,
                                               model_class=XGBRegressorModel,
                                               model_parameters=xgb_regressor_model_parameters,
                                               y_type="formation_energy")

    x, y, ids = sf.prepare_data_for_model(-1,
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