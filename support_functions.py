import logging
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

import global_flags_constanst as gf
from support_classes import Atom
import global_flags_constanst as gfc


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gf.LOGGING_LEVEL)


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

def read_geometry_file(file_path):

    f = open(file_path, "r")
    lines = f.readlines()

    # We ignore the first 3 lines in the file.
    # Start reading with index 3.
    vec_x = lines[3].split()
    vec_y = lines[4].split()
    vec_z = lines[5].split()

    vec_x = [float(vec_x[i]) for i in range(1, len(vec_x))]
    vec_y = [float(vec_y[i]) for i in range(1, len(vec_y))]
    vec_z = [float(vec_z[i]) for i in range(1, len(vec_z))]

    vectors = [vec_x, vec_y, vec_z]
    atoms = []
    atom_count = {}
    # Read the atoms.
    for i in range(6, len(lines)):

        ls = lines[i].split()
        x = float(ls[1])
        y = float(ls[2])
        z = float(ls[3])
        t = ls[4]

        if t in atom_count:
            atom_count[t] = atom_count[t] + 1
        else:
            atom_count[t] = 1

        a = Atom(x, y, z, t)
        atoms.append(a)

    return vectors, atoms, atom_count


def root_mean_squared_logarithmic_error(y_true, y_pred):
    # y_true and y_pred should

    n, _ = y_true.shape
    m, _ = y_pred.shape

    assert n == m, "y_true and y_pred shapes are not equal!"

    lpi = np.log(y_pred + 1.0)
    lai = np.log(y_true + 1.0)
    s2 = (lpi - lai) * (lpi - lai)

    rmsle = np.sqrt((1.0 / n) * np.sum(s2))

    return rmsle


def pipeline_flow(ids,
                  x,
                  formation_energy_model,
                  band_gap_model,
                  submission_file_name):

    with open(submission_file_name, "w") as f:

        f.write("id,formation_energy_ev_natom,bandgap_energy_ev\n")
        m, n = x.shape
        logger.info("m: {0}; n: {1}".format(m, n))
        for i in range(m):
            id = int(ids[i])
            fe = formation_energy_model.predict(x[i, :].reshape(1, -1))
            bg = band_gap_model.predict(x[i, :].reshape(1, -1))
            # print("id: {0}, fe: {1}, bg: {2}".format(id, fe[0][0], bg[0][0]))

            f.write("{0},{1},{2}\n".format(id, fe[0][0], bg[0][0]))
        f.close()


def cross_validate(x,
                   y,
                   model_class,
                   model_parameters=None,
                   fraction=0.1):
    """
    Perform normal corss validation.
    A fraction of the total data is used as
    the test set. If, e.g., fraction=0.1 ten
    cross validation rounds will be performed.

    :param x:
    :param y:
    :param model_class:
    :param model_parameters:
    :param fraction:
    :return:
    """


    logger.debug("Cross validating data.")

    n_samples, n_features = x.shape
    window = int(fraction*n_samples)
    n_slides = int(n_samples/window)

    train_avg = 0.0
    valid_avg = 0.0

    for i in range(n_slides):
        start_index = i*window
        end_index = i*window + window

        logger.debug("start_index: {0}".format(start_index))
        logger.debug("end_index: {0}".format(end_index))

        indexes_to_remove = [j for j in range(start_index, end_index)]

        train_data = np.delete(x, indexes_to_remove, axis=0)
        train_targets = np.delete(y, indexes_to_remove, axis=0)

        logger.debug("train_data.shape: {0}".format(train_data.shape))
        logger.debug("train_targets.shape: {0}".format(train_targets.shape))

        valid_data = x[start_index:end_index, :]
        valid_targets = y[start_index:end_index, :]

        logger.debug("valid_data.shape: {0}".format(valid_data.shape))
        logger.debug("valid_targets.shape: {0}".format(valid_targets.shape))

        # Validation data within the model are used mainly
        # for Keras base NN models.
        model_parameters["validation_data"] = (valid_data, valid_targets)
        model = model_class(**model_parameters)


        _, train_m = train_targets.shape
        if train_m == 1:
            model.fit(train_data, train_targets.ravel())
        else:
            model.fit(train_data, train_targets)

        custom_data = np.hstack((valid_data, valid_targets))
        condition = custom_data[:, gfc.LABELS["number_of_total_atoms"] - 1] == 40
        custom_data = custom_data[condition]
        custom_valid_data = custom_data[:, 0:-1]
        custom_targets_data = custom_data[:, -1].reshape(-1, 1)
        logger.info("custom_valid_data.shape: {0}".format(custom_valid_data.shape))
        logger.info("custom_targets_data.shape: {0}".format(custom_targets_data.shape))
        custom_rmsle_valid = model.evaluate(custom_valid_data, custom_targets_data)
        logger.info("custom_rmsle_valid: {0}".format(custom_rmsle_valid))

        rmsle_train = model.evaluate(train_data, train_targets)
        rmsle_valid = model.evaluate(valid_data, valid_targets)

        logger.info("i: {0}, rmsle_train: {1:.9f}, rmsle_valid: {2:.9f}".format(i, rmsle_train, rmsle_valid))

        train_avg = train_avg + rmsle_train
        valid_avg = valid_avg + rmsle_valid

    train_avg = train_avg/n_slides
    valid_avg = valid_avg / n_slides

    logger.info("train_avg: {0}, valid_avg: {1}".format(train_avg, valid_avg))

    # This printout is used by graph_performace.py to grab the
    # results of grap_performance.py. Print is simpler that logging.
    print(str(train_avg) + "x" + str(valid_avg), end="")


def one_left_cross_validation(x,
                              y,
                              model_class=None,
                              model_parameters=None):

    logger.info("One left cross validation...")
    n, m = x.shape

    train_avg = 0.0
    valid_avg = 0.0
    for i in range(n):

        train_data = np.delete(x, [i], axis=0)
        train_targets = np.delete(y, [i], axis=0)

        logger.debug("train_data.shape: {0}".format(train_data.shape))
        logger.debug("train_targets.shape: {0}".format(train_targets.shape))

        # valid_x is a single example so its shape
        # should be (1, n_features)
        valid_x = x[i, :].reshape(1, -1)
        valid_y = y[i, :].reshape(-1, 1)

        logger.debug("test_x.shape: {0}".format(valid_x.shape))
        logger.debug("test_y.shape: {0}".format(valid_y.shape))

        model_parameters["validation_data"] = (valid_x, valid_y)
        model = model_class(**model_parameters)

        _, train_m = train_targets.shape
        if train_m == 1:
            model.fit(train_data, train_targets.ravel())
        else:
            model.fit(train_data, train_targets)

        rmsle_train = model.evaluate(train_data, train_targets)
        rmsle_valid = model.evaluate(valid_x, valid_y)

        logger.info("i: {0}, rmsle_train: {1:.9f}, rmsle_valid: {2:.9f}".format(i, rmsle_train, rmsle_valid))

        train_avg = train_avg + rmsle_train
        valid_avg = valid_avg + rmsle_valid

    train_avg = train_avg/n
    valid_avg = valid_avg/n

    logger.info("train_avg: {0}, valid_avg: {1}".format(train_avg, valid_avg))



def get_percentage_of_o_atoms(percent_atom_al,
                              percent_atom_ga,
                              percent_atom_in):

    """
    This function is obsolete.
    The percentages percent_atom_al, percent_atom_ga and
    percent_atom_in always sum to one.


    :param percent_atom_al:
    :param percent_atom_ga:
    :param percent_atom_in:
    :return:
    """

    percent_atom_o = np.ones(percent_atom_al.shape)
    percent_atom_o = percent_atom_o - percent_atom_al - percent_atom_ga - percent_atom_in

    logger.info("percent_atom_o.shape: " + str(percent_atom_o.shape))
    return percent_atom_o


if __name__ == "__main__":
    file_path = "/home/tadek/Coding/Kaggle/Nomad2018/train/1/geometry.xyz"

    vectors, atoms, atom_count = read_geometry_file(file_path)

    for key, val in atom_count.items():
        print("{0}: {1}".format(key, val))