import os
import numpy as np

import matplotlib.pyplot as plt

def plot_performance(performance,
                     feature_types):

    plt.figure()

    for i in range(len(feature_types)):

        plt.plot(performance[:, i], label=feature_types[i], linewidth=3)

    n, _ = performance.shape
    plt.plot(0.09*np.ones((n, 1)), linewidth=1, linestyle="--")

    plt.legend(ncol=3)
    plt.show()

feature_types = ["standard",
                 #"rho_data",
                 #"rho_percentage_atom_data",
                 #"rho_percentage_atom_unit_cell_data",
                 #"rho_percentage_atom_unit_cell_nn_bond_parameters_data",
                 "rho_percentage_atom_unit_cell_nn_bond_parameters_symmetries_data",
                 "unit_cell_data",
                 #"unit_cell_nn_bond_parameters_data",
                 #"nn_bond_parameters_symmetries_data",
                 "unit_cell_nn_bond_parameters_symmetries_data",
                 "nn_bond_parameters_data"]

steps = 2
n_features_types = len(feature_types)
performance_train = np.zeros((steps, n_features_types))
performance_valid = np.zeros((steps, n_features_types))

for i in range(steps):
    print("i: " + str(i))

    for j in range(n_features_types):

        print("Processing " + str(feature_types[j]))
        command = "python3 main.py " + feature_types[j]
        output = os.popen(command).readlines()

        res = output[0].split("x")

        performance_train[i][j] = res[0]
        performance_valid[i][j] = res[1]

plot_performance(performance_train,
                 feature_types)

plot_performance(performance_valid,
                 feature_types)