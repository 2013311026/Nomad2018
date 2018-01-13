import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
features = np.loadtxt("train_ewald_sum_data.csv", delimiter=",", skiprows=0)
#features = np.loadtxt("train_nn_bond_parameters_data.csv", delimiter=",", skiprows=0)
#features = np.loadtxt("train_symmetries_data.csv", delimiter=",", skiprows=0)
#features = np.loadtxt("train_unit_cell_data.csv", delimiter=",", skiprows=0)
#features = np.loadtxt("train_angles_and_rs_data.csv", delimiter=",", skiprows=0)

custom_data = np.hstack((features, data))
print("custom_data.shape: {0}".format(custom_data.shape))

plt.figure()

#plt.scatter(features[:, 5], data[:, -1])
#plt.hist2d(data[:, -1], features[:, 1], bins=60)
index = 7
target = -1
bg_index = 14

plt.scatter(custom_data[custom_data[:, bg_index] == 10, index], custom_data[custom_data[:, bg_index] == 10, target], label="10")
plt.scatter(custom_data[custom_data[:, bg_index] == 20, index], custom_data[custom_data[:, bg_index] == 20, target], label="20")
plt.scatter(custom_data[custom_data[:, bg_index] == 30, index], custom_data[custom_data[:, bg_index] == 30, target], label="30")
plt.scatter(custom_data[custom_data[:, bg_index] == 40, index], custom_data[custom_data[:, bg_index] == 40, target], label="40")
plt.scatter(custom_data[custom_data[:, bg_index] == 60, index], custom_data[custom_data[:, bg_index] == 60, target], label="60")
plt.scatter(custom_data[custom_data[:, bg_index] == 80, index], custom_data[custom_data[:, bg_index] == 80, target], label="80")

plt.legend(ncol=3)
plt.show()