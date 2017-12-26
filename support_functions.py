import numpy as np

from support_classes import Atom

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
    # Read the atoms.
    for i in range(6, len(lines)):

        ls = lines[i].split()
        x = float(ls[1])
        y = float(ls[2])
        z = float(ls[3])
        t = ls[4]

        a = Atom(x, y, z, t)
        atoms.append(a)

    return vectors, atoms


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
        for i in range(m):
            id = int(ids[i])
            fe = formation_energy_model.predict(x[i, :].reshape(1, -1))
            bg = band_gap_model.predict(x[i, :].reshape(1, -1))
            print("id: {0}, fe: {1}, bg: {2}".format(id, fe[0][0], bg[0][0]))

            f.write("{0},{1},{2}\n".format(id, fe[0][0], bg[0][0]))
        f.close()