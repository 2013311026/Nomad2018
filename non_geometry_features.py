import logging
import numpy as np
import global_flags_constanst as gfc

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(gfc.LOGGING_LEVEL)


def add_number_of_symmetries(space_group_feature):

    n = len(space_group_feature)
    symmetries_data = np.zeros((n, 1))

    for i in range(n):
        sg = int(space_group_feature[i])
        logger.info("space group: {0}; number of symmetries: {1}".format(sg,
                                                                         gfc.SPACE_GROUP_PROPERTIES[sg]))

        symmetries_data[i] = gfc.SPACE_GROUP_PROPERTIES[sg]

    return symmetries_data


if __name__ == "__main__":
    data = np.loadtxt("train.csv", delimiter=",", skiprows=1)
    test_data = np.loadtxt("test.csv", delimiter=",", skiprows=1)

    ids = data[:, 0].reshape(-1, 1)
    space_group_feature = data[:, 1]

    test_ids = test_data[:, 0].reshape(-1, 1)
    test_space_group_feature = test_data[:, 1]

    symmetries_data = add_number_of_symmetries(space_group_feature)
    test_symmetries_data = add_number_of_symmetries(test_space_group_feature)

    symmetries_data = np.hstack((ids, symmetries_data))
    np.savetxt("symmetries_data.csv", symmetries_data, delimiter=",")

    test_symmetries_data = np.hstack((test_ids, test_symmetries_data))
    np.savetxt("test_symmetries_data.csv", test_symmetries_data, delimiter=",")