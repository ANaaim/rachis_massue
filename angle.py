import numpy as np
import pdb


def angle(dx, dy, list_name, name_1, name_2):
    norm = np.sqrt(dx[list_name.index(name_1)]*dx[list_name.index(name_1)] +
                   dy[list_name.index(name_1)]*dy[list_name.index(name_1)]) *  \
        np.sqrt(dx[list_name.index(name_2)]*dx[list_name.index(name_2)] +
                dy[list_name.index(name_2)]*dy[list_name.index(name_2)])

    costheta = (dx[list_name.index(name_1)]*dx[list_name.index(name_2)] +
                dy[list_name.index(name_1)]*dy[list_name.index(name_2)])/norm
    theta = np.arccos(costheta)

    return theta
