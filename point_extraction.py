import Kinetics_LBMC as lbmc
import snippet_ezc3d as snip
import numpy as np


def point_extraction(acq, general_information):
    gi = general_information
    points_temp, points_name, points_ind = snip.get_points_ezc3d(acq)

    vector_equivalent_sign_new, vector_equivalent_new = reorientation_vector_definition(
        points_temp, points_ind, general_information)

    # TODO see if we can use points already calculated in the previous function
    unit_point = acq['parameters']['POINT']['UNITS']['value'][0]

    points = lbmc.points_treatment(
        acq, gi['point_filtering'], unit_point, vector_equivalent=vector_equivalent_new, vector_sign=vector_equivalent_sign_new)

    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point

    return points, points_name, points_ind


def reorientation_vector_definition(points_temp, points_ind, general_information):
    # In order to ease the reading we reduce general_information
    gi = general_information
    # X
    temp_X = points_temp[:, points_ind[gi['Anterior_Side']], 1] - \
        points_temp[:, points_ind[gi['Posterior_Side']], 1]
    pos_X = np.argmax(np.abs(temp_X))
    sign_X = np.sign(temp_X[pos_X])
    # Y
    temp_Y = points_temp[:, points_ind[gi['Up_Side']], 1] - \
        points_temp[:, points_ind[gi['Down_Side']], 1]
    pos_Y = np.argmax(np.abs(temp_Y))
    sign_Y = np.sign(temp_Y[pos_Y])
    # Z
    temp_Z = points_temp[:, points_ind[gi['Right_Side']], 1] - \
        points_temp[:, points_ind[gi['Left_Side']], 1]
    pos_Z = np.argmax(np.abs(temp_Z))
    sign_Z = np.sign(temp_Z[pos_Z])

    vector_equivalent_sign_new = [sign_X, sign_Y, sign_Z]
    vector_equivalent_new = [pos_X, pos_Y, pos_Z]

    return vector_equivalent_sign_new, vector_equivalent_new
