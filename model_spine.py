# Different method to calculate thorax and lombaire referentiel frame
from normal_best_fitting_plan import normal_best_fitting_plan as normal_best_fitting_plan
from Kinetics_LBMC.utils.norm_vector import norm_vector as norm_vector
import numpy as np
import Kinetics_LBMC as lbmc


def Naaim_two_segments(points, points_ind, general_information):
    list_marker_lomb = general_information['list_marker_lomb']
    base_lomb = general_information['base_lomb']
    up_lomb = general_information['up_lomb']
    list_marker_thor = general_information['list_marker_thor']
    base_thor = general_information['base_thor']
    up_thor = general_information['up_thor']
    right_lomb = general_information['right_lomb']
    left_lomb = general_information['left_lomb']
    right_thor = general_information['right_thor']
    left_thor = general_information['left_thor']

    # -----------------------------------------------------------------------------------------
    # Extraction of the normal to the plane
    Normal_plan_lomb = normal_best_fitting_plan(
        points, points_ind, list_marker_lomb, right_lomb, left_lomb)
    Normal_plan_thor = normal_best_fitting_plan(
        points, points_ind, list_marker_thor, right_thor, left_thor)
    # ____________________
    # Lombaire definition
    Z_lomb = norm_vector(Normal_plan_lomb)
    rp_lomb = points[:, points_ind[up_lomb], :]
    rd_lomb = points[:, points_ind[base_lomb], :]
    u_temp_lomb = np.cross(rp_lomb-rd_lomb, Z_lomb, axisa=0, axisb=0, axisc=0)
    u_lomb = norm_vector(u_temp_lomb)
    w_temp_lomb = np.cross(u_lomb, rp_lomb-rd_lomb, axisa=0, axisb=0, axisc=0)
    w_lomb = norm_vector(w_temp_lomb)
    rm_lomb = [points[:, points_ind[base_lomb], :],
               points[:, points_ind[up_lomb], :]]
    # _________________
    # Thorax defintion
    Z_thor = norm_vector(Normal_plan_thor)
    rp_thor = points[:, points_ind[up_thor], :]
    rd_thor = points[:, points_ind[base_thor], :]
    u_temp_thor = np.cross(rp_thor-rd_thor, Z_thor, axisa=0, axisb=0, axisc=0)
    u_thor = norm_vector(u_temp_thor)
    w_temp_thor = np.cross(u_thor, rp_thor-rd_thor, axisa=0, axisb=0, axisc=0)
    w_thor = norm_vector(w_temp_thor)
    w_thor = Z_thor
    rm_thor = [points[:, points_ind[base_thor], :],
               points[:, points_ind[up_thor], :]]

    segment_lomb = lbmc.Segment(u_lomb, rp_lomb, rd_lomb, w_lomb, rm_lomb,
                                'Bwu', 'Bwu',
                                'Lomb')

    segment_thor = lbmc.Segment(u_thor, rp_thor, rd_thor, w_thor, rm_thor,
                                'Bwu', 'Bwu',
                                'Thor')
    return segment_lomb, segment_thor


def model_simple_two_segments(points, points_ind, general_information):
    base_lomb = general_information['base_lomb']
    up_lomb = general_information['up_lomb']
    base_thor = general_information['base_thor']
    up_thor = general_information['up_thor']
    right_lomb = general_information['right_lomb']
    left_lomb = general_information['left_lomb']
    right_thor = general_information['right_thor']
    left_thor = general_information['left_thor']
    # __________________
    # Laborartory frame
    # In order to be able to define some definition we need the laboratory frame
    Xref = np.zeros(points[:, 0, :].shape)
    Xref[0, :] = 1
    Yref = np.zeros(points[:, 0, :].shape)
    Yref[1, :] = 1
    Zref = np.zeros(points[:, 0, :].shape)
    Zref[2, :] = 1
    # _______________________
    # Lomb
    rp_lomb = points[:, points_ind[up_lomb], :]
    rd_lomb = points[:, points_ind[base_lomb], :]
    lateral_direction = points[:, points_ind[right_lomb],
                               :]-points[:, points_ind[left_lomb], :]
    lateral_direction = norm_vector(lateral_direction)
    u_temp_lomb = np.cross(
        rp_lomb-rd_lomb, lateral_direction, axisa=0, axisb=0, axisc=0)
    u_lomb = norm_vector(u_temp_lomb)
    w_temp_lomb = np.cross(u_lomb, rp_lomb-rd_lomb, axisa=0, axisb=0, axisc=0)
    w_lomb = norm_vector(w_temp_lomb)
    rm_lomb = [points[:, points_ind[base_lomb], :],
               points[:, points_ind[up_lomb], :]]

    # ________________________
    # Thorax
    rp_thor = points[:, points_ind[up_thor], :]
    rd_thor = points[:, points_ind[base_thor], :]
    lateral_direction = points[:, points_ind[right_thor],
                               :]-points[:, points_ind[left_thor], :]
    lateral_direction = norm_vector(lateral_direction)
    u_temp_thor = np.cross(
        rp_thor-rd_thor, lateral_direction, axisa=0, axisb=0, axisc=0)
    u_thor = norm_vector(u_temp_thor)
    w_temp_thor = np.cross(u_thor, rp_thor-rd_thor, axisa=0, axisb=0, axisc=0)
    w_thor = norm_vector(w_temp_thor)
    rm_thor = [points[:, points_ind[base_thor], :],
               points[:, points_ind[up_thor], :]]

    segment_lomb = lbmc.Segment(u_lomb, rp_lomb, rd_lomb, w_lomb, rm_lomb,
                                'Bwu', 'Bwu',
                                'Lomb')

    segment_thor = lbmc.Segment(u_thor, rp_thor, rd_thor, w_thor, rm_thor,
                                'Bwu', 'Bwu',
                                'Thor')

    return segment_lomb, segment_thor
