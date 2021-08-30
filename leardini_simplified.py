import numpy as np
import Kinetics_LBMC as lbmc
from Kinetics_LBMC.utils.norm_vector import norm_vector as norm_vector


def leardini_simplified(points, points_name, general_information):
    # TODO make the left version of the code and make the true Leardini
    LPSI_label = general_information['LPSI_label']
    RPSI_label = general_information['RPSI_label']
    LASI_label = general_information['LASI_label']
    RASI_label = general_information['RASI_label']

    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point
    # Creation segment static
    RASI = points[:, points_ind[RASI_label], :]
    LASI = points[:, points_ind[LASI_label], :]
    RPSI = points[:, points_ind[RPSI_label], :]
    LPSI = points[:, points_ind[LPSI_label], :]

    Xtemp = (RASI+LASI)/2-(RPSI+LPSI)/2
    SACR = (LPSI+RPSI)/2
    Ytemp = np.cross(RASI-SACR, LASI-SACR, axisa=0, axisb=0, axisc=0)
    Ztemp = np.cross(Xtemp, Ytemp, axisa=0, axisb=0, axisc=0)
    X_Pelvis = norm_vector(Xtemp)
    Y_Pelvis = norm_vector(Ytemp)
    Z_Pelvis = norm_vector(Ztemp)

    width_temp = LASI - RASI
    width_Pelvis = np.mean(np.sqrt(np.sum(width_temp**2, axis=0)))

    depth_temp = (RASI+LASI)/2-SACR
    depth_Pelvis = np.mean(np.sqrt(np.sum((depth_temp)**2, axis=0)))

    Harrington_R = np.array([-0.24*depth_Pelvis*1000 - 9.9,
                             -0.30*width_Pelvis*1000 - 10.9,
                             0.33*width_Pelvis*1000 + 7.3]
                            ) / 1000  # en m, d'après l'article (conclusion)
    Harrington_L = np.array([-0.24*depth_Pelvis*1000 - 9.9,
                             -0.30*width_Pelvis*1000 - 10.9,
                             -0.33*width_Pelvis*1000 - 7.3]
                            ) / 1000

    R_HJC = (RASI+LASI)/2 + (Harrington_R[0]*X_Pelvis +
                             Harrington_R[1]*Y_Pelvis +
                             Harrington_R[2]*Z_Pelvis)

    L_HJC = (RASI+LASI)/2 + (Harrington_L[0]*X_Pelvis +
                             Harrington_L[1]*Y_Pelvis +
                             Harrington_L[2]*Z_Pelvis)

#    R_HJC = (RASI+LASI)/2 - 13.9/100 * width_temp * X_Pelvis - 33.6 / \
#        100 * width_temp * Y_Pelvis + 37.2/100 * width_temp * Z_Pelvis
#    L_HJC = (RASI+LASI)/2 - 13.9/100 * width_temp * X_Pelvis - 33.6 / \
#        100 * width_temp * Y_Pelvis - 37.2/100 * width_temp * Z_Pelvis

    u_pelvis = X_Pelvis
    w_pelvis = Z_Pelvis
    rp_pelvis = SACR
    rd_pelvis = (R_HJC+L_HJC)/2
    rm_pelvis = [RASI, LASI, RPSI, LPSI]

    # Thigh
    RGT = points[:, points_ind['R_FTC'], :]
    RLE = points[:, points_ind['R_FLE'], :]
    RME = points[:, points_ind['R_FME'], :]

    rp_thigh = R_HJC
    rd_thigh = (RLE+RME)/2
    w_thigh = norm_vector(RLE-RME)
    u_thigh = norm_vector(
        np.cross(rp_thigh-rd_thigh, w_thigh, axisa=0, axisb=0, axisc=0))
    rm_thigh = [RGT, RLE, RME]

    RHF = points[:, points_ind['R_FAX'], :]
    RTT = points[:, points_ind['R_TTC'], :]
    RLM = points[:, points_ind['R_FAL'], :]
    RMM = points[:, points_ind['R_TAM'], :]

    rp_tibia = rd_thigh
    rd_tibia = (RMM+RLM)/2
    w_tibia = norm_vector(RLM-RMM)
    u_tibia = norm_vector(
        np.cross(rp_tibia-rd_tibia, w_tibia, axisa=0, axisb=0, axisc=0))
    rm_tibia = [RHF, RTT, RLM, RMM]

    RCA = points[:, points_ind['R_FCC'], :]
    RVM = points[:, points_ind['R_FM5'], :]
    RFM = points[:, points_ind['R_FM1'], :]
    RSM = points[:, points_ind['R_FM2'], :]
    rp_foot = rd_tibia
    rd_foot = RSM
    u_foot = norm_vector(RSM-RCA)
    w_foot = norm_vector(RVM-RFM)
    rm_foot = [RCA, RFM, RVM]

    segment_pelvis = lbmc.Segment(u_pelvis, rp_pelvis, rd_pelvis, w_pelvis, rm_pelvis,
                                  'Buv', 'Bwu',
                                  'Pelvis', rigid_parameter=True)
    # mettre possibilité de rien mettre pour weight sex et type
    segment_thigh = lbmc.Segment(u_thigh, rp_thigh, rd_thigh, w_thigh, rm_thigh,
                                 'Buv', 'Bwu',
                                 'Thigh', rigid_parameter=True)

    segment_tibia = lbmc.Segment(u_tibia, rp_tibia, rd_tibia, w_tibia, rm_tibia,
                                 'Buv', 'Bwu',
                                 'Tibia', rigid_parameter=True)

    segment_foot = lbmc.Segment(u_foot, rp_foot, rd_foot, w_foot, rm_foot,
                                'Buw', 'Bwu',
                                'Foot', rigid_parameter=True)

    return [segment_foot, segment_tibia, segment_thigh, segment_pelvis]


def leardini_simplified_V2(points, points_name, general_information):
    # TODO make the left version of the code and make the true Leardini
    LPSI_label = general_information['LPSI_label']
    RPSI_label = general_information['RPSI_label']
    LASI_label = general_information['LASI_label']
    RASI_label = general_information['RASI_label']

    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point
    # Creation segment static
    RASI = points[:, points_ind[RASI_label], :]
    LASI = points[:, points_ind[LASI_label], :]
    RPSI = points[:, points_ind[RPSI_label], :]
    LPSI = points[:, points_ind[LPSI_label], :]

    Xtemp = (RASI+LASI)/2-(RPSI+LPSI)/2
    SACR = (LPSI+RPSI)/2
    Ytemp = np.cross(RASI-SACR, LASI-SACR, axisa=0, axisb=0, axisc=0)
    Ztemp = np.cross(Xtemp, Ytemp, axisa=0, axisb=0, axisc=0)
    X_Pelvis = norm_vector(Xtemp)
    Y_Pelvis = norm_vector(Ytemp)
    Z_Pelvis = norm_vector(Ztemp)

    width_temp = LASI - RASI
    width_Pelvis = np.mean(np.sqrt(np.sum(width_temp**2, axis=0)))

    depth_temp = (RASI+LASI)/2-SACR
    depth_Pelvis = np.mean(np.sqrt(np.sum((depth_temp)**2, axis=0)))

    Harrington_R = np.array([-0.24*depth_Pelvis*1000 - 9.9,
                             -0.30*width_Pelvis*1000 - 10.9,
                             0.33*width_Pelvis*1000 + 7.3]
                            ) / 1000  # en m, d'après l'article (conclusion)
    Harrington_L = np.array([-0.24*depth_Pelvis*1000 - 9.9,
                             -0.30*width_Pelvis*1000 - 10.9,
                             -0.33*width_Pelvis*1000 - 7.3]
                            ) / 1000

    R_HJC = (RASI+LASI)/2 + (Harrington_R[0]*X_Pelvis +
                             Harrington_R[1]*Y_Pelvis +
                             Harrington_R[2]*Z_Pelvis)

    L_HJC = (RASI+LASI)/2 + (Harrington_L[0]*X_Pelvis +
                             Harrington_L[1]*Y_Pelvis +
                             Harrington_L[2]*Z_Pelvis)

#    R_HJC = (RASI+LASI)/2 - 13.9/100 * width_temp * X_Pelvis - 33.6 / \
#        100 * width_temp * Y_Pelvis + 37.2/100 * width_temp * Z_Pelvis
#    L_HJC = (RASI+LASI)/2 - 13.9/100 * width_temp * X_Pelvis - 33.6 / \
#        100 * width_temp * Y_Pelvis - 37.2/100 * width_temp * Z_Pelvis

    u_pelvis = X_Pelvis
    w_pelvis = Z_Pelvis
    rp_pelvis = SACR
    rd_pelvis = (R_HJC+L_HJC)/2
    rm_pelvis = [RASI, LASI, RPSI, LPSI]

    # Thigh
    RGT = points[:, points_ind['R_FTC'], :]
    RLE = points[:, points_ind['R_FLE'], :]
    RME = points[:, points_ind['R_FME'], :]

    rp_thigh = R_HJC
    rd_thigh = (RLE+RME)/2
    w_thigh = norm_vector(RLE-RME)
    u_thigh = norm_vector(
        np.cross(rp_thigh-rd_thigh, w_thigh, axisa=0, axisb=0, axisc=0))
    rm_thigh = [RGT, RLE, RME]

    RHF = points[:, points_ind['R_FAX'], :]
    RTT = points[:, points_ind['R_TTC'], :]
    RLM = points[:, points_ind['R_FAL'], :]
    RMM = points[:, points_ind['R_TAM'], :]

    rp_tibia = rd_thigh
    rd_tibia = (RMM+RLM)/2
    w_tibia = norm_vector(RLM-RMM)
    u_tibia = norm_vector(
        np.cross(rp_tibia-rd_tibia, w_tibia, axisa=0, axisb=0, axisc=0))
    rm_tibia = [RHF, RTT, RLM, RMM]

    RCA = points[:, points_ind['R_FCC'], :]
    RVM = points[:, points_ind['R_FM5'], :]
    RFM = points[:, points_ind['R_FM1'], :]
    RSM = (RVM+RFM)/2
    rp_foot = rd_tibia
    rd_foot = RSM
    u_foot = norm_vector(RSM-RCA)
    w_foot = norm_vector(RVM-RFM)
    rm_foot = [RCA, RFM, RVM]

    segment_pelvis = lbmc.Segment(u_pelvis, rp_pelvis, rd_pelvis, w_pelvis, rm_pelvis,
                                  'Buv', 'Bwu',
                                  'Pelvis', rigid_parameter=True)
    # mettre possibilité de rien mettre pour weight sex et type
    segment_thigh = lbmc.Segment(u_thigh, rp_thigh, rd_thigh, w_thigh, rm_thigh,
                                 'Buv', 'Bwu',
                                 'Thigh', rigid_parameter=True)

    segment_tibia = lbmc.Segment(u_tibia, rp_tibia, rd_tibia, w_tibia, rm_tibia,
                                 'Buv', 'Bwu',
                                 'Tibia', rigid_parameter=True)

    segment_foot = lbmc.Segment(u_foot, rp_foot, rd_foot, w_foot, rm_foot,
                                'Buw', 'Bwu',
                                'Foot', rigid_parameter=True)

    return [segment_foot, segment_tibia, segment_thigh, segment_pelvis]
