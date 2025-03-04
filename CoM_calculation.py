import numpy as np
import Kinetics_LBMC as lbmc
from Kinetics_LBMC.utils.norm_vector import norm_vector as norm_vector
import math
from inertia_Mconville import inertia_generation as inertia_generation
from scipy import signal


def virtual_points_generation(points, points_name, general_information, sexe):
    # construction of the different segment of the body based on the ISB recommendation
    # and the parameters needed for the definition of CoM
    # Adjustments to McConville et al. and Young et al. body segment inertial parameters. , 40(3), 543–553.         doi:10.1016/j.jbiomech.2006.02.013
    LPSI_label = general_information['LPSI_label']
    RPSI_label = general_information['RPSI_label']
    LASI_label = general_information['LASI_label']
    RASI_label = general_information['RASI_label']

    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point

    virtual_point = dict()
    # Virtual point
    # Calculation of the Cervical Joint Centre (CJC)
    C7 = points[:, points_ind['C7'], :]
    # IJ is equivalent to SUP in the article
    IJ = points[:, points_ind['IJ'], :]
    # To define the Sagittal plane of the thorax as we are working with scoliotic we chose to use Px
    # instead of T12
    PX = points[:, points_ind['PX'], :]
    RAC = points[:, points_ind['R_AC'], :]
    LAC = points[:, points_ind['L_AC'], :]

    X_thorax = norm_vector(IJ-C7)
    Z_thorax = norm_vector(
        np.cross(PX-C7, X_thorax, axisa=0, axisb=0, axisc=0))
    Y_thorax = norm_vector(
        np.cross(Z_thorax, X_thorax, axisa=0, axisb=0, axisc=0))

    width_thorax = np.mean(np.linalg.norm(IJ-C7, axis=0))

    if sexe == 'M':
        virtual_point['CJC'] = C7+width_thorax * \
            0.33*(math.cos(math.radians(8))*X_thorax +
                  math.sin(math.radians(8))*Y_thorax)
        virtual_point['R_GH'] = RAC + width_thorax * \
            0.33*(math.cos(math.radians(-11))*X_thorax +
                  math.sin(math.radians(-11))*Y_thorax)
        virtual_point['L_GH'] = LAC + width_thorax * \
            0.33*(math.cos(math.radians(-11))*X_thorax +
                  math.sin(math.radians(-11))*Y_thorax)
    elif sexe == 'F':
        virtual_point['CJC'] = C7+width_thorax * \
            0.53*(math.cos(math.radians(14))*X_thorax +
                  math.sin(math.radians(14))*Y_thorax)
        virtual_point['R_GH'] = RAC + width_thorax * \
            0.36*(math.cos(math.radians(-5))*X_thorax +
                  math.sin(math.radians(-5))*Y_thorax)
        virtual_point['L_GH'] = LAC + width_thorax * \
            0.36*(math.cos(math.radians(-5))*X_thorax +
                  math.sin(math.radians(-5))*Y_thorax)

    # Calculation TJC
    T12 = points[:, points_ind['T12'], :]
    T8 = points[:, points_ind['Thor_5'], :]

    Y_thor_TJC = norm_vector(T8-T12)
    Z_thor_TJC = norm_vector(
        np.cross(IJ-T12, Y_thor_TJC, axisa=0, axisb=0, axisc=0))
    X_thor_TJC = norm_vector(
        np.cross(Y_thor_TJC, Z_thor_TJC, axisa=0, axisb=0, axisc=0))

    if sexe == 'M':
        # Here it is 4° instead of 94 because we are working on X_thor instead of Y_thor
        virtual_point['TJC'] = T12 + width_thorax * \
            0.50*(math.cos(math.radians(-4))*X_thor_TJC +
                  math.sin(math.radians(-4))*Y_thor_TJC)
    elif sexe == 'F':
        virtual_point['TJC'] = T12 + width_thorax * \
            0.52*(math.cos(math.radians(-2))*X_thor_TJC +
                  math.sin(math.radians(-2))*Y_thor_TJC)

    # Calculation LJC
    RASI = points[:, points_ind[RASI_label], :]
    LASI = points[:, points_ind[LASI_label], :]
    RPSI = points[:, points_ind[RPSI_label], :]
    LPSI = points[:, points_ind[LPSI_label], :]

    SACR = (RPSI+LPSI)/2
    Z_pelvis = norm_vector(RASI-LASI)
    Y_pelvis = norm_vector(
        np.cross(Z_pelvis, (RASI+LASI)/2-SACR, axisa=0, axisb=0, axisc=0))
    X_pelvis = norm_vector(
        np.cross(Y_pelvis, Z_pelvis, axisa=0, axisb=0, axisc=0))
    width_pelvis = np.mean(np.linalg.norm(RASI-LASI, axis=0))

    if sexe == 'M':
        virtual_point['LJC'] = (RASI+LASI)/2 - 33.5/100*width_pelvis*X_pelvis + 3.2 / \
            100 * width_pelvis * Y_pelvis + 0/100 * width_pelvis * Z_pelvis
        virtual_point['R_HJC'] = (RASI+LASI)/2 - 9.5/100*width_pelvis*X_pelvis - 37 / \
            100 * width_pelvis * Y_pelvis + 36.1/100 * width_pelvis * Z_pelvis
        virtual_point['L_HJC'] = (RASI+LASI)/2 - 9.5/100*width_pelvis*X_pelvis - 37 / \
            100 * width_pelvis * Y_pelvis - 36.1/100 * width_pelvis * Z_pelvis
    elif sexe == 'F':
        virtual_point['LJC'] = (RASI+LASI)/2 - 34/100*width_pelvis*X_pelvis + 4.9/100 * \
            width_pelvis * Y_pelvis + 0/100 * width_pelvis * Z_pelvis
        virtual_point['R_HJC'] = (RASI+LASI)/2 - 13.9/100*width_pelvis*X_pelvis - 33.6 / \
            100 * width_pelvis * Y_pelvis + 37.2/100 * width_pelvis * Z_pelvis
        virtual_point['L_HJC'] = (RASI+LASI)/2 - 13.9/100*width_pelvis*X_pelvis - 33.6 / \
            100 * width_pelvis * Y_pelvis - 37.2/100 * width_pelvis * Z_pelvis

    return virtual_point


def CoM_Segment(X, Y, Z, Or, segment_name, weight, length_segment, sexe, side):
    (ms, rCs, Is, Js) = inertia_generation(
        weight, length_segment, sexe, segment_name)

    if side == 'L':
        correction = -1
    elif side == 'R':
        correction = 1
    elif side == 'no_side':
        correction = 1
    # import pdb
    # pdb.set_trace()
    CoM_coord = Or + rCs[0] * X + rCs[1] * Y + correction*rCs[2] * Z
    CoM_mass = CoM_coord * ms
    return CoM_mass, CoM_coord


def CoM_calculation(points, points_name, general_information, sexe, weight):

    virtual_point = virtual_points_generation(
        points, points_name, general_information, sexe)
    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point
    # Head
    # Superior-inferior axis from cervical joint center to
    # head vertex skin landmark
    # Sagittal plane containing cervical
    # joint center and skin landmarks on the head vertex and sellion
    # Origin at cervical joint center
    # TODO Add Head vextex on the head
    SEL = points[:, points_ind['SEL'], :]
    RCAE = points[:, points_ind['R_CAE'], :]
    LCAE = points[:, points_ind['L_CAE'], :]
    OCC = points[:, points_ind['OCC'], :]
    mid_head = (SEL+RCAE+LCAE+OCC)/4
    Y_head = norm_vector(mid_head-virtual_point['CJC'])
    Z_head = norm_vector(
        np.cross(SEL-virtual_point['CJC'], Y_head, axisa=0, axisb=0, axisc=0))
    X_head = norm_vector(np.cross(Y_head, Z_head, axisa=0, axisb=0, axisc=0))

    length_head = np.mean(
        2*np.linalg.norm(mid_head-virtual_point['CJC'], axis=0))

    CoM_head_mass, virtual_point['CoM_head_coord'] = CoM_Segment(X_head, Y_head, Z_head, virtual_point['CJC'],
                                                                 'head', weight, length_head, sexe, 'no_side')

    # Thorax
    IJ = points[:, points_ind['IJ'], :]
    PX = points[:, points_ind['PX'], :]
    C7 = points[:, points_ind['C7'], :]
    T12 = points[:, points_ind['T12'], :]

    Y_thorax = norm_vector(virtual_point['CJC']-virtual_point['TJC'])
    Z_thorax = norm_vector(
        np.cross(IJ-T12, C7-T12, axisa=0, axisb=0, axisc=0))
    X_thorax = norm_vector(
        np.cross(Y_thorax, Z_thorax, axisa=0, axisb=0, axisc=0))
    length_thorax = np.mean(
        np.linalg.norm(virtual_point['CJC']-virtual_point['TJC'], axis=0))
    # The origin is defined at CJC even at the thorax (the value given in the table have a negative value on Y)
    CoM_thorax_mass, virtual_point['CoM_thorax_coord'] = CoM_Segment(X_thorax, Y_thorax, Z_thorax, virtual_point['CJC'],
                                                                     'thorax', weight, length_thorax, sexe, 'no_side')

    # TODO Adomen
    Rlomb = points[:, points_ind['R_Lomb'], :]
    Llomb = points[:, points_ind['L_Lomb'], :]
    Y_abdomen = norm_vector(virtual_point['TJC']-virtual_point['LJC'])
    X_abdomen = norm_vector(
        np.cross(Rlomb-virtual_point['TJC'], Llomb-virtual_point['TJC'], axisa=0, axisb=0, axisc=0))
    Z_abdomen = norm_vector(
        np.cross(X_abdomen, Y_abdomen, axisa=0, axisb=0, axisc=0))
    length_thorax = np.mean(
        np.linalg.norm(virtual_point['TJC']-virtual_point['LJC'], axis=0))
    CoM_abdomen_mass, virtual_point['CoM_abdomen_coord'] = CoM_Segment(X_abdomen, Y_abdomen, Z_abdomen, virtual_point['TJC'],
                                                                       'abdomen', weight, length_thorax, sexe, 'no_side')
    # Pelvis
    LPSI_label = general_information['LPSI_label']
    RPSI_label = general_information['RPSI_label']
    LASI_label = general_information['LASI_label']
    RASI_label = general_information['RASI_label']
    LPSI = points[:, points_ind[LPSI_label], :]
    RPSI = points[:, points_ind[RPSI_label], :]
    LASI = points[:, points_ind[LASI_label], :]
    RASI = points[:, points_ind[RASI_label], :]
    mid_PSI = (LPSI+RPSI)/2
    Z_pelvis = norm_vector(RASI-LASI)
    Y_pelvis = norm_vector(
        np.cross(RASI-mid_PSI, LASI-mid_PSI, axisa=0, axisb=0, axisc=0))
    X_pelvis = norm_vector(
        np.cross(Y_pelvis, Z_pelvis, axisa=0, axisb=0, axisc=0))

    length_pelvis = np.mean(
        np.linalg.norm(RASI-LASI, axis=0))
    CoM_pelvis_mass, virtual_point['CoM_pelvis_coord'] = CoM_Segment(X_pelvis, Y_pelvis, Z_pelvis, virtual_point['LJC'],
                                                                     'pelvis', weight, length_pelvis, sexe, 'no_side')
    # Upper_limb
    # Arm
    # Right
    RLE = points[:, points_ind['R_LE'], :]
    RME = points[:, points_ind['R_ME'], :]
    R_elbow_centre = (RLE+RME)/2
    Y_Rarm = norm_vector(virtual_point['R_GH']-R_elbow_centre)
    X_Rarm = norm_vector(
        np.cross(RLE-virtual_point['R_GH'], RME-virtual_point['R_GH'], axisa=0, axisb=0, axisc=0))
    Z_Rarm = norm_vector(
        np.cross(X_Rarm, Y_Rarm, axisa=0, axisb=0, axisc=0))
    length_Rarm = np.mean(
        np.linalg.norm(virtual_point['R_GH']-R_elbow_centre, axis=0))

    CoM_Rarm_mass, virtual_point['CoM_Rarm_coord'] = CoM_Segment(X_Rarm, Y_Rarm, Z_Rarm, virtual_point['R_GH'],
                                                                 'arm', weight, length_Rarm, sexe, 'R')
    # Left
    LLE = points[:, points_ind['L_LE'], :]
    LME = points[:, points_ind['L_ME'], :]
    L_elbow_centre = (LLE+LME)/2
    Y_Larm = norm_vector(virtual_point['L_GH']-L_elbow_centre)
    X_Larm = norm_vector(
        np.cross(LLE-virtual_point['L_GH'], LME-virtual_point['L_GH'], axisa=0, axisb=0, axisc=0))
    Z_Larm = norm_vector(
        np.cross(X_Larm, Y_Larm, axisa=0, axisb=0, axisc=0))
    length_Larm = np.mean(
        np.linalg.norm(virtual_point['L_GH']-L_elbow_centre, axis=0))

    CoM_Larm_mass, virtual_point['CoM_Larm_coord'] = CoM_Segment(X_Larm, Y_Larm, Z_Larm, virtual_point['L_GH'],
                                                                 'arm', weight, length_Larm, sexe, 'L')
    # Forearm
    # Right
    RRAD = points[:, points_ind['R_RAD'], :]
    RULN = points[:, points_ind['R_ULN'], :]
    R_wrist_centre = (RRAD+RULN)/2
    Y_Rforearm = norm_vector(R_elbow_centre-R_wrist_centre)
    X_Rforearm = norm_vector(
        np.cross(RRAD-R_elbow_centre, RULN-R_elbow_centre, axisa=0, axisb=0, axisc=0))
    Z_Rforearm = norm_vector(
        np.cross(X_Rforearm, Y_Rforearm, axisa=0, axisb=0, axisc=0))
    length_Rforearm = np.mean(
        np.linalg.norm(R_elbow_centre-R_wrist_centre, axis=0))
    CoM_Rforearm_mass, virtual_point['CoM_Rforearm_coord'] = CoM_Segment(X_Rforearm, Y_Rforearm, Z_Rforearm,
                                                                         R_elbow_centre, 'forearm',
                                                                         weight, length_Rforearm, sexe, 'R')
    # Left
    LRAD = points[:, points_ind['L_RAD'], :]
    LULN = points[:, points_ind['L_ULN'], :]
    L_wrist_centre = (LRAD+LULN)/2
    Y_Lforearm = norm_vector(L_elbow_centre-L_wrist_centre)
    X_Lforearm = norm_vector(
        np.cross(LULN-L_elbow_centre, LRAD-L_elbow_centre, axisa=0, axisb=0, axisc=0))
    Z_Lforearm = norm_vector(
        np.cross(X_Lforearm, Y_Lforearm, axisa=0, axisb=0, axisc=0))
    length_Lforearm = np.mean(
        np.linalg.norm(L_elbow_centre-L_wrist_centre, axis=0))
    CoM_Lforearm_mass, virtual_point['CoM_Lforearm_coord'] = CoM_Segment(X_Lforearm, Y_Lforearm, Z_Lforearm,
                                                                         L_elbow_centre, 'forearm',
                                                                         weight, length_Lforearm, sexe, 'L')

    # Hand
    # Right
    RHM2 = points[:, points_ind['R_HM2'], :]
    RHM5 = points[:, points_ind['R_HM5'], :]
    R_dist_hand_centre = (RHM2+RHM5)/2
    Y_Rhand = norm_vector(R_wrist_centre-R_dist_hand_centre)
    X_Rhand = norm_vector(
        np.cross(RHM2-R_wrist_centre, RHM5-R_wrist_centre, axisa=0, axisb=0, axisc=0))
    Z_Rhand = norm_vector(
        np.cross(X_Rhand, Y_Rhand, axisa=0, axisb=0, axisc=0))
    length_Rhand = np.mean(
        np.linalg.norm(R_wrist_centre-R_dist_hand_centre, axis=0))
    CoM_Rhand_mass, virtual_point['CoM_Rhand_coord'] = CoM_Segment(X_Rhand, Y_Rhand, Z_Rhand,
                                                                   R_wrist_centre, 'hand',
                                                                   weight, length_Rhand, sexe, 'R')

    # Left
    LHM2 = points[:, points_ind['L_HM2'], :]
    LHM5 = points[:, points_ind['L_HM5'], :]
    L_dist_hand_centre = (LHM2+LHM5)/2
    Y_Lhand = norm_vector(L_wrist_centre-L_dist_hand_centre)
    X_Lhand = norm_vector(
        np.cross(LHM5-L_wrist_centre, LHM2-L_wrist_centre, axisa=0, axisb=0, axisc=0))
    Z_Lhand = norm_vector(
        np.cross(X_Lhand, Y_Lhand, axisa=0, axisb=0, axisc=0))
    length_Lhand = np.mean(
        np.linalg.norm(L_wrist_centre-L_dist_hand_centre, axis=0))
    CoM_Lhand_mass, virtual_point['CoM_Lhand_coord'] = CoM_Segment(X_Lhand, Y_Lhand, Z_Lhand,
                                                                   L_wrist_centre, 'hand',
                                                                   weight, length_Lhand, sexe, 'L')

    # Thigh
    # Right
    RFLE = points[:, points_ind['R_FLE'], :]
    RFME = points[:, points_ind['R_FME'], :]
    R_knee_centre = (RFLE+RFME)/2
    Y_Rthigh = norm_vector(virtual_point['R_HJC']-R_knee_centre)
    X_Rthigh = norm_vector(
        np.cross(RFLE-virtual_point['R_HJC'], RFME-virtual_point['R_HJC'], axisa=0, axisb=0, axisc=0))
    Z_Rthigh = norm_vector(
        np.cross(X_Rthigh, Y_Rthigh, axisa=0, axisb=0, axisc=0))
    length_Rthigh = np.mean(
        np.linalg.norm(virtual_point['R_HJC']-R_knee_centre, axis=0))
    CoM_Rthigh_mass, virtual_point['CoM_Rthigh_coord'] = CoM_Segment(X_Rthigh, Y_Rthigh, Z_Rthigh,
                                                                     virtual_point['R_HJC'], 'thigh',
                                                                     weight, length_Rthigh, sexe, 'R')
    # Left
    LFLE = points[:, points_ind['L_FLE'], :]
    LFME = points[:, points_ind['L_FME'], :]
    L_knee_centre = (LFLE+LFME)/2
    Y_Lthigh = norm_vector(virtual_point['L_HJC']-L_knee_centre)
    X_Lthigh = norm_vector(
        np.cross(LFME-virtual_point['L_HJC'], LFLE-virtual_point['L_HJC'], axisa=0, axisb=0, axisc=0))
    Z_Lthigh = norm_vector(
        np.cross(X_Lthigh, Y_Lthigh, axisa=0, axisb=0, axisc=0))
    length_Lthigh = np.mean(
        np.linalg.norm(virtual_point['L_HJC']-L_knee_centre, axis=0))
    CoM_Lthigh_mass, virtual_point['CoM_Lthigh_coord'] = CoM_Segment(X_Lthigh, Y_Lthigh, Z_Lthigh,
                                                                     virtual_point['L_HJC'], 'thigh',
                                                                     weight, length_Lthigh, sexe, 'L')
    # tibia
    # Right
    RTAM = points[:, points_ind['R_TAM'], :]
    RFAL = points[:, points_ind['R_FAL'], :]
    R_ankle_centre = (RTAM+RFAL)/2
    Y_Rtibia = norm_vector(R_knee_centre-R_ankle_centre)
    X_Rtibia = norm_vector(
        np.cross(RFAL-R_knee_centre, RTAM-R_knee_centre, axisa=0, axisb=0, axisc=0))
    Z_Rtibia = norm_vector(
        np.cross(X_Rtibia, Y_Rtibia, axisa=0, axisb=0, axisc=0))
    length_Rtibia = np.mean(
        np.linalg.norm(R_knee_centre-R_ankle_centre, axis=0))
    CoM_Rtibia_mass, virtual_point['CoM_Rtibia_coord'] = CoM_Segment(X_Rtibia, Y_Rtibia, Z_Rtibia,
                                                                     R_knee_centre, 'tibia',
                                                                     weight, length_Rtibia, sexe, 'R')
    # Left
    LTAM = points[:, points_ind['L_TAM'], :]
    LFAL = points[:, points_ind['L_FAL'], :]
    L_ankle_centre = (LTAM+LFAL)/2
    Y_Ltibia = norm_vector(L_knee_centre-L_ankle_centre)
    X_Ltibia = norm_vector(
        np.cross(LTAM-L_knee_centre, LFAL-L_knee_centre, axisa=0, axisb=0, axisc=0))
    Z_Ltibia = norm_vector(
        np.cross(X_Ltibia, Y_Ltibia, axisa=0, axisb=0, axisc=0))
    length_Ltibia = np.mean(
        np.linalg.norm(L_knee_centre-L_ankle_centre, axis=0))
    CoM_Ltibia_mass, virtual_point['CoM_Ltibia_coord'] = CoM_Segment(X_Ltibia, Y_Ltibia, Z_Ltibia,
                                                                     L_knee_centre, 'tibia',
                                                                     weight, length_Ltibia, sexe, 'L')
    # Foot == Quel base ? d'ou est la longueur
    # Right
    RFCC = points[:, points_ind['R_FCC'], :]
    RFM1 = points[:, points_ind['R_FM1'], :]
    RFM5 = points[:, points_ind['R_FM5'], :]
    R_foot_distal_centre = (RFM1+RFM5)/2
    X_Rfoot = norm_vector(R_foot_distal_centre-RFCC)
    Y_Rfoot = norm_vector(
        np.cross(RFM5-RFCC, RFM1-RFCC, axisa=0, axisb=0, axisc=0))
    Z_Rfoot = norm_vector(
        np.cross(X_Rfoot, Y_Rfoot, axisa=0, axisb=0, axisc=0))
    length_Rfoot = np.mean(
        np.linalg.norm(R_ankle_centre-R_foot_distal_centre, axis=0))
    CoM_Rfoot_mass, virtual_point['CoM_Rfoot_coord'] = CoM_Segment(X_Rfoot, Y_Rfoot, Z_Rfoot,
                                                                   R_ankle_centre, 'foot',
                                                                   weight, length_Rfoot, sexe, 'R')
    # Left
    LFCC = points[:, points_ind['L_FCC'], :]
    LFM1 = points[:, points_ind['L_FM1'], :]
    LFM5 = points[:, points_ind['L_FM5'], :]
    L_foot_distal_centre = (LFM1+LFM5)/2
    X_Lfoot = norm_vector(L_foot_distal_centre-LFCC)
    Y_Lfoot = norm_vector(
        np.cross(LFM1-LFCC, LFM5-LFCC, axisa=0, axisb=0, axisc=0))
    Z_Lfoot = norm_vector(
        np.cross(X_Lfoot, Y_Lfoot, axisa=0, axisb=0, axisc=0))
    length_Lfoot = np.mean(
        np.linalg.norm(L_ankle_centre-L_foot_distal_centre, axis=0))
    CoM_Lfoot_mass, virtual_point['CoM_Lfoot_coord'] = CoM_Segment(X_Lfoot, Y_Lfoot, Z_Lfoot,
                                                                   L_ankle_centre, 'foot',
                                                                   weight, length_Lfoot, sexe, 'L')
    virtual_point['CoM'] = (CoM_head_mass+CoM_thorax_mass + CoM_pelvis_mass +
                            CoM_Rarm_mass + CoM_Rforearm_mass + CoM_Rhand_mass +
                            CoM_Larm_mass + CoM_Lforearm_mass + CoM_Lhand_mass +
                            CoM_Rthigh_mass + CoM_Rtibia_mass + CoM_Rfoot_mass +
                            CoM_Lthigh_mass + CoM_Ltibia_mass + CoM_Lfoot_mass) / weight
    #import pdb
    # pdb.set_trace()
    H_pend = np.mean(virtual_point['CoM'][1, :])
    ohmega = np.sqrt(9.81/H_pend)
    fq = 100
    fq_cutoff = 10

    dt = 1/fq
    speed_CoM = np.gradient(virtual_point['CoM'], dt, axis=1)

    (b, a) = signal.butter(4, fq_cutoff/(0.5*fq), btype='lowpass')
    virtual_point['speed_CoM'] = signal.filtfilt(b, a, speed_CoM[:, :], axis=1)

    virtual_point['XCoM'] = virtual_point['CoM'] + \
        virtual_point['speed_CoM']/ohmega

    return virtual_point
