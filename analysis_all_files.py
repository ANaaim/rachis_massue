import ezc3d
import snippet_ezc3d as snip
from normal_best_fitting_plan import extract_chord_parameters as extract_chord_parameters
import numpy as np
import Kinetics_LBMC as lbmc
import pdb
from Kinetics_LBMC.utils.norm_vector import norm_vector as norm_vector
import model_spine


def rachis_all_files(filenames, list_subdivision, general_information, generate_c3d='False'):
    if len(filenames) == 1:
        value = rachis_trial(
            filenames[0], list_subdivision, general_information, generate_c3d)

    else:
        for ind_trial, trial_name in enumerate(filenames):
            if ind_trial == 0:
                value = rachis_trial(
                    filenames[ind_trial], list_subdivision, general_information, generate_c3d)
            else:
                value_temp = rachis_trial(
                    filenames[ind_trial], list_subdivision, general_information, generate_c3d)
                for task in list_subdivision.keys():
                    for key_value_temp in value_temp[task]:
                        value[task][key_value_temp] = value[task][key_value_temp] + \
                            value_temp[task][key_value_temp]
    return value


def rachis_trial(trial_name, list_subdivision, general_information, generate_c3d='False'):
    acq = ezc3d.c3d(trial_name)
    # Definition of the session interval for this trial
    session_intervals = extraction_session_from_event(acq, list_subdivision)
    # Extraction of the points to be able to reposition it correctly
    points, points_name, points_ind = point_extraction(
        acq, general_information)

    # Extraction of the information from general information to make the following code more redeable
    list_marker_lomb = general_information['list_marker_lomb']
    list_marker_lomb_spline = general_information['list_marker_lomb_spline']
    base_lomb = general_information['base_lomb']
    up_lomb = general_information['up_lomb']
    list_marker_thor = general_information['list_marker_thor']
    list_marker_thor_spline = general_information['list_marker_thor_spline']
    base_thor = general_information['base_thor']
    up_thor = general_information['up_thor']

    # dict_value_to_save will bbe used in order to save the variable and be able to cut it easily later
    dict_value_to_save = dict()

    multiseg, full_segment = kinematics_rachis_calculation(
        points, points_ind, points_name, general_information)
    for joint in multiseg.euler_rel.keys():
        for ind_kinematics, name_kinematics in enumerate(['X', 'Y', 'Z']):
            dict_value_to_save[joint+'_' +
                               name_kinematics] = multiseg.euler_rel[joint][ind_kinematics, :]
        # Kinematic calculation
        # chord calculation
    dict_value_to_save['percentage_chord_lomb'], dict_value_to_save['value_chord_lomb'] = extract_chord_parameters(
        points, points_ind, list_marker_lomb, list_marker_lomb_spline, base_lomb, up_lomb, video=False, name_video='lombaire.mp4')

    dict_value_to_save['percentage_chord_thor'], dict_value_to_save['value_chord_thor'] = extract_chord_parameters(
        points, points_ind, list_marker_thor, list_marker_thor_spline, base_thor, up_thor, video=False, name_video='thorax.mp4')

    # TODO Calcul de la position du ventre et de la fleche du ventre
    # Au final projeter les marqueurs dans le plan et faire l'analyse en 2D ?
    # Stock all the data in a dictionnary
    value_to_export = dict()
    for ind_task, task in enumerate(list_subdivision.keys()):
        value_to_export[task] = dict()
        for value_to_save in dict_value_to_save.keys():
            value_to_export[task][value_to_save] = list()

        for ind, session_interval in enumerate(session_intervals[task]):
            for value_to_save in dict_value_to_save.keys():
                value_to_export[task][value_to_save].append(
                    dict_value_to_save[value_to_save][session_interval[0]:session_interval[1]])
    if generate_c3d:
        name_file_export = trial_name[:-4]+'post_process.c3d'
        generate_c3d_with_model_and_date(
            acq, name_file_export,  full_segment, multiseg, general_information)

    return value_to_export


def extraction_session_from_event(acq, list_subdivision):
    event = snip.get_event_ezc3d(acq)
    session_interval = dict()
    for type_of_event in list_subdivision.keys():
        event_beginnning_name = list_subdivision[type_of_event][0]
        event_end_name = list_subdivision[type_of_event][1]
        if event_beginnning_name == event_end_name:
            event_begin_temp = sorted(event[event_beginnning_name])[0:-1]
            event_end_temp = sorted(event[event_beginnning_name])[1:]
        else:
            event_begin_temp = sorted(event[event_beginnning_name])
            event_end_temp = event[event_end_name][event[event_end_name]
                                                   > event_begin_temp[0]]

        # TODO this part should check for incoherent results.

        measure_temp = []
        for begin_temp, end_temp in zip(event_begin_temp, event_end_temp):
            measure_temp.append([begin_temp, end_temp])
        session_interval[type_of_event] = measure_temp
    return session_interval


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


def kinematics_rachis_calculation(points, points_ind, points_name, general_information):
    # -----------------------------------------------------------------------------------
    # Extraction of the information from general information for increase readability later
    function_lower_kinematic = general_information['model_lower_limb']
    function_rachis = general_information['model_rachis']
    # ____________________________________________________________
    # Extraction from the function defined in general information
    [segment_lomb, segment_thor] = function_rachis(
        points, points_ind, general_information)

    [segment_foot, segment_tibia, segment_thigh,
        segment_pelvis] = function_lower_kinematic(points, points_name, general_information)

    full_segment = list([segment_foot, segment_tibia, segment_thigh,
                         segment_pelvis, segment_lomb, segment_thor])

    # Creation of a the external force (here no forces)
    phi_zeros = lbmc.HomogeneousMatrix.fromHomo(
        np.zeros((4, 4, full_segment[0].u.shape[1])))
    phi_ext = [phi_zeros, phi_zeros, phi_zeros,
               phi_zeros, phi_zeros, phi_zeros]
    # Name of the joint
    name_Joint = ['Ankle', 'Knee', 'Hip',
                  'Sacro_Lombaire', 'Lombo_thoracique', ]
    # Euler sequences associated to each name_Joint
    name_rot = ['zxy', 'zxy', 'zxy', 'zxy', 'zxy']
    # Point of calcul of the Moment and Force
    point_limb = [segment_tibia.get_distal_frame_glob(),
                  segment_thigh.get_distal_frame_glob(),
                  segment_thigh.get_proximal_frame_glob(),
                  segment_pelvis.get_proximal_frame_glob(),
                  segment_lomb.get_proximal_frame_glob()]
    # Frame of expression of Moment and Force if not in JCS
    frame_limb = [segment_tibia.Tdist,
                  segment_thigh.Tdist,
                  segment_thigh.Tprox,
                  segment_pelvis.Tprox,
                  segment_lomb.Tprox]

    multiseg = lbmc.KinematicChain(full_segment, phi_ext,
                                   name_Joint, name_rot,
                                   point_limb, frame_limb,
                                   'osef')
    # TODO c3d generation from here in order to be able to visualize the data
    # TODO add export of the position of the pelvis
    return multiseg, full_segment


def generate_c3d_with_model_and_date(acq, name_file_export, full_segment, multisegment, general_information):
    points, points_name, points_ind = snip.get_points_ezc3d(acq)
    # As we are always working before in mm we have to get back to m
    correction_factor = 1

    # Data necessary to correction of the different orientation
    vector_equivalent_sign_new, vector_equivalent_new = reorientation_vector_definition(
        points, points_ind, general_information)
    [sign_X, sign_Y, sign_Z] = vector_equivalent_sign_new
    [pos_X, pos_Y, pos_Z] = vector_equivalent_new

    # copy des infos pour les points
    new_list = points_name.copy()
    new_array = acq['data']['points']*correction_factor
    nb_frame = acq['data']['points'].shape[2]

    for segment in full_segment:
        name_segment = segment.segment_name
        for ind_rm in range(len(segment.nm_list)):
            name_marker = name_segment + str(ind_rm)
            print(name_marker)

            new_list.append(name_marker)
            new_point = np.zeros((4, 1, nb_frame))

            temp = np.dot(
                segment.nm_list[ind_rm].T, segment.Q)*correction_factor
            new_point[pos_X, 0, :] = sign_X*temp[0, :]
            new_point[pos_Y, 0, :] = sign_Y*temp[1, :]
            new_point[pos_Z, 0, :] = sign_Z*temp[2, :]
            new_point[3, 0, :] = 1

            new_array = np.append(new_array, new_point, axis=1)
        list_point_to_add = [segment.rp+0.1*segment.u,
                             segment.rp, segment.rd, segment.rd+0.1*segment.w]
        list_name = ['u', 'rp', 'rd', 'w']

        for ind_point, point in enumerate(list_point_to_add):
            name_point = list_name[ind_point] + '_'+name_segment
            print(name_point)
            new_list.append(name_point)
            new_point = np.zeros((4, 1, nb_frame))
            temp = point * correction_factor
            new_point[pos_X, 0, :] = sign_X*temp[0, :]
            new_point[pos_Y, 0, :] = sign_Y*temp[1, :]
            new_point[pos_Z, 0, :] = sign_Z*temp[2, :]
            new_point[3, 0, :] = 1

            new_array = np.append(new_array, new_point, axis=1)

        homo_segment_rel_frame = segment.Tprox*segment.corr_prox

        Or = homo_segment_rel_frame.T_homo[:, 3, :]
        X = Or+0.1*homo_segment_rel_frame.T_homo[:, 0, :]
        Y = Or+0.1*homo_segment_rel_frame.T_homo[:, 1, :]
        Z = Or+0.1*homo_segment_rel_frame.T_homo[:, 2, :]

        list_point_to_add = [Or, X, Y, Z]
        list_name = ['Or', 'X', 'Y', 'Z']

        for ind_point, point in enumerate(list_point_to_add):
            name_point = list_name[ind_point] + '_'+name_segment
            print(name_point)
            new_list.append(name_point)
            new_point = np.zeros((4, 1, nb_frame))
            temp = point * correction_factor
            new_point[pos_X, 0, :] = sign_X*temp[0, :]
            new_point[pos_Y, 0, :] = sign_Y*temp[1, :]
            new_point[pos_Z, 0, :] = sign_Z*temp[2, :]
            new_point[3, 0, :] = 1

            vector_equivalent_sign_new = [sign_X, sign_Y, sign_Z]
            vector_equivalent_new = [pos_X, pos_Y, pos_Z]
            # vector_equivalent=[0, 2, 1], vector_sign=[1, 1, -1])
            new_array = np.append(new_array, new_point, axis=1)

    for joint in multisegment.euler_rel.keys():
        point = multisegment.euler_rel[joint]
        name_point = joint
        print(name_point)
        new_list.append(name_point)
        new_point = np.zeros((4, 1, nb_frame))
        new_point[0, 0, :] = point[0, :]
        new_point[1, 0, :] = -point[2, :]
        new_point[2, 0, :] = point[1, :]
        new_point[3, 0, :] = 1
        new_array = np.append(new_array, new_point, axis=1)

    c3d = ezc3d.c3d()
    # Fill it with random data
    c3d = acq
    c3d['parameters']['POINT']['LABELS']['value'] = new_list
    c3d['parameters']['POINT']['DESCRIPTIONS']['value'] = new_list.copy()
    temp_residuals = np.zeros((1, new_array.shape[1], new_array.shape[2]))
    temp_residuals[0, :acq['data']['meta_points']['residuals'].shape[1],
                   :] = acq['data']['meta_points']['residuals']
    old_camera_mask = acq['data']['meta_points']['camera_masks']
    temp_camera_mask = np.zeros(
        (old_camera_mask.shape[0], new_array.shape[1], old_camera_mask.shape[2]))
    temp_camera_mask[:, :, :] = False
    temp_camera_mask[:, :acq['data']['meta_points']
                     ['residuals'].shape[1], :] = old_camera_mask

    c3d['data']['points'] = new_array
    c3d['data']['analogs'] = -acq['data']['analogs']
    c3d['data']['meta_points']['residuals'] = temp_residuals
    c3d['data']['meta_points']['camera_masks'] = temp_camera_mask.astype(
        dtype=bool)

    c3d.write(name_file_export)
