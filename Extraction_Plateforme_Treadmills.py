import numbers
import pdb
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import snippet_ezc3d as snip
from point_extraction import reorientation_vector_definition
import ezc3d
from Kinetics_LBMC import HomogeneousMatrix


def plateforme_extraction_treadmills_massue(acq, general_information, value_threshold=30):
    # Extraction des informations de base
    # Analog frequency
    coeff_analogs = dict()
    coeff_analogs['Fx1'] = 147.9815
    coeff_analogs['Fy1'] = -148.6967
    coeff_analogs['Fz1'] = 633.7939
    coeff_analogs['Fx2'] = 149.138
    coeff_analogs['Fy2'] = -149.8195
    coeff_analogs['Fz2'] = 633.8341
    coeff_analogs['Fx3'] = 148.648
    coeff_analogs['Fy3'] = -148.2162
    coeff_analogs['Fz3'] = 631.7918
    coeff_analogs['Fx4'] = 149.1091
    coeff_analogs['Fy4'] = -149.8644
    coeff_analogs['Fz4'] = 634.4775
    coeff_analogs['Fx5'] = -149.8801
    coeff_analogs['Fy5'] = 150.3352
    coeff_analogs['Fz5'] = 634.2763
    coeff_analogs['Fx6'] = -149.7477
    coeff_analogs['Fy6'] = 148.907
    coeff_analogs['Fz6'] = 635.5663
    coeff_analogs['Fx7'] = -149.7903
    coeff_analogs['Fy7'] = 149.0624
    coeff_analogs['Fz7'] = 635.93
    coeff_analogs['Fx8'] = -149.9318
    coeff_analogs['Fy8'] = 149.9295
    coeff_analogs['Fz8'] = 633.9144
    # Analog extraction
    analog_data = acq['data']['analogs']
    analogs_names = acq['parameters']['ANALOG']['LABELS']['value']

    analog_frq = acq['header']['analogs']['frame_rate']
    point_frq = acq['header']['points']['frame_rate']
    analogs_ind = dict()
    for index_analog, name_analog in enumerate(analogs_names):
        analogs_ind[name_analog] = index_analog

    # Force plateform extraction (COP/force moment at COP)?
    origin = acq['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
    corners = acq['parameters']['FORCE_PLATFORM']['CORNERS']['value']
    list_param = ['Fx', 'Fy', 'Fz']

    nb_plateform = origin.shape[1]

    force_plateforms = dict()

    for ind_plateform in range(nb_plateform):
        # found the name of the analog parameters
        list_key_analog = analogs_ind.keys()
        list_name_analog = []
        string_plateforme = str(ind_plateform+1)
        nb_point_analog = analog_data.shape[2]
        # or name_param in list_param:
        #    for name_analog in list_key_analog:
        #        if name_param.lower() in name_analog.lower() and\
        #                string_plateforme in name_analog.lower():
        #            list_name_analog.append(name_analog)

        # if len(list_name_analog) > len(list_param):
        #    temp = [name for name in list_name_analog if (
        #        'fp' in name.lower() or 'pf' in name.lower())]
        #    list_name_analog = temp
        corners_analog = np.zeros((3, 4, analog_data.shape[2]))
        for ind_corner in range(4):
            number_analog = ind_plateform*4 + ind_corner + 1
            print(number_analog)
            corners_analog[0, ind_corner, :] = coeff_analogs['Fx'+str(number_analog)] *\
                analog_data[0, analogs_ind['Fx'+str(number_analog)], :]
            corners_analog[1, ind_corner, :] = coeff_analogs['Fy'+str(number_analog)] *\
                analog_data[0, analogs_ind['Fy'+str(number_analog)], :]
            corners_analog[2, ind_corner, :] = coeff_analogs['Fz'+str(number_analog)] *\
                analog_data[0, analogs_ind['Fz'+str(number_analog)], :]

        force_plateform_temp = dict()

        name_in_data = list_name_analog
        # corner_1 = np.array(corners[ind_plateform*12:ind_plateform*12+3])
        # corner_2 = np.array(corners[ind_plateform*12+3:ind_plateform*12+6])
        # corner_3 = np.array(corners[ind_plateform*12+6:ind_plateform*12+9])
        # corner_4 = np.array(corners[ind_plateform*12+9:ind_plateform*12+12])

        corner_1 = corners[:, 0, ind_plateform]
        corner_2 = corners[:, 1, ind_plateform]
        corner_3 = corners[:, 2, ind_plateform]
        corner_4 = corners[:, 3, ind_plateform]

        Y_platform = ((corner_1+corner_4)/2-(corner_2+corner_3)/2) / \
            np.linalg.norm((corner_1+corner_4)/2-(corner_2+corner_3)/2)
        X_platform = ((corner_1+corner_2)/2-(corner_3+corner_4)/2) / \
            np.linalg.norm((corner_1+corner_4)/2-(corner_2+corner_3)/2)
        Z_platform = np.cross(X_platform, Y_platform)
        Z_platform = Z_platform/np.linalg.norm(Z_platform)
        Y_platform = np.cross(Z_platform, X_platform,
                              axisa=0, axisb=0, axisc=0)
        # Extraction of the axis direction and orientation
        X_pos = np.argmax(abs(X_platform))
        Y_pos = np.argmax(abs(Y_platform))
        Z_pos = np.argmax(abs(Z_platform))
        position_axis = [X_pos, Y_pos, Z_pos]
        position_name = ['x', 'y', 'z']

        X_sign = np.sign(X_platform[X_pos])
        Y_sign = np.sign(Y_platform[Y_pos])
        Z_sign = np.sign(Z_platform[Z_pos])
        mult_sign = np.array([X_sign, Y_sign, Z_sign])
        # ATTENTION TODO : On met en mode ISB alors que on l'est toujours donc faut pas le faire.

        # force_plateform_temp['origin'] = np.array(
        #    origin[ind_plateform*3:(ind_plateform+1)*3])*mult_sign
        force_plateform_temp['origin'] = origin[:, ind_plateform]  # *mult_sign

        mid_corner = (corner_1+corner_2+corner_3+corner_4)/4

        force_plateform_temp['center_global'] = mid_corner
        # Here we do not need the true center everything is calculated at each center
        # force_plateform_temp['origin_global'] = force_plateform_temp['origin']+mid_corner
        force_plateform_temp['origin_global'] = mid_corner

        # for name_global, param_in_data in zip(list_param, name_in_data):
        #    if 'x' in param_in_data.lower():
        #        sign = X_sign
        #        new_direction = position_name[position_axis[0]]
        #        final_test_name = name_global.replace('x', new_direction)
        #    elif 'y' in param_in_data.lower():
        #        sign = Y_sign
        #        new_direction = position_name[position_axis[1]]
        #        final_test_name = name_global.replace('y', new_direction)
        #    elif 'z' in param_in_data.lower():
        #        sign = Z_sign
        #        new_direction = position_name[position_axis[2]]
        #        final_test_name = name_global.replace('z', new_direction)
        #    force_plateform_temp[final_test_name] = sign * \
        #        analog_data[:, analogs_ind[param_in_data], :]
        #    # Faire une transmutation

        # force_plateform_name = 'Fp'+str(ind_plateform+1)
        # nbr_analog = force_plateform_temp['Fz'].shape[1]
        # Initialisation
        # name_FCOP = [param+'_COP' for param in list_param if 'F' in param]
        # name_MCOP = [param+'_COP' for param in list_param if 'M' in param]

        # for ind in name_FCOP+name_MCOP:
        # force_plateform_temp[ind] = np.zeros(nbr_analog)

        total_force = np.sum(corners_analog[:, :, :], axis=1)
        # Mask for vertical forces above 5 newton
        mask = np.abs(total_force[2, :]) < value_threshold
        corners_analog[:, :, mask] = 0

        # Filter parameter
        Wn = 12/(analog_frq/2)
        b, a = signal.butter(4, Wn, 'lowpass', analog=False, output='ba')
        corner_analogs_filtered = signal.filtfilt(b, a, corners_analog)
        # for ind_corner in range(4):
        #    for ind_XYZ in range(3):
        #        plt.plot(corners_analog[ind_XYZ, ind_corner, :])
        #        plt.plot(corner_analogs_filtered[ind_XYZ, ind_corner, :])
        #        plt.show()

        # Transform from corners to origins
        moment_origin = np.zeros((3, nb_point_analog))
        force_origin = np.zeros((3, nb_point_analog))
        for ind_corner in range(4):
            # B=A+BA vect R

            vector = corners[:, ind_corner, ind_plateform] - \
                force_plateform_temp['origin_global']
            vector = vector[:, np.newaxis]
            moment_temp = np.cross(
                vector, corner_analogs_filtered[:, ind_corner, :], axisa=0, axisb=0, axisc=0)
            moment_origin += moment_temp
            force_origin += corner_analogs_filtered[:, ind_corner, :]

        Xor = force_plateform_temp['origin_global'][0]
        Yor = force_plateform_temp['origin_global'][1]
        # ??Her it is - because it seems that the origin is expressed in the global frame
        Zor = force_plateform_temp['origin_global'][2]

        # Calcul COP
        mask = np.abs(force_origin[2, :]) > value_threshold
        # Mx_temp = Mx[mask]
        # My_temp = My[mask]
        # Mz_temp = Mz[mask]
        # Fx_temp = Fx[mask]
        # Fy_temp = Fy[mask]
        # Fz_temp = Fz[mask]
        Mx_temp = moment_origin[0, mask]
        My_temp = moment_origin[1, mask]
        Mz_temp = moment_origin[2, mask]
        Fx_temp = force_origin[0, mask]
        Fy_temp = force_origin[1, mask]
        Fz_temp = force_origin[2, mask]
        # initialisation
        X_CoP_temp = np.zeros(nb_point_analog)
        Y_CoP_temp = np.zeros(nb_point_analog)
        Z_CoP_temp = np.zeros(nb_point_analog)

        Fx_CoP_temp = np.zeros(nb_point_analog)
        Fy_CoP_temp = np.zeros(nb_point_analog)
        Fz_CoP_temp = np.zeros(nb_point_analog)

        Mx_CoP_temp = np.zeros(nb_point_analog)
        My_CoP_temp = np.zeros(nb_point_analog)
        Mz_CoP_temp = np.zeros(nb_point_analog)

        # Calcul
        Fx_CoP_temp[mask] = Fx_temp
        Fy_CoP_temp[mask] = Fy_temp
        Fz_CoP_temp[mask] = Fz_temp

        X_CoP_temp[mask] = (-My_temp+Fz_temp*Xor-Fx_temp*Zor)/Fz_temp
        Y_CoP_temp[mask] = (Mx_temp+Fz_temp*Yor-Fy_temp*Zor)/Fz_temp

        Mz_CoP_temp[mask] = (Mz_temp + Fy_temp*(Xor-X_CoP_temp[mask])
                             - Fx_temp*(Yor - Y_CoP_temp[mask]))  # *mask

        CoP = np.array([X_CoP_temp, Y_CoP_temp, Z_CoP_temp])
        M_CoP = np.array([Mx_CoP_temp, My_CoP_temp, Mz_CoP_temp])
        F_CoP = np.array([Fx_CoP_temp, Fy_CoP_temp, Fz_CoP_temp])
        M = np.array(
            [moment_origin[0, :], moment_origin[1, :], moment_origin[2, :]])
        F = np.array(
            [force_origin[0, :], force_origin[1, :], force_origin[2, :]])
        Or = np.array([Xor, Yor, Zor])

        force_plateform_temp['CoP'] = CoP[:,
                                          0::int(round(analog_frq/point_frq))]
        force_plateform_temp['M_CoP'] = M_CoP[:,
                                              0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F_CoP'] = F_CoP[:,
                                              0::int(round(analog_frq/point_frq))]

        force_plateform_temp['origin_plateform'] = np.tile(
            Or[:, np.newaxis], (1, force_plateform_temp['CoP'].shape[1]))

        force_plateform_temp['M'] = M[:, 0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F'] = F[:, 0::int(round(analog_frq/point_frq))]

        # Creation of the ISB vrsion of COP FORCE and Moment
        points_temp, points_name, points_ind = snip.get_points_ezc3d(acq)

        # vector_sign, vector_equivalent = reorientation_vector_definition(
        #    points_temp, points_ind, general_information)
        CoP_ISB = np.zeros(force_plateform_temp['CoP'].shape)
        F_ISB = np.zeros(force_plateform_temp['F_CoP'].shape)
        M_ISB = np.zeros(force_plateform_temp['M_CoP'].shape)
        # CoP_ISB[0, :] = vector_sign[0] * \
        #    force_plateform_temp['CoP'][vector_equivalent[0]]
        # CoP_ISB[1, :] = vector_sign[1] * \
        #    force_plateform_temp['CoP'][vector_equivalent[1]]
        # CoP_ISB[2, :] = vector_sign[2] * \
        #    force_plateform_temp['CoP'][vector_equivalent[2]]
        #
        # F_ISB[0, :] = vector_sign[0] * \
        #    force_plateform_temp['F_CoP'][vector_equivalent[0]]
        # F_ISB[1, :] = vector_sign[1] * \
        #    force_plateform_temp['F_CoP'][vector_equivalent[1]]
        # F_ISB[2, :] = vector_sign[2] * \
        #    force_plateform_temp['F_CoP'][vector_equivalent[2]]
        #
        # M_ISB[0, :] = vector_sign[0] * \
        #    force_plateform_temp['M_CoP'][vector_equivalent[0]]
        # M_ISB[1, :] = vector_sign[1] * \
        #    force_plateform_temp['M_CoP'][vector_equivalent[1]]
        # M_ISB[2, :] = vector_sign[2] * \
        #    force_plateform_temp['M_CoP'][vector_equivalent[2]]

        CoP_ISB[0, :] = force_plateform_temp['CoP'][0, :]
        CoP_ISB[1, :] = force_plateform_temp['CoP'][2, :]
        CoP_ISB[2, :] = -force_plateform_temp['CoP'][1, :]

        F_ISB[0, :] = force_plateform_temp['F_CoP'][0, :]
        F_ISB[1, :] = force_plateform_temp['F_CoP'][2, :]
        F_ISB[2, :] = -force_plateform_temp['F_CoP'][1, :]

        M_ISB[0, :] = force_plateform_temp['M_CoP'][0, :]
        M_ISB[1, :] = force_plateform_temp['M_CoP'][2, :]
        M_ISB[2, :] = -force_plateform_temp['M_CoP'][1, :]

        force_plateform_temp['CoP_ISB'] = CoP_ISB
        force_plateform_temp['F_ISB'] = F_ISB
        force_plateform_temp['M_ISB'] = M_ISB

        force_plateforms['force_plateform_' +
                         str(ind_plateform+1)] = force_plateform_temp

    return force_plateforms


def plateforme_extraction_treadmills_bron(acq, general_information, value_threshold=30):
    # Extraction des informations de base
    # Analog frequency
    # traduction
    traduction = dict()
    traduction['Fx1'] = 'Channel_01'
    traduction['Fy1'] = 'Channel_02'
    traduction['Fz1'] = 'Channel_03'
    traduction['Fx2'] = 'Channel_04'
    traduction['Fy2'] = 'Channel_05'
    traduction['Fz2'] = 'Channel_06'
    traduction['Fx3'] = 'Channel_07'
    traduction['Fy3'] = 'Channel_08'
    traduction['Fz3'] = 'Channel_09'
    traduction['Fx4'] = 'Channel_10'
    traduction['Fy4'] = 'Channel_11'
    traduction['Fz4'] = 'Channel_12'
    traduction['Fx5'] = 'Channel_20'
    traduction['Fy5'] = 'Channel_21'
    traduction['Fz5'] = 'Channel_22'
    traduction['Fx6'] = 'Channel_17'
    traduction['Fy6'] = 'Channel_18'
    traduction['Fz6'] = 'Channel_19'
    traduction['Fx7'] = 'Channel_26'
    traduction['Fy7'] = 'Channel_27'
    traduction['Fz7'] = 'Channel_28'
    traduction['Fx8'] = 'Channel_23'
    traduction['Fy8'] = 'Channel_24'
    traduction['Fz8'] = 'Channel_25'

    coeff_analogs = dict()
    coeff_analogs['Fx1'] = 147.863374
    coeff_analogs['Fy1'] = -147.710487
    coeff_analogs['Fz1'] = -634.477508
    coeff_analogs['Fx2'] = 148.798452
    coeff_analogs['Fy2'] = -148.595034
    coeff_analogs['Fz2'] = -639.386189
    coeff_analogs['Fx3'] = 148.445038
    coeff_analogs['Fy3'] = -147.547031
    coeff_analogs['Fz3'] = -634.397006
    coeff_analogs['Fx4'] = 148.570749
    coeff_analogs['Fy4'] = -147.961826
    coeff_analogs['Fz4'] = -638.234292

    coeff_analogs['Fx5'] = -148.104265
    coeff_analogs['Fy5'] = 147.642881
    coeff_analogs['Fz5'] = -633.994801
    coeff_analogs['Fx6'] = -148.904805
    coeff_analogs['Fy6'] = 148.185469
    coeff_analogs['Fz6'] = -636.172785
    coeff_analogs['Fx7'] = -148.423006
    coeff_analogs['Fy7'] = 148.110846
    coeff_analogs['Fz7'] = -635.525898
    coeff_analogs['Fx8'] = -148.5553
    coeff_analogs['Fy8'] = 148.59945
    coeff_analogs['Fz8'] = -633.19192
    # Analog extraction
    analog_data = acq['data']['analogs']
    analogs_names = acq['parameters']['ANALOG']['LABELS']['value']

    analog_frq = acq['header']['analogs']['frame_rate']
    point_frq = acq['header']['points']['frame_rate']
    analogs_ind = dict()
    for index_analog, name_analog in enumerate(analogs_names):
        analogs_ind[name_analog] = index_analog

    # Force plateform extraction (COP/force moment at COP)?
    origin = acq['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
    corners = acq['parameters']['FORCE_PLATFORM']['CORNERS']['value']
    list_param = ['Fx', 'Fy', 'Fz']

    nb_plateform = origin.shape[1]

    force_plateforms = dict()

    for ind_plateform in range(nb_plateform):
        # found the name of the analog parameters
        list_key_analog = analogs_ind.keys()
        list_name_analog = []
        string_plateforme = str(ind_plateform+1)
        nb_point_analog = analog_data.shape[2]
        # or name_param in list_param:
        #    for name_analog in list_key_analog:
        #        if name_param.lower() in name_analog.lower() and\
        #                string_plateforme in name_analog.lower():
        #            list_name_analog.append(name_analog)

        # if len(list_name_analog) > len(list_param):
        #    temp = [name for name in list_name_analog if (
        #        'fp' in name.lower() or 'pf' in name.lower())]
        #    list_name_analog = temp
        corners_analog = np.zeros((3, 4, analog_data.shape[2]))
        for ind_corner in range(4):
            number_analog = ind_plateform*4 + ind_corner + 1
            print(number_analog)
            corners_analog[0, ind_corner, :] = coeff_analogs['Fx'+str(number_analog)] *\
                analog_data[0,
                            analogs_ind[traduction['Fx'+str(number_analog)]], :]
            corners_analog[1, ind_corner, :] = coeff_analogs['Fy'+str(number_analog)] *\
                analog_data[0,
                            analogs_ind[traduction['Fy'+str(number_analog)]], :]
            corners_analog[2, ind_corner, :] = coeff_analogs['Fz'+str(number_analog)] *\
                analog_data[0,
                            analogs_ind[traduction['Fz'+str(number_analog)]], :]

        force_plateform_temp = dict()

        name_in_data = list_name_analog
        # corner_1 = np.array(corners[ind_plateform*12:ind_plateform*12+3])
        # corner_2 = np.array(corners[ind_plateform*12+3:ind_plateform*12+6])
        # corner_3 = np.array(corners[ind_plateform*12+6:ind_plateform*12+9])
        # corner_4 = np.array(corners[ind_plateform*12+9:ind_plateform*12+12])

        corner_1 = corners[:, 0, ind_plateform]
        corner_2 = corners[:, 1, ind_plateform]
        corner_3 = corners[:, 2, ind_plateform]
        corner_4 = corners[:, 3, ind_plateform]

        Y_platform = ((corner_1+corner_4)/2-(corner_2+corner_3)/2) / \
            np.linalg.norm((corner_1+corner_4)/2-(corner_2+corner_3)/2)
        X_platform = ((corner_1+corner_2)/2-(corner_3+corner_4)/2) / \
            np.linalg.norm((corner_1+corner_4)/2-(corner_2+corner_3)/2)
        Z_platform = np.cross(X_platform, Y_platform)
        Z_platform = Z_platform/np.linalg.norm(Z_platform)
        Y_platform = np.cross(Z_platform, X_platform,
                              axisa=0, axisb=0, axisc=0)
        # Extraction of the axis direction and orientation
        X_pos = np.argmax(abs(X_platform))
        Y_pos = np.argmax(abs(Y_platform))
        Z_pos = np.argmax(abs(Z_platform))
        position_axis = [X_pos, Y_pos, Z_pos]
        position_name = ['x', 'y', 'z']

        X_sign = np.sign(X_platform[X_pos])
        Y_sign = np.sign(Y_platform[Y_pos])
        Z_sign = np.sign(Z_platform[Z_pos])
        mult_sign = np.array([X_sign, Y_sign, Z_sign])
        # ATTENTION TODO : On met en mode ISB alors que on l'est toujours donc faut pas le faire.

        # force_plateform_temp['origin'] = np.array(
        #    origin[ind_plateform*3:(ind_plateform+1)*3])*mult_sign
        force_plateform_temp['origin'] = origin[:, ind_plateform]  # *mult_sign

        mid_corner = (corner_1+corner_2+corner_3+corner_4)/4

        force_plateform_temp['center_global'] = mid_corner
        # Here we do not need the true center everything is calculated at each center
        # force_plateform_temp['origin_global'] = force_plateform_temp['origin']+mid_corner
        force_plateform_temp['origin_global'] = mid_corner

        # for name_global, param_in_data in zip(list_param, name_in_data):
        #    if 'x' in param_in_data.lower():
        #        sign = X_sign
        #        new_direction = position_name[position_axis[0]]
        #        final_test_name = name_global.replace('x', new_direction)
        #    elif 'y' in param_in_data.lower():
        #        sign = Y_sign
        #        new_direction = position_name[position_axis[1]]
        #        final_test_name = name_global.replace('y', new_direction)
        #    elif 'z' in param_in_data.lower():
        #        sign = Z_sign
        #        new_direction = position_name[position_axis[2]]
        #        final_test_name = name_global.replace('z', new_direction)
        #    force_plateform_temp[final_test_name] = sign * \
        #        analog_data[:, analogs_ind[param_in_data], :]
        #    # Faire une transmutation

        # force_plateform_name = 'Fp'+str(ind_plateform+1)
        # nbr_analog = force_plateform_temp['Fz'].shape[1]
        # Initialisation
        # name_FCOP = [param+'_COP' for param in list_param if 'F' in param]
        # name_MCOP = [param+'_COP' for param in list_param if 'M' in param]

        # for ind in name_FCOP+name_MCOP:
        # force_plateform_temp[ind] = np.zeros(nbr_analog)

        total_force = np.sum(corners_analog[:, :, :], axis=1)
        # Mask for vertical forces above 5 newton
        mask = np.abs(total_force[2, :]) < value_threshold
        corners_analog[:, :, mask] = 0

        # Filter parameter
        Wn = 12/(analog_frq/2)
        b, a = signal.butter(4, Wn, 'lowpass', analog=False, output='ba')
        corner_analogs_filtered = signal.filtfilt(b, a, corners_analog)
        # for ind_corner in range(4):
        #    for ind_XYZ in range(3):
        #        plt.plot(corners_analog[ind_XYZ, ind_corner, :])
        #        plt.plot(corner_analogs_filtered[ind_XYZ, ind_corner, :])
        #        plt.show()

        # Transform from corners to origins
        moment_origin = np.zeros((3, nb_point_analog))
        force_origin = np.zeros((3, nb_point_analog))
        for ind_corner in range(4):
            # B=A+BA vect R

            vector = corners[:, ind_corner, ind_plateform] - \
                force_plateform_temp['origin_global']
            vector = vector[:, np.newaxis]
            moment_temp = np.cross(
                vector, corner_analogs_filtered[:, ind_corner, :], axisa=0, axisb=0, axisc=0)
            moment_origin += moment_temp
            force_origin += corner_analogs_filtered[:, ind_corner, :]

        Xor = force_plateform_temp['origin_global'][0]
        Yor = force_plateform_temp['origin_global'][1]
        # ??Her it is - because it seems that the origin is expressed in the global frame
        Zor = force_plateform_temp['origin_global'][2]

        # Calcul COP
        mask = np.abs(force_origin[2, :]) > value_threshold
        # Mx_temp = Mx[mask]
        # My_temp = My[mask]
        # Mz_temp = Mz[mask]
        # Fx_temp = Fx[mask]
        # Fy_temp = Fy[mask]
        # Fz_temp = Fz[mask]
        Mx_temp = moment_origin[0, mask]
        My_temp = moment_origin[1, mask]
        Mz_temp = moment_origin[2, mask]
        Fx_temp = force_origin[0, mask]
        Fy_temp = force_origin[1, mask]
        Fz_temp = force_origin[2, mask]
        # initialisation
        X_CoP_temp = np.zeros(nb_point_analog)
        Y_CoP_temp = np.zeros(nb_point_analog)
        Z_CoP_temp = np.zeros(nb_point_analog)

        Fx_CoP_temp = np.zeros(nb_point_analog)
        Fy_CoP_temp = np.zeros(nb_point_analog)
        Fz_CoP_temp = np.zeros(nb_point_analog)

        Mx_CoP_temp = np.zeros(nb_point_analog)
        My_CoP_temp = np.zeros(nb_point_analog)
        Mz_CoP_temp = np.zeros(nb_point_analog)

        # Calcul
        Fx_CoP_temp[mask] = Fx_temp
        Fy_CoP_temp[mask] = Fy_temp
        Fz_CoP_temp[mask] = Fz_temp

        X_CoP_temp[mask] = (-My_temp+Fz_temp*Xor-Fx_temp*Zor)/Fz_temp
        Y_CoP_temp[mask] = (Mx_temp+Fz_temp*Yor-Fy_temp*Zor)/Fz_temp

        Mz_CoP_temp[mask] = (Mz_temp + Fy_temp*(Xor-X_CoP_temp[mask])
                             - Fx_temp*(Yor - Y_CoP_temp[mask]))  # *mask

        CoP = np.array([X_CoP_temp, Y_CoP_temp, Z_CoP_temp])
        M_CoP = np.array([Mx_CoP_temp, My_CoP_temp, Mz_CoP_temp])
        F_CoP = np.array([Fx_CoP_temp, Fy_CoP_temp, Fz_CoP_temp])
        M = np.array(
            [moment_origin[0, :], moment_origin[1, :], moment_origin[2, :]])
        F = np.array(
            [force_origin[0, :], force_origin[1, :], force_origin[2, :]])
        Or = np.array([Xor, Yor, Zor])

        force_plateform_temp['CoP'] = CoP[:,
                                          0::int(round(analog_frq/point_frq))]
        force_plateform_temp['M_CoP'] = M_CoP[:,
                                              0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F_CoP'] = F_CoP[:,
                                              0::int(round(analog_frq/point_frq))]

        force_plateform_temp['origin_plateform'] = np.tile(
            Or[:, np.newaxis], (1, force_plateform_temp['CoP'].shape[1]))

        force_plateform_temp['M'] = M[:, 0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F'] = F[:, 0::int(round(analog_frq/point_frq))]

        # Creation of the ISB vrsion of COP FORCE and Moment
        points_temp, points_name, points_ind = snip.get_points_ezc3d(acq)

        # vector_sign, vector_equivalent = reorientation_vector_definition(
        #    points_temp, points_ind, general_information)
        CoP_ISB = np.zeros(force_plateform_temp['CoP'].shape)
        F_ISB = np.zeros(force_plateform_temp['F_CoP'].shape)
        M_ISB = np.zeros(force_plateform_temp['M_CoP'].shape)
        # CoP_ISB[0, :] = vector_sign[0] * \
        #    force_plateform_temp['CoP'][vector_equivalent[0]]
        # CoP_ISB[1, :] = vector_sign[1] * \
        #    force_plateform_temp['CoP'][vector_equivalent[1]]
        # CoP_ISB[2, :] = vector_sign[2] * \
        #    force_plateform_temp['CoP'][vector_equivalent[2]]
        #
        # F_ISB[0, :] = vector_sign[0] * \
        #    force_plateform_temp['F_CoP'][vector_equivalent[0]]
        # F_ISB[1, :] = vector_sign[1] * \
        #    force_plateform_temp['F_CoP'][vector_equivalent[1]]
        # F_ISB[2, :] = vector_sign[2] * \
        #    force_plateform_temp['F_CoP'][vector_equivalent[2]]
        #
        # M_ISB[0, :] = vector_sign[0] * \
        #    force_plateform_temp['M_CoP'][vector_equivalent[0]]
        # M_ISB[1, :] = vector_sign[1] * \
        #    force_plateform_temp['M_CoP'][vector_equivalent[1]]
        # M_ISB[2, :] = vector_sign[2] * \
        #    force_plateform_temp['M_CoP'][vector_equivalent[2]]

        CoP_ISB[0, :] = force_plateform_temp['CoP'][0, :]
        CoP_ISB[1, :] = force_plateform_temp['CoP'][2, :]
        CoP_ISB[2, :] = -force_plateform_temp['CoP'][1, :]

        F_ISB[0, :] = force_plateform_temp['F_CoP'][0, :]
        F_ISB[1, :] = force_plateform_temp['F_CoP'][2, :]
        F_ISB[2, :] = -force_plateform_temp['F_CoP'][1, :]

        M_ISB[0, :] = force_plateform_temp['M_CoP'][0, :]
        M_ISB[1, :] = force_plateform_temp['M_CoP'][2, :]
        M_ISB[2, :] = -force_plateform_temp['M_CoP'][1, :]

        force_plateform_temp['CoP_ISB'] = CoP_ISB
        force_plateform_temp['F_ISB'] = F_ISB
        force_plateform_temp['M_ISB'] = M_ISB

        force_plateforms['force_plateform_' +
                         str(ind_plateform+1)] = force_plateform_temp
        import pdb
        pdb.set_trace()
    return force_plateforms


def plateforme_extraction_treadmills_bron_c3d_only(acq, general_information, value_threshold=30):
    # Data to perfom the calculation
    # label_analog = acq['parameters']['ANALOG']['LABELS']['value']
    cal_matrix = acq['parameters']['FORCE_PLATFORM']['CAL_MATRIX']['value']
    origin = acq['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
    corners = acq['parameters']['FORCE_PLATFORM']['CORNERS']['value']
    channels = acq['parameters']['FORCE_PLATFORM']['CHANNEL']['value']
    # Extraction des informations de base
    # Analog frequency
    # Analog extraction
    analog_data = acq['data']['analogs']
    # The analog data are arranged in the same order as indicated in acq['parameters']['FORCE_PLATEFORM']['CHANNEL']['value']
    analogs_names = acq['parameters']['ANALOG']['LABELS']['value']

    analog_frq = acq['header']['analogs']['frame_rate']
    point_frq = acq['header']['points']['frame_rate']
    analogs_ind = dict()
    for index_analog, name_analog in enumerate(analogs_names):
        analogs_ind[name_analog] = index_analog

    # Force plateform extraction (COP/force moment at COP)?
    origin = acq['parameters']['FORCE_PLATFORM']['ORIGIN']['value']
    corners = acq['parameters']['FORCE_PLATFORM']['CORNERS']['value']

    nb_plateform = origin.shape[1]

    force_plateforms = dict()

    for ind_plateform in range(nb_plateform):
        # found the name of the analog parameters
        nb_point_analog = analog_data.shape[2]
        corners_analog = np.zeros((3, 4, analog_data.shape[2]))
        for ind_corner in range(4):
            ind_begin = ind_corner*3
            ind_end = (ind_corner+1)*3

            coeff_analog = np.diag(
                cal_matrix[ind_begin:ind_end, ind_begin:ind_end, ind_plateform])
            # We have to remove 1 here because in python 1 correspond to the indices 0
            channels_corner = channels[ind_begin:ind_end, ind_plateform]-1
            corners_analog[:, ind_corner, :] = coeff_analog[:,
                                                            np.newaxis]*analog_data[0, channels_corner, :]

        force_plateform_temp = dict()

        corner_1 = corners[:, 0, ind_plateform]
        corner_2 = corners[:, 1, ind_plateform]
        corner_3 = corners[:, 2, ind_plateform]
        corner_4 = corners[:, 3, ind_plateform]

        X_platform = ((corner_1+corner_4)/2-(corner_2+corner_3)/2) / \
            np.linalg.norm((corner_1+corner_4)/2-(corner_2+corner_3)/2)
        Y_platform = ((corner_1+corner_2)/2-(corner_3+corner_4)/2) / \
            np.linalg.norm((corner_1+corner_4)/2-(corner_2+corner_3)/2)
        Z_platform = np.cross(X_platform, Y_platform)
        Z_platform = Z_platform/np.linalg.norm(Z_platform)
        Y_platform = np.cross(Z_platform, X_platform,
                              axisa=0, axisb=0, axisc=0)
        Y_platform = Y_platform/np.linalg.norm(Y_platform)
        rot_platform = np.zeros((3, 3))
        rot_platform[0, :] = X_platform
        rot_platform[1, :] = Y_platform
        rot_platform[2, :] = Z_platform
        # Extraction of the axis direction and orientation
        X_pos = np.argmax(abs(X_platform))
        Y_pos = np.argmax(abs(Y_platform))
        Z_pos = np.argmax(abs(Z_platform))
        position_axis = [X_pos, Y_pos, Z_pos]
        position_name = ['x', 'y', 'z']

        X_sign = np.sign(X_platform[X_pos])
        Y_sign = np.sign(Y_platform[Y_pos])
        Z_sign = np.sign(Z_platform[Z_pos])
        mult_sign = np.array([X_sign, Y_sign, Z_sign])
        # ATTENTION TODO : On met en mode ISB alors que on l'est toujours donc faut pas le faire.

        # force_plateform_temp['origin'] = np.array(
        #    origin[ind_plateform*3:(ind_plateform+1)*3])*mult_sign
        force_plateform_temp['origin'] = origin[:, ind_plateform]  # *mult_sign

        mid_corner = (corner_1+corner_2+corner_3+corner_4)/4
        # in order to calculate the CoP at the surface
        mid_corner[Z_pos] = 0

        force_plateform_temp['center_global'] = mid_corner
        # Here we do not need the true center everything is calculated at each center
        # force_plateform_temp['origin_global'] = force_plateform_temp['origin']+mid_corner
        force_plateform_temp['origin_global'] = mid_corner

        # for name_global, param_in_data in zip(list_param, name_in_data):
        #    if 'x' in param_in_data.lower():
        #        sign = X_sign
        #        new_direction = position_name[position_axis[0]]
        #        final_test_name = name_global.replace('x', new_direction)
        #    elif 'y' in param_in_data.lower():
        #        sign = Y_sign
        #        new_direction = position_name[position_axis[1]]
        #        final_test_name = name_global.replace('y', new_direction)
        #    elif 'z' in param_in_data.lower():
        #        sign = Z_sign
        #        new_direction = position_name[position_axis[2]]
        #        final_test_name = name_global.replace('z', new_direction)
        #    force_plateform_temp[final_test_name] = sign * \
        #        analog_data[:, analogs_ind[param_in_data], :]
        #    # Faire une transmutation

        # force_plateform_name = 'Fp'+str(ind_plateform+1)
        # nbr_analog = force_plateform_temp['Fz'].shape[1]
        # Initialisation
        # name_FCOP = [param+'_COP' for param in list_param if 'F' in param]
        # name_MCOP = [param+'_COP' for param in list_param if 'M' in param]

        # for ind in name_FCOP+name_MCOP:
        # force_plateform_temp[ind] = np.zeros(nbr_analog)

        total_force = np.sum(corners_analog[:, :, :], axis=1)
        # Mask for vertical forces above 5 newton
        mask = np.abs(total_force[2, :]) < value_threshold
        corners_analog[:, :, mask] = 0
        # Filter parameter
        Wn = 12/(analog_frq/2)
        b, a = signal.butter(4, Wn, 'lowpass', analog=False, output='ba')
        corner_analogs_filtered = signal.filtfilt(b, a, corners_analog)
        # We remove the value where it is supposed to be 0
        corner_analogs_filtered[:, :, mask] = 0

        # for ind_XYZ in range(3):
        #    plt.plot(np.sum(corner_analogs_filtered[ind_XYZ, :, :], axis=0))
        #    plt.show()
        # for ind_corner in range(4):
        #    for ind_XYZ in range(3):
        #        plt.plot(corners_analog[ind_XYZ, ind_corner, :])
        #        plt.plot(corner_analogs_filtered[ind_XYZ, ind_corner, :])
        #        plt.show()
        # Express corner in force plateform frame
        # Transform from corners to origins
        moment_origin = np.zeros((3, nb_point_analog))
        force_origin = np.zeros((3, nb_point_analog))
        for ind_corner in range(4):
            # B=A+BA vect R
            vector = corners[:, ind_corner, ind_plateform] - \
                force_plateform_temp['origin_global']

            vector_plateform = vector[0]*rot_platform[:, 0] + \
                vector[1]*rot_platform[:, 1] + \
                vector[2]*rot_platform[:, 2]
            vector_plateform = vector_plateform[:, np.newaxis]
            moment_temp = np.cross(
                vector_plateform, corner_analogs_filtered[:, ind_corner, :], axisa=0, axisb=0, axisc=0)
            moment_origin += moment_temp
            force_origin += corner_analogs_filtered[:, ind_corner, :]

        Xor = force_plateform_temp['origin_global'][0]
        Yor = force_plateform_temp['origin_global'][1]
        # ??Her it is - because it seems that the origin is expressed in the global frame
        Zor = force_plateform_temp['origin_global'][2]

        # Calcul COP
        mask = np.abs(force_origin[2, :]) > value_threshold
        # Mx_temp = Mx[mask]
        # My_temp = My[mask]
        # Mz_temp = Mz[mask]
        # Fx_temp = Fx[mask]
        # Fy_temp = Fy[mask]
        # Fz_temp = Fz[mask]
        Mx_temp = moment_origin[0, mask]
        My_temp = moment_origin[1, mask]
        Mz_temp = moment_origin[2, mask]
        Fx_temp = force_origin[0, mask]
        Fy_temp = force_origin[1, mask]
        Fz_temp = force_origin[2, mask]
        # initialisation
        X_CoP_temp = np.zeros(nb_point_analog)
        Y_CoP_temp = np.zeros(nb_point_analog)
        Z_CoP_temp = np.zeros(nb_point_analog)

        Fx_CoP_temp = np.zeros(nb_point_analog)
        Fy_CoP_temp = np.zeros(nb_point_analog)
        Fz_CoP_temp = np.zeros(nb_point_analog)

        Mx_CoP_temp = np.zeros(nb_point_analog)
        My_CoP_temp = np.zeros(nb_point_analog)
        Mz_CoP_temp = np.zeros(nb_point_analog)

        # Calcul
        Fx_CoP_temp[mask] = Fx_temp
        Fy_CoP_temp[mask] = Fy_temp
        Fz_CoP_temp[mask] = Fz_temp

        X_CoP_temp[mask] = (-My_temp+Fz_temp*Xor-Fx_temp*Zor)/Fz_temp
        Y_CoP_temp[mask] = (Mx_temp+Fz_temp*Yor-Fy_temp*Zor)/Fz_temp

        Mz_CoP_temp[mask] = (Mz_temp + Fy_temp*(Xor-X_CoP_temp[mask])
                             - Fx_temp*(Yor - Y_CoP_temp[mask]))  # *mask

        CoP = np.array([X_CoP_temp, Y_CoP_temp, Z_CoP_temp])
        M_CoP = np.array([Mx_CoP_temp, My_CoP_temp, Mz_CoP_temp])
        F_CoP = np.array([Fx_CoP_temp, Fy_CoP_temp, Fz_CoP_temp])
        M = np.array(
            [moment_origin[0, :], moment_origin[1, :], moment_origin[2, :]])
        F = np.array(
            [force_origin[0, :], force_origin[1, :], force_origin[2, :]])
        Or = np.array([Xor, Yor, Zor])

        force_plateform_temp['CoP'] = CoP[:,
                                          0::int(round(analog_frq/point_frq))]
        force_plateform_temp['M_CoP'] = M_CoP[:,
                                              0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F_CoP'] = F_CoP[:,
                                              0::int(round(analog_frq/point_frq))]

        force_plateform_temp['origin_plateform'] = np.tile(
            Or[:, np.newaxis], (1, force_plateform_temp['CoP'].shape[1]))

        force_plateform_temp['M'] = M[:, 0::int(round(analog_frq/point_frq))]
        force_plateform_temp['F'] = F[:, 0::int(round(analog_frq/point_frq))]

        CoP_ISB = np.zeros(force_plateform_temp['CoP'].shape)
        F_ISB = np.zeros(force_plateform_temp['F_CoP'].shape)
        M_ISB = np.zeros(force_plateform_temp['M_CoP'].shape)
        pdb.set_trace()
        CoP_ISB = force_plateform_temp['CoP'][0, :]*X_platform +\
            force_plateform_temp['CoP'][1, :]*Y_platform +\
            force_plateform_temp['CoP'][2, :]*Z_platform
        F_ISB = force_plateform_temp['F_CoP'][0, :]*X_platform +\
            force_plateform_temp['F_CoP'][1, :]*Y_platform +\
            force_plateform_temp['F_CoP'][2, :]*Z_platform
        M_ISB = force_plateform_temp['M_CoP'][0, :]*X_platform +\
            force_plateform_temp['M_CoP'][1, :]*Y_platform +\
            force_plateform_temp['M_CoP'][2, :]*Z_platform

        CoP_ISB[0, :] = force_plateform_temp['CoP'][0, :]
        CoP_ISB[1, :] = force_plateform_temp['CoP'][2, :]
        CoP_ISB[2, :] = -force_plateform_temp['CoP'][1, :]

        F_ISB[0, :] = force_plateform_temp['F_CoP'][0, :]
        F_ISB[1, :] = force_plateform_temp['F_CoP'][2, :]
        F_ISB[2, :] = -force_plateform_temp['F_CoP'][1, :]

        M_ISB[0, :] = force_plateform_temp['M_CoP'][0, :]
        M_ISB[1, :] = force_plateform_temp['M_CoP'][2, :]
        M_ISB[2, :] = -force_plateform_temp['M_CoP'][1, :]

        force_plateform_temp['CoP_ISB'] = CoP_ISB
        force_plateform_temp['F_ISB'] = F_ISB
        force_plateform_temp['M_ISB'] = M_ISB
        force_plateform_temp['Force_Plateform_Frame'] = rot_platform

        force_plateforms['force_plateform_' +
                         str(ind_plateform+1)] = force_plateform_temp
    # Point to add
    # copy des infos pour les points
    new_list = points_name.copy()
    new_array = acq['data']['points']  # *correction_factor
    nb_frame = acq['data']['points'].shape[2]

    for ind_point, (name_point, value_point) in enumerate(virtual_point.items()):
        new_point = np.zeros((4, 1, nb_frame))
        temp = value_point * correction_factor
        new_list.append(name_point)
        new_point[pos_X, 0, :] = sign_X*temp[0, :]
        new_point[pos_Y, 0, :] = sign_Y*temp[1, :]
        new_point[pos_Z, 0, :] = sign_Z*temp[2, :]
        new_point[3, 0, :] = 1
        vector_equivalent_sign_new = [sign_X, sign_Y, sign_Z]
        vector_equivalent_new = [pos_X, pos_Y, pos_Z]
        # vector_equivalent=[0, 2, 1], vector_sign=[1, 1, -1])
        new_array = np.append(new_array, new_point, axis=1)

    # Ajout dans le C3D
    c3d = ezc3d.c3d()
    c3d = acq
    # Copy des informations contenu dans force plateform
    corners = acq['parameters']['FORCE_PLATFORM']['CORNERS']['value']
    for ind_plateforme in range(corners.shape[2]):
        for ind_corner in range(corners.shape[1]):
            c3d.add_parameter("Plateform", "Corner_"+str(ind_plateforme*4+ind_corner+1),
                              acq['parameters']['FORCE_PLATFORM']['CORNERS']['value'][:, ind_corner, ind_plateforme].tolist())

    cal_matrix = acq['parameters']['FORCE_PLATFORM']['CAL_MATRIX']['value']
    pdb.set_trace()
    for ind_plateforme in range(cal_matrix.shape[2]):
        cal_temp = list()
        for ind_cal_mat in range(cal_matrix.shape[1]):
            cal_temp.append(acq['parameters']['FORCE_PLATFORM']['CAL_MATRIX']
                            ['value'][ind_cal_mat, ind_cal_mat, ind_plateforme])
        c3d.add_parameter("Plateform", "CAL_MATRIX_" +
                          str(ind_plateforme+1), cal_temp)

    channel_from_acq = acq['parameters']['FORCE_PLATFORM']['CHANNEL']['value']
    for ind_plateforme in range(channel_from_acq.shape[1]):
        c3d.add_parameter("Plateform", "Channel_" +
                          str(ind_plateforme+1), channel_from_acq[:, ind_plateforme].tolist())

    del c3d['parameters']['FORCE_PLATFORM']
    c3d['parameters']['POINT']['LABELS']['value'] = new_list
    c3d['parameters']['POINT']['DESCRIPTIONS']['value'] = new_list.copy()

    return force_plateforms
