import json

# TODO add in the general information the name of the different marker for the leardini model.
# TODO do a json specific pour each part : General information, model information (lower limb,rachis,uupper limb)


def general_information_generation(name_json_export='general_information.json'):
    # How to define a acqusition
    # Define general parameters
    # Probably a dictionary that could be used as an variable in the different function
    general_information_acqusition = dict()
    # Main direction
    general_information_acqusition['point_filtering'] = 10
    # General information for model
    general_information_acqusition['LPSI_label'] = 'L_IPS'
    general_information_acqusition['RPSI_label'] = 'R_IPS'
    general_information_acqusition['LASI_label'] = 'L_IAS'
    general_information_acqusition['RASI_label'] = 'R_IAS'
    # The main direction parameters will be used in order to define the correct X Y Z axis
    general_information_acqusition['Anterior_Side'] = 'IJ'
    general_information_acqusition['Posterior_Side'] = 'C7'
    general_information_acqusition['Up_Side'] = 'C7'
    general_information_acqusition['Down_Side'] = 'S1'
    general_information_acqusition['Left_Side'] = 'LAC'
    general_information_acqusition['Right_Side'] = 'RAC'

    # Definition of the different segment
    general_information_acqusition['base_lomb'] = 'S1'
    general_information_acqusition['up_lomb'] = 'T12'
    general_information_acqusition['left_lomb'] = 'L_Lomb'
    general_information_acqusition['right_lomb'] = 'R_Lomb'
    general_information_acqusition['list_marker_lomb'] = [
        'S1', 'L5', 'L3', 'L1', 'T12']
    general_information_acqusition['list_marker_lomb_spline'] = [
        'S1', 'L5', 'L3', 'L1', 'T12', 'T10']

    general_information_acqusition['base_thor'] = 'T12'
    general_information_acqusition['up_thor'] = 'T2'
    general_information_acqusition['left_thor'] = 'L_Thorax'
    general_information_acqusition['right_thor'] = 'R_Thorax'
    general_information_acqusition['list_marker_thor'] = [
        'T12', 'T10', 'T8', 'T6', 'T4', 'T2']
    general_information_acqusition['list_marker_thor_spline'] = ['L1', 'T12',
                                                                 'T10', 'T8', 'T6', 'T4', 'T2', 'C7']
    general_information_acqusition['base_spine'] = 'S1'
    general_information_acqusition['up_spine'] = 'T2'
    general_information_acqusition['left_spine'] = 'L_Thorax'
    general_information_acqusition['right_spine'] = 'R_Thorax'
    general_information_acqusition['list_marker_spine'] = ['L_IAS', 'R_IAS', 'R_IPS', 'L_IPS',
                                                           'S1', 'L5', 'L3', 'L1', 'T12',
                                                           'T10', 'T8', 'T6', 'T4', 'T2', 'C7']
    general_information_acqusition['list_marker_spine_spline'] = ['S1', 'L5', 'L3', 'L1', 'T12',
                                                                  'T10', 'T8', 'T6', 'T4', 'T2', 'C7']
    # model choice for lower limb kinematics (can be a function in the dictionary)

    # Serializing json
    json_object = json.dumps(general_information_acqusition, indent=4)
    print(json_object)
    with open("general_information.json", "w") as outfile:
        json.dump(general_information_acqusition, outfile)
    return general_information_acqusition


def general_information_generation_V_Hovannes2(name_json_export='general_information.json'):
    # How to define a acqusition
    # Define general parameters
    # Probably a dictionary that could be used as an variable in the different function
    general_information_acqusition = dict()
    # Main direction
    general_information_acqusition['point_filtering'] = 10
    # General information for model
    general_information_acqusition['LPSI_label'] = 'L_IPS'
    general_information_acqusition['RPSI_label'] = 'R_IPS'
    general_information_acqusition['LASI_label'] = 'L_IAS'
    general_information_acqusition['RASI_label'] = 'R_IAS'
    # The main direction parameters will be used in order to define the correct X Y Z axis
    general_information_acqusition['Anterior_Side'] = 'SJN'
    general_information_acqusition['Posterior_Side'] = 'CV7'
    general_information_acqusition['Up_Side'] = 'CV7'
    general_information_acqusition['Down_Side'] = 'LV1'
    general_information_acqusition['Left_Side'] = 'L_SAE'
    general_information_acqusition['Right_Side'] = 'R_SAE'

    # Definition of the different segment
    general_information_acqusition['base_lomb'] = 'LV5'
    general_information_acqusition['up_lomb'] = 'MAI'
    general_information_acqusition['left_lomb'] = 'L_LOMB'
    general_information_acqusition['right_lomb'] = 'R_LOMB'
    general_information_acqusition['list_marker_lomb'] = [
        'LV5', 'LV3', 'LV1', 'MAI']
    general_information_acqusition['list_marker_lomb_spline'] = [
        'LV5', 'LV3', 'LV1', 'MAI']

    general_information_acqusition['base_thor'] = 'MAI'
    general_information_acqusition['up_thor'] = 'TV2'
    general_information_acqusition['left_thor'] = 'L_THOR'
    general_information_acqusition['right_thor'] = 'R_THOR'
    general_information_acqusition['list_marker_thor'] = [
        'LV1', 'MAI', 'TV2']
    general_information_acqusition['list_marker_thor_spline'] = [
        'LV1', 'MAI', 'TV2']
    #general_information_acqusition['base_spine'] = 'S1'
    #general_information_acqusition['up_spine'] = 'T2'
    #general_information_acqusition['left_spine'] = 'L_Thorax'
    #general_information_acqusition['right_spine'] = 'R_Thorax'
    # general_information_acqusition['list_marker_spine'] = ['L_IAS', 'R_IAS', 'R_IPS', 'L_IPS',
    #                                                       'S1', 'L5', 'L3', 'L1', 'T12',
    #                                                       'T10', 'T8', 'T6', 'T4', 'T2', 'C7']
    # general_information_acqusition['list_marker_spine_spline'] = ['S1', 'L5', 'L3', 'L1', 'T12',
    #                                                              'T10', 'T8', 'T6', 'T4', 'T2', 'C7']
    # model choice for lower limb kinematics (can be a function in the dictionary)

    return general_information_acqusition
