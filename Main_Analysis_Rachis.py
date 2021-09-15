import os
import pdb
from leardini_simplified import leardini_simplified as leardini_simplified
from leardini_simplified import leardini_simplified_V2 as leardini_simplified_V2
from analysis_all_files import rachis_all_files as rachis_all_files
import numpy as np
import plot_function
from normalisation_global_analysis import normalisation_global_analysis as normalisation_global_analysis
from general_information_generation import general_information_generation as general_information_generation
from general_information_generation import general_information_generation_V_Hovannes2 as general_information_generation_V_Hovannes2
import json
import model_spine
from collections import defaultdict
# Leardini to correct
# Study on the effect of s on the interpolation
# How to model correctly spine
# Different model for the spine ==> very simple to more complex
# TODO Faire un interface pour çà avec :
#   - Selection des fichiers
#   - Défintion des taches
#   - Sélection des découpages
#   - Choix des variables

# --------------------------------------------------------------------------------
# General information generation
# --------------------------------------------------------------------------------
# Save data with a json
if os.path.exists('general_information.json'):
    f = open('general_information.json',)
    general_information_acqusition = json.load(f)
else:
    general_information_acqusition = general_information_generation()
# As function cannot be stored in a json we have to add it here
general_information_acqusition['model_lower_limb'] = leardini_simplified
general_information_acqusition['model_rachis'] = model_spine.Naaim_two_segments
generate_c3d = True
# -------------------------------------------------------------------------------
# Extraction Data
# -------------------------------------------------------------------------------
# TODO : Reflexion sur l'ordre dans lequel sont stocké les données.
# Pour le moment [Task]==>[Condition]==>[Division temporel] ==> [Type de valeur (Hip flexion knee abdiction etc...)]
# TODO : How to integrate subdivision that could be integrated in the plotting

# Global analysis will contain all the data extracted non normalised on 100 frames
Global_analysis = defaultdict(dict)
test_hovannes = False
if test_hovannes:
    general_information_acqusition = general_information_generation_V_Hovannes2()
    general_information_acqusition['model_lower_limb'] = leardini_simplified_V2
    general_information_acqusition['model_rachis'] = model_spine.model_simple_two_segments
    # ______________
    # Task 1 Walking
    # Condition 1 : normal
    filenames_walking = ['TESTOU.c3d']
    # Definition of the temporal subdivision of the file
    list_subdivision_walking = dict()
    list_subdivision_walking['Right'] = [
        'RHS', 'RHS']
    list_subdivision_walking['Left'] = [
        'LHS', 'LHS']
    # We create the dictionnary associated with the task
    Global_analysis['Walking'] = dict()
    Global_analysis['Walking']['Normal'] = rachis_all_files(filenames_walking, list_subdivision_walking,
                                                            general_information_acqusition)
else:
    # ______________
    # Task 1 Walking
    # Condition 1 : normal
    filenames_walking = [os.path.join('Data', 'Gait_01.c3d')]
    # Condition 2 : rigid back
    filenames_walking_rigid = [os.path.join('Data', 'Gait_01_Rigid_Back.c3d')]
    # Definition of the temporal subdivision of the file
    list_subdivision_walking = dict()
    list_subdivision_walking['Right'] = [
        'Right Foot Strike', 'Right Foot Strike']
    # list_subdivision_walking['Left'] = [
    #    'Left Foot Strike', 'Left Foot Strike']
    # We create the dictionnary associated with the task
    #Global_analysis['Walking'] = dict()
    Global_analysis['Walking']['Normal'] = rachis_all_files(filenames_walking, list_subdivision_walking,
                                                            general_information_acqusition, generate_c3d)
    Global_analysis['Walking']['Rigid Back'] = rachis_all_files(filenames_walking_rigid, list_subdivision_walking,
                                                                general_information_acqusition, generate_c3d)
    general_information_acqusition['model_rachis'] = model_spine.model_simple_two_segments
    Global_analysis['Walking']['Normal simple'] = rachis_all_files(filenames_walking, list_subdivision_walking,
                                                                   general_information_acqusition, generate_c3d)
    Global_analysis['Walking']['Rigid Back simple'] = rachis_all_files(filenames_walking_rigid, list_subdivision_walking,
                                                                       general_information_acqusition, generate_c3d)

    general_information_acqusition['model_rachis'] = model_spine.Naaim_two_segments
    # _____________________
    # Task 2 : Sit to stand
    # Extract filenames
    filenames_sit_to_stand = [os.path.join('Data', 'Sit_to_stand.c3d')]
    # Define event names of the different subdivision
    list_subdivision_sit_to_stand = dict()
    list_subdivision_sit_to_stand['Sit'] = [
        'General Begin_Sit', 'General End_Sit']
    list_subdivision_sit_to_stand['Stand'] = [
        'General Begin_Stand', 'General End_Stand']
    #Global_analysis['Sit_to_Stand'] = dict()
    Global_analysis['Sit_to_Stand']['Normal'] = rachis_all_files(filenames_sit_to_stand, list_subdivision_sit_to_stand,
                                                                 general_information_acqusition, generate_c3d)

    # ___________________________________
    # Task 3: Ground different condition
    # Ergo condition
    filenames_ground_ergo = [os.path.join('Data', 'Ground_Ergo.c3d')]
    filenames_ground_knight = [os.path.join('Data', 'Ground_Ergo_Knight.c3d')]
    filenames_up_limb = [os.path.join('Data', 'Ground_Ergo_Up_limb.c3d')]
    # Define event names of the different subdivision
    list_subdivision_ground = dict()
    list_subdivision_ground['Grab'] = [
        'General Begin_Ground', 'General End_Ground']
    #Global_analysis['Ground'] = dict()
    Global_analysis['Ground']['Ergo'] = rachis_all_files(filenames_ground_ergo, list_subdivision_ground,
                                                         general_information_acqusition, generate_c3d)
    Global_analysis['Ground']['Up_Limb'] = rachis_all_files(filenames_up_limb, list_subdivision_ground,
                                                            general_information_acqusition, generate_c3d)

    general_information_acqusition['model_rachis'] = model_spine.model_simple_two_segments
    Global_analysis['Ground']['Ergo_simple'] = rachis_all_files(filenames_ground_ergo, list_subdivision_ground,
                                                                general_information_acqusition, generate_c3d)
    Global_analysis['Ground']['Up_Limb_simple'] = rachis_all_files(filenames_up_limb, list_subdivision_ground,
                                                                   general_information_acqusition, generate_c3d)

# ____________________________________________________________
# Normalisation on 100 frame for all data and mean calculation
value_final = normalisation_global_analysis(Global_analysis)

# --------------------------------------------------------------------------------------
# Plot of the result
# --------------------------------------------------------------------------------------
task_to_plot_1 = 'Ground'
value_to_plot_1 = ['Hip_X', 'Hip_Y', 'Hip_Z',
                   'Knee_X', 'Knee_Y', 'Knee_Z',
                   'Ankle_X', 'Ankle_Y', 'Ankle_Z']
task_to_plot_2 = 'Ground'
value_to_plot_2 = ['Knee_X', 'Knee_Y', 'Hip_X',
                   'Sacro_Lombaire_X', 'Sacro_Lombaire_Y', 'Sacro_Lombaire_Z',
                   'Lombo_thoracique_X', 'Lombo_thoracique_Y', 'Lombo_thoracique_Z']
task_to_plot_3 = 'Ground'
value_to_plot_3 = ['percentage_chord_thor', 'value_chord_thor', 'curvature_thor',
                   'percentage_chord_lomb', 'value_chord_lomb', 'curvature_lomb']

plot_function.plot_function_temp(value_final, task_to_plot_1, value_to_plot_1)
plot_function.plot_function_temp(value_final, task_to_plot_2, value_to_plot_2)
plot_function.plot_function_temp(value_final, task_to_plot_3, value_to_plot_3)
