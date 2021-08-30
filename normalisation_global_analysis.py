import numpy as np


def normalisation_global_analysis(Global_analysis):
    value_final = dict()
    for task in Global_analysis:
        value_final[task] = dict()
        for condition in Global_analysis[task]:
            value_final[task][condition] = dict()
            for division in Global_analysis[task][condition]:
                value_final[task][condition][division] = dict()
                for value in Global_analysis[task][condition][division]:
                    value_final[task][condition][division][value] = dict()
                    nb_trial = len(
                        Global_analysis[task][condition][division][value])
                    value_final[task][condition][division][value]['values'] = np.zeros(
                        (101, nb_trial))
                    print(value)
                    for ind_trial in range(nb_trial):
                        nb_frame = Global_analysis[task][condition][division][value][ind_trial].shape[0]
                        value_to_interpolate = Global_analysis[task][condition][division][value][ind_trial]

                        x = np.linspace(0, nb_frame, 101)
                        xp = np.linspace(0, nb_frame, nb_frame)
                        value_final[task][condition][division][value]['values'][:,
                                                                                ind_trial] = np.interp(x, xp, value_to_interpolate)
                    value_final[task][condition][division][value]['mean'] = np.mean(
                        value_final[task][condition][division][value]['values'], axis=1)
                    value_final[task][condition][division][value]['std'] = np.std(
                        value_final[task][condition][division][value]['values'], axis=1)
    return value_final
