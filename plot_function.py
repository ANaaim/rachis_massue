import matplotlib.pyplot as plt
import math


def plot_function_temp(value_final, task_to_plot, values_to_plot):
    name_of_color = ['tab:blue', 'tab:orange',
                     'tab:green', 'tab:red', 'tab:purple']
    nb_row = math.ceil(len(values_to_plot)/3)

    what_to_plot = 'values'
    for ind_value, value_to_plot in enumerate(values_to_plot):
        if len(value_to_plot) > 0:
            for ind_condition, condition_to_plot in enumerate(value_final[task_to_plot]):
                for ind_division_to_plot, division_to_plot in enumerate(value_final[task_to_plot][condition_to_plot]):
                    # there is a plus 1 here because indices in matplotlib begin at 1
                    ax_temp = plt.subplot(nb_row, 3, ind_value+1)
                    ax_temp.plot(value_final[task_to_plot][condition_to_plot]
                                 [division_to_plot][value_to_plot][what_to_plot],
                                 color=name_of_color[ind_condition])

                    plt.title(value_to_plot)
                    # When everything is plotted we can add two fake curve to have a singular legend
                    if ind_condition == len(value_final[task_to_plot].keys())-1 and ind_value == 0:
                        for ind_condition_legend, condition_to_plot_legend in enumerate(value_final[task_to_plot]):
                            plt.plot(value_final[task_to_plot][condition_to_plot_legend]
                                     [division_to_plot][value_to_plot][what_to_plot][:, 0],
                                     color=name_of_color[ind_condition_legend],
                                     label=condition_to_plot_legend)
                        plt.legend()
    plt.show()
