import ipywidgets as widgets
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from EDAspy.optimization import EdaResult

def violin_plot(input_data: EdaResult = None , fig_size: tuple = (20,12), plt_suptitle: str='KDE for each variable', x_label: str = 'Variables', y_label: str = 'Values',
                          y_ticks: np.ndarray = np.arange(-120, 121, 20), best_ind_color: str = 'red', superimpose_sampled_individuals: bool = False, variables_list: list = None):
    
    '''This function returns violin plot with KDE evolution for each variable of the selected individuals each gen of input data.

        :param input_data: eda_result object with sel_inds_hist attribute
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_suptitle: string with sup title
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param y_ticks: tuple with y ticks for variables
        :param best_ind_color: string with color for best individual
        :param superimpose_sampled_individuals: bool If True, another KDE will be plotted for each variable showing all sampled individuals
        :param variables_list: list of strings with variables names. Same order as passe to EDA
        
        :type input_data: EdaResult
        :type fig_size: tuple
        :type plt_suptitle: str
        :type x_label: str
        :type y_label: str
        :type y_ticks: numpy.ndarray
        :type best_ind_color: str
        :type superimpose_sampled_individuals: bool
        :type variables_list: list
        :return: Figure.    '''
    
    assert isinstance(input_data, EdaResult), 'Input object is not EdaResult class'
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert type(plt_suptitle)==str, 'plt_suptitle must be a string'
    assert type(x_label)==str, 'x_label must be a string'
    assert type(y_label)==str, 'y_label must be a string'
    assert type(y_ticks)==np.ndarray, 'y_ticks must be a numpy.ndarray'
    assert type(best_ind_color)==str, 'best_ind_color must be a string'
    assert type(superimpose_sampled_individuals)==bool, 'superimpose_sampled_individuals must a boolean'
    
    n_variables = input_data.sel_inds_hist[:,:,:].shape[-1] - 1
    if variables_list:
        assert type(variables_list)==list, 'variables_list must be a list'
        assert len(variables_list)==n_variables, 'number of listed variables and number of EDAresult variables do not match'

    def update_plot(gen):
        plt.figure(figsize=fig_size)
        plt.rcParams.update({'font.size':20})
        plt.clf()  # Clear previous plot
        
        edadf = pd.DataFrame(input_data.sel_inds_hist[gen,:,:-1], columns=variables_list if variables_list is not None else np.arange(0, n_variables, 1).tolist())
        edadf['ev'] = input_data.sel_inds_hist[gen,:,-1]

        

        #Find the row index with the smallest 'ev' value
        min_ev_row_index = edadf['ev'].idxmin()
        # Extract the data of the row with the smallest 'ev' value
        min_ev_row_data = edadf.iloc[min_ev_row_index, :-1].tolist()

        data = []
        for col in edadf:
            data.append(edadf[col])
        
        plt.violinplot(data[:-1], showmeans=True)
        # Adding a line representing the row with the smallest 'ev' value
        plt.plot(np.arange(1, len(min_ev_row_data) + 1), min_ev_row_data, label='Best ind', color= 'k' if superimpose_sampled_individuals else best_ind_color)

        if superimpose_sampled_individuals:
            eda2df = pd.DataFrame(input_data.all_inds_hist[gen,:,:], columns=np.arange(0, n_variables, 1).tolist())
            data1 = []
            for col in eda2df:
                data1.append(eda2df[col])
                
            b = plt.violinplot(data1, showmeans=True)
            # Use bodies as the return key
            for body in b['bodies']:
                body.set(color='pink', alpha=0.4)
            for lines in ('cbars', 'cmaxes', 'cmins', 'cmeans'):
                b[lines].set_edgecolor('lightcoral')

            plt.legend(handles = [Patch(facecolor='lightsteelblue', edgecolor='cornflowerblue',
                                label='Selected individuals'), Patch(facecolor='pink', edgecolor='coral',
                                label='Sampled individuals'), Line2D([0], [0], color='black', lw=2, label='Best individual')], loc='best')   

        # Adding legend
        if not superimpose_sampled_individuals:
            plt.legend(handles = [Patch(facecolor='lightsteelblue', edgecolor='cornflowerblue',
                                label='Selected individuals'), Line2D([0], [0], color=best_ind_color, lw=2, label='Best individual')], loc='best')
        plt.xlabel(x_label)
        plt.xticks(np.arange(1, n_variables+1, 1), labels=variables_list if variables_list is not None else np.arange(1,n_variables+1,1))
        #plt.set_xticks_labels(variables_list if variables_list is not None else np.arange(1,n_variables+1,1))
        plt.ylabel(y_label)
        plt.yticks(y_ticks)
        plt.title(f"Gen {gen + 1}")  # Adjust title according to gen

        plt.suptitle(plt_suptitle, fontsize=20)
        plt.subplots_adjust(top=0.92)
        #plt.show()


    def save_plot(_):
        save_location = save_location_text.value#+".pdf"
        update_plot(generation_slider.value)
        plt.savefig(save_location)  # Save the plot using the provided location
        plt.clf()
        plt.close()

    generation_slider = widgets.IntSlider(value=0, min=0, max=len(input_data.sel_inds_hist)-1,
                                        step=1, description='Generation', layout=widgets.Layout(width='100%'))


    save_location_text = widgets.Text(placeholder='Enter save location here...', description='Save Loc:', layout=widgets.Layout(width='50%'))
    save_button = widgets.Button(description='Save Plot')
    save_button.on_click(save_plot)

    # Use interact to connect the slider with the update_plot function
    interact(update_plot, gen=generation_slider)
    display(save_location_text, save_button)
