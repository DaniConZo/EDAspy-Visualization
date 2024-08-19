import numpy as np
import pandas as pd
from EDAspy.optimization import EdaResult
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.colors as mpcol
from pandas.plotting import parallel_coordinates
#from matplotlib import cm

def parall_coord(input_data: EdaResult = None , fig_size: tuple = (20,12), plt_suptitle: str='Parallel coordinates', cmap=plt.cm.viridis_r, x_label: str = 'Variables', y_label: str = 'Values',
                          y_ticks: np.ndarray = np.arange(-120, 121, 20), best_ind_color: str = 'red', variables_list: list = None):
    
    '''This function returns parallel coordinates plot of input data.

        :param input_data: eda_result object with sel_inds_hist attribute
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_suptitle: string with sup title
        :param cmap plt.cm color
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param y_ticks: tuple with y ticks for variables
        :param best_ind_color: string with color for best individual
        :param variables_list: list of strings with variables names. Same order as passe to EDA
        
        :type input_data: EdaResult
        :type fig_size: tuple
        :type plt_suptitle: str
        :type cmap: cmap plt.cm color
        :type x_label: str
        :type y_label: str
        :type y_ticks: numpy.ndarray
        :type best_ind_color: str
        :type variables_list: list
        :return: Figure.'''
    
    assert isinstance(input_data, EdaResult), 'Input object is not EdaResult class'
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert type(plt_suptitle)==str, 'plt_suptitle must be a string'
    assert type(x_label)==str, 'x_label must be a string'
    assert type(y_label)==str, 'y_label must be a string'
    assert type(y_ticks)==np.ndarray, 'y_ticks must be a numpy.ndarray'
    assert type(best_ind_color)==str, 'best_ind_color must be a string'
    
    n_variables = input_data.sel_inds_hist[:,:,:].shape[-1] - 1
    if variables_list:
        assert type(variables_list)==list, 'variables_list must be a list'
        assert len(variables_list)==n_variables, 'number of listed variables and number of EDAresult variables do not match'
    

    def update_plot(gen):
        plt.figure(figsize=fig_size)
        plt.clf()  # Clear previous plot
        
        edadframe = pd.DataFrame(input_data.sel_inds_hist[gen,:,:-1], columns=variables_list if variables_list is not None else np.arange(0, n_variables, 1).tolist())
        edadframe['ev'] = input_data.sel_inds_hist[gen,:,-1]

        # Create a colormap
        #cmap = cmap#plt.cm.viridis_r

        # Normalize values to map to the colormap
        norm = mpcol.Normalize(vmin=min(edadframe['ev'].values), vmax=max(edadframe['ev'].values))

        # Create a list of colors corresponding to each row
        colors = [cmap(norm(val)) for val in edadframe['ev']]

        # Plotting parallel coordinates
        parallel_coordinates(edadframe[::-1], class_column='ev', color=colors, alpha=0.5)
        parallel_coordinates(edadframe[0:1], class_column='ev', color=best_ind_color, lw=3, alpha=1)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.yticks(y_ticks)
        plt.grid()
        plt.title(f"Gen {gen}")  # Adjust title according to gen
        plt.legend().remove()

        # Create a colorbar as the legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Dummy empty array for the colorbar

        # Create an axis for the colorbar
        cbar_ax = plt.axes([0.91, 0.105, 0.025, 0.84])  # Adjust the position and size of the colorbar

        # Add the colorbar to the figure using the created axis
        plt.colorbar(sm, cax=cbar_ax, label='Fitness')

        plt.suptitle(plt_suptitle, fontsize=20)
        plt.subplots_adjust(top=0.92)
        #plt.show

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
