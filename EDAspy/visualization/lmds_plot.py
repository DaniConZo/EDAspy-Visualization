import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.colors as mpcol
import matplotlib.pyplot as plt
from typing import Union
from matplotlib.lines import Line2D

def lmds_plot(input_data: Union[np.ndarray, dict] = None , fig_size: tuple = (20,12), plt_suptitle: str='LMDS dimensionality reduction', cmap=plt.cm.viridis_r, x_label: str = 'Var 1', y_label: str = 'Var 2', x_lim: tuple = (-1,1), y_lim: tuple = (-1 ,1),
         plot_trajectory: bool = True, trajectory_color: str = 'r-', trajectory_labels: int = 20 ):
    
    '''This function plots 2D dimensionality reduction of the solutions visited by the algorithm using LMDS in lmds.py function
       It returns a sequence of 2D scatter plots (one per generation) and shows the trajectory of the best individual if desired.
       
        :param input_data: array or dictionary of arrays with embedded data. 3D (gen, inds, var1++var2+score)
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_suptitle: string with sup title
        :param cmap plt.cm color
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param x_lim: tuple with lims for x axis
        :param y_lim: tuple with lims for y axis 
        :param trajectory_color: string with color for trajectory of best individual
        :param trajectory_labels: int. Distancing between printed labels (generation number) along trajectory
                  
        :type input_data: Union[dict, EdaResult]
        :type fig_size: tuple
        :type plt_suptitle: string
        :type cmap: cmap plt.cm color
        :type x_label: str
        :type y_label: str
        :type x_lim: tuple
        :type y_lim: tuple
        :type trajectory_color: str
        :type trajectory_labels: int
        :return: Figure.'''
    
    assert isinstance(input_data, Union[np.ndarray, dict]), 'Input object is not numpy.ndarray, or dictionary'
    if isinstance(input_data, dict):
        for keys in input_data:
            assert isinstance(input_data[keys], np.ndarray), 'dict.values() must be np.ndarrays'
            assert len(input_data) <= 6, "Can't compare more than 6 EDAs !!!"
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert type(plt_suptitle)==str, 'plt_suptitle must be a string'
    assert type(x_label)==str, 'x_label must be a string'
    assert type(y_label)==str, 'y_label must be a string'
    assert type(x_lim)==tuple, 'x_lim must be a tuple'
    assert type(y_lim)==tuple, 'y_lim must be a tuple'
    assert type(plot_trajectory)== bool, 'plot_trajectory must be a boolean'
    assert type(trajectory_color)== str, 'trajectory_color must be a string like "r-" or "k--"'
    assert type(trajectory_labels)==int, 'trajectory_labels must be an integer'

    #If input is single EdaResult
    if isinstance(input_data, np.ndarray):
          
        def update_plot(gen):
            plt.figure(figsize=fig_size)
            plt.rcParams.update({'font.size':20})
            plt.clf()  # Clear previous plot

            # Normalize values to map to the colormap
            norm = mpcol.Normalize(vmin=min(input_data[gen,:,-1]), vmax=max(input_data[gen,:,-1]))

            # Create a list of colors corresponding to each row
            colors = [cmap(norm(val)) for val in input_data[gen,:,-1]]
            plt.scatter(input_data[gen,:,0], input_data[gen,:,1], color=colors)

            #Plot trajectory of best individual
            if plot_trajectory:
                plt.plot(input_data[0:gen, 0, 0], input_data[:gen, 0, 1], trajectory_color, alpha=0.4, label='Best ind trajectory')
                # Add labels every ten generations
                for i in range(0, gen):
                    if i % trajectory_labels == 0:
                        plt.text(input_data[i, 0, 0], input_data[i, 0, 1], f'{i}', fontsize=12)
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(fontsize=20)
            plt.title(f"Gen {gen-1}")  # Adjust title according to gen
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.suptitle(plt_suptitle, fontsize=20)
            plt.subplots_adjust(top=0.92)
            # Create a colorbar as the legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Dummy empty array for the colorbar

            # Create an axis for the colorbar
            cbar_ax = plt.axes([0.91, 0.105, 0.025, 0.78])  # Adjust the position and size of the colorbar

            # Add the colorbar to the figure using the created axis
            plt.colorbar(sm, cax=cbar_ax, label='Fitness')

        def save_plot(_):
            save_location = save_location_text.value#+".pdf"
            update_plot(generation_slider.value)
            plt.savefig(save_location)  # Save the plot using the provided location
            plt.clf()
            plt.close()
        
        #Slider
        generation_slider = widgets.IntSlider(value=0, min=0, max=len(input_data)-1,
                                                step=1, description='Generation', layout=widgets.Layout(width='100%'))
        
        #Text, Button
        save_location_text = widgets.Text(placeholder='Enter save location here...', description='Save Loc:', layout=widgets.Layout(width='50%'))
        save_button = widgets.Button(description='Save Plot')
        save_button.on_click(save_plot)

        # Use interact to connect the slider with the update_plot function
        interact(update_plot, gen=generation_slider) 
        display(save_location_text, save_button)


    #If input_data is dictionary    
    if isinstance(input_data, dict):

        """Now the actual plot"""
        #Default markers and colors
        mk_col = ['r', 'b', 'k', 'g', 'y', 'm']
        markers = ['o', 's', 'P', 'd', '<','>']

        # Create a colormap
        cmap = plt.cm.viridis_r

        def update_plot(gen):       
            plt.figure(figsize=fig_size)
            plt.rcParams.update({'font.size':20})
            plt.clf()  # Clear previous plot

            # Normalize values to map to the colormap. We compare individuals in the same generation for all EDAs.
            minscore = np.inf
            for arrays in input_data.values():
                if gen > len(arrays)-1: #EDAs can have different gen sizes. So max gen is limited for each EDAs max.
                    gen = len(arrays)-1
                minscore = min(minscore, arrays[gen, 0, -1]) #arrays[gen, 0, -1] is the best individual for each EDA

            maxscore = -np.inf
            for arrays in input_data.values():
                if gen > len(arrays)-1:
                    gen = len(arrays)-1
                maxscore = max(maxscore, arrays[gen, -1, -1]) #arrays[gen, -1, -1] is the worst (selected) ind for each EDA

            norm = mpcol.Normalize(vmin=minscore, vmax=maxscore)
            
            c = 0 #counter to iterate colors and markers
            for keys in input_data:
                if gen > len(input_data[keys])-1:
                    gen = len(input_data[keys])-1
                # Create a list of colors corresponding to each row for each EDA according to normalization
                colors = [cmap(norm(val)) for val in input_data[keys][gen,:,-1]]
                plt.scatter(input_data[keys][gen,:,0], input_data[keys][gen,:,1], color=colors, marker=markers[c])#, label=keys)
                plt.plot(input_data[keys][0:gen, 0, 0], input_data[keys][:gen, 0, 1], f'{mk_col[c]}-', alpha=0.4)#, label=f'{keys} best trajectory')
                c += 1
                # Add labels every ten generations
                for i in range(0, gen):
                    if i % trajectory_labels == 0:
                        plt.text(input_data[keys][i, 0, 0], input_data[keys][i, 0, 1], f'{i}', fontsize=12)

            legend_elements = []
            c=0
            for keys in input_data:
                legend_elements.append(Line2D([0],[0], marker=markers[c], markerfacecolor='black', label=keys, color='w'))
                legend_elements.append(Line2D([0],[0], color=mk_col[c], label=keys))
                c+=1

            plt.legend(handles=legend_elements, loc='upper right', fontsize=20)    
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Gen {gen}")  # Adjust title according to gen
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.suptitle(plt_suptitle, fontsize=20)
            plt.subplots_adjust(top=0.92)

            # Create a colorbar as the legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Dummy empty array for the colorbar

            # Create an axis for the colorbar
            cbar_ax = plt.axes([0.91, 0.105, 0.025, 0.78])  # Adjust the position and size of the colorbar

            # Add the colorbar to the figure using the created axis
            plt.colorbar(sm, cax=cbar_ax, label='Fitness')
            
        def save_plot(_):
            save_location = save_location_text.value#+".pdf"
            update_plot(generation_slider.value)
            plt.savefig(save_location)  # Save the plot using the provided location
            plt.clf()
            plt.close()

        #Text, Button
        save_location_text = widgets.Text(placeholder='Enter save location here...', description='Save Loc:', layout=widgets.Layout(width='50%'))
        save_button = widgets.Button(description='Save Plot')
        save_button.on_click(save_plot)

        maxgens = []  
        for keys in input_data:
            maxgens.append(len(input_data[keys]))
        maxgen=max(maxgens)

        #Slider
        generation_slider = widgets.IntSlider(value=0, min=0, max=maxgen-1,
                                            step=1, description='Generation', layout=widgets.Layout(width='100%'))


        # Use interact to connect the slider with the update_plot function
        interact(update_plot, gen=generation_slider) 
        display(save_location_text, save_button)
        