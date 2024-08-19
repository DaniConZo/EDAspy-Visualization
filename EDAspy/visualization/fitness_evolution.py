import matplotlib.pyplot as plt
import numpy as np
from EDAspy.optimization import EdaResult
from typing import Union

def fitness_evolution(input_data: Union[EdaResult, dict] = None, fig_size: tuple = (20,12), plt_title: str='Fitness evolution',  x_label: str = 'Generation', y_label: str = 'Fitness', current_best_color: str = 'r-'
                    , best_ever_color: str = 'b--', x_lim: tuple = None, y_lim: tuple = None, save_location: Union[str, None] = None):
    
    '''This function returns fitness evolution plot of input data. Best individual ever vs best sampled individual.
       If input is a dictionary with multiple EdaResults, it returns fitness evolution of the different EDA executions, allowing comparision for up to 6 different EDAs.
       If input is a dictionary, color choices are automated and no customization is allowed.
       You can use plt.__ after calling this function and further change the plot.


        :param input_data: eda_result object (or dictionary containig multiple) with sel_inds_hist and history attribute
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_title: string with sup title
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param current_best_color: string with color for best created individual per generation
        :param best_ever_color: string with color for best individual
        :param x_lim: tuple with x axis limits passed to plt.xlim(x_lim)
        :param y_lim: tuple with y axis limits passed to plt.ylim(y_lim)
        :param save_location: [str, None] str with saving location path

        
        :type input_data: Union[dict, EdaResult]
        :type fig_size: tuple
        :type plt_title: str
        :type x_label: str
        :type y_label: str
        :type current_best_color: str
        :type best_ever_color: str
        :type x_lim: tuple
        :type y_lim: tuple
        :type save_location: str

        :return: Figure.'''
    
    assert isinstance(input_data, Union[EdaResult, dict]), 'Input object is not EdaResult class, or dictionary'
    if isinstance(input_data, dict):
        for keys in input_data:
            assert isinstance(input_data[keys], EdaResult), 'dict.values() must be EdaResult objects'
            assert len(input_data) <= 6, "Can't compare more than 6 EDAs !!!"
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert type(plt_title)==str, 'plt_suptitle must be a string'
    assert type(x_label)==str, 'x_label must be a string'
    assert type(y_label)==str, 'y_label must be a string'
    assert type(current_best_color)==str, 'current_best_color must be a string'
    assert type(best_ever_color)==str, 'best_ever_color must be a string'
    assert isinstance(save_location, Union[str, None]), 'save_location must be a string or None'
    if x_lim is not None:
        assert type(x_lim) is tuple, 'x_lim must be a tuple'
    if y_lim is not None:
        assert type(y_lim) is tuple, 'y_lim must be a tuple'


    plt.figure(figsize=fig_size)
    #If input is single EdaResult
    if isinstance(input_data, EdaResult):
        plt.rcParams.update({'font.size':20})
        plt.plot(np.arange(0, len(input_data.history), 1), (input_data.history), current_best_color, label = 'Best sampled')
        plt.plot(np.arange(0, len(input_data.history)-1, 1), input_data.sel_inds_hist[:,0,-1], best_ever_color, label='Best ever')
        plt.xticks(np.arange(0,len((input_data.history)), 10))
        
    #If input is dictionary
    if isinstance(input_data, dict):
        colors = ['r', 'b', 'k', 'g', 'y', 'm']
        c=0
        maxgens=[]
        for keys in input_data:
            plt.rcParams.update({'font.size':20})
            plt.plot(np.arange(0, len(input_data[keys].history), 1), (input_data[keys].history), colors[c]+'-', label=f'Best sampled {keys}')
            plt.plot(np.arange(0, len(input_data[keys].history)-1, 1), input_data[keys].sel_inds_hist[:,0,-1], colors[c]+'--', label=f'Best ever {keys}')
            c+=1
            maxgens.append(len(input_data[keys].history))
        maxgen = max(maxgens)
        plt.xticks(np.arange(0,maxgen, 10))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    plt.legend()
    plt.title(plt_title);
    if save_location is not None:
        plt.savefig(save_location)