import numpy as np
from EDAspy.optimization import EdaResult
import matplotlib.pyplot as plt
from typing import Union

def best_ind_var_ev(input_data: EdaResult = None , fig_size: tuple = (20,12), plt_suptitle: str='Best individual variables evolution', x_label: str = 'Generation', y_label: str = 'Values',
                          y_ticks: np.ndarray = np.arange(-120, 121, 20), save_location: Union[str, None] = None, variables_list: list = None):
    
    '''This function returns current best individual variables (different from best individual ever) evolution plot of input data. 

        :param input_data: eda_result object with sel_inds_hist attribute
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_suptitle: string with sup title
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param y_ticks: tuple with y ticks for variables
        :param save_location: [str, None] string with saving location path
        :param variables_list: list of strings with variables names. Same order as passed to EDA
        
        :type input_data: EdaResult
        :type fig_size: tuple
        :type plt_suptitle: str
        :type x_label: str
        :type y_label: str
        :type y_ticks: numpy.ndarray
        :type save_location: str
        :type variables_list: list

        :return: Figure.'''
    
    assert isinstance(input_data, EdaResult), 'Input object is not EdaResult class'
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert type(plt_suptitle)==str, 'plt_suptitle must be a string'
    assert type(x_label)==str, 'x_label must be a string'
    assert type(y_label)==str, 'y_label must be a string'
    assert type(y_ticks)==np.ndarray, 'must be a numpy.ndarray'
    assert isinstance(save_location, Union[str, None]), 'save_location must be a string or None'


    n_variables = input_data.sel_inds_hist[:,:,:].shape[-1] - 1
    if variables_list:
        assert type(variables_list)==list, 'variables_list must be a list'
        assert len(variables_list)==n_variables, 'number of listed variables and number of EDAresult variables do not match'
    


    plt.figure(figsize=fig_size)

    for var in range(n_variables):
        plt.plot(np.arange(0, len(input_data.history), 1), input_data.best_ind_hist[:, var], label=variables_list[var] if variables_list is not None else f'var {var}')

    plt.legend(loc=(0.9, 0.15))
    plt.xlabel(x_label)
    plt.yticks(y_ticks)
    plt.title(plt_suptitle)
    plt.grid(True)
    plt.ylabel(y_label)
    if save_location:
        plt.savefig(save_location)
