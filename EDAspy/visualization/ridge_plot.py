import joypy
from matplotlib import cm
from EDAspy.optimization import EdaResult
import pandas as pd

def ridge_plot(input_data: EdaResult = None , var_list: list = None, vars_to_plot: list = None, kind: str = 'counts' , overlap: float = 1.5, bins: int = 40 ,gen_interval: tuple = (0, 200), gen_label: int = 10,fig_size: tuple = (20,12), plt_suptitle: str='Variable distribution evolution', x_label: str = 'Values', y_label: str = 'Generation', superimpose_sampled_individuals: bool = False):
    
    '''This function returns ridge plot of selected variables showing search evoulution.

        :param input_data: eda_result object with sel_inds_hist attribute
        :param var_list: list of all variables in the same order as passed to EDA
        :param vars_to_plot: list of variables to plot
        :param kind: string indicating kind of density calculation, between ['kde', 'counts', 'normalized_counts'], 'lognorm', 'values']
        :param overlap: float adjusting the overlap in plot between generations
        :param bins: integer, number of bins for counts
        :param gen_interval: tuple with inf and sup limits of generation for plot visualization
        :param gen_label: interval to label the generation axis
        :param fig_size: tuple wiht figsize
        :param plt_suptitle: string with sup title
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param superimpose_sampled_individuals: bool, If True, distribution of all sampled individuals will be shown as well
        
        :type input_data: EdaResult
        :type var_list: list
        :type vars_to_plot: list
        :type kind: str
        :type overlap: float
        :type bins: int
        :type gen_interval: tuple
        :type gen_label: tuple
        :type fig_size: tuple
        :type plt_suptitle: str
        :type x_label: str
        :type y_label: str
        :type superimpose_sampled_individuals: bool
        :return: Figure.    '''
    
    assert isinstance(input_data, EdaResult), 'Input object is not EdaResult class'
    assert type(fig_size)==tuple, 'fig_size must be a tuple'
    assert type(plt_suptitle)==str, 'plt_suptitle must be a string'
    assert type(x_label)==str, 'x_label must be a string'
    assert type(y_label)==str, 'y_label must be a string'
    assert type(superimpose_sampled_individuals)==bool, 'superimpose_sampled_individuals must a boolean'
    
    #n_variables = input_data.sel_inds_hist[:,:,:].shape[-1] - 1)
    # Reshape the array to have individuals as rows and variables as columns
    reshaped_array = input_data.sel_inds_hist.reshape(-1, input_data.sel_inds_hist.shape[-1])

    # Create a list to hold the generation numbers
    generations = []
    for gen in range(input_data.sel_inds_hist.shape[0]):
        generations.extend([gen] * input_data.sel_inds_hist[gen, :, :].shape[0])

    # Create a DataFrame from the reshaped array and generations list
    if var_list is None:
        columns = [f"var_{i+1}" for i in range(input_data.sel_inds_hist.shape[-1]-1)] + ['ev']
    if var_list:
        columns = var_list + ['ev']
        
    EDAFrame = pd.DataFrame(data=reshaped_array, columns=columns)
    EDAFrame['gen'] = generations

    if vars_to_plot is None:
        vars_to_plot = columns[:-1]

    if superimpose_sampled_individuals:
        #Creating generation list for all sampled individuals
        gen_all = []
        for gen in range(input_data.all_inds_hist.shape[0]):
            gen_all.extend([gen] * input_data.all_inds_hist[gen, :, :].shape[0])

        #Adding new columns to EDAFrame with all sampled individuals for desired variables only
        indices = []
        for vars in vars_to_plot:
            indices.append(columns.index(vars)) #list with desired variable index in the 3D array to create DF from it 

        new_columns = [] #New Column names. We are differentiating var_1 of sel inds from allvar_1 from all inds 
        for vars in vars_to_plot:
            new_columns.append('all'+vars)     

        Alldata = input_data.all_inds_hist[:,:,indices].reshape(-1, input_data.all_inds_hist[:,:,indices].shape[-1])
        AllEDAFrame = pd.DataFrame(data=Alldata, columns=new_columns)
        AllEDAFrame['allgen'] = gen_all

        NewEDAFrame = pd.concat([EDAFrame, AllEDAFrame], axis=1)        
        
        # #Adding actual data from all individuals in new columns
        # for vars in new_columns:
        #     EDAFrame[vars] = input_data.all_inds_hist[:,:,vars].reshape(-1, input_data.all_inds_hist.shape[-1])

        for vars in vars_to_plot:
            labels=[int(g) if g%gen_label==0 else None for g in list(NewEDAFrame[(NewEDAFrame['gen'] > gen_interval[0])&(NewEDAFrame['gen'] < gen_interval[1])].gen.unique())]

            fig, ax = joypy.joyplot(NewEDAFrame[(NewEDAFrame['gen']>gen_interval[0])&(NewEDAFrame['gen'] < gen_interval[1])], by='gen', column=[vars, 'all'+vars],
                                title=f'{vars} Distribution. Selected vs All', figsize=fig_size,
                            hist=False, overlap=overlap, labels=labels, range_style='own',
                            #colormap=cm.autumn_r,
                             #fade=True,
                              grid=True, kind=kind,
                                bins=bins, legend=True, alpha=0.4, color=['cornflowerblue', 'indianred'])

    if not superimpose_sampled_individuals:
        for vars in vars_to_plot:
            labels=[int(g) if g%gen_label==0 else None for g in list(EDAFrame[(EDAFrame['gen'] > gen_interval[0])&(EDAFrame['gen'] < gen_interval[1])].gen.unique())]

            fig, ax = joypy.joyplot(EDAFrame[(EDAFrame['gen']>gen_interval[0])&(EDAFrame['gen'] < gen_interval[1])], by='gen', column=vars,
                                title=f'{vars} Distribution', figsize=fig_size,
                            hist=False, overlap=overlap, labels=labels, range_style='own',
                            colormap=cm.autumn_r, fade=True, grid=True, kind=kind, bins=bins, legend=True)
