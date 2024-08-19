import numpy as np
#from mds import MDS, landmark_MDS
from scipy.spatial import distance
from EDAspy.optimization import EdaResult
import random
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.colors as mpcol
import matplotlib.pyplot as plt
from typing import Union
from matplotlib.lines import Line2D

##############################################################################################
############################    MDS.py  ######################################################
# Author: Danilo Motta  -- <ddanilomotta@gmail.com>

# This is an implementation of the technique described in:
# Sparse multidimensional scaling using landmark points
# http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf

import numpy as np
import scipy as sp

def MDS(D,dim=[]):
	# Number of points
	n = len(D)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(D**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim!=[]:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	L = np.diag(np.sqrt(evals[w]))
	V = evecs[:,w]
	Y = V.dot(L)
	return Y

def landmark_MDS(D, lands, dim):
	Dl = D[:,lands]
	n = len(Dl)

	# Centering matrix
	H = - np.ones((n, n))/n
	np.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(Dl**2).dot(H)/2

	# Diagonalize
	evals, evecs = np.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = np.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = np.where(evals > 0)
	if dim:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if np.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	if w.size==0:
		print('Error: matrix is negative definite.')
		return []

	V = evecs[:,w]
	L = V.dot(np.diag(np.sqrt(evals[w]))).T
	N = D.shape[1]
	Lh = V.dot(np.diag(1./np.sqrt(evals[w]))).T
	Dm = D - np.tile(np.mean(Dl,axis=1),(N, 1)).T
	dim = w.size
	X = -Lh.dot(Dm)/2.
	X -= np.tile(np.mean(X,axis=1),(N, 1)).T

	_, evecs = sp.linalg.eigh(X.dot(X.T))

	return (evecs[:,::-1].T.dot(X)).T

############################################################################################
############################################################################################

def lmds(input_data: Union[EdaResult, dict] = None , fig_size: tuple = (20,12), plt_suptitle: str='LMDS dimensionality reduction', cmap=plt.cm.viridis_r, x_label: str = 'var 1', y_label: str = 'var 2', x_lim: tuple = (-1,1), y_lim: tuple = (-1 ,1),
         plot_trajectory: bool = True, trajectory_color: str = 'r-', trajectory_labels: int = 20 ,landmark_points: float = 0.1, landmark_points_seed: int= 0,return_embedded_data: bool = False):
    
    '''This function computes and plots 2D dimensionality reduction of the solutions visited by the algorithm using LMDS.
       It returns a sequence of 2D scatter plots (one per generation) and shows the trajectory of the best individual if desired.
       If return_embedded_data = True lmds function returns a 3D numpy array (generation, individuals, var1+var2+score) with the embedded data that can be saved or passed to a variable

        :param input_data: eda_result object (or dictionary containig multiple)
        :param fig_size: tuple wiht figsize passed to plt.figure(figsize)
        :param plt_suptitle: string with sup title
        :param cmap plt.cm color
        :param x_label: string for x axis
        :param y_label: string for y axis
        :param x_lim: tuple with lims for x axis
        :param y_lim: tuple with lims for y axis 
        :param trajectory_color: string with color for trajectory of best individual
        :param trajectory_labels: int. Distancing between printed labels (generation number) along trajectory
        :param landmark_points: proportion of landmark points 0-1. 1 will be a regular MDS
        :param landmark_points_seed: int to set seed to pick landmark points randomly
        :param return_embedded_data: boolean to decided whether or not to return embedded data
                  
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
        :type landmark_points: float
        :type landmark_points_seed: int
        :type return_embedded_data: bool
        :return: Figure.'''
    
    assert isinstance(input_data, Union[EdaResult, dict]), 'Input object is not EdaResult class, or dictionary'
    if isinstance(input_data, dict):
        for keys in input_data:
            assert isinstance(input_data[keys], EdaResult), 'dict.values() must be EdaResult objects'
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
    assert type(landmark_points)==float, 'number of landmark_points must be a float'
    assert type(landmark_points_seed) is int, 'seed must be an integer'
    assert type(return_embedded_data) is bool,'return_embedded_data must be boolean'

    #If input is single EdaResult
    if isinstance(input_data, EdaResult):
          
        if input_data.sel_inds_hist[:,:,:-1].shape[-1] < 3:
            return('Your input is already 1D or 2D!!!!')
        
        #Reshape the array to have individuals as rows and variables as columns
        indstoembedLMDS = input_data.sel_inds_hist[:, :, :]
        reshaped_arrayLMDS = indstoembedLMDS.reshape(-1, indstoembedLMDS.shape[-1])

        ## Landmark points are chosen randomly.  10% of total points by default
        random.seed(landmark_points_seed)
        lands = random.sample(range(0,reshaped_arrayLMDS.shape[0],1),int(np.ceil(landmark_points * reshaped_arrayLMDS.shape[0]))) 
        lands = np.array(lands,dtype=int)
        Dl2 = distance.cdist(reshaped_arrayLMDS[lands,:-1], reshaped_arrayLMDS[:,:-1], 'euclidean') #distance matrix between landmarks and rest of points
        reduced_points = landmark_MDS(Dl2,lands,2) # 2D points

        #Once we have reduced the data, we reshape the array to 3D again [gen, ind, vars+fitness]. Adding the fitness of each individual to 2D points

        lmds_embed = np.column_stack((reduced_points, reshaped_arrayLMDS[:,-1]))
        lmds_embed = np.reshape(lmds_embed, (input_data.sel_inds_hist.shape[0], input_data.sel_inds_hist.shape[1], 2+1)) 

        def update_plot(gen):
            plt.figure(figsize=fig_size)
            plt.clf()  # Clear previous plot

            # Normalize values to map to the colormap
            norm = mpcol.Normalize(vmin=min(lmds_embed[gen,:,-1]), vmax=max(lmds_embed[gen,:,-1]))

            # Create a list of colors corresponding to each row
            colors = [cmap(norm(val)) for val in lmds_embed[gen,:,-1]]
            plt.scatter(lmds_embed[gen,:,0], lmds_embed[gen,:,1], color=colors)

            #Plot trajectory of best individual
            if plot_trajectory:
                plt.plot(lmds_embed[0:gen, 0, 0], lmds_embed[:gen, 0, 1], trajectory_color, alpha=0.4)
                # Add labels every ten generations
                for i in range(0, gen):
                    if i % trajectory_labels == 0:
                        plt.text(lmds_embed[i, 0, 0], lmds_embed[i, 0, 1], f'{i}', fontsize=10)
            
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
        
        #Slider
        generation_slider = widgets.IntSlider(value=0, min=0, max=len(input_data.sel_inds_hist)-1,
                                                step=1, description='Generation', layout=widgets.Layout(width='100%'))
        
        #Text, Button
        save_location_text = widgets.Text(placeholder='Enter save location here...', description='Save Loc:', layout=widgets.Layout(width='50%'))
        save_button = widgets.Button(description='Save Plot')
        save_button.on_click(save_plot)

        # Use interact to connect the slider with the update_plot function
        interact(update_plot, gen=generation_slider) 
        display(save_location_text, save_button)

        if return_embedded_data:
            return lmds_embed

    #If input_data is dictionary    
    if isinstance(input_data, dict):

        """First, we convert the 3D individual arrays in dict to 2D by removing the dimension corresponding to the generation.
        Each 2D array from each EDA is saved in a list."""
        
        reshaped_original_data = [] #list containing 2D arrays of each EDA where generation dimension is lost 
        for keys in input_data:
            indstoembed = input_data[keys].sel_inds_hist[:,:,:]
            reshapedinds = indstoembed.reshape(-1, indstoembed.shape[-1])
            reshaped_original_data.append(reshapedinds)

        """Now we need to stack those 2D arrays one below the other, to have a large 2D array that contains all individuals
         from the EDAs at once with their variables and evaluations."""

        all_edas_inds2D = np.concatenate(reshaped_original_data, axis=0) #array with all individuals from different EDAs without mixing.
        #Pasted on top of each other with same number of columns (vars + score)

        """Now we select the landmark points from among all the points. It's 10% of the total points, by default.
          Then, we calculate the distance matrix from those points to all others, and afterwards, LMDS"""
        
        random.seed(landmark_points_seed)
        lands = random.sample(range(0,all_edas_inds2D.shape[0],1),int(landmark_points*all_edas_inds2D.shape[0])) #by default 10% 
        lands = np.array(lands,dtype=int)
        Dl2 = distance.cdist(all_edas_inds2D[lands,:-1], all_edas_inds2D[:,:-1], 'euclidean')#last column is the score so we don't take it
        embedding = landmark_MDS(Dl2,lands,2)

        """Once we have reduced the data, we reshape the array again.
          Adding the fitness of each individual to the side."""
        
        embedplusscore = np.column_stack((embedding, all_edas_inds2D[:,-1]))

        """The following is delicate. We want to construct a lmds_dictionary whit the embedded data for each eda,
        with the same keys than the eda dictionary passed at the beggining.
        embedplusscore is a 2D array where the first rows correspond to the first eda in input_data.
        The next rows correspond to the next eda, ... and so on.
        Every time we pass the values of an EDA in a reshaped way [3D (gen , inds, vars+score) in each case]
        to the lmds_dictionary we eliminate those individuals (all belonging to the same EDA) form the embedplussocre array.
        So each time the first individuals in embedplusscore are the ones that have to be passed to the lmds_dictionary"""

        lmds_dict = {} #This dictionary will store the embedded data for each EDA

        for keys in input_data:
            lmds_dict[keys] = np.reshape(embedplusscore[:input_data[keys].sel_inds_hist.shape[0]*input_data[keys].sel_inds_hist.shape[1],:],
                (input_data[keys].sel_inds_hist.shape[0], input_data[keys].sel_inds_hist.shape[1], 2+1)) #to 3D (gen, inds, vars+score) again
            
            rows_index_to_delete =  [range(input_data[keys].sel_inds_hist.shape[0]*input_data[keys].sel_inds_hist.shape[1])] #we will delete
            #the first gen*indspergen, this is, all the elements from the current input_data[keys]
            
            embedplusscore = np.delete(embedplusscore,rows_index_to_delete ,axis=0)

        """After this loop embedplusscore is empty"""

        """Now the actual plot"""
        #Default markers and colors
        mk_col = ['r', 'b', 'k', 'g', 'y', 'm']
        markers = ['o', 's', 'P', 'd', '<','>']

        # Create a colormap
        cmap = plt.cm.viridis_r

        def update_plot(gen):       
            plt.figure(figsize=fig_size)
            plt.clf()  # Clear previous plot

            # Normalize values to map to the colormap. We compare individuals in the same generation for all EDAs.
            minscore = np.infty
            for arrays in lmds_dict.values():
                if gen > len(arrays)-1: #EDAs can have different gen sizes. So max gen is limited for each EDAs max.
                    gen = len(arrays)-1
                minscore = min(minscore, arrays[gen, 0, -1]) #arrays[gen, 0, -1] is the best individual for each EDA

            maxscore = -np.infty
            for arrays in lmds_dict.values():
                if gen > len(arrays)-1:
                    gen = len(arrays)-1
                maxscore = max(maxscore, arrays[gen, -1, -1]) #arrays[gen, -1, -1] is the worst (selected) ind for each EDA

            norm = mpcol.Normalize(vmin=minscore, vmax=maxscore)
            
            c = 0 #counter to iterate colors and markers
            for keys in lmds_dict:
                if gen > len(lmds_dict[keys])-1:
                    gen = len(lmds_dict[keys])-1
                # Create a list of colors corresponding to each row for each EDA according to normalization
                colors = [cmap(norm(val)) for val in lmds_dict[keys][gen,:,-1]]
                plt.scatter(lmds_dict[keys][gen,:,0], lmds_dict[keys][gen,:,1], color=colors, marker=markers[c])#, label=keys)
                plt.plot(lmds_dict[keys][0:gen, 0, 0], lmds_dict[keys][:gen, 0, 1], f'{mk_col[c]}-', alpha=0.4)#, label=f'{keys} best trajectory')
                c += 1
                # Add labels every ten generations
                for i in range(0, gen):
                    if i % trajectory_labels == 0:
                        plt.text(lmds_dict[keys][i, 0, 0], lmds_dict[keys][i, 0, 1], f'{i}', fontsize=10)

            legend_elements = []
            c=0
            for keys in lmds_dict:
                legend_elements.append(Line2D([0],[0], marker=markers[c], markerfacecolor='black', label=keys, color='w'))
                legend_elements.append(Line2D([0],[0], color=mk_col[c], label=keys))
                c+=1

            plt.legend(handles=legend_elements, loc='upper right')    
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
        for keys in lmds_dict:
            maxgens.append(len(lmds_dict[keys]))
        maxgen=max(maxgens)

        #Slider
        generation_slider = widgets.IntSlider(value=0, min=0, max=maxgen-1,
                                            step=1, description='Generation', layout=widgets.Layout(width='100%'))


        # Use interact to connect the slider with the update_plot function
        interact(update_plot, gen=generation_slider) 
        display(save_location_text, save_button)
        
        if return_embedded_data:
             return lmds_dict