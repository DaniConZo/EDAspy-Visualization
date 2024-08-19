import numpy as np
from scipy.spatial import distance
from EDAspy.optimization import EdaResult
import random
from typing import Union

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

def lmds_calc(input_data: Union[EdaResult, dict] = None, landmark_points: float = 0.1, landmark_points_seed: int= 0):
    
    '''This function computes 2D dimensionality reduction of the solutions visited by the algorithm using LMDS.
       If input_data is a single EdaResult, it returns a 3D numpy array with the embedded data (gen, ind, var1+var2+score).
       If input_data is a dictionary with multiple EdaResults, it returns a dictionary with embedded data for each EDA with same keys. In this case
       the embedding is calculated with all points visited by all EDAs allowing comparision of search trajectories


        :param input_data: eda_result object (or dictionary containig multiple)
        :param landmark_points: proportion of landmark points 0-1. 1 will be a regular MDS
        :param landmark_points_seed: int to set seed to pick landmark points randomly
                  
        :type input_data: Union[dict, EdaResult]
        :type landmark_points: float
        :type landmark_points_seed: int
        :return: Union[numpy.ndarray, dict]'''
    
    assert isinstance(input_data, Union[EdaResult, dict]), 'Input object is not EdaResult class, or dictionary'
    if isinstance(input_data, dict):
        for keys in input_data:
            assert isinstance(input_data[keys], EdaResult), 'dict.values() must be EdaResult objects'
            assert len(input_data) <= 6, "Can't embed more than 6 EDAs !!!"
    assert type(landmark_points)==float, 'number of landmark_points must be a float'
    assert type(landmark_points_seed) is int, 'seed must be an integer'
    
    #If input is single EdaResult
    if isinstance(input_data, EdaResult):
          
        if input_data.sel_inds_hist[:,:,:-1].shape[-1] < 3:
            return('Your input is already 1D or 2D!!!!')
        
        #Reshape the array to have individuals as rows and variables as columns
        indstoembedLMDS = input_data.sel_inds_hist[:, :, :]
        reshaped_arrayLMDS = indstoembedLMDS.reshape(-1, indstoembedLMDS.shape[-1])

        ## Landmark points are chosen randomly.  10% of total points by default
        random.seed(landmark_points_seed)
        lands = random.sample(range(0,reshaped_arrayLMDS.shape[0],1),int(landmark_points * reshaped_arrayLMDS.shape[0])) 
        lands = np.array(lands,dtype=int)
        Dl2 = distance.cdist(reshaped_arrayLMDS[lands,:-1], reshaped_arrayLMDS[:,:-1], 'euclidean') #distance matrix between landmarks and rest of points. Not taking last column because it is the score
        reduced_points = landmark_MDS(Dl2,lands,2) # 2D points

        #Once we have reduced the data, we reshape the array to 3D again [gen, ind, vars+fitness]. Adding the fitness of each individual to 2D points

        lmds_embed = np.column_stack((reduced_points, reshaped_arrayLMDS[:,-1]))
        lmds_embed = np.reshape(lmds_embed, (input_data.sel_inds_hist.shape[0], input_data.sel_inds_hist.shape[1], 2+1)) 
        
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

        """After this loop embedplusscore is empty, and the dictionary is complete"""

        return lmds_dict