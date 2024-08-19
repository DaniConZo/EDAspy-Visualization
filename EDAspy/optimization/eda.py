#!/usr/bin/env python
# coding: utf-8

import numpy as np
from abc import ABC
from .eda_result import EdaResult
from .custom.probabilistic_models import ProbabilisticModel
from .custom.initialization_models import GenInit
from .utils import _parallel_apply_along_axis
from time import process_time
from .custom.probabilistic_models import UniGauss
import copy

class EDA(ABC):

    """
    Abstract class which defines the general performance of the algorithms. The baseline of the EDA
    approach is defined in this object. The specific configurations is defined in the class of each
    specific algorithm.
    """

    _pm = None
    _init = None

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 tolerance: float = 0.001,
                 elite_factor: float = 0.4,
                 disp: bool = True,
                 parallelize: bool = False,
                 init_data: np.array = None,
                 *args, **kwargs):

        self.disp = disp
        self.size_gen = size_gen
        self.max_iter = max_iter
        self.alpha = alpha
        self.n_variables = n_variables
        self.truncation_length = int(size_gen * alpha)
        self.elite_factor = elite_factor
        self.elite_length = int(size_gen * elite_factor)
        self.parallelize = parallelize

        assert dead_iter <= self.max_iter, 'dead_iter must be lower than max_iter'
        self.dead_iter = dead_iter
        self.tolerance = tolerance

        self.best_mae_global = 999999999999
        self.best_ind_global = np.array([0]*self.n_variables)

        self.evaluations = np.array(0)
        self.evaluations_elite = np.array(0)

        self.generation = None
        self.elite_temp = None

        if parallelize:
            self._check_generation = self._check_generation_parallel
        else:
            self._check_generation = self._check_generation_no_parallel

        # allow initialize EDA with data
        if init_data is not None:
            assert init_data.shape[1] == n_variables, 'The inserted data shape and the number of variables do not match'
            # assert init_data.shape[0] == size_gen, 'The inserted data shape and the generation size do not match'

            self.init_data = init_data
            self._initialize_generation = self._initialize_generation_with_data
        else:
            self._initialize_generation = self._initialize_generation_with_init

    def _new_generation(self):
        # self.generation = np.concatenate([self.pm.sample(size=self.size_gen), self.elite_temp])
        self.generation = self.pm.sample(size=self.size_gen)

    def _initialize_generation_with_data(self) -> np.array:
        return self.init_data

    def _initialize_generation_with_init(self) -> np.array:
        return self.init.sample(size=self.size_gen)

    def _initialize_generation(self) -> np.array:
        raise Exception('Not implemented function')

    def _truncation(self):
        """
        Selection of the best individuals of the actual generation.
        """
        # first add the elite selection to be considered
        self.generation = np.concatenate([self.generation, self.elite_temp])
        self.evaluations = np.append(self.evaluations, self.evaluations_elite)

        # now we truncate
        ordering = self.evaluations.argsort()
        best_indices_truc = ordering[: self.truncation_length]
        best_indices_elit = ordering[: self.elite_length]
        self.elite_temp = self.generation[best_indices_elit, :]
        self.generation = self.generation[best_indices_truc, :]
        self.evaluations_elite = np.take(self.evaluations, best_indices_elit)
        self.evaluations = np.take(self.evaluations, best_indices_truc)

    # check each individual of the generation
    def _check_generation(self, objective_function: callable):
        """
        Check the cost of each individual in the cost function implemented by the user, and updates the
        generation DataFrame.
        """
        raise Exception('Not implemented function')

    def _check_generation_parallel(self, objective_function: callable):
        self.evaluations = _parallel_apply_along_axis(objective_function, 1, self.generation)

    def _check_generation_no_parallel(self, objective_function: callable):
        self.evaluations = np.apply_along_axis(objective_function, 1, self.generation)

    def _update_pm(self):
        """
        Learn the probabilistic model from the best individuals of previous generation.
        """
        self.pm.learn(dataset=self.generation)

    def export_settings(self) -> dict:
        """
        Export the configuration of the algorithm to an object to be loaded in other execution.

        :return: configuration dictionary.
        :rtype: dict
        """
        return {
            "size_gen": self.size_gen,
            "max_iter": self.max_iter,
            "dead_iter": self.dead_iter,
            "n_variables": self.n_variables,
            "alpha": self.alpha,
            "elite_factor": self.elite_factor,
            "disp": self.disp,
            "parallelize": self.parallelize
        }

    def minimize(self, cost_function: callable, output_runtime: bool = True, *args, **kwargs) -> EdaResult:
        """
        Minimize function to execute the EDA optimization. By default, the optimizer is designed to minimize a cost
        function; if maximization is desired, just add a minus sign to your cost function.

        :param cost_function: cost function to be optimized and accepts an array as argument.
        :param output_runtime: true if information during runtime is desired.
        :return: EdaResult object with results and information.
        :rtype: EdaResult
        """

        history = []
        not_better = 0

        t1 = process_time()
        self.generation = self._initialize_generation()
        self._check_generation(cost_function)

        # select just one item to be the elite selection if first iteration
        self.elite_temp = np.array([self.generation[0, :]])
        self.evaluations_elite = np.array([self.evaluations.item(0)])

        best_mae_local = min(self.evaluations)
        self.best_mae_global = min(self.evaluations)
        history.append(best_mae_local)
        aux = np.where(self.evaluations == best_mae_local)[0][0]
        best_ind_local = aux
        self.best_ind_global = aux
        
        ###############Modified by Daniel Conde###########
        if self.disp:
            print('IT: ',0, '\tBest cost: ', self.best_mae_global) ## If desired printing the best element in starting gen
            
        # array to store best individual created in each generation. Different from best global individual
        best_ind_hist = self.generation[best_ind_local]
        ##this is a np array of 1D we reshape it so rows will be individuals and columns will be variables
        best_ind_hist = np.reshape(best_ind_hist, (1, best_ind_hist.size))


        ### numpy array to store selected individuals in each generation for visualization purposes
        sel_inds_hist = []

        ### numpy array to store all individuals in each generation
        all_inds_hist = []

        white_noise = np.random.normal(0, .5, size=(self.truncation_length, self.n_variables))

        ### list to store pm object in each iteration
        prob_mod_hist = []
        prob_clone_hist = []
        #####################################################

        for _ in range(1, self.max_iter):
        

            ############################### Modified by Daniel Conde ###################################
            lista = np.expand_dims(self.generation, axis=0) # Will not contain the evaluations
            all_inds_hist.append(lista) #requires converting to 3D np.array after the for loop
            #########################################################################################

            self._truncation()
            
            #################################################################################################################
            ##### Added white noise to data to prevent 0 variance other EDAs than UMDA (UMDA has a minimum of 0.5 defined)
            #print("(b) Mean: ", self.generation.mean(axis=0), "\tStd.: ", self.generation.std(axis=0))
            if _ > 1:
            	self.generation += white_noise
            #print("(a) Mean: ", self.generation.mean(axis=0), "\tStd.: ", self.generation.std(axis=0))

            #####################################################################################################################
            
            
            self._update_pm()

            ##########Modified by Daniel Conde############
            #Saving selected individuals with their evaluation after truncation in a list
            #first we add evaluation column to generation array to create a (inds, variables+eval) array
            list = np.column_stack((self.generation, self.evaluations))
            #Now we add a dimension (generation), axis at the beginning
            list = np.expand_dims(list, axis=0)
            sel_inds_hist.append(list)
            # requires converting to 3D np.array after the for loop

            ### Saving probalistic model object using deepcopy fails often. Used only with UMDA
            if isinstance(self.pm, UniGauss):
                pm_temp = copy.deepcopy(self.pm)
                prob_clone_hist.append(pm_temp)
            else:
                ### Saving probabilistic model object using clone method from pybnesian
                pm_clone_temp = self.pm.pm.clone()
                prob_clone_hist.append(pm_clone_temp)

            ##############################################

            self._new_generation()
            self._check_generation(cost_function)

            best_mae_local = min(self.evaluations)
            history.append(best_mae_local)

            best_ind_local = np.where(self.evaluations == best_mae_local)[0][0]
            best_ind_local = self.generation[best_ind_local]

            ###############Modified by Daniel Conde###########
            # update array to store best individual in each generation
            #first we reshape the new best ind array
            best_ind_local = np.reshape(best_ind_local, (1, best_ind_local.size))
            ## now we append it to the hist array
            best_ind_hist = np.append(best_ind_hist, best_ind_local, axis=0)
            ##################################################

            # update the best values ever
            if best_mae_local < self.best_mae_global*(1+self.tolerance):
                self.best_mae_global = best_mae_local
                self.best_ind_global = best_ind_local
                not_better = 0

            else:
                not_better += 1
                if not_better == self.dead_iter:
                    break

            if output_runtime:
                print('IT: ', _, '\tBest cost: ', self.best_mae_global)
                ### Before IT: _, we were not printing the best individual in the starting gen so we had a mismatch ###
                ### in the number of best individuals ###

        if self.disp:
            print("\tNFEVALS = " + str(len(history) * self.size_gen) + " F = " + str(self.best_mae_global))
            print("\tX = " + str(self.best_ind_global))

        t2 = process_time()

        ##############Modified by Daniel Conde#########################
        # converting list to 3D numpy array
        sel_inds_hist = np.concatenate(sel_inds_hist, axis=0)
        all_inds_hist = np.concatenate(all_inds_hist, axis=0)
        ###########################################################

        ####################Modified by Daniel Conde ##############
        ##### added best_ind_hist, sel_inds_hist, all_inds_hist, prob_mod_hist, prob_clone_hist to eda_result >>> requires changes in eda_result.py #######
        eda_result = EdaResult(self.best_ind_global, self.best_mae_global, len(history) * self.size_gen,
                               history, best_ind_hist, sel_inds_hist, all_inds_hist , prob_clone_hist ,self.export_settings(), t2-t1)

        return eda_result

    @property
    def pm(self) -> ProbabilisticModel:
        """
        Returns the probabilistic model used in the EDA implementation.

        :return: probabilistic model.
        :rtype: ProbabilisticModel
        """
        return self._pm

    @pm.setter
    def pm(self, value):
        if isinstance(value, ProbabilisticModel):
            self._pm = value
        else:
            raise ValueError('The object you try to set as a probabilistic model does not extend the '
                             'class ProbabilisticModel provided by EDAspy.')

        if len(value.variables) != self.n_variables:
            raise Exception('The number of variables of the probabilistic model is not equal to the number of '
                            'variables of the EDA.')

    @property
    def init(self) -> GenInit:
        """
        Returns the initializer used in the EDA implementation.

        :return: initializer.
        :rtype: GenInit
        """
        return self._init

    @init.setter
    def init(self, value):
        if isinstance(value, GenInit):
            self._init = value
        else:
            raise ValueError('The object you try to set as an initializator does not extend the '
                             'class GenInit provided by EDAspy')

        if value.n_variables != self.n_variables:
            raise Exception('The number of variables of the initializator is not equal to the number of '
                            'variables of the EDA')
