#!/usr/bin/env python
# coding: utf-8

import numpy as np


class EdaResult:

    """
    Object used to encapsulate the result and information of the EDA during the execution
    """

    def __init__(self,
                 best_ind: np.array,
                 best_cost: float,
                 n_fev: int,
                 history: list,
                 best_ind_hist: np.array,###Modified by Daniel Conde, added best_ind_hist >> check minimize in eda.py ###
                 sel_inds_hist: np.array, ###Modified by Daniel Conde, added sel_inds_hist >> check minimize in eda.py ###
                 all_inds_hist: np.array, ###Modified by Daniel Conde, added all_inds_hist >> check minimize in eda.py ###
                 #prob_mod_hist: list, ###Modified by Daniel Conde, added prob_mod_hist >> check minimize in eda.py ###
                 prob_clone_hist: list, ##Modified by Daniel Conde, added prob_clone_hist >> check minimize in eda.py ###
                 settings: dict,
                 cpu_time: float):

        """

        :param best_ind: Best result found in the execution.
        :param best_cost: Cost of the best result found.
        :param n_fev: Number of cost function evaluations.
        :param history: Best result found in each iteration of the algorithm.
        :param best_ind_hist: array with best individuals in each iteration/generation ## Added by Daniel Conde ##
        :param sel_inds_hist: array with selected individuals in each iterarion/generation ## Added by Daniel Conde ##
        :param all_inds_hist: array with generated individuals in each iteration/generation ## Added by Daniel Conde ##
        :param prob_mod_hist: list with probabilistic model object from selected individuals in each generation ## Added by Daniel Conde ##
        :param prob_mod_hist: list with probabilistic model object from selected individuals in each generation ## Added by Daniel Conde ##
        :param settings: Configuration of the parameters of the EDA.
        :param cpu_time: CPU time invested in the optimization.
        """

        self.best_ind = best_ind
        self.best_cost = best_cost
        self.n_fev = n_fev
        self.history = history
        self.best_ind_hist = best_ind_hist ###Modified by Daniel Conde, added best_ind_hist >> check minimize in eda.py ###
        self.sel_inds_hist = sel_inds_hist ###Modified by Daniel Conde, added sel_inds_hist >> check minimize in eda.py ###
        self.all_inds_hist = all_inds_hist ###Modified by Daniel Conde, added all_inds_hist >> check minimize in eda.py ###
        #self.prob_mod_hist = prob_mod_hist ###Modified by Daniel Conde, added prob_mod_hist >> check minimize in eda.py ###
        self.prob_clone_hist = prob_clone_hist ###Modified by Daniel Conde, added prob_clone_hist >> check minimize in eda.py ###
        self.settings = settings
        self.cpu_time = cpu_time

    def __str__(self):
        string = "\tNFVALS = " + str(self.n_fev) + " F = " + str(self.best_cost) + "CPU time (s) = " + \
                 str(self.cpu_time) + "\n\tX = " + str(self.best_ind) + "\n\tSettings: " + str(self.settings) + \
                 "\n\tHistory best cost per iteration: " + str(self.history)
        return string
