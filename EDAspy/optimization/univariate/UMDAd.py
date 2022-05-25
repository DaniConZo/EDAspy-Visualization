#!/usr/bin/env python
# coding: utf-8

import numpy as np
from .UMDA import UMDA


class UMDAd(UMDA):
    """
    Univariate marginal Estimation of Distribution algorithm binary. New individuals are sampled
    from a univariate binary probabilistic model. It can be used for hyper-parameter optimization
    or to optimize a function.

    UMDA [1] is a specific type of Estimation of Distribution Algorithm (EDA) where new individuals
    are sampled from univariate binary distributions and are updated in each iteration of the
    algorithm by the best individuals found in the previous iteration. In this implementation each
    individual is an array of 0s and 1s so new individuals are sampled from a univariate probabilistic
    model updated in each iteration. Optionally it is possible to set lower and upper bound to the
    probabilities to avoid premature convergence.

    This approach has been widely used and shown to achieve very good results in a wide range of
    problems such as Feature Subset Selection or Portfolio Optimization.

    Example:

        This short example runs UMDAd for a toy example of the One-Max problem.

        .. code-block:: python

            from EDAspy.benchmarks import one_max
            from EDAspy.optimization import UMDAc, UMDAd

            def one_max_min(array):
                return -one_max(array)

            umda = UMDAd(size_gen=100, max_iter=100, dead_iter=10, n_variables=10)
            # We leave bound by default
            best_sol, best_cost, cost_evals = umda.minimize(one_max_min, True)

    References:

        [1]: Mühlenbein, H., & Paass, G. (1996, September). From recombination of genes to the
        estimation of distributions I. Binary parameters. In International conference on parallel
        problem solving from nature (pp. 178-187). Springer, Berlin, Heidelberg.
    """

    best_mae_global = 999999999999
    best_ind_global = -1

    history = []
    evaluations = np.array(0)

    def __init__(self,
                 size_gen: int,
                 max_iter: int,
                 dead_iter: int,
                 n_variables: int,
                 alpha: float = 0.5,
                 vector: np.array = None,
                 lower_bound: float = 0.2,
                 upper_bound: float = 0.8,
                 elite_factor: float = 0.4,
                 disp: bool = True):
        r"""
        Args:
            size_gen: Population size of each generation.
            max_iter: Maximum number of function evaluations.
            dead_iter: Stopping criteria. Number of iterations after with no improvement after which EDA stops.
            n_variables: Number of variables to be optimized.
            alpha: Percentage of population selected to update the probabilistic model.
            vector: Array with shape (n_variables, ) where rows are mean and std of the parameters to be optimized.
            lower_bound: Lower bound imposed to the probabilities of the variables. If not desired, set to 0.
            upper_bound: Upper bound imposed to the probabilities of the variables. If not desired, set to 1.
            elite_factor: Percentage of previous population selected to add to new generation (elite approach).
            disp: Set to True to print convergence messages.
        """

        super().__init__(size_gen, max_iter, dead_iter, n_variables, alpha, elite_factor, disp)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if vector is not None:
            assert vector.shape == (self.n_variables, )
            self.vector = vector
        else:
            self.vector = self._initialize_vector()

        # initialization of generation
        self.generation = np.random.random((self.size_gen, self.n_variables))
        self.generation = self.generation < self.vector
        self.generation = np.array(self.generation, dtype=int)

    def _initialize_vector(self):
        return np.array([0.5]*self.n_variables)

    # build a generation of size SIZE_GEN from prob vector
    def _new_generation(self):
        """
        Build a new generation sampled from the vector of probabilities. Updates the generation pandas dataframe
        """
        gen = np.random.random((self.size_gen, self.n_variables))
        gen = gen < self.vector
        gen = np.array(gen, dtype=int)

        self.generation = self.generation[: int(self.elite_factor * len(self.generation))]
        self.generation = np.vstack((self.generation, gen))

    # update the probability vector
    def _update_vector(self):
        """
        From the best individuals update the vector of univariate distributions in order to the next
        generation can sample from it. Update the vector of univariate binary distributions.
        """
        self.vector = sum(self.generation) / len(self.generation)
        self.vector[self.vector < self.lower_bound] = self.lower_bound
        self.vector[self.vector < self.upper_bound] = self.upper_bound
