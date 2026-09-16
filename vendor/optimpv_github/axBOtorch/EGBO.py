#!/usr/bin/env python3

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from logging import Logger
from typing import Any
import numpy as np

import torch
from torch import Tensor
from ax.core.search_space import SearchSpaceDigest
from ax.generators.torch.botorch_modular.acquisition import Acquisition
from ax.generators.torch.botorch_modular.surrogate import Surrogate
from ax.generators.torch_base import TorchOptConfig
from ax.utils.common.logger import get_logger
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.models.model import ModelList
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

import pymoo
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.problem import Problem 
from pymoo.core.termination import NoTermination
from pymoo.algorithms.moo.unsga3 import UNSGA3

CLAMP_TOL = 1e-2
logger: Logger = get_logger(__name__)

class EGBOAcquisition(Acquisition):
    """
    Implement the acquisition function of Evolution-Guided Bayesian Optimization (EGBO).  

    Based on the following paper:  
    Low, A.K.Y., Mekki-Berrada, F., Gupta, A. et al. Evolution-guided Bayesian optimization for constrained multi-objective optimization in self-driving labs. npj Comput Mater 10, 104 (2024). https://doi.org/10.1038/s41524-024-01274-x

    Code inspired the repository:  
    https://github.com/andrelowky/CMOO-Algorithm-Development/

    """

    def __init__(
        self,
        surrogate: Surrogate,
        search_space_digest: SearchSpaceDigest,
        torch_opt_config: TorchOptConfig,
        botorch_acqf_class: type[AcquisitionFunction],
        botorch_acqf_options: dict[str, Any],
        botorch_acqf_classes_with_options: list[
            tuple[type[AcquisitionFunction], dict[str, Any]]
        ]
        | None = None,
        n: int | None = None,
        options: dict[str, Any] | None = None,
        pop_size: int = 256,
        EA_algo: str = "UNSGA3",
    ) -> None:
        tkwargs: dict[str, Any] = {"dtype": surrogate.dtype, "device": surrogate.device}
        options = {} if options is None else options
        surrogate_f = deepcopy(surrogate)
        surrogate_f._model = ModelList(surrogate.model) #

        self.pop_size = pop_size
        self.EA_algo = EA_algo

        # check if EA_algo is valid
        if self.EA_algo not in ["UNSGA3"]:
            raise ValueError(f"EA algorithm {self.EA_algo} not recognized.")

        super().__init__(
            surrogate=surrogate_f,
            search_space_digest=search_space_digest,
            torch_opt_config=torch_opt_config,
            botorch_acqf_class=qLogNoisyExpectedHypervolumeImprovement,
            botorch_acqf_options=botorch_acqf_options,
            options=options,
        )

    def optimize(
        self,
        n: int,
        search_space_digest: SearchSpaceDigest,
        inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        fixed_features: dict[int, float] | None = None,
        rounding_func: Callable[[Tensor], Tensor] | None = None,
        optimizer_options: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a set of candidates via multi-start optimization. Obtains
        candidates and their associated acquisition function values.

        Args:
            n: The number of candidates to generate.
            search_space_digest: A ``SearchSpaceDigest`` object containing search space
                properties, e.g. ``bounds`` for optimization.
            inequality_constraints: A list of tuples (indices, coefficients, rhs),
                with each tuple encoding an inequality constraint of the form
                ``sum_i (X[indices[i]] * coefficients[i]) >= rhs``.
            fixed_features: A map `{feature_index: value}` for features that
                should be fixed to a particular value during generation.
            rounding_func: A function that post-processes an optimization
                result appropriately (i.e., according to `round-trip`
                transformations).
            optimizer_options: Options for the optimizer function, e.g. ``sequential``
                or ``raw_samples``.

        Returns:
            A three-element tuple containing an `n x d`-dim tensor of generated
            candidates, a tensor with the associated acquisition values, and a tensor
            with the weight for each candidate.
        """
        # Get the device information and the optimizer options
        device = self.device
        dtype = self.dtype
        tkwargs = {"device": device, "dtype": dtype}
        acq_func = self.acqf
        _tensorize = partial(torch.tensor, dtype=self.dtype, device=self.device)
        ssd = search_space_digest
        bounds = _tensorize(ssd.bounds).t()

        # First, we optimize the acquisition function using the standard method
        qnehvi_x, expected_acquisition_value, weights = super().optimize(
            n=n,
            search_space_digest=search_space_digest,
            inequality_constraints=inequality_constraints,
            fixed_features=fixed_features,
            rounding_func=rounding_func,
            optimizer_options=optimizer_options,
        )

        ############################################################################
        # Evolutionary Algorithm
        ############################################################################

        # Next, we optimize the acquisition function using NSGA3
        # get the training data
        Xs = []
        Ys = []
        for dataset in self.surrogate.training_data:
            Xs.append(dataset.X)
            Ys.append(dataset.Y)
        x = Xs[0]
        y = Ys[0]
        n_obj = y.shape[1] # number of objectives
        n_var = x.shape[1] # number of variables
        n_constr = 0 # number of constraints

        # we pick out the best points so far to form parents
        pareto_mask = is_non_dominated(y)
        pareto_x = x[pareto_mask].cpu().numpy()
        pareto_y = -y[pareto_mask].cpu().numpy()

        hv=Hypervolume(ref_point=-self.acqf.ref_point)

        if self.EA_algo == "UNSGA3": # see https://pymoo.org/algorithms/moo/unsga3.html
            algorithm = UNSGA3(
                                pop_size=self.pop_size,
                                ref_dirs=get_reference_directions("energy", n_obj, n, seed=None),
                                sampling=pareto_x,
                                #    sampling = qnehvi_x.cpu().numpy(),
                                #crossover=SimulatedBinaryCrossover(eta=30, prob=1.0),
                                #mutation=PolynomialMutation(eta=20, prob=None),
                            )
        else:
            raise ValueError(f"EA algorithm {self.EA_algo} not recognized.")
        
        # make xl, xu from the bounds
        xl = bounds[0].cpu().numpy()
        xu = bounds[1].cpu().numpy()

        # Define the pymoo problem with constraints
        if inequality_constraints and len(inequality_constraints) > 0:
            n_constr = len(inequality_constraints)
            
            # Define the constraint evaluation function
            def evaluate_constraints(x):
                # Convert numpy array to tensor
                x_tensor = torch.tensor(x, **tkwargs)
                
                # Initialize constraint values
                g = np.zeros((x.shape[0], n_constr))
                
                # For each constraint
                for i, (indices, coefficients, rhs) in enumerate(inequality_constraints):
                    # Convert BoTorch constraint (sum_i coef[i]*x[idx[i]] >= rhs)
                    # to pymoo constraint (rhs - sum_i coef[i]*x[idx[i]] <= 0)
                    idx = indices.cpu().numpy()
                    coef = coefficients.cpu().numpy()
                    
                    # Calculate constraint value for each point
                    for j in range(x.shape[0]):
                        constraint_value = rhs - np.sum(coef * x[j, idx])
                        g[j, i] = constraint_value
                        
                return g
            
            # Create pymoo problem with constraints
            pymooproblem = Problem(n_var=n_var, n_obj=n_obj, n_constr=n_constr,
                                   xl=xl, xu=xu, evaluate_constraints=evaluate_constraints)
        else:
            # No constraints case
            n_constr = 0
            pymooproblem = Problem(n_var=n_var, n_obj=n_obj, n_constr=n_constr, 
                                   xl=xl, xu=xu)
        
        # set the algorithm
        algorithm.setup(pymooproblem, termination=NoTermination())
        
        # set the 1st population to the current evaluated population
        pop = algorithm.ask()
        pop.set("F", pareto_y)
        # pop.set("G", pareto_y)
        algorithm.tell(infills=pop)

        # propose children based on tournament selection -> crossover/mutation
        newpop = algorithm.ask()
        nsga3_x = torch.tensor(newpop.get("X"), **tkwargs)
        
        # total pool of candidates for sorting
        candidates = torch.cat([qnehvi_x, nsga3_x])
        acq_value_list = []
        for i in range(0, candidates.shape[0]):
            with torch.no_grad():
                acq_value = acq_func(candidates[i].unsqueeze(dim=0))
                acq_value_list.append(acq_value.item())

        sorted_x = candidates.cpu().numpy()[np.argsort(acq_value_list)]
        
        acqf_values = torch.tensor(acq_value_list, **tkwargs)
        candidates = torch.tensor(sorted_x[-n:], **tkwargs)
        acqf_values = acqf_values[-n:]
        expected_acquisition_value = acqf_values

        return candidates, expected_acquisition_value, weights
