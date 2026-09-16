"""scipyOptimizer module. This module contains the ScipyOptimizer class, which is used to optimize a given function using the scipy library."""
######### Package Imports #########################################################################
import sys, math, copy, os, shutil
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from optimpv import *
from optimpv.general.BaseAgent import BaseAgent
from scipy.optimize import minimize, least_squares

######### Optimizer Definition #######################################################################

class ScipyOptimizer(BaseAgent):
    """
    Initialize the ScipyOptimizer class. This class is used to optimize a given function using the scipy library.
    
    Parameters
    ----------
    params : list of Fitparam() objects, optional
        List of Fitparam() objects, by default None
    agents : list of Agent() objects, optional
        List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
    method : str, optional
        Optimization method to use with scipy.minimize, by default 'L-BFGS-B'
    options : dict, optional
        Options to pass to scipy.minimize, by default None
    name : str, optional
        Name of the optimization process, by default 'scipy_opti'
    **kwargs : dict
        Additional keyword arguments including:
        - parallel_agents: bool, whether to run agents in parallel
        - max_parallelism: int, maximum number of parallel processes
        - verbose_logging: bool, whether to log verbose information
    """
    def __init__(self, params=None, agents=None, method='L-BFGS-B', options=None, name='scipy_opti', **kwargs):
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        self.method = method
        self.options = options if options is not None else {}
        self.name = name
        self.kwargs = kwargs
        self.results = None
        self.all_metrics = None
        self.all_evaluations = []
        
        # Set defaults from kwargs
        self.parallel_agents = kwargs.get('parallel_agents', True)
        self.max_parallelism = kwargs.get('max_parallelism', os.cpu_count()-1)
        self.verbose_logging = kwargs.get('verbose_logging', True)
    
    def create_search_space(self, params):
        """
        Create a search space suitable for scipy.optimize.minimize based on parameters.
        
        Parameters
        ----------
        params : list of Fitparam() objects
            List of parameter objects
            
        Returns
        -------
        tuple
            A tuple containing (x0, bounds) where x0 is the initial parameter vector
            and bounds is a list of (lower, upper) bound tuples
        """
        x0 = []
        bounds = []
        self.param_mapping = []  # Store mapping of parameter indices to names for reconstruction
        
        for i, param in enumerate(params):
            if param.type == 'fixed':
                # Fixed parameters are not included in the optimization
                continue
                
            self.param_mapping.append(param.name)
            
            if param.value_type == 'float':
                if param.force_log:
                    # For log-scale parameters, we optimize in log space
                    x0.append(np.log10(param.value))
                    bounds.append((np.log10(param.bounds[0]), np.log10(param.bounds[1])))
                else:
                    # For regular float parameters
                    scale_factor = param.fscale if hasattr(param, 'fscale') else 1.0
                    x0.append(param.value / scale_factor)
                    bounds.append((param.bounds[0] / scale_factor, param.bounds[1] / scale_factor))
            
            elif param.value_type == 'int':
                # For integer parameters, we optimize in integer space
                # but will round to integers when evaluating
                step_size = param.stepsize if hasattr(param, 'stepsize') else 1
                x0.append(param.value / step_size)
                bounds.append((param.bounds[0] / step_size, param.bounds[1] / step_size))
            
            # Note: categorical, string, and boolean parameters are not directly 
            # supported by scipy.optimize.minimize and require special handling
        
        return np.array(x0), bounds
    
    def reconstruct_params(self, x):
        """
        Reconstruct a parameter dictionary from an optimization vector.
        
        Parameters
        ----------
        x : array-like
            Parameter vector from the optimizer
            
        Returns
        -------
        dict
            Dictionary mapping parameter names to values
        """
        param_dict = {}
        x_idx = 0
        
        for i, param in enumerate(self.params):
            if param.type == 'fixed':
                # Include fixed parameters with their fixed values
                param_dict[param.name] = param.value
                continue
                
            # Get the optimized value for this parameter
            if param.value_type == 'float':
                if param.force_log:
                    # Convert back from log space
                    param_dict[param.name] = 10 ** x[x_idx]
                else:
                    # Apply scaling factor
                    scale_factor = param.fscale if hasattr(param, 'fscale') else 1.0
                    param_dict[param.name] = x[x_idx] * scale_factor
            
            elif param.value_type == 'int':
                # Convert back to integer with proper step size
                step_size = param.stepsize if hasattr(param, 'stepsize') else 1
                param_dict[param.name] = int(round(x[x_idx] * step_size))
            
            x_idx += 1
            
        return param_dict
    
    def create_objective(self, multi_objective=False):
        """
        Create an objective function for scipy.minimize based on agent evaluations.
        
        Parameters
        ----------
        multi_objective : bool, optional
            Whether to handle multiple objectives, by default False

        Returns
        -------
        callable
            Objective function that takes a parameter vector and returns a scalar value
        """
        # Define the objective function
        def objective(x):
            # Map parameter vector to dictionary
            
            param_dict = {}
            idx = 0
            for i, param in enumerate(self.params):
                if param.type != 'fixed':
                #     # Include fixed parameters with their fixed values
                #     param_dict[param.name] = param.value
                #     continue
                # else:
                    param_dict[param.name] = x[idx]
                    idx += 1
                    
            # Evaluate all agents
            results = {}
            for agent in self.agents:
                agent_results = agent.run_Ax(param_dict)
                results.update(agent_results)
            
            # Store evaluation results for later analysis
            self.all_evaluations.append({
                'params': param_dict.copy(),
                'results': results.copy()
            })
            
            if not multi_objective:
                # Combine results according to objectives (assuming minimization)
                combined_result = 0
                for agent in self.agents:
                    for i in range(len(agent.all_agent_metrics)):
                        value = results[agent.all_agent_metrics[i]]
                        # Adjust sign for maximization objectives
                        sign = 1 if agent.minimize[i] else -1
                        combined_result += sign * value
                
                return combined_result/len(self.agents)  # Average over agents
            else:
                # For multi-objective, return all objective values
                objective_values = []
                for agent in self.agents:
                    for i  in range(len(agent.all_agent_metrics)):
                        value = results[agent.all_agent_metrics[i]]
                        # Adjust sign for maximization objectives
                        sign = 1 if agent.minimize[i] else -1
                        objective_values.append(sign * value)
                
                return np.array(objective_values)
        
        return objective
    
    def create_metrics_list(self):
        """
        Create a list of all metrics from all agents.
        
        Returns
        -------
        list
            List of metric names
        """
        metrics = []
        for agent in self.agents:
            for i in range(len(agent.all_agent_metrics)):
                metrics.append(agent.all_agent_metrics[i])
        return metrics
    
    def evaluate(self, args):
        """
        Evaluate the agent on a parameter point.
        
        Parameters
        ----------
        args : tuple
            Tuple containing the index of the agent, the agent, the index of the parameter point and the parameter point
        
        Returns
        -------
        tuple
            Tuple containing the index of the parameter point and the results of the agent on the parameter point
        """
        idx, agent, p_idx, p = args
        res = agent.run_Ax(p)
        return p_idx, res
    
    def optimize(self, multi_objective=False):
        """
        Run the optimization process using scipy.minimize.
        
        Parameters
        ----------
        multi_objective : bool, optional
            Whether to use multi-objective optimization, by default False
        
        Returns
        -------
        object
            The optimization results
        """
        if self.verbose_logging:
            print(f"Starting optimization using {self.method} method")
        if 'tol' in self.options:
            tol = self.options['tol']
        else:
            tol = None

        # Create the metrics list
        if self.all_metrics is None:
            self.all_metrics = self.create_metrics_list()
        
        # Get initial parameter values and bounds using the new method
        x0, bounds = self.create_search_space(self.params)
        
        # Create the objective function
        objective = self.create_objective(multi_objective=multi_objective)
        
        # Run the optimization
        if not multi_objective:
            # Single objective optimization
            result = minimize(
                objective,
                x0,
                method=self.method,
                bounds=bounds,
                options=self.options
            )
            
            self.results = result
            
            # Update parameter values with the optimal solution
            param_dict = self.reconstruct_params(result.x)
            for param in self.params:
                if param.name in param_dict:
                    param.value = param_dict[param.name]
            
            if self.verbose_logging:
                print(f"Optimization completed with status: {result.message}")
                print(f"Final objective value: {result.fun}")
        else:
            # For multi-objective, we'll use a weighted sum approach
            # This is a simplified approach; more advanced methods would be better
            # for true multi-objective optimization
            weights = self.kwargs.get('objective_weights', None)
            if weights is None:
                weights = np.ones(len(self.all_metrics)) / len(self.all_metrics)
            else:
                weights = np.asarray(weights)
                if len(weights) != len(self.all_metrics):
                    raise ValueError("Weights length must match the number of metrics.")
                
            def weighted_objective(x):
                obj_values = objective(x)
                return np.sum(weights * obj_values)
            
            result = minimize(
                weighted_objective,
                x0,
                method=self.method,
                bounds=bounds,
                options=self.options,
                tol=tol
            )
            
            self.results = result
            
            # Update parameter values with the optimal solution
            param_dict = self.reconstruct_params(result.x)
            for param in self.params:
                if param.name in param_dict:
                    param.value = param_dict[param.name]
            
            if self.verbose_logging:

                print(f"Optimization completed with status: {result.message}")
                print(f"Final objective value: {result.fun}")
        
        return result

    def optimize_least_squares(self, multi_objective=False):
        """
        Run the optimization process using scipy.optimize.least_squares.
        
        Parameters
        ----------
        multi_objective : bool, optional
            Whether to use multi-objective optimization, by default False
        
        Returns
        -------
        object
            The optimization results
        """
        if self.verbose_logging:
            print(f"Starting optimization using {self.method} method")

        if self.method not in ['trf', 'dogbox', 'lm']:
            raise ValueError("Method must be one of 'trf', 'dogbox', or 'lm' for least_squares optimization.")
        
        # Create the metrics list
        if self.all_metrics is None:
            self.all_metrics = self.create_metrics_list()
        
        # Get initial parameter values and bounds using the new method
        x0, bounds = self.create_search_space(self.params)
        # Convert bounds format for least_squares (expects separate lower and upper arrays)
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds = (lower_bounds, upper_bounds)
        # Create the objective function
        objective = self.create_objective(multi_objective=multi_objective)
        
        # Run the optimization
        if not multi_objective:
            # Single objective optimization
            result = least_squares(
                objective,
                x0,
                method=self.method,
                bounds=bounds,
                **self.options
            )
            
            self.results = result
            
            # Update parameter values with the optimal solution
            param_dict = self.reconstruct_params(result.x)
            for param in self.params:
                if param.name in param_dict:
                    param.value = param_dict[param.name]
            
            if self.verbose_logging:
                print(f"Optimization completed with status: {result.message}")
                print(f"Final objective value: {result.fun}")
        else:
            # For multi-objective, we'll use a weighted sum approach
            # This is a simplified approach; more advanced methods would be better
            # for true multi-objective optimization
            weights = self.kwargs.get('objective_weights', None)
            if weights is None:
                weights = np.ones(len(self.all_metrics)) / len(self.all_metrics)
            else:
                weights = np.asarray(weights)
                if len(weights) != len(self.all_metrics):
                    raise ValueError("Weights length must match the number of metrics.")
                
            def weighted_objective(x):
                obj_values = objective(x)
                return np.sum(weights * obj_values)

            result = least_squares(
                weighted_objective,
                x0,
                method=self.method,
                bounds=bounds,
                **self.options
            )
            
            self.results = result
            
            # Update parameter values with the optimal solution
            param_dict = self.reconstruct_params(result.x)
            for param in self.params:
                if param.name in param_dict:
                    param.value = param_dict[param.name]
            
            if self.verbose_logging:

                print(f"Optimization completed with status: {result.message}")
                print(f"Final objective value: {result.fun}")
        
        return result
    
    def update_params_with_best_balance(self, return_best_balance=False):
        """
        Update the parameters with the best balance of all metrics.
        The best balance is defined by ranking the results for each metric and taking the parameters that has the lowest sum of ranks.
        
        Parameters
        ----------
        return_best_balance : bool, optional
            Whether to return the best balance index and parameters, by default False
            
        Returns
        -------
        tuple, optional
            Tuple containing the best balance index and parameters, if return_best_balance is True
            
        Raises
        ------
        ValueError
            We need at least one metric to update the parameters
        """
        if not self.all_evaluations:
            raise ValueError("No evaluations have been performed.")
        
        if len(self.all_metrics) == 0:
            raise ValueError("We need at least one metric to update the parameters")
        
        # If we have one objective, just use the best result
        if len(self.all_metrics) == 1:
            best_idx = np.argmin([eval_result['results'][self.all_metrics[0]] for eval_result in self.all_evaluations])
            best_params = self.all_evaluations[best_idx]['params']
            
            # Update parameters
            for p in self.params:
                if p.name in best_params:
                    p.value = best_params[p.name]
                    
            if return_best_balance:
                return best_idx, best_params
        # If we have multiple objectives, find the best balance
        else:
            # Collect all results
            metrics_values = []
            for metric in self.all_metrics:
                metric_vals = [eval_result['results'][metric] for eval_result in self.all_evaluations]
                metrics_values.append(metric_vals)
            
            # Determine minimize/maximize for each metric
            minimizes = []
            for agent in self.agents:
                for i in range(len(agent.minimize)):
                    minimizes.append(agent.minimize[i])
            
            # Calculate ranks for each metric
            ranks = []
            for i in range(len(self.all_metrics)):
                # Rank values (ascending if minimize, descending if maximize)
                vals = np.array(metrics_values[i])
                rank = np.argsort(np.argsort(vals if minimizes[i] else -vals))
                ranks.append(rank)
            
            # Find the best balance (lowest sum of ranks)
            sum_ranks = np.sum(ranks, axis=0)
            best_idx = np.argmin(sum_ranks)
            best_params = self.all_evaluations[best_idx]['params']
            
            # Update parameters
            for p in self.params:
                if p.name in best_params:
                    p.value = best_params[p.name]
            
            if return_best_balance:
                return best_idx, best_params

