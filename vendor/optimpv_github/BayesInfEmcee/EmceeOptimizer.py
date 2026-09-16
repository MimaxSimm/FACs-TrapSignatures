import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

import copy, os, warnings
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from logging import Logger
import time
import logging
# logging.basicConfig(format='[%(levelname)s %(asctime)s] %(name)s: %(message)s', level=logging.INFO)

# logger = logging.getLogger(__name__)
from optimpv.general.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger('EmceeOptimizer')
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)
# Assuming other necessary imports like the base optimizer class or model interface exist
from optimpv.general.BaseAgent import BaseAgent # Import BaseAgent

class EmceeOptimizer(BaseAgent): # Inherit from BaseAgent
    """
    Optimizer using the emcee library for MCMC Bayesian inference.
    Inherits from BaseAgent and interacts with Agent objects.
    """
    def __init__(self, params=None, agents=None, nwalkers=20, nsteps=1000, burn_in=100, progress=True, name='emcee_opti', **kwargs):
        """_summary_

        Parameters
        ----------
        params : list of Fitparam() objects, optional
            List of Fitparam() objects, by default None
        agents : list of Agent() objects, optional
            List of Agent() objects see optimpv/general/BaseAgent.py for a base class definition, by default None
        nwalkers : int, optional
            Number of walkers in the MCMC ensemble, by default 20
        nsteps : int, optional
            Number of MCMC steps per walker, by default 1000
        burn_in : int, optional
            Number of steps to discard as burn-in, by default 100
        progress : bool, optional
            Whether to display the progress bar during sampling, by default True
        name : str, optional
            Name for the optimization process, by default 'emcee_opti'
        **kwargs : dict, optional
            Additional keyword arguments (e.g., parallel processing settings), by default None

        Raises
        ------
        ValueError
            Agents must minimize all targets. Please set minimize=True for all targets.
        ValueError
            Parameter must be of type 'float'. Please set value_type='float' for all parameters.
        ValueError
            Number of dimensions (parameters) cannot be determined.


        """        

        # super().__init__() # Call BaseAgent init if needed
        self.params = params
        if not isinstance(agents, list):
            agents = [agents]
        self.agents = agents
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.burn_in = burn_in
        self.progress = progress
        self.name = name
        self.kwargs = kwargs

        # make sure all agents target are minimize
        for agent in self.agents:
            if hasattr(agent, 'minimize'):
                for i in range(len(agent.minimize)):
                    if not agent.minimize[i]:
                        raise ValueError(f"Agent {agent.name} must minimize all targets. Please set minimize=True for all targets.")
        # make sure all of the params val_type are floats
        for param in self.params:
            if param.value_type != 'float':
                raise ValueError(f"Parameter {param.name} must be of type 'float'. Please set value_type='float' for all parameters.")
            
        # Extract settings from kwargs
        self.use_pool = kwargs.get('use_pool', True) # Control whether to use multiprocessing Pool
        self.max_parallelism = kwargs.get('max_parallelism', os.cpu_count() - 1)

        # Process parameters to get dimensions, bounds, initial guess, names
        self.x0, self.bounds, self.param_mapping, self.log_params_indices = self.create_search_space(self.params)
        self.ndim = len(self.x0)

        self.param_names = [p.display_name if hasattr(p,'display_name') else p.name for p in self.params if p.name in self.param_mapping] # Use display names if available

        if self.ndim == 0:
            raise ValueError("Number of dimensions (parameters) cannot be determined.")

        self.sampler = None
        self.chain = None
        self.flat_samples = None
        self.results = None
        self.all_metrics = self.create_metrics_list() # Helper to get metric names

    # def _get_all_metrics(self):
    #     """ Get a list of all metric names from the agents. """
    #     metrics = []
    #     for agent in self.agents:
    #         for i in range(len(agent.metric)):
    #             if hasattr(agent, 'exp_format'):
    #                 metrics.append(f"{agent.name}_{agent.exp_format[i]}_{agent.metric[i]}")
    #             else:
    #                 metrics.append(f"{agent.name}_{agent.metric[i]}")
    #     return metrics
    
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
    

    def create_search_space(self, params):
        """Create search space details (initial vector, bounds, mapping) from FitParam list.

        Parameters
        ----------
        params : list of FitParam
            List of FitParam objects defining the parameters to optimize.

        Returns
        -------
        tuple
            x0 : array
                Initial parameter vector for optimization (potentially log-transformed).
            bounds : list of tuples
                List of (lower, upper) bound tuples for optimization vector.
            param_mapping : list
                List of parameter names corresponding to x0 elements.
            log_params_indices : list
                Indices of parameters optimized in log10 space.

        Raises
        ------
        ValueError
            If a parameter type is unsupported (not 'float').
        """
        # Initialize empty lists for x0, bounds, and parameter mapping
        x0 = []
        bounds = []
        param_mapping = []
        log_params_indices = []

        for i, param in enumerate(params):
            if param.type == 'fixed':
                continue

            param_mapping.append(param.name)
            current_index = len(x0) # Index in the optimization vector 'x'

            if param.value_type == 'float':
                if param.force_log:
                    log_params_indices.append(current_index)
                    x0.append(np.log10(param.value))
                    # Ensure bounds are positive before log10
                    lower_bound = np.log10(param.bounds[0]) if param.bounds[0] > 0 else -np.inf
                    upper_bound = np.log10(param.bounds[1]) if param.bounds[1] > 0 else np.inf
                    bounds.append((lower_bound, upper_bound))
                else:
                    scale_factor = param.fscale if hasattr(param, 'fscale') and param.fscale is not None else 1.0
                    x0.append(param.value / scale_factor)
                    bounds.append((param.bounds[0] / scale_factor, param.bounds[1] / scale_factor))

            else:
                raise ValueError(f"Unsupported parameter type: {param.value_type}. Only 'float' is supported.")

        return np.array(x0), bounds, param_mapping, log_params_indices

    def reconstruct_params(self, x_opt):
        """Reconstruct a full parameter dictionary from an optimization vector x_opt.

        Parameters
        ----------
        x_opt : array-like
            Parameter vector from the optimizer (potentially log-transformed).

        Returns
        -------
        dict
            Dictionary mapping full parameter names to their values.

        Raises
        ------
        ValueError
            If a parameter type is unsupported (not 'float').
        """        
        # Initialize empty dictionary for reconstructed parameters
        param_dict = {}
        opt_idx = 0

        for param in self.params:
            if param.type == 'fixed':
                param_dict[param.name] = param.value
            else:
                # Find the corresponding value in x_opt
                current_val = x_opt[opt_idx]

                if param.value_type == 'float':
                    if opt_idx in self.log_params_indices:
                        if param.force_log:
                            # If log10 transformed, convert back to original scale
                            param_dict[param.name] = 10**current_val
                            
                        else:
                            param_dict[param.name] = current_val
                    else:
                        scale_factor = param.fscale if hasattr(param, 'fscale') and param.fscale is not None else 1.0
                        param_dict[param.name] = current_val * scale_factor
                else:
                    raise ValueError(f"Unsupported parameter type: {param.value_type}. Only 'float' is supported.")

                opt_idx += 1

        return param_dict


    def _log_likelihood(self, theta, agents=None):
        """Calculate the log-likelihood based on agent evaluations.
        Assumes agent.run_Ax returns a dictionary where keys match self.all_metrics
        and values are loss/metric values (e.g., sum of squared errors).
        Converts loss to log-likelihood assuming Gaussian noise.

        Parameters
        ----------
        theta : array-like
            contains the parameters to evaluate
        agents : list of Agent() objects, optional
            List of Agent() objects to evaluate the likelihood, by default None

        Returns
        -------
        float
            Log-likelihood value. Returns -np.inf for invalid evaluations (e.g., NaN, Inf).
        """        

        # param_dict = self.reconstruct_params(theta)
        param_dict = {}
        idx = 0
        for i, param in enumerate(self.params):
            if param.type == 'fixed':
                param_dict[param.name] = param.value
            else:
                param_dict[param.name] = theta[idx]
                idx += 1

        total_log_like = 0.0
        all_results = {}

        # Evaluate all agents for the given parameter set
        # Note: This part is sequential. Parallelism happens at the walker level in emcee.
        try:
            for agent in agents:
                # Assuming run_Ax needs the parameter dictionary
                agent_results = agent.run_Ax(param_dict)
                all_results.update(agent_results)

            # Combine results into a single log-likelihood value
            # Simple approach: sum of negative losses (assuming loss ~ -2*logL)
            for metric_name in self.all_metrics:
                if metric_name in all_results:
                    loss_val = all_results[metric_name]
                    if np.isnan(loss_val) or not np.isfinite(loss_val):
                        return -np.inf # Penalize NaNs or Infs heavily
                    
                    log_like_contribution = -0.5 * loss_val 
                    total_log_like += log_like_contribution
                else:
                    # Metric not found in results, indicates an issue
                    warnings.warn(f"Metric {metric_name} not found in agent results for params {param_dict}, something went wrong.")
                    return -np.inf
            if not np.isfinite(total_log_like):
                 return -np.inf

            return total_log_like

        except Exception as e:
            # Handle potential errors during agent evaluation (e.g., simulation crashes)
            # print(f"Error during agent evaluation: {e}") # Optional: log error
            return -np.inf # Penalize parameters causing errors


    def _log_prior(self, theta):
        """
        Calculate the log-prior probability of the parameters (in optimization space).
        Uses bounds defined during initialization. Assumes uniform prior within bounds.

        Parameters
        ----------
        theta : array-like
            Parameter vector in optimization space.

        Returns
        -------
        float
            Log-prior value. Returns -np.inf for invalid evaluations (e.g., outside bounds).
        """
        for i in range(self.ndim):
            min_val, max_val = self.bounds[i]
            if not (min_val <= theta[i] <= max_val):
                return -np.inf
        # Add other priors if necessary (e.g., Gaussian priors on specific parameters)
        # Remember theta is potentially log10 transformed for some parameters.
        # Priors should be defined on the space you are sampling (theta).
        return 0.0 # Flat prior within bounds

    def _log_probability(self, theta, agents=None):
        """
        Calculate the total log-probability (log-prior + log-likelihood).
        This is the function called by emcee.
        It combines the prior and likelihood evaluations.
        Handles potential issues with likelihood evaluation and prior violations.

        Parameters
        ----------
        theta : array-like
            Parameter vector in optimization space.
        agents : list of Agent() objects, optional
            List of Agent() objects to evaluate the likelihood, by default None

        Returns
        -------
        float
            Total log-probability value. Returns -np.inf for invalid evaluations (e.g., NaN, Inf).
        """
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf # Prior is violated

        # Likelihood calculation might fail, handle potential errors
        try:
            ll = self._log_likelihood(theta, agents=agents)
            if not np.isfinite(ll):
                return -np.inf # Likelihood calculation failed or returned non-finite value
        except Exception as e:
            # Catch unexpected errors during likelihood calculation
            warnings.warn(f"Unexpected error in log_likelihood: {e}")
            return -np.inf
        return lp + ll
    
    def initialize_walkers(self):
        # use LHS to initialize walkers
        # LHS is a sampling method that ensures the samples are evenly distributed in the parameter space
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=self.ndim)
        unit_samples = sampler.random(n=self.nwalkers)
        lower_bounds, upper_bounds = np.array(self.bounds).T
        scaled_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

        return scaled_samples

    def optimize(self):
        """
        Run the MCMC optimization using emcee.
        """
        verbose_logging = self.kwargs.get('verbose_logging',True)
        # Initialize walkers
        # Start walkers in a small ball around the initial guess x0
        # pos = self.x0 + 1e-4 * np.random.randn(self.nwalkers, self.ndim)
        
        # # Ensure initial positions respect bounds
        # for i in range(self.nwalkers):
        #     for j in range(self.ndim):
        #         pos[i, j] = np.clip(pos[i, j], self.bounds[j][0], self.bounds[j][1])
        #     # Optional: Re-check if any clipped position violates prior (e.g., if bounds were -inf/inf)
        #     while not np.isfinite(self._log_prior(pos[i])):
        #         # Resample if prior is still violated (should be rare with clipping if bounds are finite)
        #         pos[i] = self.x0 + 1e-3 * np.random.randn(self.ndim)
        #         for j in range(self.ndim):
        #             pos[i, j] = np.clip(pos[i, j], self.bounds[j][0], self.bounds[j][1])
        pos = self.initialize_walkers()
        if verbose_logging:
            # Log initial positions
            print("----------------------------------------------------\n")
            logger.info(f"Running MCMC with {self.nwalkers} walkers for {self.nsteps} steps...")

        # Setup multiprocessing pool if enabled
        pool = None
        map_fn = map
        if self.use_pool:
            pool = Pool(processes=self.max_parallelism)
            map_fn = pool.map # Use pool's map for parallelization

        # Create the sampler
        # Pass the pool to EnsembleSampler for parallel likelihood evaluations
        njobs = min(self.nwalkers, self.max_parallelism) if self.use_pool else 1

        # with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            self.nwalkers, self.ndim, self._log_probability, pool=pool, kwargs={'agents': self.agents}
        )

        # Burn-in phase
        state = sampler.run_mcmc(pos, self.burn_in, progress=True)
        sampler.reset()
        # Run MCMC
        sampler.run_mcmc(state, self.nsteps, progress=self.progress)

        # Close the pool if it was used
        if pool is not None:
            pool.close()
            pool.join()

        self.sampler = sampler

        if verbose_logging:
            logger.info("MCMC run complete.")

        # Process results
        self.chain = sampler.get_chain()
        # Adjust thin parameter as needed, based on autocorrelation time analysis if performed
        autocorr_time = sampler.get_autocorr_time(tol=0) # Basic estimate
        thin_factor = int(np.mean(autocorr_time) / 2) if np.all(np.isfinite(autocorr_time)) else 15
        thin_factor = max(1, thin_factor) # Ensure thin >= 1
        self.log_prob_samples = self.sampler.get_log_prob(discard=self.burn_in, thin=thin_factor, flat=True) # Use same thinning
        self.flat_samples = sampler.get_chain(discard=self.burn_in, thin=thin_factor, flat=True)

        # Store results (e.g., median parameters and uncertainties in original parameter space)
        self.results = {}
        # Get median parameters in optimization space
        median_opt_params = np.median(self.flat_samples, axis=0)
        # Convert median parameters back to original space
        median_params_dict = self.reconstruct_params(median_opt_params)

        if verbose_logging:
            # Log median parameters
            logger.info("MCMC Results (Median & 16th/84th Percentiles)")
        for i, name in enumerate(self.param_mapping):
             # Get samples for this parameter in optimization space
             param_samples_opt = self.flat_samples[:, i]

             # Transform samples back to original space if necessary
             if i in self.log_params_indices:
                param_samples_orig = 10**param_samples_opt
             else:
                # Check if scaling was applied
                original_param = next(p for p in self.params if p.name == name)
                scale_factor = original_param.fscale if hasattr(original_param, 'fscale') and original_param.fscale is not None else 1.0
                param_samples_orig = param_samples_opt * scale_factor
                if original_param.value_type == 'int':
                    # Keep as float for distribution analysis, or round if needed
                    pass # param_samples_orig = np.round(param_samples_orig)


             # Calculate percentiles in original space
             mcmc = np.percentile(param_samples_orig, [16, 50, 84])
             q = np.diff(mcmc)
             self.results[name] = {'median': mcmc[1], '16th': mcmc[0], '84th': mcmc[2], 'lower_err': q[0], 'upper_err': q[1]}
             # Find the display name for printing
             display_name = next((p.display_name for p in self.params if p.name == name and hasattr(p,'display_name')), name)
             if verbose_logging:
                # Log results
                logger.info(f"{display_name} ({name}): {mcmc[1]:.4g} (+{q[1]:.3g} / -{q[0]:.3g})")


        # Update self.params with median values
        self.update_params_with_best_balance() # Use max likelihood by default 
        if verbose_logging:
            print("----------------------------------------------------\n")
        return self.results

    def get_best_params(self, method='max_likelihood'):
        """Return the 'best' parameters based on the MCMC samples.
        This method allows the user to specify how to determine the 'best' parameters

        Parameters
        ----------
        method : str
            How to determine 'best' params ('median', 'mean', 'max_likelihood').
            'median' - median of the samples
            'mean' - mean of the samples
            'max_likelihood' - maximum likelihood estimate (MLE) based on the log-probability samples
            'max_likelihood' is the default method.

        Returns
        -------
        dict
            Dictionary of best parameter values in original space.

        Raises
        ------
        ValueError
            If the method is not one of 'median', 'mean', or 'max_likelihood'.
        """        
        # Check if optimization has been run
        if self.flat_samples is None:
            print("Optimization has not been run yet.")
            return None

        if method == 'median':
            best_opt_params = np.median(self.flat_samples, axis=0)
        elif method == 'mean':
            best_opt_params = np.mean(self.flat_samples, axis=0)
        elif method == 'max_likelihood':
            max_prob_index = np.argmax(self.log_prob_samples)
            best_opt_params = self.flat_samples[max_prob_index]
        else:
            raise ValueError("Method must be 'median', 'mean', or 'max_likelihood'")

        # Reconstruct to original parameter space
        best_params_dict = self.reconstruct_params(best_opt_params)
        return best_params_dict

    def update_params_with_best_balance(self, method='max_likelihood', return_best_balance=False):
        """Update the parameters with the best balance based on MCMC results.
        This method updates the parameters in self.params with the best values
        determined by the specified method. It can also return the best parameters
        dictionary if requested.

        Parameters
        ----------
        method : str, optional
            method to determine 'best' params ('median', 'mean', 'max_likelihood'), by default 'max_likelihood'
        return_best_balance : bool, optional
            If True, return the best parameters dictionary, by default False
        """
        if self.results is None:
            raise ValueError("Optimization has not run or results not processed.")

        best_params_dict = self.get_best_params(method=method)

        # Update the FitParam objects in self.params
        for param in self.params:
            if param.name in best_params_dict:
                param.value = best_params_dict[param.name]

        if return_best_balance:
            return best_params_dict # Return the dictionary used for updating

    def get_chain(self, **kwargs):
        """Return the MCMC chain. kwargs passed to sampler.get_chain()"""
        if self.sampler:
            return self.sampler.get_chain(**kwargs)
        return None

    def get_flat_samples(self):
        """Return the flattened samples after burn-in and thinning."""
        return self.flat_samples

    # Add plotting methods if desired (e.g., corner plots, walker traces)
    def plot_corner(self, **kwargs):
        """Generate a corner plot of the posterior distribution."""
        
        title_fmt = kwargs.get('title_fmt', ".4e")
        if self.flat_samples is None:
            print("Optimization has not been run yet.")
            return None

        # Get samples in original parameter space for plotting
        samples_orig = []
        labels_orig = []
        truths_orig = kwargs.get('True_params', None) # Dictionary to hold true values for parameters
        

        for i, name in enumerate(self.param_mapping):
            param_samples_opt = self.flat_samples[:, i]
            original_param = next(p for p in self.params if p.name == name)
            labels_orig.append(original_param.display_name if hasattr(original_param,'display_name') else name)

            if i in self.log_params_indices:
                samples_orig.append(10**param_samples_opt)
            else:
                scale_factor = original_param.fscale if hasattr(original_param, 'fscale') and original_param.fscale is not None else 1.0
                samples_orig.append(param_samples_opt * scale_factor)

        samples_orig_array = np.vstack(samples_orig).T

        # Prepare truths list in the correct order for corner
        if truths_orig is None:
            truths_list = [None] * len(self.param_mapping)
        else:
            truths_list = [truths_orig.get(name, None) for name in self.param_mapping]


        # Default corner plot settings
        corner_kwargs = {
            'labels': labels_orig,
            'show_titles': True,
            'title_kwargs': {"fontsize": 10},
            'quantiles': [0.16, 0.5, 0.84],
            'truths': truths_list,
            'truth_color': 'red',
            'color': 'darkblue',
            'hist2d_kwargs': {
            'cmap': plt.get_cmap('Blues'),
            },
            'hist_kwargs': {
            'color': 'darkblue',
            },
        }
        corner_kwargs.update(kwargs) # Allow user to override defaults
        params_axis_type = []
        for param in self.params:
            if hasattr(param, 'axis_type'):
                params_axis_type.append(param.axis_type)
            else:
                params_axis_type.append('linear')
        fig = corner.corner(samples_orig_array,  axes_scale=params_axis_type,title_fmt=title_fmt,**corner_kwargs)
        return fig

    def plot_traces(self, **kwargs):
        """Plot the MCMC traces for each parameter."""
        import matplotlib.pyplot as plt

        if self.chain is None:
             print("Optimization has not been run yet or chain not stored.")
             return None

        n_steps, n_walkers, n_dim = self.chain.shape
        labels = [p.display_name if hasattr(p,'display_name') else p.name for p in self.params if p.name in self.param_mapping]

        fig, axes = plt.subplots(n_dim, figsize=(10, 2 * n_dim), sharex=True)
        if n_dim == 1: # Handle case with only one parameter
             axes = [axes]

        for i,param in enumerate(self.params):
            ax = axes[i]
            # Plot traces for all walkers
            if param.force_log:
                # If log10 transformed, plot in original space
                ax.plot(10**self.chain[:, :, i], "k", alpha=0.2)
                ax.set_yscale('log')
            else:
                ax.plot(self.chain[:, :, i], "k", alpha=0.2)
            ax.set_xlim(0, n_steps)
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
            # Add burn-in line
            ax.axvline(self.burn_in, color='blue', linestyle='--', lw=1, label=f'Burn-in ({self.burn_in})')
            if i == 0:
                ax.legend(loc='upper right')

        axes[-1].set_xlabel("Step number")
        plt.tight_layout()
        return fig

