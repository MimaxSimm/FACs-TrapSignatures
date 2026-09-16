"""Provides general functionality for Agent objects for Rate Equation simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants
from joblib import Parallel, delayed

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.general.BaseAgent import BaseAgent
from optimpv.RateEqfits.RateEqModel import *
from optimpv.RateEqfits.Pumps import *

from logging import Logger

from optimpv.general.logger import get_logger, _round_floats_for_logging

logger: Logger = get_logger('RateEqModel')
ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES: int = 6
round_floats_for_logging = partial(
    _round_floats_for_logging,
    decimal_places=ROUND_FLOATS_IN_LOGS_TO_DECIMAL_PLACES,
)

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

######### Agent Definition #######################################################################
class RateEqAgent(BaseAgent):
    """Agent object for fitting rate equation models. 
    Available models are:  

    - Bimolecular-Trapping (BT)
    - Bimolecular-Trapping-detrapping (BTD)

    see optimpv.RateEqfits.RateEqModel.py for more details about the available models

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
    y : array-like
        1-D array containing the current values.
    pump_model : function, optional
        Function to get the generated carrier density profile, by default initial_carrier_density.
    pump_args : dict, optional
        Arguments for the pump_model function, by default {}.
    exp_format : str or list of str, optional
        Format of the experimental data, cane be ['trPL','TAS','trMC'], by default 'trPL'.
    exp_format_agrs : dict, optional
        Arguments dependent on the exp_format function, by default {}.
    metric : str or list of str, optional
        Metric to evaluate the model, see optimpv.general.calc_metric for options, by default 'mse'.
    loss : str or list of str, optional
        Loss function to use, see optimpv.general.loss_function for options, by default 'linear'.
    threshold : int or list of int, optional
        Threshold value for the loss function used when doing multi-objective optimization, by default 100.
    minimize : bool or list of bool, optional
        If True then minimize the loss function, if False then maximize the loss function (note that if running a fit minize should be True), by default True.
    yerr : array-like or list of array-like, optional
        Errors in the current values, by default None.
    weight : array-like or list of array-like, optional
        Weights used for fitting if weight is None and yerr is not None, then weight = 1/yerr**2, by default None.
    tracking_metric : str or list of str, optional
        Additional metrics to track and report in run_Ax output, by default None.
    tracking_loss : str or list of str, optional
        Loss functions to apply to tracking metrics, by default None.
    tracking_exp_format : str or list of str, optional
        Experimental formats for tracking metrics, by default None.
    tracking_X : array-like or list of array-like, optional
        X values for tracking metrics, by default None.
    tracking_y : array-like or list of array-like, optional
        y values for tracking metrics, by default None.
    tracking_weight : array-like or list of array-like, optional
        Weights for tracking metrics, by default None.
    name : str, optional
        Name of the agent, by default 'RateEq'.
    **kwargs : dict
        Additional keyword arguments.
    """    
    def __init__(self, params, X, y, model = BT_model, pump_model = initial_carrier_density, pump_args = {}, exp_format = 'trPL', fixed_model_args = {}, metric = 'mse', loss = 'linear', threshold = 100, minimize = True, yerr = None, weight = None, tracking_metric = None, tracking_loss = None, tracking_exp_format = None, tracking_X = None, tracking_y = None, tracking_weight = None, name = 'RateEq', equilibrate = True, detection_limit=None,**kwargs):

        self.params = params
        self.X = X # voltage and Gfrac
        self.y = y
        self.pump_model = pump_model
        self.pump_args = pump_args
        self.model = model
        self.fixed_model_args = fixed_model_args
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.equilibrate = equilibrate
        self.detection_limit = detection_limit
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format
        self.tracking_X = tracking_X
        self.tracking_y = tracking_y
        self.tracking_weight = tracking_weight

        self.yerr = yerr
        self.weight = weight
        self.name = name
        self.kwargs = kwargs

        # Set and validate experiment format
        self.exp_format = exp_format
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]
            
        # Check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['trPL','TAS','trMC']:
                raise ValueError(f'{form} is an invalid exp_format, must be either "trPL", "TAS" or "trMC"')

        # Process main metrics and loss functions
        if self.loss is None:
            self.loss = 'linear'
        if self.metric is None:
            self.metric = 'mse'

        if isinstance(metric, str):
            self.metric = [metric]
        if isinstance(loss, str):
            self.loss = [loss]
        if isinstance(threshold, (int,float)):
            self.threshold = [threshold]
        if isinstance(minimize, bool):
            self.minimize = [minimize]
        
        # Process weights
        if weight is not None:
            # check that weight has the same length as y
            if not len(weight) == len(y):
                raise ValueError('weight must have the same length as y')
            self.weight = []
            for w in weight:
                if isinstance(w, (list, tuple)):
                    self.weight.append(np.asarray(w))
                else:
                    self.weight.append(w)
        else:
            if yerr is not None:
                # check that yerr has the same length as y
                if not len(yerr) == len(y):
                    raise ValueError('yerr must have the same length as y')
                self.weight = []
                for yer in yerr:
                    self.weight.append(1/np.asarray(yer)**2)
            else:
                self.weight = [None]*len(y)
                
        # Check that primary data dimensions match
        if not len(self.exp_format) == len(self.metric) == len(self.loss) == len(self.threshold) == len(self.minimize) == len(self.X) == len(self.y) == len(self.weight):
            raise ValueError('exp_format, metric, loss, threshold and minimize must have the same length')
        self.all_agent_metrics = self.get_all_agent_metric_names()     

        # Process tracking metrics
        if self.tracking_metric is not None:
            if isinstance(self.tracking_metric, str):
                self.tracking_metric = [self.tracking_metric]
            
            if self.tracking_loss is None:
                self.tracking_loss = ['linear'] * len(self.tracking_metric)
            elif isinstance(self.tracking_loss, str):
                self.tracking_loss = [self.tracking_loss] * len(self.tracking_metric)
                
            # Ensure tracking_metric and tracking_loss have the same length
            if len(self.tracking_metric) != len(self.tracking_loss):
                raise ValueError('tracking_metric and tracking_loss must have the same length')
                
            # Process tracking_exp_format
            if self.tracking_exp_format is None:
                # Default to the main experiment formats if not specified
                self.tracking_exp_format = self.exp_format
            elif isinstance(self.tracking_exp_format, str):
                self.tracking_exp_format = [self.tracking_exp_format]
                
            # Check that all elements in tracking_exp_format are valid
            for form in self.tracking_exp_format:
                if form not in ['trPL','TAS','trMC']:
                    raise ValueError(f'{form} is an invalid tracking_exp_format, must be either "trPL", "TAS" or "trMC"')
            
            # Process tracking_X and tracking_y
            all_formats_in_main = all(fmt in self.exp_format for fmt in self.tracking_exp_format)
            if self.tracking_X is None or self.tracking_y is None:
                if not all_formats_in_main:
                    raise ValueError('tracking_X and tracking_y must be provided when tracking_exp_format contains formats not in exp_format')
                
                # Construct tracking_X and tracking_y from main X and y based on matching formats
                self.tracking_X = []
                self.tracking_y = []
                
                for fmt in self.tracking_exp_format:
                    fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                    if fmt_indices:
                        # Use the first matching format's data
                        idx = fmt_indices[0]
                        self.tracking_X.append(self.X[idx])
                        self.tracking_y.append(self.y[idx])
            
            # Ensure tracking_X and tracking_y are lists
            if not isinstance(self.tracking_X, list):
                self.tracking_X = [self.tracking_X]
            if not isinstance(self.tracking_y, list):
                self.tracking_y = [self.tracking_y]
                
            # Check that tracking_X and tracking_y have the right lengths
            if len(self.tracking_X) != len(self.tracking_exp_format) or len(self.tracking_y) != len(self.tracking_exp_format):
                raise ValueError('tracking_X and tracking_y must have the same length as tracking_exp_format')
            
            # Process tracking_weight
            if self.tracking_weight is None and all_formats_in_main:
                # Use the main weights if available
                self.tracking_weight = []
                for fmt in self.tracking_exp_format:
                    fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                    if fmt_indices:
                        idx = fmt_indices[0]
                        self.tracking_weight.append(self.weight[idx])
                    else:
                        self.tracking_weight.append(None)
            elif self.tracking_weight is None:
                self.tracking_weight = [None] * len(self.tracking_exp_format)
            elif not isinstance(self.tracking_weight, list):
                self.tracking_weight = [self.tracking_weight]
                
            # Ensure tracking_weight has the right length
            if len(self.tracking_weight) != len(self.tracking_exp_format):
                raise ValueError('tracking_weight must have the same length as tracking_exp_format')

            # Check that tracking dimensions match
            if not len(self.tracking_exp_format) == len(self.tracking_metric) == len(self.tracking_loss):
                raise ValueError('tracking_exp_format, tracking_metric and tracking_loss must have the same length')
        self.all_agent_tracking_metrics = self.get_all_agent_tracking_metric_names()
        # Process compare_type parameter
        self.compare_type = self.kwargs.get('compare_type', 'linear')
        if 'compare_type' in self.kwargs.keys():
            self.kwargs.pop('compare_type')
        self.do_G_frac_transform = self.kwargs.get('do_G_frac_transform', False)
        if 'do_G_frac_transform' in self.kwargs.keys():
            self.kwargs.pop('do_G_frac_transform')
   
        # Validate compare_type
        if self.compare_type not in ['linear', 'log', 'normalized', 'normalized_log', 'sqrt']:
            raise ValueError('compare_type must be either linear, log, normalized, normalized_log, or sqrt')
    
    def run_RateEq(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    
        parallel = self.kwargs.get('parallel', False)
        max_jobs = self.kwargs.get('max_jobs', 1)

        # get Gfracs from X
        if 'QE' in parameters.keys():
            QE = parameters['QE']
        elif 'QE' in self.fixed_model_args.keys():
            QE = self.fixed_model_args['QE']
        else:
            QE = 1

        if 'N0' in parameters.keys():
            N0 = parameters['N0']
        elif 'N0' in self.pump_args.keys():
            N0 = self.pump_args['N0']
        else:
            N0 = 0
        # remove N0 from pump_args if it exists
        # self.pump_args.pop('N0', None)

        Gfracs = []
        got_gfrac_none = False
        len_X = len(self.X[0])
        x_default = self.X[0]
        for xx in self.X:
            if len(xx.shape) == 1:
                Gfracs = None
                got_gfrac_none = True
                if len(xx) != len_X:
                    raise ValueError('all X elements should have the same shape')
                if not np.allclose(xx,x_default):
                    raise ValueError('all X elements should be the same, if they are different then create a separate agent for each X element')
            else:
                if got_gfrac_none:
                    raise ValueError('all X elements should have the same shape and Gfrac should be provided for all elements if specified for one')
                # append np.unique(xx[:,1]) to Gfracs list
                if len(xx.shape) == 2:
                    Gfrac, index = np.unique(xx[:,1],return_index=True)
                    Gfrac = Gfrac[np.argsort(index)]
                    for g in Gfrac:
                        if g not in Gfracs:
                            Gfracs.append(g)
                else:
                    Gfracs = None
                if len(xx) != len_X:
                    raise ValueError('all X elements should have the same shape')

                # check all elements in xx[:,0] are the same
                for Gfrac in Gfracs:
                    if not np.allclose(xx[xx[:,1]==Gfrac,0],x_default[x_default[:,1]==Gfrac,0]):
                # if not np.allclose(xx[:,0],x_default[:,0]):
                        raise ValueError('all X elements should be the same, if they are different then create a separate agent for each X element')

        if Gfracs is not None:
            Gfracs = np.asarray(Gfracs)    
        ns, ps = None, None

        if Gfracs is None:
            t = self.X[0] # time axis should be the same for all elements
            tmax = 0.99999*1/self.pump_args['fpu'] # maximum time for the pump
            t_span = np.linspace(0,tmax,len(t)) # time axis for the simulation, here we need a different time axis for the simulation in case there is any equilibration to make sure that the full pulse is included and to reproduce the accumulated carrier density properly
            t = self.X[0]
            tmax = 0.99999*1/self.pump_args['fpu']
            t_span = t
            
            if t_span[-1] < tmax:
                dum = np.linspace(t[-1],tmax,100)
                dum = dum[1:]
                t_span = np.hstack((t,dum))

            pump = copy.deepcopy(self.pump_args)
            if 'N0' in pump.keys():
                pump.pop('N0')
            
            if 'G_frac' in self.pump_args.keys():
                G_frac = self.pump_args['G_frac']
            else:
                G_frac = 1

            # get the pump profile
            Generation = self.pump_model(t_span, N0=N0, G_frac=G_frac, **pump) * QE # multiply by the quantum efficiency
            
            # if 'N0' in self.pump_args.keys():
            #     N0 = self.pump_args['N0']
            # else:
            #     N0 = 0

            ns, ps = self.model(parameters, t, Generation, t_span, N0 = N0, equilibrate = self.equilibrate, G_frac = G_frac, **self.kwargs)
            Gfrac_list = np.ones(len(t))
        else:
            if parallel:
                num_cores = min(max_jobs, len(Gfracs),os.cpu_count())

                results = Parallel(n_jobs=num_cores,backend="loky")(delayed(self._run_single_Gfrac)(parameters, Gfrac, N0,QE) for Gfrac in Gfracs)

                for ns_, ps_, t, Gfrac in results:
                    try:                   
                        if ns is None:
                            ns = ns_
                            ps = ps_
                            t_list = t
                            Gfrac_list = np.ones(len(t))*Gfrac
                        else:
                            if len(ns_.shape) == 1:
                                ns = np.hstack((ns,ns_))
                                ps = np.hstack((ps,ps_))
                            else:
                                ns = np.vstack((ns,ns_))
                                ps = np.vstack((ps,ps_))
                            t_list = np.hstack((t_list,t))
                            Gfrac_list = np.hstack((Gfrac_list,np.ones(len(t))*Gfrac))
                    except Exception as e:
                        logger.error(f"The simulation failed for {parameters}\n{e}")
                        return np.nan
            else:
                for Gfrac in Gfracs:
                    ns_, ps_, t, Gfrac = self._run_single_Gfrac(parameters, Gfrac, N0, QE)
                    
                    #check for np.nan in ns_ or ps_
                    if np.isnan(ns_).any() or np.isnan(ps_).any():
                        # if something went wrong there is no point in continuing
                        return np.nan

                    try:                   
                        if ns is None:
                            ns = ns_
                            ps = ps_
                            t_list = t
                            Gfrac_list = np.ones(len(t))*Gfrac
                        else:
                            if len(ns_.shape) == 1:
                                ns = np.hstack((ns,ns_))
                                ps = np.hstack((ps,ps_))
                            else:
                                ns = np.vstack((ns,ns_))
                                ps = np.vstack((ps,ps_))
                            t_list = np.hstack((t_list,t))
                            Gfrac_list = np.hstack((Gfrac_list,np.ones(len(t))*Gfrac))
                    except Exception as e:
                        logger.error(f"The simulation failed for {parameters}\n{e}")
                        return np.nan
        
        ## MSimmonds edit for his fits. Remove if necessary
        try:
            PL_ave = np.trapezoid(ns*ps, axis = 1)
            n_ave = np.trapezoid(ns, axis = 1)
            p_ave = np.trapezoid(ps, axis = 1)
            starts = len(t)
            PLs_zero = np.array([PL_ave[i*(starts)] for i in range(len(Gfracs))])
            
            log_ps = np.log(np.unique(np.array(Gfrac_list)))
            a, c = np.polyfit(log_ps, np.log(PLs_zero), deg = 1)
            
            if(a < 1.7):
                #print("Slope too small, p=", a)
                # Return noise floor instead of NaN — produces bad but finite NRMSE
                n_floor = np.ones_like(ns) * 1e10  # small constant carrier density
                p_floor = np.ones_like(ps) * 1e10
                dum_dict = {}
                dum_dict['n'] = list(n_floor)
                dum_dict['p'] = list(p_floor)
                dum_dict['t'] = t_list
                dum_dict['G_frac'] = Gfrac_list
                df = pd.DataFrame(dum_dict)
                return df
                #logger.error(f"The simulation failed for {parameters}\n{lines}")

                return np.nan
        except Exception as e:
            #a, c = np.polyfit(np.unique(np.array(Gfrac_list), )
            print("Failed Polyfit")
            print(e)

        dum_dict = {}
        dum_dict['n'] = list(ns)
        dum_dict['p'] = list(ps)
            
        if Gfracs is None:
            dum_dict['t'] = t
            dum_dict['G_frac'] = Gfrac_list
            
        else:
            dum_dict['t'] = t_list
            dum_dict['G_frac'] = Gfrac_list
        
        try:
            df = pd.DataFrame(dum_dict)
        except:
            dum_dict = {}
            lines = ''
            for key in dum_dict.keys():
                lines += f"{key}: {len(dum_dict[key])}, "
            logger.error(f"The simulation failed for {parameters}\n{lines}")

            return np.nan

        return df
    
    def reformat_data(self,df,X,parameters,exp_format='trPL'):
        """Reformat the data to the experimental format

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the simulation results.
        X : array-like
            1-D or 2-D array containing the voltage (1st column) and if specified the Gfrac (2nd column) values.
        parameters : dict
            Dictionary of parameter names and values.
        exp_format : str, optional
            Format of the experimental data, cane be ['trPL','TAS','trMC'], by default 'trPL'.

        Returns
        -------
        DataFrame
            DataFrame containing the simulation results in the experimental format.
        """  

        # check if Gfrac is None is unique in df
        Gfracs, index = np.unique(df['G_frac'],return_index=True)

        if len(Gfracs) == 1:
            Gfracs = None 
        else:
            Gfracs = Gfracs[np.argsort(index)]

        Xfit,yfit = None,None # will be np.array
        do_interp = True

        if exp_format == 'trPL': # transient photoluminescence
            if 'k_direct' in parameters.keys():
                k_direct = parameters['k_direct']
            elif 'k_direct' in self.fixed_model_args.keys():
                k_direct = self.fixed_model_args['k_direct']
            else:
                raise ValueError('k_direct should be provided in the parameters or fixed_model_args')
            
            if 'I_factor_PL' in parameters.keys():
                I_factor = parameters['I_factor_PL']   
            elif 'I_factor_PL' in self.fixed_model_args.keys():
                I_factor = self.fixed_model_args['I_factor_PL']
            else:
                raise ValueError('I_factor_PL should be provided in the parameters or fixed_model_args')
            
            if 'N_A' in parameters.keys():
                N_A = parameters['N_A']
            elif 'N_A' in self.fixed_model_args.keys():
                N_A = self.fixed_model_args['N_A']
            else:
                N_A = 0

            if Gfracs is None:
                # check if we have an array
                if not isinstance(df['n'].iloc[0], (np.ndarray, list)):
                    # if n and p are not arrays, then we can use them directly  
                    signal = I_factor * k_direct * df['n'] * (df['p'] + N_A)
                else:
                    n_dens = np.asarray(df['n'].values.tolist())
                    p_dens = np.asarray(df['p'].values.tolist())
                    Rrad_calc = (n_dens * (p_dens + N_A) * k_direct)
                    signal = np.sum((Rrad_calc[:, 1:] + Rrad_calc[:, :-1]) / 2, axis=1) * I_factor # need to integrate over the space axis

                # signal = I_factor * k_direct * df['n'] * (df['p'] + N_A)
                t = df['t']

                # check if t = X
                if len(t) == len(X):
                    if np.allclose(t,X):
                        do_interp = False

                if do_interp:
                    try:
                        tck = interpolate.splrep(t,signal)
                        signal = interpolate.splev(X,tck)
                    except:
                        f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                        yfit = f(X)
                else:
                    yfit = np.asarray(signal)
                Xfit = np.asarray(X)
            else:
                for Gfrac in Gfracs:
                    dum_df = df[df['G_frac'] == Gfrac]
                    if not isinstance(dum_df['n'].iloc[0], (np.ndarray, list)):
                        # if n and p are not arrays, then we can use them directly
                        signal = I_factor * k_direct * dum_df['n'] * (dum_df['p'] + N_A)
                    else:
                        n_dens = np.asarray(dum_df['n'].values.tolist())
                        p_dens = np.asarray(dum_df['p'].values.tolist())
                        Rrad_calc = (n_dens * (p_dens + N_A) * k_direct)
                        signal = np.sum((Rrad_calc[:, 1:] + Rrad_calc[:, :-1]) / 2, axis=1) * I_factor # need to integrate over the space axis
                    t = dum_df['t']
                    X_ = X[X[:,1] == Gfrac,0]

                    if do_interp:
                        try:
                            tck = interpolate.splrep(t,signal)
                            yfit_ = interpolate.splev(X_,tck)
                        except Exception as e:
                            f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                            yfit_ = f(X_)
                    else:
                        yfit_ = signal

                    if yfit is None:
                        yfit = np.asarray(yfit_)
                    else:
                        yfit = np.hstack((yfit,yfit_))

                    if Xfit is None:
                        Xfit = X
                    else:
                        Xfit = np.hstack((Xfit,X))

                Xfit = np.asarray(Xfit)
                yfit = np.asarray(yfit)
        elif exp_format == 'trMC': # transient microwave conductivity

            if 'I_factor_MC' in parameters.keys():
                I_factor = parameters['I_factor_MC']
            elif 'I_factor_MC' in self.fixed_model_args.keys():
                I_factor = self.fixed_model_args['I_factor_MC']
            else:
                raise ValueError('I_factor_MC should be provided in the parameters or fixed_model_args')
                
            if 'ratio_mu' in parameters.keys():
                ratio_mu = parameters['ratio_mu']
            elif 'ratio_mu' in self.fixed_model_args.keys():
                ratio_mu = self.fixed_model_args['ratio_mu']
            else:
                raise ValueError('ratio_mu should be provided in the parameters or fixed_model_args')

            if Gfracs is None:
                if not isinstance(df['n'].iloc[0], (np.ndarray, list)):
                    # if n and p are not arrays, then we can use them directly
                    signal = I_factor * (ratio_mu * df['n'] + df['p'])
                else:
                    n_dens = np.asarray(df['n'].values.tolist())
                    p_dens = np.asarray(df['p'].values.tolist())
                    signal = (ratio_mu * n_dens + p_dens)
                    signal = np.mean((signal), axis=1) * I_factor # need to integrate over the space axis, no need to 

                # signal = I_factor*(ratio_mu*df['n'] + df['p'])
                t = df['t']

                # check if t = X
                if len(t) == len(X):
                    if np.allclose(t,X):
                        do_interp = False

                if do_interp:
                    try:
                        tck = interpolate.splrep(t,signal)
                        signal = interpolate.splev(X,tck)
                    except:
                        f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                        yfit = f(X)
                else:
                    yfit = np.asarray(signal)
                Xfit = np.asarray(X)
            else:
                for Gfrac in Gfracs:
                    dum_df = df[df['G_frac'] == Gfrac]
                    # signal = I_factor*(ratio_mu*dum_df['n'] + dum_df['p'])
                    if not isinstance(dum_df['n'].iloc[0], (np.ndarray, list)):
                        # if n and p are not arrays, then we can use them directly
                        signal = I_factor * (ratio_mu * dum_df['n'] + dum_df['p'])
                    else:
                        n_dens = np.asarray(dum_df['n'].values.tolist())
                        p_dens = np.asarray(dum_df['p'].values.tolist())
                        signal = (ratio_mu * n_dens + p_dens)
                        signal = np.mean((signal), axis=1) * I_factor # need to integrate over the space axis, no need to 
                    t = dum_df['t']
                    X_ = X[X[:,1] == Gfrac,0]

                    if do_interp:
                        try:
                            tck = interpolate.splrep(t,signal)
                            yfit_ = interpolate.splev(X_,tck)
                        except Exception as e:
                            f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                            yfit_ = f(X_)
                    else:
                        yfit_ = signal

                    if yfit is None:
                        yfit = np.asarray(yfit_)
                    else:
                        yfit = np.hstack((yfit,yfit_))

                    if Xfit is None:
                        Xfit = X
                    else:
                        Xfit = np.hstack((Xfit,X))

                Xfit = np.asarray(Xfit)
                yfit = np.asarray(yfit)
        elif exp_format == 'TAS': # transient absorption spectroscopy
                
                if 'I_factor_TAS' in parameters.keys():
                    I_factor = parameters['I_factor_TAS']
                elif 'I_factor_TAS' in self.fixed_model_args.keys():
                    I_factor = self.fixed_model_args['I_factor_TAS']
                else:
                    I_factor = 1 # we usually don't have a factor for TAS and correct with cross-sections and thickness
                    # raise ValueError('I_factor_TAS should be provided in the parameters or fixed_model_args')
                    
                if 'cross_section' in parameters.keys():
                    cross_section = parameters['cross_section']
                elif 'cross_section' in self.fixed_model_args.keys():
                    cross_section = self.fixed_model_args['cross_section']
                else:
                    raise ValueError('cross_section should be provided in the parameters or fixed_model_args')
                
                if 'L' in parameters.keys():
                    L = parameters['L']
                elif 'L' in self.fixed_model_args.keys():
                    L = self.fixed_model_args['L']
                else:
                    raise ValueError('L should be provided in the parameters or fixed_model_args')
                                
                if Gfracs is None:
                    # check if we have an array
                    if not isinstance(df['n'].iloc[0], (np.ndarray, list)):
                        # if n is not an array, then we can use it directly  
                        signal = df['n']
                    else:
                        signal = np.mean(np.asarray(df['n'].values.tolist()))
                    # signal = df['n']
                    t = df['t']
    
                    # check if t = X
                    if len(t) == len(X):
                        if np.allclose(t,X):
                            do_interp = False
    
                    if do_interp:
                        try:
                            tck = interpolate.splrep(t,signal)
                            signal = interpolate.splev(X,tck)
                        except:
                            f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                            yfit = f(X)
                    else:
                        yfit = np.asarray(signal)

                    yfit -= np.mean(yfit) # simulate AC coupling 
                    yfit = I_factor*(np.exp(-cross_section*L*yfit) - 1) # \Delta T/T = -\sigma*L*\Delta \n - 1 (the minus one comes from the fact that we are looking at the change in transmission and since Beer-Lambert law is A = exp(-\sigma*L*n) 

                    Xfit = np.asarray(X)
                else:
                    for Gfrac in Gfracs:
                        dum_df = df[df['G_frac'] == Gfrac]
                        signal = dum_df['n']
                        t = dum_df['t']
                        X_ = X[X[:,1] == Gfrac,0]
    
                        if do_interp:
                            try:
                                tck = interpolate.splrep(t,signal)
                                yfit_ = interpolate.splev(X_,tck)
                            except Exception as e:
                                f = interpolate.interp1d(t,signal,kind='linear',fill_value='extrapolate')
                                yfit_ = f(X_)
                        else:
                            yfit_ = signal

                        yfit_ -= np.mean(yfit_) # simulate AC coupling
                        yfit_ = I_factor*(np.exp(-cross_section*L*yfit_) - 1) # \Delta T/T = -\sigma*L*\Delta \n

                        if yfit is None:
                            yfit = np.asarray(yfit_)
                        else:
                            yfit = np.hstack((yfit,yfit_))
    
                        if Xfit is None:
                            Xfit = X
                        else:
                            Xfit = np.hstack((Xfit,X))
    
                    Xfit = np.asarray(Xfit)
                    
        if self.detection_limit is not None:
            yfit = yfit + self.detection_limit

        return Xfit,yfit

    def _run_single_Gfrac(self, parameters, Gfrac, N0, QE):
        t = self.X[0][self.X[0][:,1] == Gfrac,0]
        tmax = 0.99999*1/self.pump_args['fpu']
        t_span = t
        
        if t_span[-1] < tmax:
            dum = np.linspace(t[-1],tmax,100)
            dum = dum[1:]
            t_span = np.hstack((t,dum))
        
        pump = copy.deepcopy(self.pump_args)
        if 'G_frac' in pump.keys():
            pump.pop('G_frac')
        
        if 'N0' in pump.keys():
            pump.pop('N0')
        
        Generation = self.pump_model(t_span, G_frac = Gfrac, N0=N0, **pump) * QE
        
        # if 'N0' in self.pump_args.keys():
        #     N0 = self.pump_args['N0']
        # else:
        #     N0 = 0

        ns_, ps_ = self.model(parameters, t, Generation, t_span, N0 = N0, equilibrate = self.equilibrate, G_frac = Gfrac, **self.kwargs)
        if type(ns_) is list:
            ns_ = np.asarray(ns_)
            ps_ = np.asarray(ps_)

        return ns_, ps_, t, Gfrac
    
    def run(self,parameters,X=None,exp_format='trPL'):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        array-like
            Array containing the current values.
        """    

        parameters_rescaled = self.params_rescale(parameters, self.params)
        
        df = self.run_RateEq(parameters_rescaled)

        if df is np.nan:
            return np.nan

        if X is None:
            X = self.X[0]
        
        Xfit, yfit = self.reformat_data(df, X, parameters_rescaled, exp_format)
        
        return yfit


    def run_Ax(self,parameters):
        """Run the diode model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        float
            Loss function value.
        """    
                
        parameters_rescaled = self.params_rescale(parameters, self.params)
        # print('Running RateEqAgent with parameters:', parameters_rescaled)
        df = self.run_RateEq(parameters_rescaled)
        # print('Finished running RateEqAgent with parameters:', parameters_rescaled)

        if df is np.nan:
            dum_dict = {}
            for i in range(len(self.exp_format)):
                dum_dict[self.all_agent_metrics[i]] = np.nan

            # Add NaN values for tracking metrics
            if self.tracking_metric is not None:
                for j in range(len(self.tracking_metric)):
                    dum_dict[self.all_agent_tracking_metrics[j]] = np.nan
                        
            return dum_dict
        
        dum_dict = {}
        
        # First loop: calculate main metrics for each exp_format
        for i in range(len(self.exp_format)):
            Xfit, yfit = self.reformat_data(df, self.X[i], parameters_rescaled, self.exp_format[i])
            
            # Transform both true and predicted values together using the enhanced function
            # Pass both X values as well for future extensibility
            if self.compare_type == 'linear':
                metric_value = calc_metric(
                self.y[i],
                yfit,
                sample_weight=self.weight[i], 
                metric_name=self.metric[i]
            )
            else:
                y_true_transformed, y_pred_transformed = transform_data(
                    self.y[i], 
                    yfit, 
                    X=self.X[i],
                    X_pred=Xfit,
                    transform_type=self.compare_type,
                    do_G_frac_transform=self.do_G_frac_transform
                )
            
                # Calculate metric with transformed data
                metric_value = calc_metric(
                    y_true_transformed, 
                    y_pred_transformed, 
                    sample_weight=self.weight[i], 
                    metric_name=self.metric[i]
                )
            dum_dict[self.all_agent_metrics[i]] = loss_function(metric_value, loss=self.loss[i])

        # Second loop: calculate all tracking metrics using tracking_X and tracking_y
        if self.tracking_metric is not None:
            for j in range(len(self.tracking_metric)):
                exp_fmt = self.tracking_exp_format[j]
                metric_name = self.tracking_metric[j]
                loss_type = self.tracking_loss[j]
                
                Xfit, yfit = self.reformat_data(df, self.tracking_X[j], parameters_rescaled, exp_fmt)
                
                # Transform data once for each format
                y_true_transformed, y_pred_transformed = transform_data(
                    self.tracking_y[j], 
                    yfit, 
                    X=self.tracking_X[j],
                    X_pred=Xfit,
                    transform_type=self.compare_type,
                    do_G_frac_transform=self.do_G_frac_transform
                )
                
                metric_value = calc_metric(
                    y_true_transformed, 
                    y_pred_transformed, 
                    sample_weight=self.tracking_weight[j], 
                    metric_name=metric_name
                )
                
                dum_dict[self.all_agent_tracking_metrics[j]] = loss_function(metric_value, loss=loss_type)

        return dum_dict