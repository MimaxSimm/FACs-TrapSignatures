"""IMPSAgent class for steady-state IMPS simulations"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd
import os, uuid, sys, copy, time, warnings
from scipy import interpolate

from optimpv import *
from optimpv.general.general import *
from optimpv.DDfits.SIMsalabimAgent import SIMsalabimAgent
from pySIMsalabim import *
from pySIMsalabim.experiments.imps import *

######### Agent Definition #######################################################################
class IMPSAgent(SIMsalabimAgent):  
    """IMPSAgent class for IMPS simulations with SIMsalabim

    Parameters
    ----------
    params : list of Fitparam() objects
        List of Fitparam() objects.
    X : array-like
        1-D or 2-D array containing the voltage values.
    y : array-like
        1-D array containing the current values.
    session_path : str
        Path to the session directory.
    f_min : float
        Minimum frequency for the IMPS simulation in Hz.
    f_max : float
        Maximum frequency for the IMPS simulation in Hz.
    f_steps : float, optional
        Number of frequency steps for the IMPS simulation (log spaced), by default 30.
    V : float, optional
        Voltage value for the simulation, by default 0.
    G_frac : float, optional
        Fractional light intensity, by default 1.
    GStep : float, optional
        Applied generation rate increase at t=0, by default 0.05.
    simulation_setup : str, optional
        Path to the simulation setup file, if None then use the default file 'simulation_setup.txt'in the session_path directory, by default None.
    exp_format : str, optional
        Format of the IMPS data, possible values are: 'ReY', 'ImY', 'ColeCole', by default 'ImY'.
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
        Name of the agent, by default 'IMPS'.
    **kwargs : dict
        Additional keyword arguments.
    """   
    def __init__(self, params, X, y, session_path, f_min, f_max, f_steps=30, V=0, G_frac=1, GStep=0.05, 
                 simulation_setup=None, exp_format='ImY', metric='mse', loss='linear', threshold=100, minimize=True, 
                 yerr=None, weight=None, tracking_metric=None, tracking_loss=None, tracking_exp_format=None, 
                 tracking_X=None, tracking_y=None, tracking_weight=None, name='IMPS', **kwargs):    

        self.params = params
        self.session_path = session_path  
        if simulation_setup is None:
            self.simulation_setup = os.path.join(session_path,'simulation_setup.txt')
        else:
            self.simulation_setup = simulation_setup

        if not isinstance(X, (list, tuple)):
            X = [np.asarray(X)]
        if not isinstance(y, (list, tuple)):
            y = [np.asarray(y)]

        self.X = X
        self.y = y
        self.yerr = yerr
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format
        self.tracking_X = tracking_X
        self.tracking_y = tracking_y
        self.tracking_weight = tracking_weight

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

        self.kwargs = kwargs
        self.f_min = f_min
        self.f_max = f_max
        self.f_steps = f_steps
        self.V = V
        self.GStep = GStep
        self.G_frac = G_frac
        self.name = name
        
        self.exp_format = exp_format
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]

        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form.lower() not in ['rey', 'imy', 'colecole']:
                raise ValueError(f'{form} is an invalid IMPS format. Possible values are: ReY, ImY, ColeCole')

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

        # check that exp_format, metric, loss, threshold and minimize have the same length
        if not len(self.exp_format) == len(self.metric) == len(self.loss) == len(self.threshold) == len(self.minimize) == len(self.X) == len(self.y) == len(self.weight):
            raise ValueError('exp_format, metric, loss, threshold and minimize must have the same length')
        
        # Process tracking metrics and losses
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
                
            # check that all elements in tracking_exp_format are valid
            for form in self.tracking_exp_format:
                if form.lower() not in ['rey', 'imy', 'colecole']:
                    raise ValueError(f'{form} is an invalid tracking_exp_format. Possible values are: ReY, ImY, ColeCole')
            
            # Process tracking_X and tracking_y
            # Check if all tracking formats are in main exp_format
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
                if all_formats_in_main:
                    for fmt in self.tracking_exp_format:
                        fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                        if fmt_indices:
                            idx = fmt_indices[0]
                            self.tracking_weight.append(self.weight[idx])
                        else:
                            self.tracking_weight.append(None)
                else:
                    self.tracking_weight = [None] * len(self.tracking_exp_format)
            elif not isinstance(self.tracking_weight, list):
                self.tracking_weight = [self.tracking_weight]
                
            # Ensure tracking_weight has the right length
            if len(self.tracking_weight) != len(self.tracking_exp_format):
                raise ValueError('tracking_weight must have the same length as tracking_exp_format')

        if tracking_exp_format is not None:
            # check that tracking_exp_format, tracking_metric and tracking_loss have the same length
            if not len(self.tracking_exp_format) == len(self.tracking_metric) == len(self.tracking_loss):
                raise ValueError('tracking_exp_format, tracking_metric and tracking_loss must have the same length')
                
        # Add compare_type parameter
        self.compare_type = self.kwargs.get('compare_type', 'linear')
        if 'compare_type' in self.kwargs.keys():
            self.kwargs.pop('compare_type')
            
        # Validate compare_type
        if self.compare_type not in ['linear', 'log', 'normalized', 'normalized_log', 'sqrt']:
            raise ValueError('compare_type must be either linear, log, normalized, normalized_log, or sqrt')
        
        # check if simulation_setup file exists
        if not os.path.exists(os.path.join(self.session_path,self.simulation_setup)):
            raise ValueError('simulation_setup file does not exist: {}'.format(os.path.join(self.session_path,self.simulation_setup)))
        if os.name != 'nt':
            try:
                dev_par, layers = load_device_parameters(session_path, simulation_setup, run_mode = False)
            except Exception as e:
                raise ValueError('Error loading device parameters check that all the input files are in the right directory. \n Error: {}'.format(e))
        else:
            warning_timeout = self.kwargs.get('warning_timeout', 10)
            exit_timeout = self.kwargs.get('exit_timeout', 60)
            t_wait = 0
            while True: # need this to be thread safe
                try:
                    dev_par, layers = load_device_parameters(session_path, simulation_setup, run_mode = False)
                    break
                except Exception as e:
                    time.sleep(0.002)
                    t_wait = t_wait + 0.002
                    if t_wait > warning_timeout:
                        print('Warning: SIMsalabim is not responding, please check that all the input files are in the right directory')
                    if t_wait > exit_timeout:
                        raise ValueError('Error loading device parameters check that all the input files are in the right directory. \n Error: {}'.format(e))
        
        self.dev_par = dev_par
        self.layers = layers
        SIMsalabim_params  = {}

        for layer in layers:
            SIMsalabim_params[layer[1]] = ReadParameterFile(os.path.join(session_path,layer[2]))

        self.SIMsalabim_params = SIMsalabim_params
        pnames = list(SIMsalabim_params[list(SIMsalabim_params.keys())[0]].keys())
        pnames = pnames + list(SIMsalabim_params[list(SIMsalabim_params.keys())[1]].keys())
        self.pnames = pnames    


    def target_metric(self, y, yfit, metric_name, X=None, Xfit=None, sample_weight=None):
        """Calculate the target metric depending on self.metric

        Parameters
        ----------
        y : array-like
            1-D array containing the target values.
        yfit : array-like
            1-D array containing the fitted values.
        metric_name : str
            Metric to evaluate the model, see optimpv.general.calc_metric for options.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        Xfit : array-like, optional
            1-D array containing the x axis values, by default None.
        sample_weight : array-like, optional
            1-D array containing the weights, by default None.

        Returns
        -------
        float
            Target metric value.
        """        
        if metric_name.lower() == 'mmeud':
            if Xfit is None:
                raise ValueError('Xfit must be specified for the mmed metric')
            return mean_min_euclidean_distance(X,y,Xfit,yfit)
        elif metric_name.lower() == 'dmeud':
            if Xfit is None:
                raise ValueError('Xfit must be specified for the med metric')
            return direct_mean_euclidean_distance(X,y,Xfit,yfit)
        else:
            return  calc_metric(y,yfit,sample_weight=sample_weight,metric_name=metric_name)
    

    def run_Ax(self, parameters):
        """Function to run the simulation with the parameters and return the target metric value for Ax optimization

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dict
            Dictionary with the target metric value and any tracking metrics.
        """  
        df = self.run_IMPS_simulation(parameters)
        if df is np.nan:
            dum_dict = {}
            for i in range(len(self.exp_format)):
                dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = np.nan
                
                # Add NaN values for tracking metrics
                if self.tracking_metric is not None:
                    for j in range(len(self.tracking_metric)):
                        dum_dict[self.name+'_'+self.tracking_exp_format[j]+'_tracking_'+self.tracking_metric[j]] = np.nan
                    
            return dum_dict
        
        dum_dict = {}
        
        # First loop: calculate main metrics for each exp_format
        for i in range(len(self.exp_format)):
            Xfit, yfit = self.reformat_IMPS_data(df, self.X[i], exp_format=self.exp_format[i])
            
            # Apply data transformation based on compare_type
            if self.compare_type == 'linear':
                metric_value = self.target_metric(
                    self.y[i],
                    yfit,
                    self.metric[i],
                    self.X[i],
                    Xfit,
                    sample_weight=self.weight[i]
                )
            else:
                y_true_transformed, y_pred_transformed = transform_data(
                    self.y[i], 
                    yfit, 
                    X=self.X[i],
                    X_pred=Xfit,
                    transform_type=self.compare_type
                )
                
                # Calculate metric with transformed data
                metric_value = self.target_metric(
                    y_true_transformed, 
                    y_pred_transformed, 
                    self.metric[i],
                    self.X[i],
                    Xfit,
                    sample_weight=self.weight[i]
                )
            
            dum_dict[self.name+'_'+self.exp_format[i]+'_'+self.metric[i]] = loss_function(metric_value, loss=self.loss[i])
        
        # Second loop: calculate all tracking metrics
        if self.tracking_metric is not None:
            for j in range(len(self.tracking_metric)):
                exp_fmt = self.tracking_exp_format[j]
                metric_name = self.tracking_metric[j]
                loss_type = self.tracking_loss[j]
                
                Xfit, yfit = self.reformat_IMPS_data(df, self.tracking_X[j], exp_format=exp_fmt)
                
                # Apply data transformation based on compare_type
                if self.compare_type == 'linear':
                    metric_value = self.target_metric(
                        self.tracking_y[j],
                        yfit,
                        metric_name,
                        self.tracking_X[j],
                        Xfit,
                        sample_weight=self.tracking_weight[j]
                    )
                else:
                    # Transform data for each format
                    y_true_transformed, y_pred_transformed = transform_data(
                        self.tracking_y[j], 
                        yfit, 
                        X=self.tracking_X[j],
                        X_pred=Xfit,
                        transform_type=self.compare_type
                    )
                    
                    # Calculate metric with transformed data
                    metric_value = self.target_metric(
                        y_true_transformed, 
                        y_pred_transformed, 
                        metric_name,
                        self.tracking_X[j],
                        Xfit,
                        sample_weight=self.tracking_weight[j]
                    )
                
                dum_dict[self.name+'_'+exp_fmt+'_tracking_'+metric_name] = loss_function(metric_value, loss=loss_type)
        
        return dum_dict
    
    def run_IMPS_simulation(self, parameters):
        """Run the simulation with the parameters and return the simulated values

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.

        Returns
        -------
        dataframe
            Dataframe with the simulated IMPS values.
        """    

        parallel = self.kwargs.get('parallel', False)
        max_jobs = self.kwargs.get('max_jobs', 1)
        # output_file = self.kwargs.get('output_file', 'freqY.dat')

        VarNames,custom_pars,clean_pars = [],[],[]
                
        # check if cmd_pars is in kwargs
        if 'cmd_pars' in self.kwargs:
            cmd_pars = self.kwargs['cmd_pars']
            for cmd_par in cmd_pars:
                if (cmd_par['par'] not in self.SIMsalabim_params['l1'].keys()) and (cmd_par['par'] not in self.SIMsalabim_params['setup'].keys()):
                    custom_pars.append(cmd_par)
                else:
                    clean_pars.append(cmd_par)
                VarNames.append(cmd_par['par'])
        else:
            cmd_pars = []


        # prepare the cmd_pars for the simulation
        custom_pars, clean_pars, VarNames = self.prepare_cmd_pars(parameters, custom_pars, clean_pars, VarNames)

        # check if there are any custom_pars that are energy level offsets
        clean_pars = self.energy_level_offsets(custom_pars, clean_pars)

        # check if there are any duplicated parameters in cmd_pars
        self.check_duplicated_parameters(clean_pars)
        
        # Run the IMPS simulation
        UUID = self.kwargs.get('UUID',str(uuid.uuid4()))

        # remove UUID and output_file and cmd_pars from kwargs
        dummy_kwargs = copy.deepcopy(self.kwargs)
        if 'UUID' in dummy_kwargs:
            dummy_kwargs.pop('UUID')
        if 'output_file' in dummy_kwargs:
            dummy_kwargs.pop('output_file')
        if 'cmd_pars' in dummy_kwargs:
            dummy_kwargs.pop('cmd_pars')

        ret, mess = run_IMPS_simu(self.simulation_setup, self.session_path, self.f_min, self.f_max, self.f_steps, self.V, self.G_frac, self.GStep,  run_mode=False, output_file = 'freqY.dat', UUID=UUID, cmd_pars=clean_pars, **dummy_kwargs)
        
        if type(ret) == int: 
            if not ret == 0 :
                print('Error in running SIMsalabim: '+mess)
                return np.nan
        elif isinstance(ret, subprocess.CompletedProcess):
            
            if not(ret.returncode == 0 or ret.returncode == 95):
                # print('Error in running SIMsalabim: '+mess)
                return np.nan
        else:
            if not all([(res == 0 or res == 95) for res in ret]):
                # print('Error in running SIMsalabim: '+mess)
                return np.nan
        try:
            df = pd.read_csv(os.path.join(self.session_path, 'freqY_'+UUID+'.dat'), sep=r'\s+')
        except:
            print('No IMPS data found for UUID '+UUID + ' and cmd_pars '+str(cmd_pars))
            return np.nan

        return df

    def run(self, parameters,X=None,exp_format='ImY'):
        """Run the simulation with the parameters and return an array with the simulated values in the format specified by exp_format (default is 'Cf')

        Parameters
        ----------
        parameters : dict
            Dictionary with the parameter names and values.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'Cf'.

        Returns
        -------
        array-like
            1-D array with the simulated current values.
        """     

        df = self.run_IMPS_simulation(parameters)
        if df is np.nan:
            return np.nan

        if X is None:
            X = self.X[0]

        Xfit, yfit = self.reformat_IMPS_data(df, X, exp_format)

        return yfit


    def reformat_IMPS_data(self,df,X,exp_format='IMPS'):
        """ Reformat the data depending on the exp_format and X values
        Also interpolates the data if the simulation did not return the same points as the experimental data (i.e. if some points did not converge)

        Parameters
        ----------
        df : dataframe
            Dataframe with the IMPS dara from run_IMPS_simulation function.
        X : array-like, optional
            1-D array containing the x axis values, by default None.
        exp_format : str, optional
            Format of the experimental data, by default 'IMPS'.

        Returns
        -------
        tuple
            Tuple with the reformatted Xfit and yfit values.

        Raises
        ------
        ValueError
            If the exp_format is not valid.
        """     
        Xfit,yfit = [],[]
        do_interp = True
        if exp_format.lower() == 'rey':
            
            if len(X) == len(df['freq'].values):
                if np.allclose(X, np.asarray(df['freq'].values)):
                    do_interp = False
            
            if do_interp:

                # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                try:
                    tck = interpolate.splrep(df['freq'], df['ReY'].values, s=0)
                    yfit = interpolate.splev(X, tck, der=0)
                except:
                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                    f = interpolate.interp1d(df['freq'], df['ReY'].values, kind='linear', fill_value='extrapolate')
                    yfit = f(X)
            else:
                Xfit = X
                yfit = np.asarray(df['ReY'].values)

        elif exp_format.lower() == 'imy':
                

                if len(X) == len(df['freq'].values):
                    if np.allclose(X, np.asarray(df['freq'].values)):
                        do_interp = False
                
                if do_interp:
    
                    # Do interpolation in case SIMsalabim did not return the same number of points as the experimental data
                    try:
                        tck = interpolate.splrep(df['freq'], df['ImY'].values, s=0)
                        yfit = interpolate.splev(X, tck, der=0)
                    except:
                        warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
                        f = interpolate.interp1d(df['freq'], df['ImY'].values, kind='linear', fill_value='extrapolate')
                        yfit = f(X)
                else:
                    Xfit = X
                    yfit = np.asarray(df['ImY'].values)
        
        elif exp_format.lower() == 'colecole':

            if self.metric.lower() == 'mmeud' or self.metric.lower() == 'dmeud':
                Xfit = np.asarray(df['ReY'].values)
                yfit = np.asarray(df['ImY'].values)
            else:
                raise ValueError('Invalid metric for Cole-Cole analysis. Possible values are: MMEUD, DMEUD. if you want to fit the ReZ and ImZ values, please for a MO analysis using the ReY and ImY exp_format')
            
            if len(X) == len(Xfit):
                if np.allclose(X, Xfit):
                    do_interp = False

            if do_interp:
                freqs = self.kwargs.get('freqs',None)
                if freqs is None:
                    raise ValueError('freqs must be specified for a Cole-Cole analysis in case not all frequencies are returned by SIMsalabim')

                try:
                    # interpolate ReZ
                    dum_freqs = np.asarray(df['freq'].values)
                    dum_Re = np.asarray(df['ReY'].values)
                    dum_Im = np.asarray(df['ImY'].values)
                    # check if the frequencies are in descending order and reverse them if necessary
                    if dum_freqs[0] > dum_freqs[-1]:
                        dum_freqs = dum_freqs[::-1]
                        dum_Re = dum_Re[::-1]
                        dum_Im = dum_Im[::-1]

                    tck = interpolate.splrep(dum_freqs, dum_Re, s=0)
                    yfit = interpolate.splev(freqs, tck, der=0, ext=0)
                    # interpolate ImZ
                    tck = interpolate.splrep(dum_freqs, dum_Im, s=0)
                    yfit2 = interpolate.splev(freqs, tck, der=0, ext=0)

                    Xfit = yfit
                    yfit = yfit2

                except Exception as e:
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ReY'].values), fill_value='extrapolate', kind='linear')
                    yfit = f(freqs)
                    f = interpolate.interp1d(np.asarray(df['freq'].values), np.asarray(df['ImY'].values), fill_value='extrapolate', kind='linear')
                    yfit2 = f(freqs)
                    # put ReZ and ImZ in the same array and double the length of Xfit
                    Xfit = yfit
                    yfit = yfit2

                    warnings.warn('Spline interpolation failed, using linear interpolation', UserWarning)
        else:
            raise ValueError('Invalid IMPS format. Possible values are: IMPS.')

        return Xfit, yfit