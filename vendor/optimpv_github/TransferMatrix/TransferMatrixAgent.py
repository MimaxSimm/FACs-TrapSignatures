"""Provides general functionality for Agent objects for transfer matrix simulations"""
######### Package Imports #########################################################################

import os, uuid, sys, copy, warnings
import numpy as np
import pandas as pd
from scipy import interpolate, constants

from optimpv import *
from optimpv.general.general import calc_metric, loss_function, transform_data
from optimpv.general.BaseAgent import BaseAgent
from optimpv.TransferMatrix.TransferMatrixModel import *

## Physics constants
q = constants.value(u'elementary charge')
eps_0 = constants.value(u'electric constant')
kb = constants.value(u'Boltzmann constant in eV/K')

######### Agent Definition #######################################################################
class TransferMatrixAgent(BaseAgent):
    """Initialize the TransferMatrixAgent

        Parameters
        ----------
        params : dict
            Dictionary of parameters.
        y : array-like, optional
            Experimental data, by default None.
        layers : list, optional
            List of material names in the stack, by default None. Note that these names will be used to find the refractive index files in the mat_dir. The filenames must be in the form of 'nk_materialname.txt'.
        thicknesses : list, optional
            List of thicknesses of the layers in the stack in meters, by default None.
        lambda_min : float, optional
            Start wavelength in m, by default 350e-9.
        lambda_max : float, optional
            Stop wavelength in m, by default 800e-9.
        lambda_step : float, optional
            Wavelength step in m, by default 1e-9.
        x_step : float, optional
            Step size for the x position in the stack in m, by default 1e-9.
        activeLayer : int, optional
            Index of the active layer in the stack, i.e. the layer where the generation profile will be calculated. Counting starts at 0, by default None.
        spectrum : string, optional
            Name of the file that contains the spectrum, by default None.
        mat_dir : string, optional
            Path to the directory where the refractive index files and the spectrum file are located, by default None.
        photopic_file : string, optional
            Name of the file that contains the photopic response (must be in the same directory as the refractive index files), by default None.
        exp_format : str or list, optional
            Expected format of the output, by default 'Jsc'.
        metric : str or list, optional
            Metric to be used for optimization, by default None.
        loss : str or list, optional
            Loss function to be used for optimization, by default None.
        threshold : int, float or list, optional
            Threshold value for the loss function, by default 10.
        minimize : bool or list, optional
            Whether to minimize the loss function, by default False.
        tracking_metric : str or list of str, optional
            Additional metrics to track and report in run_Ax output, by default None.
        tracking_loss : str or list of str, optional
            Loss functions to apply to tracking metrics, by default None.
        tracking_exp_format : str or list of str, optional
            Experimental formats for tracking metrics, by default None.
        tracking_y : array-like or list of array-like, optional
            y values for tracking metrics, by default None.
        compare_type : str, optional
            Type of comparison to use for metrics, by default 'linear'.
            Options: 'linear', 'log', 'normalized', 'normalized_log', 'sqrt'.
        name : str, optional
            Name of the agent, by default 'TM'.

        Raises
        ------
        ValueError
            If any of the required parameters are not defined or if there is a mismatch in the lengths of metric, loss, threshold, minimize, and exp_format.
    """    
    def __init__(self, params, y = None, layers = None, thicknesses=None, activeLayer=None,
                 lambda_min=350e-9, lambda_max=800e-9, lambda_step=1e-9, x_step=1e-9, 
                 mat_dir=None, spectrum=None, photopic_file=None, exp_format='Jsc', 
                 metric=None, loss=None, threshold=10, minimize=False,
                 tracking_metric=None, tracking_loss=None, tracking_exp_format=None,
                 tracking_y=None, compare_type='linear', name='TM'):
    
        self.params = params
        self.y = y
        self.layers = layers
        self.thicknesses = thicknesses
        self.activeLayer = activeLayer
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_step = lambda_step
        self.x_step = x_step
        self.mat_dir = mat_dir
        self.spectrum = spectrum
        self.photopic_file = photopic_file
        self.exp_format = exp_format
        self.metric = metric
        self.loss = loss
        self.threshold = threshold
        self.minimize = minimize
        self.name = name
        self.compare_type = compare_type
        self.tracking_metric = tracking_metric
        self.tracking_loss = tracking_loss
        self.tracking_exp_format = tracking_exp_format
        self.tracking_y = tracking_y

        if isinstance(metric, str):
            self.metric = [metric]
        if isinstance(loss, str):
            self.loss = [loss]
        if isinstance(threshold, (int,float)):
            self.threshold = [threshold]
        if isinstance(minimize, bool):
            self.minimize = [minimize]
        if isinstance(exp_format, str):
            self.exp_format = [exp_format]
        
        # check that all elements in exp_format are valid
        for form in self.exp_format:
            if form not in ['Jsc','AVT','LUE']:
                raise ValueError(f'{form} is an invalid impedance format. Possible values are: Jsc, AVT, LUE')
            
        if len(self.metric) != len(self.loss) or len(self.metric) != len(self.threshold) or len(self.metric) != len(self.minimize) or len(self.metric) != len(self.exp_format):
            raise ValueError('metric, loss, threshold, minimize and exp_format must have the same length')
        
        # for i in range(len(self.metric)):
        #     if self.metric[i] is None:
        #         self.metric[i] =         
        # Validate compare_type
        if self.compare_type not in ['linear', 'log', 'normalized', 'normalized_log', 'sqrt']:
            raise ValueError('compare_type must be either linear, log, normalized, normalized_log, or sqrt')

        self.all_agent_metrics = self.get_all_agent_metric_names()       

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
                if form not in ['Jsc','AVT','LUE']:
                    raise ValueError(f'{form} is an invalid tracking_exp_format, must be one of: Jsc, AVT, LUE')
            
            # Process tracking_y
            # Check if all tracking formats are in main exp_format
            all_formats_in_main = all(fmt in self.exp_format for fmt in self.tracking_exp_format)
            if self.tracking_y is None:
                
                if not all_formats_in_main:
                    raise ValueError('tracking_y must be provided when tracking_exp_format contains formats not in exp_format')
                
                # Construct tracking_y from main y based on matching formats
                self.tracking_y = []
                
                for fmt in self.tracking_exp_format:
                    fmt_indices = [i for i, main_fmt in enumerate(self.exp_format) if main_fmt == fmt]
                    if fmt_indices:
                        # Use the first matching format's data
                        idx = fmt_indices[0]
                        if isinstance(self.y, list):
                            self.tracking_y.append(self.y[idx])
                        else:
                            self.tracking_y.append(self.y)
                    else:
                        self.tracking_y.append(None)
            
            # Ensure tracking_y is a list
            if not isinstance(self.tracking_y, list):
                self.tracking_y = [self.tracking_y]

            # Check that tracking_y has the right length
            if len(self.tracking_y) != len(self.tracking_exp_format):
                raise ValueError('tracking_y must have the same length as tracking_exp_format')
            
            # check that tracking_exp_format, tracking_metric and tracking_loss have the same length
            if not len(self.tracking_exp_format) == len(self.tracking_metric) == len(self.tracking_loss):
                raise ValueError('tracking_exp_format, tracking_metric and tracking_loss must have the same length')

        self.all_agent_tracking_metrics = self.get_all_agent_tracking_metric_names()    
        # for i in range(len(self.tracking_metric)):
        #     if self.tracking_metric[i] is None:
        #         self.tracking_metric[i] = ''

        # check that layers, thicknesses and activeLayer and spectrum are not None
        if self.layers is None:
            raise ValueError('layers must be defined')
        if self.thicknesses is None:
            raise ValueError('thicknesses must be defined')
        if self.activeLayer is None:
            raise ValueError('activeLayer must be defined')
        if self.spectrum is None:
            raise ValueError('spectrum must be defined')
        if self.photopic_file is None and ('AVT' in self.exp_format or 'LUE' in self.exp_format):
            raise ValueError('photopic_file must be defined to calculate AVT or LUE')
        if self.mat_dir is None:
            raise ValueError('mat_dir must be defined')
        
        # check that layers, thicknesses have the same length
        if len(self.layers) != len(self.thicknesses):
            raise ValueError('layers and thicknesses must have the same length')
        # check that activeLayer is in layers
        if self.activeLayer > len(self.layers):
            raise ValueError('activeLayer must be in layers')
        
    def target_metric(self,y,yfit=None,metric_name=None):
        """Calculates the target metric based on the metric, loss, threshold and minimize values

        Parameters
        ----------
        y : array-like
            1-D array containing the current values.
        yfit : array-like
            1-D array containing the fitted current values.
        metric_name : str
            Metric to evaluate the model, see optimpv.general.calc_metric for options.

        Returns
        -------
        float
            Target metric value.
        """        
        if metric_name is None or metric_name == '':
            return y
        else:
            return calc_metric(y,yfit,metric_name=metric_name)
    
    def run(self,parameters):
        """Run the transfer matrix model and calculate the Jsc, AVT and LUE

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        Jsc : float
            Short circuit current density in A/m^2.
        AVT : float
            Average visible transmittance.
        LUE : float
            Light utilization efficiency.
        """        

        parameters_rescaled = self.params_rescale(parameters, self.params)
        Jsc, AVT, LUE = TMM(parameters_rescaled, self.layers, self.thicknesses, self.lambda_min, self.lambda_max, self.lambda_step, self.x_step, self.activeLayer, self.spectrum, self.mat_dir, self.photopic_file)

        return Jsc, AVT, LUE
    
    def run_Ax(self,parameters):
        """Run the transfer matrix model and calculate the loss function

        Parameters
        ----------
        parameters : dict
            Dictionary of parameter names and values.

        Returns
        -------
        dict
            Dictionary of loss functions.
        """    

        Jsc, AVT, LUE = self.run(parameters)
        res_dict = {'Jsc':Jsc,'AVT':AVT,'LUE':LUE}

        dum_dict = {}
        # First loop: calculate main metrics for each exp_format
        for i in range(len(self.exp_format)):
            if self.loss[i] is None:
                dum_dict[self.all_agent_metrics[i]] = self.target_metric(res_dict[self.exp_format[i]], metric_name=self.metric[i])
            else:
                result = res_dict[self.exp_format[i]]
                
                # Apply data transformation based on compare_type
                if self.compare_type == 'linear':
                    if isinstance(self.y, list):
                        metric_value = self.target_metric(self.y[i], result, metric_name=self.metric[i])
                    else:
                        metric_value = self.target_metric(self.y, result, metric_name=self.metric[i])
                else:
                    y_true = self.y[i] if isinstance(self.y, list) else self.y
                    y_true_transformed, y_pred_transformed = transform_data(
                        y_true, 
                        result, 
                        transform_type=self.compare_type
                    )
                    
                    # Calculate metric with transformed data
                    metric_value = self.target_metric(
                        y_true_transformed, 
                        y_pred_transformed, 
                        metric_name=self.metric[i]
                    )
    
                # Calculate the loss function based on the metric value
                dum_dict[self.all_agent_metrics[i]] = loss_function(metric_value, loss=self.loss[i])
        
        # Second loop: calculate all tracking metrics
        if self.tracking_metric is not None:
            for j in range(len(self.all_agent_tracking_metrics)):
                exp_fmt = self.tracking_exp_format[j]
                metric_name = self.tracking_metric[j]
                loss_type = self.tracking_loss[j]
                result = res_dict[exp_fmt]
                
                if loss_type is None:
                    dum_dict[self.all_agent_tracking_metrics[j]] = self.target_metric(result, metric_name=metric_name)
                else:
                    # Apply data transformation based on compare_type
                    if self.compare_type == 'linear':
                        metric_value = self.target_metric(
                            self.tracking_y[j],
                            result,
                            metric_name=metric_name
                        )
                    else:
                        # Transform data for each format
                        y_true_transformed, y_pred_transformed = transform_data(
                            self.tracking_y[j], 
                            result, 
                            transform_type=self.compare_type
                        )
                        
                        # Calculate metric with transformed data
                        metric_value = self.target_metric(
                            y_true_transformed, 
                            y_pred_transformed, 
                            metric_name=metric_name
                        )

                    # Calculate the loss function based on the metric value
                    dum_dict[self.all_agent_tracking_metrics[j]] = loss_function(metric_value, loss=loss_type)
        
        return dum_dict






