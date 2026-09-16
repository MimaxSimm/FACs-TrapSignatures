""" Test EQE module with pySIMsalabim"""

######### Package Imports #########################################################################

import warnings, os, sys, shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch, copy, uuid
import ax, logging

try:
    from optimpv import *
    from optimpv.axBOtorch.axUtils import *
    from optimpv.Diodefits.DiodeAgent import DiodeAgent
    from optimpv.Diodefits.DiodeModel import *
except Exception as e:
    # Add the parent directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from optimpv import *
    from optimpv.axBOtorch.axUtils import *
    from optimpv.Diodefits.DiodeAgent import DiodeAgent
    from optimpv.Diodefits.DiodeModel import *


######### Test Functions #########################################################################

def test_SOO_diode_fit():
    """Test the single-objective optimization of a diode model using scipyOptimizer with scipy.optimize.minimize."""
    try:
        params = []

        J0 = FitParam(name = 'J0', value = 1e-5, bounds = [1e-6,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$J_0$', unit='A m$^{-2}$', axis_type = 'log',force_log=True)
        params.append(J0)

        n = FitParam(name = 'n', value = 1.5, bounds = [1,2], log_scale = False, value_type = 'float', type='range', display_name=r'$n$', unit='', axis_type = 'linear')
        params.append(n)

        R_series = FitParam(name = 'R_series', value = 1e-4, bounds = [1e-5,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$R_{\text{series}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_series)

        R_shunt = FitParam(name = 'R_shunt', value = 1e-1, bounds = [1e-2,1e2], log_scale = True, rescale = False, value_type = 'float', type='fixed', display_name=r'$R_{\text{shunt}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_shunt)

        # original values
        params_orig = copy.deepcopy(params)

        # Create JV to fit
        X = np.linspace(0.001,1,100)
        y = NonIdealDiode_dark(X, J0.value, n.value, R_series.value, R_shunt.value)

        # Define the Agent and the target metric/loss function
        metric = 'mse' # can be 'nrmse', 'mse', 'mae'
        loss = 'soft_l1' # can be 'linear', 'huber', 'soft_l1'
        exp_format = 'dark' # can be 'dark', 'light' depending on the type of data you have
        use_pvlib = False # if True, use pvlib to calculate the diode model if not use the implementation in DiodeModel.py

        diode = DiodeAgent(params, X, y, metric = metric, loss = loss, minimize=True,exp_format=exp_format,use_pvlib=use_pvlib,compare_type='log')

        from optimpv.scipyOpti.scipyOptimizer import ScipyOptimizer
        optimizer = ScipyOptimizer(params=params, agents=diode, method='L-BFGS-B', options={}, name='scipy_opti', parallel_agents=True, max_parallelism=os.cpu_count()-1, verbose_logging=True)
        optimizer.optimize() # run the optimization with ax

        assert True

    except Exception as e:
        assert False, "Error occurred during diode fitting: {}".format(e)

def test_SOO_diode_fit():
    """Test the single-objective optimization of a diode model using scipyOptimizer with scipy.optimize.least_squares."""
    try:
        params = []

        J0 = FitParam(name = 'J0', value = 1e-5, bounds = [1e-6,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$J_0$', unit='A m$^{-2}$', axis_type = 'log',force_log=True)
        params.append(J0)

        n = FitParam(name = 'n', value = 1.5, bounds = [1,2], log_scale = False, value_type = 'float', type='range', display_name=r'$n$', unit='', axis_type = 'linear')
        params.append(n)

        R_series = FitParam(name = 'R_series', value = 1e-4, bounds = [1e-5,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$R_{\text{series}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_series)

        R_shunt = FitParam(name = 'R_shunt', value = 1e-1, bounds = [1e-2,1e2], log_scale = True, rescale = False, value_type = 'float', type='fixed', display_name=r'$R_{\text{shunt}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_shunt)

        # original values
        params_orig = copy.deepcopy(params)

        # Create JV to fit
        X = np.linspace(0.001,1,100)
        y = NonIdealDiode_dark(X, J0.value, n.value, R_series.value, R_shunt.value)

        # Define the Agent and the target metric/loss function
        metric = 'mse' # can be 'nrmse', 'mse', 'mae'
        loss = 'soft_l1' # can be 'linear', 'huber', 'soft_l1'
        exp_format = 'dark' # can be 'dark', 'light' depending on the type of data you have
        use_pvlib = False # if True, use pvlib to calculate the diode model if not use the implementation in DiodeModel.py

        diode = DiodeAgent(params, X, y, metric = metric, loss = loss, minimize=True,exp_format=exp_format,use_pvlib=use_pvlib,compare_type='log')

        from optimpv.scipyOpti.scipyOptimizer import ScipyOptimizer
        optimizer = ScipyOptimizer(params=params, agents=diode, method='dogbox', options={}, name='scipy_opti', parallel_agents=True, max_parallelism=os.cpu_count()-1, verbose_logging=True)
        optimizer.optimize_least_squares() # run the optimization with ax

        assert True

    except Exception as e:
        assert False, "Error occurred during diode fitting: {}".format(e)

