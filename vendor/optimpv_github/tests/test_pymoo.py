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
    from optimpv.Diodefits.DiodeAgent import DiodeAgent
    from optimpv.Diodefits.DiodeModel import *
except Exception as e:
    # Add the parent directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from optimpv import *
    from optimpv.Diodefits.DiodeAgent import DiodeAgent
    from optimpv.Diodefits.DiodeModel import *

from optimpv.RateEqfits.RateEqAgent import RateEqAgent
from optimpv.RateEqfits.RateEqModel import *
from optimpv.RateEqfits.Pumps import *
from optimpv.pymooOpti.pymooOptimizer import PymooOptimizer
from optimpv.TransferMatrix.TransferMatrixAgent import TransferMatrixAgent
from optimpv.general.SuggestOnlyAgent import SuggestOnlyAgent

######### Test Functions #########################################################################

def test_SOO_diode_fit_pymoo():
    """Test the single-objective optimization of a diode model using axBOtorchOptimizer."""
    try:
        params = []

        J0 = FitParam(name = 'J0', value = 1e-5, bounds = [1e-6,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$J_0$', unit='A m$^{-2}$', axis_type = 'log',force_log=True)
        params.append(J0)

        n = FitParam(name = 'n', value = 1.5, bounds = [1,2], log_scale = False, value_type = 'float', type='range', display_name=r'$n$', unit='', axis_type = 'linear')
        params.append(n)

        R_series = FitParam(name = 'R_series', value = 1e-4, bounds = [1e-5,1e-3], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$R_{\text{series}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
        params.append(R_series)

        R_shunt = FitParam(name = 'R_shunt', value = 1e-1, bounds = [1e-2,1e2], log_scale = True, rescale = False, value_type = 'float', type='range', display_name=r'$R_{\text{shunt}}$', unit=r'$\Omega$ m$^2$', axis_type = 'log',force_log=True)
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

        # Create PymooOptimizer for single-objective optimization
        optimizer = PymooOptimizer(params=params, agents=[diode], algorithm='GA', pop_size=10, n_gen=10, name='pymoo_single_obj', verbose_logging=False)

        optimizer.optimize() # run the optimization with ax

        assert True

    except Exception as e:
        assert False, "Error occurred during diode fitting: {}".format(e)

def test_MOO_TransferMatrix_pymoo():
    """Test the multi-objective optimization of a rate equation model using PymooOptimizer."""

    # Define the parameters for the rate equation model
    try:
        params = []
        d_3 = FitParam(name = 'd_3', value = 80e-9, bounds = [40e-9, 200e-9], log_scale = False, rescale = True, value_type = 'float', type='range', display_name='d_3',unit='m')
        params.append(d_3)

        d_6 = FitParam(name = 'd_6', value =  10e-9, bounds = [5e-9, 20e-9], log_scale = False, rescale = True, value_type = 'float', type='range', display_name='d_6',unit='m')
        params.append(d_6)

        d_7 = FitParam(name = 'd_7', value =  100e-9, bounds = [50e-9, 200e-9], log_scale = False, rescale = True, value_type = 'float', type='range', display_name='d_7',unit='m')
        params.append(d_7)

        d_8 = FitParam(name = 'd_8', value =  10e-9, bounds = [5e-9, 20e-9], log_scale = False, rescale = True, value_type = 'float', type='range', display_name='d_8',unit='m')
        params.append(d_8)

        d_9 = FitParam(name = 'd_9', value =  100e-9, bounds = [50e-9, 200e-9], log_scale = False, rescale = True, value_type = 'float', type='range', display_name='d_9',unit='m')
        params.append(d_9)

        # Initialize the agent and default device stack
        layers = ['SiOx' , 'ITO' , 'ZnO' , 'PCE10_FOIC_1to1' , 'MoOx' , 'Ag', 'MoOx', 'LiF','MoOx', 'LiF','Air'] # list of layers (need to be the same than the name nk_*.csv file in the matdata folder)
        thicknesses =  [0 , 100e-9 , 30e-9  , 100e-9 , 9e-9 , 8e-9, 100e-9, 100e-9, 100e-9, 100e-9, 100e-9]# list of thicknesses in nm
        mat_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),'Data','matdata') # path to the folder containing the nk_*.csv files
        lambda_min = 350e-9 # start of the wavelength range
        lambda_max = 800e-9 # end of the wavelength range
        lambda_step = 1e-9 # wavelength step
        x_step = 1e-9 # x step
        activeLayer = 3 # active layer index
        spectrum = os.path.join(mat_dir ,'AM15G.txt') # path to the AM15G spectrum file
        photopic_file = os.path.join(mat_dir ,'photopic_curve.txt') # path to the photopic spectrum file

        TMAgent = TransferMatrixAgent(params, [None,None], layers=layers, thicknesses=thicknesses, lambda_min=lambda_min, lambda_max=lambda_max, lambda_step=lambda_step, x_step=x_step, activeLayer=activeLayer, spectrum=spectrum, mat_dir=mat_dir, photopic_file=photopic_file, exp_format=['Jsc', 'AVT'],metric=[None,None],loss=[None,None],threshold=[4,0.1],minimize=[False,False])

        # Define the optimizer
        optimizer = PymooOptimizer(params=params, agents=TMAgent, algorithm='NSGA2', pop_size=5, n_gen=5, name='pymoo_single_obj', verbose_logging=True,max_parallelism=100, )

        res = optimizer.optimize() # run the optimization with ax

        assert True
    except Exception as e:
        assert False, "Error occurred during transfer matrix stack optimization: {}".format(e)


def test_DOE_SOO_pymoo():

    try:
        # Define the path to the data 
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir =os.path.join(parent_dir,'Data','6D_pero_opti') # path to the data directory

        # Load the data
        df = pd.read_csv(os.path.join(data_dir,'6D_pero_opti.csv'),sep=r'\s+') # load the data

        params = [] # list of parameters to be optimized

        Spin_Speed_1 = FitParam(name = 'Spin_Speed_1', value = 1000, bounds = [500,3000], value_type = 'float', display_name='Spin Speed 1', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_1)

        Duration_t1 = FitParam(name = 'Duration_t1', value = 10, bounds = [5,35], value_type = 'float', display_name='Duration t1', unit='s', axis_type = 'linear')
        params.append(Duration_t1)

        Spin_Speed_2 = FitParam(name = 'Spin_Speed_2', value = 1000, bounds = [1000,3000], value_type = 'float', display_name='Spin Speed 2', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_2)

        Dispense_Speed = FitParam(name = 'Dispense_Speed', value = 100, bounds = [10,400], value_type = 'float', display_name='Dispense Speed', unit='rpm', axis_type = 'linear')
        params.append(Dispense_Speed)

        Duration_t3 = FitParam(name = 'Duration_t3', value = 10, bounds = [5,35], value_type = 'float', display_name='Duration t3', unit='s', axis_type = 'linear')
        params.append(Duration_t3)

        Spin_Speed_3 = FitParam(name = 'Spin_Speed_3', value = 3000, bounds = [2000,5000], value_type = 'float', display_name='Spin Speed 3', unit='rpm', axis_type = 'linear')
        params.append(Spin_Speed_3)

        suggest = SuggestOnlyAgent(params,exp_format='Pmax',minimize=False,tracking_exp_format=['Jsc','Voc','FF'],name=None)

        # Define the optimizer
        optimizer = PymooOptimizer(params=params, agents=suggest, algorithm='GA', pop_size=6, n_gen=1, name='pymoo_single_obj', verbose_logging=True,max_parallelism=20,existing_data=df, suggest_only=True)

        to_run_next = optimizer.optimize() 

        assert True
    except Exception as e:
        assert False, "Error occurred during DOE single-objective optimization: {}".format(e)


def test_DOE_MOO_pymoo():

    try:
        # Define the path to the data 
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        data_dir =os.path.join(parent_dir,'Data','pero_MOO_opti') # path to the data directory

        # Load the data
        df = pd.read_csv(os.path.join(data_dir,'Pero_PLQY_FWHM.csv'),sep=r',') # load the data

        stepsize_fraction = 0.05
        stepsize_spin_speed = 100

        params = [] # list of parameters to be optimized

        Cs_fraction = FitParam(name = 'Cs_fraction', value = 0, bounds = [0,1], value_type = 'float', display_name='Cs fraction', unit='', axis_type = 'linear')
        params.append(Cs_fraction)

        Fa_fraction = FitParam(name = 'Fa_fraction', value = 0, bounds = [0,1], value_type = 'float', display_name='Fa fraction', unit='', axis_type = 'linear')
        params.append(Fa_fraction)

        Spin_duration_Antisolvent = FitParam(name = 'Spin_duration_Antisolvent', value = 10, bounds = [5,30], value_type = 'float', display_name='Spin duration Antisolvent', unit='s', axis_type = 'linear')
        params.append(Spin_duration_Antisolvent)

        Spin_duration_High_Speed = FitParam(name = 'Spin_duration_High_Speed', value = 20, bounds = [15,60], value_type = 'float', display_name='Spin duration High Speed', unit='s', axis_type = 'linear')
        params.append(Spin_duration_High_Speed)

        Spin_speed = FitParam(name = 'Spin_speed', value = 1000, bounds = [1000,5000], value_type = 'float', display_name='Spin speed', unit='rpm', axis_type = 'linear')
        params.append(Spin_speed)

        threshold = [0.5, 50] # thresholds for the metrics
        suggest = SuggestOnlyAgent(params,exp_format=['PLQY', 'FWHM'],minimize=[False,True],name=None,threshold=threshold)

        # Define the optimizer
        parameter_constraints =[f'Cs_fraction + Fa_fraction <= 1']

        optimizer = PymooOptimizer(params=params, agents=suggest, algorithm='NSGA2', pop_size=6, n_gen=1, name='pymoo_single_obj', verbose_logging=True,max_parallelism=20,existing_data=df, suggest_only=True, parameter_constraints=parameter_constraints)

        to_run_next = optimizer.optimize() 
        assert True
    except Exception as e:
        assert False, "Error occurred during DOE multi-objective optimization: {}".format(e)

