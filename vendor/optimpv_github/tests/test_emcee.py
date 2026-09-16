""" Test EQE module with pySIMsalabim"""

######### Package Imports #########################################################################

import warnings, os, sys, shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from copy import deepcopy
import torch, copy, uuid
import ax, logging

try:
    from optimpv import *

except Exception as e:
    # Add the parent directory to the system path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from optimpv import *

from optimpv.BayesInfEmcee.EmceeOptimizer import EmceeOptimizer
import pySIMsalabim as sim
from pySIMsalabim.experiments.JV_steady_state import *
from optimpv.DDfits.JVAgent import JVAgent
######### Test Functions #########################################################################

def test_SOO_JV_fit_emcee():
    """Test the single-objective optimization of a diode model using axBOtorchOptimizer."""
    try:
        params = [] # list of parameters to be optimized

        mun = FitParam(name = 'l2.mu_n', value = 7e-8, bounds = [1e-9,1e-6], log_scale = True, value_type = 'float', fscale = None, rescale = False, display_name=r'$\mu_n$', unit='m$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', force_log = True)
        params.append(mun)

        mup = FitParam(name = 'l2.mu_p', value = 5e-8, bounds = [1e-9,1e-6], log_scale = True, value_type = 'float', fscale = None, rescale = False, display_name=r'$\mu_p$', unit=r'm$^2$ V$^{-1}$s$^{-1}$', axis_type = 'log', force_log = True)
        params.append(mup)

        preLangevin = FitParam(name = 'l2.preLangevin', value = 1e-2, bounds = [0.005,1], log_scale = True, value_type = 'float', fscale = None, rescale = False, display_name=r'$\gamma_{pre}$', unit=r'', axis_type = 'log', force_log = True)
        params.append(preLangevin)


        # Set the session path for the simulation and the input files
        session_path = os.path.join(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),'SIMsalabim','SimSS'))
        input_path = os.path.join(os.path.join(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),'Data','simsalabim_test_inputs','fakeOPV')))
        simulation_setup_filename = 'simulation_setup_fakeOPV.txt'
        simulation_setup = os.path.join(session_path, simulation_setup_filename) 

        # path to the layer files defined in the simulation_setup file
        l1 = 'ZnO.txt'
        l2 = 'ActiveLayer.txt'
        l3 = 'BM_HTL.txt'
        l1 = os.path.join(input_path, l1)
        l2 = os.path.join(input_path, l2)
        l3 = os.path.join(input_path, l3)

        # copy this files to session_path
        force_copy = True
        if not os.path.exists(session_path):
            os.makedirs(session_path)
        for file in [l1,l2,l3,simulation_setup_filename]:
            file = os.path.join(input_path, os.path.basename(file))
            if force_copy or not os.path.exists(os.path.join(session_path, os.path.basename(file))):
                shutil.copyfile(file, os.path.join(session_path, os.path.basename(file)))
            else:
                print('File already exists: ',file)

        # reset simss
        # Set the JV parameters
        Gfracs = [0.1,0.5,1] # Fractions of the generation rate to simulate (None if you want only one light intensity as define in the simulation_setup file)
        UUID = str(uuid.uuid4()) # random UUID to avoid overwriting files

        cmd_pars = [] # see pySIMsalabim documentation for the command line parameters
        # Add the parameters to the command line arguments
        for param in params:
            cmd_pars.append({'par':param.name, 'val':str(param.value)})

        # Run the JV simulation
        ret, mess = run_SS_JV(simulation_setup, session_path, JV_file_name = 'JV.dat', G_fracs = Gfracs, parallel = True, max_jobs = 3, UUID=UUID, cmd_pars=cmd_pars)

        # save data for fitting
        X,y = [],[]
        X_orig,y_orig = [],[]
        if Gfracs is None:
            data = pd.read_csv(os.path.join(session_path, 'JV_'+UUID+'.dat'), sep=r'\s+') # Load the data
            Vext = np.asarray(data['Vext'].values)
            Jext = np.asarray(data['Jext'].values)
            G = np.ones_like(Vext)
            rng = default_rng()#
            noise = rng.standard_normal(Jext.shape) * 0.01 * Jext
            Jext = Jext + noise
            X = Vext
            y = Jext

            plt.figure()
            plt.plot(X,y)
            plt.show()
        else:
            for Gfrac in Gfracs:
                data = pd.read_csv(os.path.join(session_path, 'JV_Gfrac_'+str(Gfrac)+'_'+UUID+'.dat'), sep=r'\s+') # Load the data
                Vext = np.asarray(data['Vext'].values)
                Jext = np.asarray(data['Jext'].values)
                G = np.ones_like(Vext)*Gfrac
                rng = default_rng()#
                noise = rng.standard_normal(Jext.shape) * 0.005 * Jext

                if len(X) == 0:
                    X = np.vstack((Vext,G)).T
                    y = Jext + noise
                    y_orig = Jext 
                else:
                    X = np.vstack((X,np.vstack((Vext,G)).T))
                    y = np.hstack((y,Jext+ noise))
                    y_orig = np.hstack((y_orig,Jext))

            # remove all the current where Jext is higher than a given value
            X = X[y<200]
            X_orig = copy.deepcopy(X)
            y_orig = y_orig[y<200]
            y = y[y<200]
    
        # Define the Agent and the target metric/loss function
        
        metric = 'mse' # can be 'nrmse', 'mse', 'mae'
        loss = 'linear' # can be 'linear', 'huber', 'soft_l1'

        # create a different params list for the agent
        params_agent = copy.deepcopy(params)
        #select a random value between the bounds, we do this because the walkers will be randomly initialized from the param.value
        for param in params_agent:
            if param.force_log:
                param.value =10**np.random.uniform(np.log10(param.bounds[0]),np.log10(param.bounds[1]))
            else:
                param.value = np.random.uniform(param.bounds[0],param.bounds[1])
        
        jv = JVAgent(params, X, y, session_path, simulation_setup, parallel = True, max_jobs = 3, metric = metric, loss = loss)

        # Calulate the target metric for the original parameters
        best_fit_possible = loss_function(calc_metric(y,y_orig, metric_name = metric),loss)

        

        # Define the Bayesian Inference object
        optimizer = EmceeOptimizer(params = params, agents = jv, nwalkers=20, nsteps=20, burn_in=10, progress=True, name='emcee_opti')

        optimizer.optimize()

        # Clean up the output files (comment out if you want to keep the output files)
        sim.clean_all_output(session_path)
        sim.delete_folders('tmp',session_path)
        # uncomment the following lines to delete specific files
        sim.clean_up_output('nk_',session_path)
        sim.clean_up_output('ZnO',session_path)
        sim.clean_up_output('ActiveLayer',session_path)
        sim.clean_up_output('BM_HTL',session_path)
        sim.clean_up_output('simulation_setup_fakeOPV',session_path)
        assert True

    except Exception as e:
        assert False, "Error occurred during JV fitting: {}".format(e)