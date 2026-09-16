# Import necessary libraries
import warnings, os, sys, shutil
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / 'data' / 'raw'
_RESULTS_DIR = _REPO_ROOT / 'results'
if str(_REPO_ROOT / 'vendor') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'vendor'))
# remove warnings from the output
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import torch, copy, uuid
import pySIMsalabim as sim
from pySIMsalabim.experiments.JV_steady_state import *
import ax, logging
from ax.utils.notebook.plotting import init_notebook_plotting, render

#Imported the github folder here.
# optimpv_github is vendored at vendor/optimpv_github and added to sys.path above
from optimpv_github import *
from optimpv_github.RateEqfits.RateEqAgent import RateEqAgent
from optimpv_github.RateEqfits.RateEqModel import *
from optimpv_github.RateEqfits.Pumps import *

# Check that it is the right optimPV imported
import importlib
spec = importlib.util.find_spec('optimpv_github')

import httpimport
import sys
with httpimport.github_repo('MimaxSimm', 'trPL_Analysis', ref='master'):
    import trPL_importClass

from optimpv_github.axBOtorch.axBOtorchOptimizer import axBOtorchOptimizer
from botorch.acquisition.logei import qLogNoisyExpectedImprovement 
from ax.adapter.transforms.standardize_y import StandardizeY
from ax.adapter.transforms.unit_x import UnitX
from ax.adapter.transforms.remove_fixed import RemoveFixed
from ax.adapter.transforms.log import Log
from ax.generators.torch.botorch_modular.utils import ModelConfig
from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec
from gpytorch.kernels import MaternKernel
from gpytorch.kernels import ScaleKernel
from botorch.models import SingleTaskGP

import copy
import pickle
from optimpv.general.general import *
from optimpv.scipyOpti.scipyOptimizer import ScipyOptimizer
import time


def importALL_data(dirs, L_layer = 800e-9, alpha = 36300*1e2, plot = False):

    trPL_list = []
    N0_list = []
    data2fit_list = []
    if (plot):
        fig, axes = plt.subplots(nrows=len(dirs), ncols=4, figsize=(12, 12))

    for idy, dir in enumerate(dirs):
        # Note that the BG and thicknesses dont affect the raw import, only processing for other funstionalities of the library.
        trPL = trPL_importClass.trPL_measurement_series(dir, BG = 1.55, thickness = L_layer, TRPL_denoise = 10, mode = "auto", retime = True, importPL=True, importSPV=False)
        trPL_list.append(trPL)
        # import Data, the :4 is to select only the last N.

        #Adapt N0 value
        Fluence = np.array(trPL.TRPL_powers)*trPL.BD_ratio*trPL.lambda_laser/(trPL.spot_area*np.array(trPL.TRPL_reprates_Hz)*trPL.hc)
        z_array = np.linspace(0, L_layer, 100)
        ns = alpha * Fluence[3]*np.exp(-alpha*z_array)
        plt.semilogy(z_array, ns)
        N0_ave = np.trapezoid(ns, x = z_array)/L_layer
        N0_list.append(N0_ave)
        
        power = []
        pmax = np.amax(trPL.TRPL_powers)
        cut = 1
        if(idy > -1):
            cut = 2e-5
        for idx, file in enumerate(trPL.TRPLs_files[:4]):
            t = trPL.TRPLs_ts[:,idx]
            PL = trPL.TRPLs_subsMean[:,idx]
            PL = PL[(t>=0) & (t<=cut)]
            t = t[(t>=0) & (t<=cut)]
            # interpolate the data to have a logarithmically spaced time axis
            t_log = np.logspace(np.log10(t[1]), np.log10(t[-1]), num=1000)
            t_log = np.insert(t_log, 0, 0)  # add 0 to the time array
            # interpolate the trPL values
            trPL_log = np.interp(t_log, t, PL)

            power.append(trPL.TRPL_powers[idx])
            if idx == 0:
                data2fit = {'t': t_log , 'trPL': trPL_log, 'G_frac': power[-1] / pmax * np.ones_like(t_log)}
            else:
                data2fit['t'] = np.concatenate((data2fit['t'], t_log ))
                data2fit['trPL'] = np.concatenate((data2fit['trPL'], trPL_log))
                data2fit['G_frac'] = np.concatenate((data2fit['G_frac'], power[-1] / pmax * np.ones_like(t_log)))
            
            if (plot):
                # plot the data
                ax = axes[idy, idx]
                ax.plot(t, PL, 'o', label='raw')
                ax.plot(t_log, trPL_log, '*', label='Interpolated trPL')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('trPL [a.u.]')
                ax.set_title(f'P = {power[-1]:.2e} $\\mu$W')
                ax.legend()

        data2fit= pd.DataFrame(data2fit)
        data2fit_list.append(data2fit)
    if (plot):
        plt.tight_layout()
        plt.show()

    return trPL_list, N0_list, data2fit_list

def define_baseParams(Lval = 800e-9, alphaval = 36300*1e2, Eg_val = 1.553, num_traps = 3):
    # Define the parameters to be fitted
    params = []

    Eg = FitParam(name = 'Eg', value = Eg_val, bounds = [0.5,2.0], log_scale = False, rescale = True, value_type = 'float', type='fixed', display_name=r'$E_g$', unit='eV', axis_type = 'linear')
    params.append(Eg)

    L = FitParam(name = 'L', value = Lval, bounds = [400e-9,1e-6], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$L$', unit='m', axis_type = 'linear',force_log=True)
    params.append(L)

    alpha = FitParam(name = 'alpha', value = alphaval, bounds = [1e6,1e8], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$\alpha$', unit='m$^{-1}$', axis_type = 'log',)
    params.append(alpha)

    N_cv = FitParam(name = 'N_cv', value = 2e24, bounds = [1e19,1e26], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$N_{cv}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
    params.append(N_cv)

    k_direct = FitParam(name = 'k_direct', value = 1.96e-17, bounds = [5e-18,5e-17], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(k_direct)

    mu_n = FitParam(name = 'mu_n', value = 1.2e-4, bounds = [1e-6,1e-3], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_n$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(mu_n) # 4e-1*1e-4

    mu_p = FitParam(name = 'mu_p', value = 4e-5, bounds = [1e-6,1e-3], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$\mu_p$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type = 'log',force_log=True)
    params.append(mu_p) # 4e-1*1e-4


    for i in range(num_traps):
        N_t_bulk = FitParam(name = 'N_t_bulk'+f'_{(i+1):d}', value = 1.85e23, bounds = [1e19,1e26], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type = 'log',force_log=True)
        params.append(N_t_bulk)

        C_n = FitParam(name = 'C_n'+f'_{(i+1):d}', value = 4.24e-15, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{n,}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(C_n)

        C_p = FitParam(name = 'C_p'+f'_{(i+1):d}', value = 8.85e-19, bounds = [1e-22,1e-12], log_scale = True, rescale = True, value_type = 'float', type='range', display_name=r'$C_{p,}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=True)
        params.append(C_p)
  
        E_t_bulk = FitParam(name = 'E_t_bulk'+f'_{(i+1):d}', value = 1.0, bounds = [Eg.value/2,Eg.value-0.02], log_scale = False, rescale = True, value_type = 'float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type = 'log',force_log=False)
        params.append(E_t_bulk)

    I_factor_PL = FitParam(name = 'I_factor_PL', value = 1.275e-22, bounds = [1e-27,1e-20], log_scale = True, rescale = True, value_type = 'float', type='fixed', display_name=r'$I_{\text{PL}}$', unit='-', axis_type = 'log', force_log=True)
    params.append(I_factor_PL) # in the following we weill fit the PL with the normalized log transformation so this factor is not useful and can be fixed to any value

    # original values
    params_orig = copy.deepcopy(params)
    num_free_params = 0
    dum_dic = {}
    for i in range(len(params)):
        if params[i].force_log:
            dum_dic[params[i].name] = np.log10(params[i].value)
        else:
            dum_dic[params[i].name] = params[i].value/params[i].fscale
    # we need this just to run the model to generate some fake data

        if params[i].type != 'fixed':
            num_free_params += 1

    return params, params_orig

def define_rateEq_andOpti(params, data_2fit, N0, parameter_constraints = [f'mu_n - mu_p >= 0']):
    # original values
    params_orig = copy.deepcopy(params)
    num_free_params = 0
    dum_dic = {}
    for i in range(len(params)):
        if params[i].force_log:
            dum_dic[params[i].name] = np.log10(params[i].value)
        else:
            dum_dic[params[i].name] = params[i].value/params[i].fscale
    # we need this just to run the model to generate some fake data
        if params[i].type != 'fixed'    :
            num_free_params += 1

    # Plot the data to be fitted and the initial guess
    time = data_2fit['t'].values # time in seconds
    X = np.asarray(data_2fit[['t', 'G_frac']])
    y = np.asarray(data_2fit['trPL'])
    fpu = 10e3 # Frequency of the pump laser in Hz
    N0 = N0
    background = 0e28 # Background illumination 

    # Define the Agent and the target metric/loss function
    metric = 'nrmse'
    loss = 'linear' # 'nrmse' or 'mse' or 'soft_l1' or 'linear'
    pump_args = { 'fpu': fpu , 'background' : background, 'N0': N0,}
    exp_format = 'trPL' # experiment format
    model = partial(DBTD_multi_trap, method='BDF', dimensionless=True, timeout=90, timeout_solve=90, use_jacobian=True)

    RateEq = RateEqAgent(params, [X], [y], model = model, pump_model = initial_carrier_density, pump_args = pump_args, fixed_model_args = {}, metric = metric, 
                                    loss = loss,minimize=True,exp_format=exp_format,detection_limit=0e-5,  compare_type ='normalized_log',do_G_frac_transform=True,parallel=False)

    model_gen_kwargs_list = None
    # Here we add some constraints to the parameters to help the optimizer
    
    model_kwargs_list = [{},{"torch_device":torch.device("cuda" if torch.cuda.is_available() else "cpu"),'botorch_acqf_class':qLogNoisyExpectedImprovement,'transforms':[RemoveFixed, Log,UnitX, StandardizeY],
    'surrogate_spec':SurrogateSpec(model_configs=[ModelConfig(botorch_model_class=SingleTaskGP,covar_module_class=ScaleKernel, covar_module_options={'base_kernel':MaternKernel(nu=2.5, ard_num_dims=num_free_params)})])}]

    if not(parameter_constraints == None):
        optimizer_turbo = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,600], batch_size = [10,4], 
        ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parallel_agents= True, parameter_constraints = parameter_constraints)
    else:
        # For optimization down the line where mun and mup are fixed.
        optimizer_turbo = axBOtorchOptimizer(params = params, agents = RateEq, models = ['SOBOL','BOTORCH_MODULAR'],n_batches = [1,600], batch_size = [10,4], 
        ax_client = None,  max_parallelism = 100, model_kwargs_list = model_kwargs_list, model_gen_kwargs_list = model_gen_kwargs_list, name = 'ax_opti',parallel_agents= True)

    return RateEq, optimizer_turbo

if __name__ == "__main__":
    print("Imported optimpv", spec)

    # Import the Data
    dir_Cs00 = str(_DATA_DIR / 'CsSeries' / 'Cs00')
    dir_Cs15 = str(_DATA_DIR / 'CsSeries' / 'Cs15')
    dir_Cs30 = str(_DATA_DIR / 'CsSeries' / 'Cs30')

    savedir = str(_RESULTS_DIR / 'FitResults' / 'CsSeries' / 'Run5_12345Traps_L500nm')

    dirs = [dir_Cs00, dir_Cs15, dir_Cs30]
    #dirs = [dir_Cs15]
    alpha = 31000*1e2   #From TA data
    L_layer = 500e-9    #Define Absorber length
    Eg_list = [1.563, 1.585, 1.563] #From TA data
    trPL_list, N0_list, data2fit_list = importALL_data(dirs, L_layer = L_layer, alpha = alpha, plot=True) #L and alpha are measured from the films.

    names = ["-Cs00", "-Cs15", "-Cs30"]
    exp_format = 'trPL'
    #First loops, do 3 optimisatiosn for each condition, save the data appropriately.
    for n_traps in [1,2,3,4,5]:#[3,2,4,5]:
        for i in range(15):
            for id_samp, (N0, data, Eg) in enumerate(zip(N0_list, data2fit_list, Eg_list)):
                print("----------------------------"+names[id_samp]+"; Number of traps:", n_traps, "-------------------------------")
                params, params_orig = define_baseParams(num_traps = n_traps, Lval=L_layer, alphaval=alpha, Eg_val=Eg)
                params_loop = copy.deepcopy(params)
                nrmse_best = 1e5
                RateEq_best = None

                y_experimental = np.asarray(data['trPL'])
                X_experimental = np.asarray(data[['t', 'G_frac']])

                if (n_traps == 1):
                    constraints =  [f'mu_n - mu_p >= 0',f' -N_t_bulk_1 - 0.5 * C_n_1 - 0.5 * C_p_1 <= -5']
                else:
                    constraints =  [f'mu_n - mu_p >= 0',f' -N_t_bulk_1 - 0.5 * C_n_1 - 0.5 * C_p_1 - N_t_bulk_2 - 0.5 * C_n_2 - 0.5 * C_p_2 <= -5',f'E_t_bulk_1-E_t_bulk_2 <= -0.01'] #try a new one.

                if (n_traps > 2):
                    string1 = f' -N_t_bulk_1 - 0.5 * C_n_1 - 0.5 * C_p_1 - N_t_bulk_2 - 0.5 * C_n_2 - 0.5 * C_p_2'
                    for j in range(3, n_traps+1):
                        string1 = string1 + f' - N_t_bulk_{j:d} - 0.5 * C_n_{j:d} - 0.5 * C_p_{j:d}'
                        constraints.append(f'E_t_bulk_{j-1:d}-E_t_bulk_{j:d} <= -0.01')
                    string1 = string1 + f'<= -5'
                    constraints[1] = string1
                        
                RateEq, optimizer_turbo = define_rateEq_andOpti(params_loop, data, N0, parameter_constraints = constraints)

                turbo = copy.deepcopy(optimizer_turbo)
        
                try:
                    turbo.optimize_turbo(force_continue=False,kwargs_turbo_state={'failure_tolerance':8}) # run the optimization with turbo

                    ax_client = turbo.ax_client # get the ax client
                    turbo.update_params_with_best_balance() # update the params list in the turbo with the best parameters
                    RateEq.params = turbo.params # update the params list in the agent with the best parameters
                    
                    y_test = RateEq.run({},exp_format=exp_format)
                    y_transformed, y_pred_transformed = transform_data(y_experimental,y_test, transform_type='normalized_log', do_G_frac_transform=True, X=X_experimental)
                    nrmse = calc_metric(y_transformed, y_pred_transformed, metric_name='nrmse')

                    # Save result
                    file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-{n_traps:d}traps-nrmse{nrmse*10000:05.0f}-optimizer.pickle')
                    with open(file_path, 'wb') as file:
                        pickle.dump(turbo, file)

                    file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-{n_traps:d}traps-nrmse{nrmse*10000:05.0f}-RateEqAgent.pickle')
                    with open(file_path, 'wb') as file:
                        pickle.dump(RateEq, file)

                except:
                    print("---------------------------- Failed Opti Round", i, "for", names[id_samp])
                    nrmse = 0.1
                    # Save result
                    file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-{n_traps:d}traps-nrmse{nrmse*10000:05.0f}-optimizer.pickle')
                    with open(file_path, 'wb') as file:
                        pickle.dump(turbo, file)

                    file_path = os.path.join(savedir,'0'+str(i)+names[id_samp]+f'-{n_traps:d}traps-nrmse{nrmse*10000:05.0f}-RateEqAgent.pickle')
                    with open(file_path, 'wb') as file:
                        pickle.dump(RateEq, file)
                    print("---------------------------- Failed Opti Round", i, "for", names[id_samp])