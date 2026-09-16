# Import necessary libraries
import warnings, os, sys, shutil
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / 'data' / 'raw'
_RESULTS_DIR = _REPO_ROOT / 'results'
if str(_REPO_ROOT / 'vendor') not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / 'vendor'))
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import copy, pickle, time
from functools import partial

# optimpv_github is vendored at vendor/optimpv_github and added to sys.path above
from optimpv_github import *
from optimpv_github.RateEqfits.RateEqAgent import RateEqAgent
from optimpv_github.RateEqfits.RateEqModel import *
from optimpv_github.RateEqfits.Pumps import *

import httpimport
with httpimport.github_repo('MimaxSimm', 'trPL_Analysis', ref='master'):
    import trPL_importClass

# =====================================================================
# MCMC-specific imports
# =====================================================================
from optimpv.BayesInfEmcee.EmceeOptimizer import EmceeOptimizer
from optimpv.general.general import calc_metric, transform_data

import importlib
spec = importlib.util.find_spec('optimpv_github')

# =====================================================================
# MCMC configuration
# =====================================================================
SIGMA_NRMSE = 0.01 #0.005          # Likelihood width (~0.2 × best NRMSE)
N_WALKERS = 44               # Number of emcee walkers
N_STEPS = 30000              # MCMC steps per walker
BURN_IN = 2000               # Burn-in steps
N_CORES = 22                 # CPU cores (i9-14900K: 24 cores, use 22)
WARM_START = True            # Initialize from TuRBO best-fit
WARM_SPREAD = 0.001           # Gaussian spread for warm-start

# =====================================================================
# Data import (unchanged from your TuRBO script)
# =====================================================================
def importALL_data(dirs, L_layer=800e-9, alpha=36300*1e2, plot=False):
    trPL_list = []
    N0_list = []
    data2fit_list = []
    if plot:
        fig, axes = plt.subplots(nrows=len(dirs), ncols=4, figsize=(12, 12))

    for idy, dir in enumerate(dirs):
        trPL = trPL_importClass.trPL_measurement_series(dir, BG=1.55, thickness=L_layer, TRPL_denoise=50, mode="auto", retime=True, importPL=True, importSPV=False)
        trPL_list.append(trPL)

        Fluence = np.array(trPL.TRPL_powers)*trPL.BD_ratio*trPL.lambda_laser/(trPL.spot_area*np.array(trPL.TRPL_reprates_Hz)*trPL.hc)
        z_array = np.linspace(0, L_layer, 100)
        ns = alpha * Fluence[3]*np.exp(-alpha*z_array)
        N0_ave = np.trapezoid(ns, x=z_array)/L_layer
        N0_list.append(N0_ave)

        power = []
        pmax = np.amax(trPL.TRPL_powers)
        cut = 1
        if idy > 1:
            cut = 1e-5
        for idx, file in enumerate(trPL.TRPLs_files[:4]):
            t = trPL.TRPLs_ts[:, idx]
            PL = trPL.TRPLs_subsMean[:, idx]
            PL = PL[(t >= 0) & (t <= cut)]
            t = t[(t >= 0) & (t <= cut)]
            t_log = np.logspace(np.log10(t[1]), np.log10(t[-1]), num=1000)
            t_log = np.insert(t_log, 0, 0)
            trPL_log = np.interp(t_log, t, PL)
            power.append(trPL.TRPL_powers[idx])
            if idx == 0:
                data2fit = {'t': t_log, 'trPL': trPL_log, 'G_frac': power[-1] / pmax * np.ones_like(t_log)}
            else:
                data2fit['t'] = np.concatenate((data2fit['t'], t_log))
                data2fit['trPL'] = np.concatenate((data2fit['trPL'], trPL_log))
                data2fit['G_frac'] = np.concatenate((data2fit['G_frac'], power[-1] / pmax * np.ones_like(t_log)))
            if plot:
                ax = axes[idy, idx]
                ax.plot(t, PL, 'o', label='raw')
                ax.plot(t_log, trPL_log, '*', label='Interpolated trPL')
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('Time [s]'); ax.set_ylabel('trPL [a.u.]')
                ax.set_title(f'P = {power[-1]:.2e} $\\mu$W')
                ax.legend()
        data2fit = pd.DataFrame(data2fit)
        data2fit_list.append(data2fit)
    if plot:
        plt.tight_layout(); plt.show()
    return trPL_list, N0_list, data2fit_list


# =====================================================================
# Parameter definition (unchanged from your TuRBO script)
# =====================================================================
def define_baseParams(Lval=800e-9, alphaval=36300*1e2, num_traps=3):
    params = []

    Eg = FitParam(name='Eg', value=1.553, bounds=[0.5, 2.0], log_scale=False, rescale=True, value_type='float', type='fixed', display_name=r'$E_g$', unit='eV', axis_type='linear')
    params.append(Eg)
    L = FitParam(name='L', value=Lval, bounds=[400e-9, 1e-6], log_scale=True, rescale=True, value_type='float', type='fixed', display_name=r'$L$', unit='m', axis_type='linear', force_log=True)
    params.append(L)
    alpha = FitParam(name='alpha', value=alphaval, bounds=[1e6, 1e8], log_scale=True, rescale=True, value_type='float', type='fixed', display_name=r'$\alpha$', unit='m$^{-1}$', axis_type='log')
    params.append(alpha)
    N_cv = FitParam(name='N_cv', value=2e24, bounds=[1e19, 1e26], log_scale=True, rescale=True, value_type='float', type='fixed', display_name=r'$N_{cv}$', unit='m$^{-3}$', axis_type='log', force_log=True)
    params.append(N_cv)
    k_direct = FitParam(name='k_direct', value=1.96e-17, bounds=[1e-18, 5e-17], log_scale=True, rescale=True, value_type='float', type='range', display_name=r'$k_{\text{direct}}$', unit='m$^{3}$ s$^{-1}$', axis_type='log', force_log=True)
    params.append(k_direct)
    mu_n = FitParam(name='mu_n', value=1.2e-4, bounds=[1e-6, 1e-3], log_scale=True, rescale=True, value_type='float', type='range', display_name=r'$\mu_n$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type='log', force_log=True)
    params.append(mu_n)
    mu_p = FitParam(name='mu_p', value=4e-5, bounds=[1e-6, 1e-3], log_scale=True, rescale=True, value_type='float', type='range', display_name=r'$\mu_p$', unit='m$^{2}$ V$^{-1}$ s$^{-1}$', axis_type='log', force_log=True)
    params.append(mu_p)

    for i in range(num_traps):
        N_t_bulk = FitParam(name=f'N_t_bulk_{i+1}', value=1.85e23, bounds=[1e19, 1e26], log_scale=True, rescale=True, value_type='float', type='range', display_name=r'N_{t,\text{bulk}}$', unit='m$^{-3}$', axis_type='log', force_log=True)
        params.append(N_t_bulk)
        C_n = FitParam(name=f'C_n_{i+1}', value=4.24e-15, bounds=[1e-22, 1e-10], log_scale=True, rescale=True, value_type='float', type='range', display_name=r'$C_{n,}$', unit='m$^{3}$ s$^{-1}$', axis_type='log', force_log=True)
        params.append(C_n)
        C_p = FitParam(name=f'C_p_{i+1}', value=8.85e-19, bounds=[1e-22, 1e-10], log_scale=True, rescale=True, value_type='float', type='range', display_name=r'$C_{p,}$', unit='m$^{3}$ s$^{-1}$', axis_type='log', force_log=True)
        params.append(C_p)
        E_t_bulk = FitParam(name=f'E_t_bulk_{i+1}', value=1.0, bounds=[Eg.value/2, Eg.value-0.02], log_scale=False, rescale=True, value_type='float', type='range', display_name=r'$E_{t,\text{bulk}}$', unit='m$^{3}$ s$^{-1}$', axis_type='log', force_log=False)
        params.append(E_t_bulk)

    I_factor_PL = FitParam(name='I_factor_PL', value=1.275e-22, bounds=[1e-27, 1e-20], log_scale=True, rescale=True, value_type='float', type='fixed', display_name=r'$I_{\text{PL}}$', unit='-', axis_type='log', force_log=True)
    params.append(I_factor_PL)

    params_orig = copy.deepcopy(params)
    return params, params_orig


# =====================================================================
# Agent definition (unchanged — same RateEqAgent as TuRBO)
# =====================================================================
def define_rateEq_agent(params, data_2fit, N0):
    """Create the RateEqAgent (same as TuRBO, no optimizer)."""
    X = np.asarray(data_2fit[['t', 'G_frac']])
    y = np.asarray(data_2fit['trPL'])
    fpu = 10e3
    background = 0e28
    pump_args = {'fpu': fpu, 'background': background, 'N0': N0}
    model = partial(DBTD_multi_trap, method='BDF', dimensionless=True,
                    timeout=90, timeout_solve=90, use_jacobian=True)

    RateEq = RateEqAgent(
        params, [X], [y], model=model, pump_model=initial_carrier_density,
        pump_args=pump_args, fixed_model_args={}, metric='nrmse',
        loss='linear', minimize=True, exp_format='trPL',
        detection_limit=0e-5, compare_type='normalized_log',
        do_G_frac_transform=True, parallel=False)
    return RateEq


# =====================================================================
# Fix the EmceeOptimizer log-likelihood (the critical correction)
# =====================================================================
class PatchedEmceeOptimizer(EmceeOptimizer):
    _patch_sigma = 0.005
    
    def _log_likelihood(self, theta, agents=None):
        if agents is None:
            agents = self.agents
        sigma = self._patch_sigma
        
        param_dict = {}
        idx = 0
        for param in self.params:
            if param.type == 'fixed':
                continue  # ← skip! run_Ax handles fixed params internally
            else:
                param_dict[param.name] = theta[idx]
                idx += 1

        total_log_like = 0.0
        try:
            for agent in agents:
                agent_results = agent.run_Ax(param_dict)
                for metric_name in self.all_metrics:
                    if metric_name in agent_results:
                        loss_val = agent_results[metric_name]
                        if np.isnan(loss_val) or not np.isfinite(loss_val):
                            return -np.inf
                        total_log_like += -0.5 * (loss_val / sigma) ** 2
                    else:
                        return -np.inf
            if not np.isfinite(total_log_like):
                return -np.inf
            return total_log_like
        except Exception:
            return -np.inf

def patch_log_likelihood(emcee_opt, sigma_nrmse):
    """Set sigma as both class and instance attribute."""
    PatchedEmceeOptimizer._patch_sigma = sigma_nrmse
    emcee_opt._patch_sigma = sigma_nrmse
    print(f"  [PATCHED] σ={sigma_nrmse}")


# =====================================================================
# Tighten bounds from Turbo
# =====================================================================
def tighten_bounds_around_turbo(params, turbo_params, total_log_orders=3.0,
                                linear_frac=0.1, min_linear_width=0.02,
                                k_direct_log_orders=2.0):
    turbo_map = {p.name: p for p in turbo_params}
    half_decades = total_log_orders / 2.0
    factor = 10 ** half_decades
    for p in params:
        if p.type == 'fixed':
            continue
        if p.name not in turbo_map:
            continue
        orig_lo, orig_hi = p.bounds
        best_val = turbo_map[p.name].value
        if not np.isfinite(best_val):
            continue
        if p.name == 'k_direct':
            if k_direct_log_orders is None:
                continue  # skip entirely (old behavior)
            half_dec_k = k_direct_log_orders / 2.0
            factor_k = 10 ** half_dec_k
            new_lo = best_val / factor_k
            new_hi = best_val * factor_k
        elif getattr(p, 'force_log', False):
            if best_val <= 0:
                print(f"  [WARN] Cannot tighten log bounds for {p.name}: best_val={best_val}")
                continue
            new_lo = best_val / factor
            new_hi = best_val * factor
        else:
            half_width = 0.5 * linear_frac * (orig_hi - orig_lo)
            half_width = max(half_width, min_linear_width)
            new_lo = best_val - half_width
            new_hi = best_val + half_width
        if np.isfinite(new_lo) and np.isfinite(new_hi) and (new_hi > new_lo):
            p.bounds = [new_lo, new_hi]
            print(f"  [TIGHTEN] {p.name:>15s}: [{orig_lo:.3e}, {orig_hi:.3e}] "
                  f"-> [{new_lo:.3e}, {new_hi:.3e}] (best={best_val:.3e})")
        else:
            print(f"  [WARN] Invalid tightened bounds for {p.name}")

# =====================================================================
# Warm-start from TuRBO pickle
# =====================================================================
def load_turbo_best_params(turbo_pickle_path):
    """Load a TuRBO optimizer pickle and extract the best parameters."""
    try:
        with open(turbo_pickle_path, 'rb') as f:
            turbo = pickle.load(f)
        
        # Extract best params from Ax experiment
        ax_client = turbo.ax_client
        experiment = ax_client._maybe_experiment or ax_client.experiment
        data = experiment.fetch_data()
        df = data.df
        best_row = df.loc[df["mean"].idxmin()]
        best_trial_idx = int(best_row["trial_index"])
        best_nrmse = best_row["mean"]
        best_params = dict(experiment.trials[best_trial_idx].arm.parameters)
        
        # Also update the param values in the turbo object
        turbo.update_params_with_best_balance()
        
        print(f"  [OK] Loaded TuRBO best: NRMSE={best_nrmse:.5f} from trial #{best_trial_idx}")
        return best_params, best_nrmse, turbo.params
    except Exception as e:
        print(f"  [WARN] Could not load TuRBO pickle: {e}")
        return None, None, None


def warm_start_from_turbo(emcee_opt, turbo_best_params, spread=0.02):
    """
    Initialize emcee walkers near the TuRBO best-fit.
    turbo_best_params is a dict in Ax space (log10 for most params).
    EmceeOptimizer's create_search_space uses force_log, so optimization
    space is also log10 for the same params → direct mapping.
    """
    ndim = emcee_opt.ndim
    bounds = np.array(emcee_opt.bounds)

    # Map TuRBO best params to the EmceeOptimizer's optimization vector
    x0 = np.zeros(ndim)
    idx = 0
    for param in emcee_opt.params:
        if param.type == 'fixed':
            continue
        name = param.name
        if name in turbo_best_params:
            ax_val = turbo_best_params[name]
            if param.force_log:
                # Ax stores log10, EmceeOptimizer also uses log10 via force_log
                x0[idx] = ax_val
            else:
                # E_t_bulk: Ax stores the raw value, EmceeOptimizer scales by fscale
                scale = param.fscale if hasattr(param, 'fscale') and param.fscale else 1.0
                # Ax value is already in the rescaled space matching what 
                # create_search_space produces: value / scale_factor
                x0[idx] = ax_val  # Already in correct space
        else:
            x0[idx] = emcee_opt.x0[idx]  # Fallback
        idx += 1

    # Clip to bounds
    x0 = np.clip(x0, bounds[:, 0] + 1e-10, bounds[:, 1] - 1e-10)

    # Verify prior is finite
    if not np.isfinite(emcee_opt._log_prior(x0)):
        print("  [WARN] TuRBO best-fit violates prior. Falling back to LHS.")
        return emcee_opt.initialize_walkers()

    # Generate walkers around x0
    param_ranges = bounds[:, 1] - bounds[:, 0]
    pos = np.zeros((emcee_opt.nwalkers, ndim))
    n_valid = 0

    for _ in range(emcee_opt.nwalkers * 200):
        if n_valid >= emcee_opt.nwalkers:
            break
        candidate = x0 + spread * param_ranges * np.random.randn(ndim)
        candidate = np.clip(candidate, bounds[:, 0] + 1e-10, bounds[:, 1] - 1e-10)
        if np.isfinite(emcee_opt._log_prior(candidate)):
            pos[n_valid] = candidate
            n_valid += 1

    if n_valid < emcee_opt.nwalkers:
        print(f"  [WARN] Only {n_valid}/{emcee_opt.nwalkers} valid walkers.")
        lhs = emcee_opt.initialize_walkers()
        pos[n_valid:] = lhs[:emcee_opt.nwalkers - n_valid]

    print(f"  [OK] Warm-started {n_valid} walkers (spread={spread})")
    return pos

# Add this near the top, with the other module-level globals:
_WARM_POS = None

def _warm_initialize_walkers():
    """Module-level warm-start initializer (picklable)."""
    return _WARM_POS


def plot_corner_fixed(emcee_opt, savepath=None):
    """Fixed corner plot — only uses free parameters."""
    import corner
    if emcee_opt.flat_samples is None:
        print("No samples to plot")
        return None

    samples_orig = []
    labels = []
    axes_scale = []

    for i, name in enumerate(emcee_opt.param_mapping):
        param = next(p for p in emcee_opt.params if p.name == name)
        labels.append(param.display_name if hasattr(param, 'display_name') else name)

        if i in emcee_opt.log_params_indices:
            samples_orig.append(10 ** emcee_opt.flat_samples[:, i])
        else:
            scale = param.fscale if hasattr(param, 'fscale') and param.fscale else 1.0
            samples_orig.append(emcee_opt.flat_samples[:, i] * scale)

        axes_scale.append(getattr(param, 'axis_type', 'linear'))

    data = np.vstack(samples_orig).T

    fig = corner.corner(data, labels=labels, show_titles=True,
                        quantiles=[0.16, 0.5, 0.84], title_fmt=".4e",
                        axes_scale=axes_scale, color='darkblue',
                        title_kwargs={"fontsize": 10})
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {savepath}")
    return fig


def plot_traces_fixed(emcee_opt, savepath=None):
    """Fixed trace plot — only iterates over free parameters."""
    if emcee_opt.chain is None:
        print("No chain to plot")
        return None

    n_steps, n_walkers, n_dim = emcee_opt.chain.shape
    fig, axes = plt.subplots(n_dim, figsize=(10, 2 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]

    for i, name in enumerate(emcee_opt.param_mapping):
        param = next(p for p in emcee_opt.params if p.name == name)
        ax = axes[i]
        if i in emcee_opt.log_params_indices:
            ax.plot(10 ** emcee_opt.chain[:, :, i], "k", alpha=0.2)
            ax.set_yscale('log')
        else:
            ax.plot(emcee_opt.chain[:, :, i], "k", alpha=0.2)
        ax.set_ylabel(param.display_name if hasattr(param, 'display_name') else name)
        ax.axvline(emcee_opt.burn_in, color='blue', ls='--', lw=1,
                   label=f'Burn-in ({emcee_opt.burn_in})' if i == 0 else None)
        if i == 0:
            ax.legend(loc='upper right')

    axes[-1].set_xlabel("Step number")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {savepath}")
    return fig

from multiprocessing import Pool
            
def test_worker_sigma(theta):
    from optimpv.BayesInfEmcee.EmceeOptimizer import EmceeOptimizer
    # Import our module to get the subclass
    from run_mcmc_from_turbo_2 import PatchedEmceeOptimizer
    return PatchedEmceeOptimizer._patch_sigma

def test_worker_loglik(args):
    """Test that the actual log-likelihood uses correct sigma in workers."""
    emcee_opt, theta = args
    return emcee_opt._log_likelihood(theta, agents=emcee_opt.agents)

# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    print("Imported optimpv", spec)

    # ---- Data directories (same as your TuRBO script) ----
    # dir2Q3_before = os.path.abspath('/home/qrb/Documents/Scripts/Fitting_Review/PowerDep-Before/Fitted')
    # dir2Q3_after21H = os.path.abspath('/home/qrb/Documents/Scripts/Fitting_Review/PowerDep-After20Hours1SUN')
    # dir3Q2_before = os.path.abspath('/home/qrb/Documents/Scripts/Fitting_Review/3Q2/Before')
    # dir3Q2_after21H = os.path.abspath('/home/qrb/Documents/Scripts/Fitting_Review/3Q2/After21H')
    dir_Cs00 = str(_DATA_DIR / 'CsSeries' / 'Cs00')
    dir_Cs15 = str(_DATA_DIR / 'CsSeries' / 'Cs15')
    dir_Cs30 = str(_DATA_DIR / 'CsSeries' / 'Cs30')


    savedir = str(_RESULTS_DIR / 'MCMCResults' / 'UsesBestNRMSEs_Run2' / 'Sigma01')
    os.makedirs(savedir, exist_ok=True)

    # ---- Directory where your TuRBO pickle results are saved ----
    turbo_results_dir = str(_RESULTS_DIR / 'FitResults' / 'CsSeries' / 'Run5_12345Traps_L500nm')

    dirs = [dir_Cs00, dir_Cs15, dir_Cs30]
    names = ["-Cs00", "-Cs15", "-Cs30"]

    trPL_list, N0_list, data2fit_list = importALL_data(dirs, L_layer=500e-9, alpha=36300*1e2)

    exp_format = 'trPL'

    ntrap_list = [3,3,2]

    # ---- Loop over trap counts and samples ----
    for n_traps in [2]:
        for id_samp, (N0, data) in enumerate(zip(N0_list, data2fit_list)):
            if not(id_samp == 2):
                continue

            n_traps = ntrap_list[id_samp]
            n_traps = 3 #Try with n = 3

            print(f"\n{'='*65}")
            print(f"  MCMC: {names[id_samp]}, {n_traps} trap(s)")
            print(f"{'='*65}")

            # ---- Find the best TuRBO pickle for this sample/trap combo ----
            turbo_pickle = None
            turbo_best_params = None
            if WARM_START:
                # Search for the best pickle (lowest NRMSE in filename)
                import glob
                import re
                pattern = os.path.join(turbo_results_dir, f'*{names[id_samp]}-{n_traps}traps-*-optimizer.pickle')
                candidates = sorted(glob.glob(pattern))
                
                if candidates:
                    # Pick the one with lowest NRMSE (extracted from filename)
                    def extract_nrmse(path):
                        m = re.search(r'nrmse(\d+)', os.path.basename(path))
                        return int(m.group(1)) if m else 99999
                    turbo_pickle = min(candidates, key=extract_nrmse)
                    print(f"  TuRBO pickle: {os.path.basename(turbo_pickle)}")
                    turbo_best_params, turbo_best_nrmse, turbo_params = load_turbo_best_params(turbo_pickle)
                else:
                    print(f"  [WARN] No TuRBO pickle found for {names[id_samp]} {n_traps}-traps")

            # ---- Define parameters and agent (identical to TuRBO) ----
            params, params_orig = define_baseParams(num_traps=n_traps)

            # If we have TuRBO results, seed parameter values from best fit
            if turbo_best_params is not None and turbo_params is not None:
                for p in params:
                    for tp in turbo_params:
                        if p.name == tp.name:
                            p.value = tp.value
                            break
            
            fix_params = ['mu_n', 'mu_p']
            for p in params:
                if p.name in fix_params:
                    p.type = 'fixed'
                    # no need to touch force_log or value anymore

            tighten_bounds_around_turbo(
                params,
                turbo_params,
                total_log_orders=6.0,   # total width = 6 decades
                linear_frac=0.50,       # total width = 10% of original linear interval
                min_linear_width=0.02, 
                k_direct_log_orders=2.0
            )

            # Load TuRBO pickle and use its agent's data
            with open(turbo_pickle, 'rb') as f:
                turbo_obj = pickle.load(f)

            turbo_obj.update_params_with_best_balance()
            turbo_agent = turbo_obj.agents if not isinstance(turbo_obj.agents, list) else turbo_obj.agents[0]

            # Use TuRBO's exact X and y
            X_turbo = turbo_agent.X[0]
            y_turbo = turbo_agent.y[0]

            # Build data2fit DataFrame from TuRBO's data
            data = pd.DataFrame({'t': X_turbo[:, 0], 'trPL': y_turbo, 'G_frac': X_turbo[:, 1]})

            # Also get N0 from TuRBO's pump_args
            N0 = turbo_agent.pump_args['N0']

            # Now create the MCMC agent with identical data
            RateEq = define_rateEq_agent(params, data, N0)

            # ---- Create EmceeOptimizer ----
            emcee_opt = PatchedEmceeOptimizer(
                params=params,
                agents=RateEq,
                nwalkers=N_WALKERS,
                nsteps=N_STEPS,
                burn_in=BURN_IN,
                progress=True,
                name='mcmc_opti',
                use_pool=True,
                max_parallelism=N_CORES,
            )

            # ---- Diagnostic: check what params_rescale produces ----
            print("\n  --- Diagnostic: parameters_rescaled ---")
            test_params_rescaled = RateEq.params_rescale({}, RateEq.params)
            for k, v in sorted(test_params_rescaled.items()):
                p = next((pp for pp in params if pp.name == k), None)
                fl = getattr(p, 'force_log', False) if p else '?'
                typ = p.type if p else '?'
                print(f"    {k:>15s} = {v:>12.4e}  (force_log={fl}, type={typ})")

            # Add this right after creating RateEq and before creating emcee_opt:
            print("  Direct agent test...")
            test_result = RateEq.run_Ax(parameters={})
            print(f"  run_Ax with default params: {test_result}")

            # Also test with explicit TuRBO best params in Ax space:
            print(f"  run_Ax with turbo_best_params: {RateEq.run_Ax(turbo_best_params)}")

            # ---- Apply the log-likelihood fix ----
            sigma = SIGMA_NRMSE
            # if turbo_best_nrmse is not None:
            #     sigma = max(0.003, 0.1 * turbo_best_nrmse)  # Scale to best NRMSE
            #     print(f"  σ_NRMSE = {sigma:.5f} (0.1 × best TuRBO NRMSE = {turbo_best_nrmse:.5f})")
            patch_log_likelihood(emcee_opt, sigma)

            # ---- Diagnostic: compare data between TuRBO agent and MCMC agent ----
            print("\n  --- Data comparison ---")

            print(f"  TuRBO agent X shape: {turbo_agent.X[0].shape}")
            print(f"  MCMC  agent X shape: {RateEq.X[0].shape}")
            print(f"  TuRBO agent y shape: {turbo_agent.y[0].shape}")
            print(f"  MCMC  agent y shape: {RateEq.y[0].shape}")
            print(f"  TuRBO X[:5]:  {turbo_agent.X[0][:5]}")
            print(f"  MCMC  X[:5]:  {RateEq.X[0][:5]}")
            print(f"  Same X? {np.allclose(turbo_agent.X[0], RateEq.X[0]) if turbo_agent.X[0].shape == RateEq.X[0].shape else 'DIFFERENT SHAPES'}")
            print(f"  Same y? {np.allclose(turbo_agent.y[0], RateEq.y[0]) if turbo_agent.y[0].shape == RateEq.y[0].shape else 'DIFFERENT SHAPES'}")

            # Also test the TuRBO's own agent directly
            turbo_obj.update_params_with_best_balance()
            turbo_agent.params = turbo_obj.params
            turbo_result = turbo_agent.run_Ax({})
            print(f"\n  TuRBO's own agent run_Ax: {turbo_result}")
            print(f"  MCMC's agent run_Ax:     {RateEq.run_Ax({})}")

           # ---- Warm-start walkers from TuRBO ----
            if WARM_START and turbo_best_params is not None:
                _WARM_POS = warm_start_from_turbo(emcee_opt, turbo_best_params, spread=WARM_SPREAD)
                emcee_opt.initialize_walkers = _warm_initialize_walkers

            # ---- Debug & timing test ----
            print(f"\n  --- Debug: checking parameter space ---")
            print(f"  ndim = {emcee_opt.ndim}")
            print(f"  x0 = {emcee_opt.x0}")
            print(f"  bounds:")
            for i, name in enumerate(emcee_opt.param_mapping):
                lo, hi = emcee_opt.bounds[i]
                x0i = emcee_opt.x0[i]
                in_bounds = lo <= x0i <= hi
                log_tag = " [log10]" if i in emcee_opt.log_params_indices else ""
                print(f"    {name:>15s}: x0={x0i:.6f}  bounds=[{lo:.4f}, {hi:.4f}]  "
                      f"in_bounds={in_bounds}{log_tag}")

            # Test prior separately
            lp = emcee_opt._log_prior(emcee_opt.x0)
            print(f"\n  log_prior(x0) = {lp}")

            if np.isfinite(lp):
                # Test likelihood separately_X_experimental
                print(f"  Testing log_likelihood...", end=" ", flush=True)
                t0 = time.time()
                ll = emcee_opt._log_likelihood(emcee_opt.x0, agents=emcee_opt.agents)
                t_eval = time.time() - t0
                print(f"log_likelihood = {ll:.4f} ({t_eval:.2f}s)")
                test_lp = lp + ll
            else:
                print(f"  [ERROR] Prior is -inf! x0 is outside bounds.")
                # Find which parameter(s) violate bounds
                for i in range(emcee_opt.ndim):
                    lo, hi = emcee_opt.bounds[i]
                    if not (lo <= emcee_opt.x0[i] <= hi):
                        print(f"    VIOLATION: {emcee_opt.param_mapping[i]} "
                              f"x0={emcee_opt.x0[i]} not in [{lo}, {hi}]")
                
                # Try direct model call to test agent
                print(f"\n  Trying direct agent.run_Ax with reconstructed params...")
                test_params = emcee_opt.reconstruct_params(emcee_opt.x0)
                print(f"  Reconstructed: {test_params}")
                try:
                    result = emcee_opt.agents[0].run_Ax(test_params)
                    print(f"  run_Ax result: {result}")
                except Exception as e:
                    print(f"  run_Ax error: {e}")
                
                test_lp = -np.inf
                t_eval = 0

            total_evals = N_WALKERS * (N_STEPS + BURN_IN)
            est = total_evals * max(t_eval, 0.001) / N_CORES
            print(f"\n  log_prob(x0) = {test_lp:.4f}")
            print(f"  Estimated wall time: {est/3600:.1f} hrs ({total_evals:,} evals, {N_CORES} cores)")

            if not np.isfinite(test_lp):
                print(f"  [WARN] x0 gives -inf. MCMC will likely fail.")
                print(f"  Skipping this system. Check parameter mapping.\n")
                continue

            # ---- Test a few walkers before launching ----
            print(f"\n  Testing first 3 walkers...")
            test_pos = emcee_opt.initialize_walkers()
            for w in range(min(3, len(test_pos))):
                lp = emcee_opt._log_probability(test_pos[w], agents=emcee_opt.agents)
                print(f"    Walker {w}: log_prob = {lp:.4f} (finite={np.isfinite(lp)})")
                if not np.isfinite(lp):
                    print(f"      theta = {test_pos[w]}")
                    # Test prior and likelihood separately
                    pri = emcee_opt._log_prior(test_pos[w])
                    print(f"      prior = {pri}")
                    if np.isfinite(pri):
                        ll = emcee_opt._log_likelihood(test_pos[w], agents=emcee_opt.agents)
                        print(f"      likelihood = {ll}")

            # ---- Test that workers use correct sigma ----
        
            print(f"\n  --- Testing log-likelihood in workers ---")
            print(f"  Main process: {emcee_opt._log_likelihood(emcee_opt.x0, agents=emcee_opt.agents):.2f}")
            with Pool(2) as pool:
                worker_lls = pool.map(test_worker_loglik, 
                    [(emcee_opt, emcee_opt.x0), (emcee_opt, emcee_opt.x0)])
            print(f"  Worker 1: {worker_lls[0]:.2f}")
            print(f"  Worker 2: {worker_lls[1]:.2f}")
            
            expected_ll = -0.5 * (turbo_best_nrmse / sigma) ** 2
            if abs(worker_lls[0] - expected_ll) < 1.0:
                print(f"  [OK] Workers compute correct log-likelihood!")
            else:
                print(f"  [FAIL] Expected ~{expected_ll:.1f}, got {worker_lls[0]:.2f}")
                continue


            # ---- Run MCMC ----
            print(f"\n  Running MCMC: {N_WALKERS} walkers × {N_STEPS} steps...")
            t_start = time.time()
            try:
                results = emcee_opt.optimize()
                wall_time = time.time() - t_start
                print(f"  Done in {wall_time:.0f}s ({wall_time/3600:.1f} hrs)")

                # ---- Save results ----
                # Evaluate best NRMSE (same pattern as TuRBO script)
                # Right after: RateEq = define_rateEq_agent(params, data, N0)
                y_experimental = np.asarray(data['trPL'])
                X_experimental = np.asarray(data[['t', 'G_frac']])
                emcee_opt.update_params_with_best_balance()
                RateEq.params = emcee_opt.params
                try:
                    y_test = RateEq.run({}, exp_format=exp_format)
                    if y_test is not None and not np.any(np.isnan(y_test)):
                        y_tr, y_pr = transform_data(y_experimental, y_test,
                                                    transform_type='normalized_log',
                                                    do_G_frac_transform=True, X=X_experimental)
                        nrmse = calc_metric(y_tr, y_pr, metric_name='nrmse')
                    else:
                        nrmse = 0.999
                except Exception as e:
                    print(f"  [WARN] NRMSE eval failed: {e}")
                    nrmse = 0.999

                print(f"  MCMC best NRMSE: {nrmse:.5f}")

                # Save EmceeOptimizer pickle
                tag = f'{names[id_samp]}-{n_traps}traps-nrmse{nrmse*10000:05.0f}'
                fpath = os.path.join(savedir, f'mcmc{tag}-EmceeOptimizer.pickle')
                with open(fpath, 'wb') as f:
                    pickle.dump(emcee_opt, f)
                print(f"  [SAVED] {os.path.basename(fpath)}")

                # Save flat samples as CSV
                if emcee_opt.flat_samples is not None:
                    df_samples = pd.DataFrame(emcee_opt.flat_samples,
                                              columns=emcee_opt.param_mapping)
                    csv_path = os.path.join(savedir, f'mcmc{tag}-samples.csv')
                    df_samples.to_csv(csv_path, index=False)
                    print(f"  [SAVED] {os.path.basename(csv_path)}")

                # Save corner plot
                try:
                    fig = plot_corner_fixed(emcee_opt,
                          savepath=os.path.join(savedir, f'mcmc{tag}-corner.pdf'))
                    if fig: plt.close(fig)
                except Exception as e:
                    print(f"  [WARN] Corner plot failed: {e}")

                # Save trace plot
                try:
                    fig = plot_traces_fixed(emcee_opt,
                          savepath=os.path.join(savedir, f'mcmc{tag}-traces.pdf'))
                    if fig: plt.close(fig)
                except Exception as e:
                    print(f"  [WARN] Trace plot failed: {e}")

                # Print autocorrelation times
                try:
                    tau = emcee_opt.sampler.get_autocorr_time(tol=0)
                    print(f"\n  Autocorrelation times:")
                    for i, name_p in enumerate(emcee_opt.param_mapping):
                        status = "OK" if np.isfinite(tau[i]) and tau[i] < N_STEPS / 50 else "WARN"
                        print(f"    {name_p:>15s}: τ={tau[i]:.1f} [{status}]")
                except Exception:
                    print("  [WARN] Autocorrelation estimation failed")

            except Exception as e:
                wall_time = time.time() - t_start
                print(f"  [FAILED] after {wall_time:.0f}s: {e}")
                # Save what we have
                fpath = os.path.join(savedir, f'mcmc{names[id_samp]}-{n_traps}traps-FAILED-EmceeOptimizer.pickle')
                with open(fpath, 'wb') as f:
                    pickle.dump(emcee_opt, f)

    print(f"\n{'='*65}")
    print(f"  All MCMC runs complete. Results in: {savedir}")
    print(f"{'='*65}")
