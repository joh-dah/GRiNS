#%%
# Import libraries
import os
import sys
import glob
import time
import itertools
from importlib import import_module
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, Event, steady_state_event
import jax.numpy as jnp
from jax import vmap, pmap
import pandas as pd
import xarray as xr

def ODE_topo(topo_name, simdir):
    """
    Loads an ODE system from a specified topology module and returns it as an ODETerm.

    Parameters:
        topo_name (str): The name of the topology module to import.
        simdir (str): The directory path where the topology module is located.
        
    Returns:
        term (ODETerm): An instance of ODETerm initialized with the ODE system for specified topo.
    """
    sys.path.append(f'{simdir}/{topo_name}')
    mod = import_module(f'{topo_name}')
    term = ODETerm(getattr(mod, "odesys"))
    return term

# ODE solver wrapper function
def solve_ode(term, solver, t0, t1, dt0, y0, args):
    """
    Gets the steady state of an ordinary differential equation (ODE) by solving using the specified solver and parameters.

    Parameters:
        term (callable): The term representing the ODE to be solved.
        solver (object): The solver to be used for solving the ODE.
        t0 (float): The initial time.
        t1 (float): The final time.
        dt0 (float): The initial time step.
        y0 (array-like): The initial state of the system.
        args (array-like): Additional arguments to be passed to the ODE term.

    Returns:
        sol.ys (jnp.array): The final state of the system at time t1.
    """
    # Add line to convert y0 to a tuple
    y0 = tuple(y0)
    # Diffrax solve function
    sol = diffeqsolve(term, solver, t0, t1, dt0, y0, args, saveat=SaveAt(t1=True), stepsize_controller=PIDController(rtol=1e-5, atol=1e-6), max_steps=None, event=Event(steady_state_event()))
    # Return the solution
    return jnp.array(sol.ys).flatten()

def get_steady_states(init_cond, param_vals, term):
    """
    Calculate the steady states of a system of ordinary differential equations (ODEs).

    Parameters:
        init_cond (DataFrame): Initial conditions for the ODEs.
        param_vals (DataFrame): Parameter values for the ODEs.
        term (callable): The ODE system to solve.

    Returns:
        sol_li (DataFrame): DataFrame containing the steady state solutions of the ODEs.
    """
    # Get the node list
    node_li = list(init_cond.columns)
    # Merge the initial conditions and parameter values dataframes with combinations of inital conditions and parameter values
    comb_arr = jnp.array(param_vals.merge(init_cond, how="cross"))
    # Run the solve_ode function over the combinations array
    start = time.time()
    sol_li = vmap(
        lambda i: solve_ode(
            term,
            Tsit5(),
            0,
            200,
            0.1,
            i[-len(node_li):], #Indexing changed
            i[:-len(node_li)],
        ),
    )(comb_arr)
    print(f"Time taken to solve the ODEs: {time.time() - start}")
    # Convert to dataframe
    sol_li = pd.DataFrame(sol_li, columns=node_li, index = pd.MultiIndex.from_product([param_vals.index, init_cond.index], names=["ParaNum", "InitCondNum"]))
    return sol_li
#%%
def solve_ode_timeseries(term, solver, y0, args, t):
    """
    Solve the ordinary differential equations (ODEs) and return the time series data.

    Parameters:
        term (callable): The term representing the ODE to be solved.
        solver (object): The solver to be used for solving the ODE.
        y0 (array-like): The initial state of the system.
        args (array-like): Additional arguments to be passed to the ODE term.
        t (array-like): The time points at which to evaluate the solution.

    Returns:
        sol.ys (jnp.array): The time series data of the system.
    """
    # Add line to convert y0 to a tuple
    y0 = tuple(y0)
    # Diffrax solve function
    sol = diffeqsolve(term, solver, t[0], t[-1], (t[1]-t[0]), y0, args, saveat=SaveAt(ts=t), stepsize_controller=PIDController(rtol=1e-5, atol=1e-6), max_steps=None)
    # Return the solution
    return jnp.array(sol.ys).T
#%%
def ode_timeseries_param(init_cond, param_vals, t, term):
    """
    Solve a system of ordinary differential equations (ODEs) over a given time range for multiple combinations of initial conditions and parameter values.

    Parameters:
        init_cond (DataFrame): DataFrame containing the initial conditions for each node.
        param_vals (DataFrame): DataFrame containing the parameter values for each combination.
        t (array-like): Array-like object representing the time range.
        term (callable): Callable representing the ODE system.

    Returns:
        sol_li (DataArray): xarray DataArray containing the solutions of the ODEs for each combination of initial conditions and parameter values.
    """
    # Get the node list
    node_li = list(init_cond.columns)
    # Merge the initial conditions and parameter values dataframes with combinations of inital conditions and parameter values
    comb_arr = jnp.array(param_vals.merge(init_cond, how="cross"))
    # Run the solve_ode function over the combinations array
    start = time.time()
    sol_li = vmap(
        lambda i: solve_ode_timeseries(
            term,
            Tsit5(),
            i[-len(node_li):], #Indexing changed
            i[:-len(node_li)],
            t
        ),
    )(comb_arr)
    print(f"Time taken to solve the ODEs: {time.time() - start}")
    # Convert to xarray dataarray
    sol_li = xr.DataArray(sol_li.transpose(1, 2, 0), coords=[t,node_li, pd.MultiIndex.from_product([param_vals.index, init_cond.index])], dims=[ "Time","Node", "Combination"])
    return sol_li
# %%
