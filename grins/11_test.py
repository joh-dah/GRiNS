from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Tsit5,
    PIDController,
    Event,
    steady_state_event,
)

import jax.numpy as jnp
import numpy as np
from jax import lax, clear_caches, jit
from jax.tree import map
import pandas as pd
import glob
import os
import time
from multiprocessing import Pool, cpu_count
import gc


# Positive Shifted Hill function
def psH(nod, fld, thr, hill):
    return (fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))) / fld


# Negative Shifted Hill function
def nsH(nod, fld, thr, hill):
    return fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))


# ODE system function for diffrax
@jit
def odesys_11(t, y, args):
    (A, B, C, D) = y
    (
        Prod_A,
        Prod_B,
        Prod_C,
        Prod_D,
        Deg_A,
        Deg_B,
        Deg_C,
        Deg_D,
        InhFld_B_A,
        Thr_B_A,
        Hill_B_A,
        ActFld_C_A,
        Thr_C_A,
        Hill_C_A,
        InhFld_A_B,
        Thr_A_B,
        Hill_A_B,
        ActFld_A_D,
        Thr_A_D,
        Hill_A_D,
    ) = args
    d_A = (
        Prod_A
        * nsH(B, InhFld_B_A, Thr_B_A, Hill_B_A)
        * psH(C, ActFld_C_A, Thr_C_A, Hill_C_A)
        - Deg_A * A
    )
    d_B = Prod_B * nsH(A, InhFld_A_B, Thr_A_B, Hill_A_B) - Deg_B * B
    d_C = Prod_C - Deg_C * C
    d_D = Prod_D * psH(A, ActFld_A_D, Thr_A_D, Hill_A_D) - Deg_D * D
    d_y = (d_A, d_B, d_C, d_D)
    return d_y


def _gen_combinations(num_init_conds, num_params):
    # Generate the combinations more efficiently using numpy
    i = jnp.repeat(jnp.arange(num_init_conds), num_params)
    p = jnp.tile(jnp.arange(num_params), num_init_conds)
    icprm_comb = jnp.stack([i, p], axis=1)
    return icprm_comb


# A closure to parameterise all the non initial and paramter values in the ODE system
def parameterise_solveode(
    odesys,
    inicond,
    paramvals,
    solver=Tsit5(),
    t0=0,
    t1=250,
    dt0=0.1,
    ts=None,
    retol=1e-5,
    atol=1e-6,
    max_steps=None,
):
    ode_term = ODETerm(odesys)
    if ts is None:
        saveat = SaveAt(t1=True)
    else:
        # Make sure the time steps are sorted
        ts = sorted(ts)
        if t1 < ts[-1]:
            t1 = ts[-1]
        # Make saveat to be steps
        saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=retol, atol=atol)
    # Convert the initial conditions and parameter values to a jax array
    inicond = jnp.array(inicond)
    paramvals = jnp.array(paramvals)
    # Generate the combinations of the indices of the initial conditions and parameter values
    icprm_comb = _gen_combinations(len(inicond), len(paramvals))

    # Check if number of time steps to save is None
    if ts is None:
        # Function to solve the ODEs
        def solve_ode(pi_row):
            sol = diffeqsolve(
                ode_term,  # ODETerm
                solver,  # Solver
                t0,  # Start time
                t1,  # End time
                dt0,  # Time step
                tuple(inicond[pi_row[0]][:-1]),  # Initial conditions
                tuple(paramvals[pi_row[1]][:-1]),  # Parameters
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
                event=Event(steady_state_event()),
            )
            return (
                sol.ys,  # The steady state vals, Nodes as list elements
                sol.ts[-1],  # The time point at which the steady state was reached
                sol.event_mask,  # The steady state event mask
            )

        # Function to format steady state solutions
        def _format_solution_chunk(steady_state_array, round, ts, icprm_chunk):
            if ts is not None:
                raise ValueError("Time series values are not None.")
            # Concatenate the steady state values with the end time and event mask in one shot
            # Sol_li is a list of tuples
            # Each of the tuples has a length equal to the numner of initial conditions-parameter combinations for that chunk
            # First n (n = len(node_li)) elements are the steady state values
            # The n+1 element is the end time and the n+2 element is the event mask
            steady_state_array = jnp.concatenate(
                [
                    jnp.squeeze(jnp.array(steady_state_array[:-2])),
                    jnp.squeeze(jnp.array(steady_state_array[-2:])),
                ],
                axis=0,
            )
            # Round the final steady state array
            steady_state_array = jnp.round(steady_state_array.T, round)
            # Getting the initial condition and parameter numbers and concatenating them with the steady state values
            steady_state_array = jnp.column_stack(
                [
                    steady_state_array,
                    inicond[icprm_chunk[:, 0], -1],
                    paramvals[icprm_chunk[:, 1], -1],
                ],
            )
            # Returning as a numpy array to save VRAM space
            return np.array(steady_state_array)
    else:
        # Function to solve the ODEs
        def solve_ode(pi_row):
            sol = diffeqsolve(
                ode_term,  # ODETerm
                solver,  # Solver
                t0,  # Start time
                t1,  # End time
                dt0,  # Time step
                tuple(inicond[pi_row[0]][:-1]),  # Initial conditions
                tuple(paramvals[pi_row[1]][:-1]),  # Parameters
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=max_steps,
                # event=Event(steady_state_event()), # Removed as suing this will give inf values for time points after steady state is reached
            )
            return (
                sol.ys,  # The time series values Node-Columns, Time-Rows
            )

        # Function to format time series solutions
        def _format_solution_chunk(time_series_array, round, ts, icprm_chunk):
            # Convert all elements to JAX arrays if they are not already, and stack along axis 1
            time_series_array = jnp.stack(
                map(jnp.array, time_series_array[0]), axis=2
            )  # Stack along time dimension (axis 2)
            # Saving the number of initial conditions-parameter combinations
            num_icprm = time_series_array.shape[0]
            # Concatenate the time seires values. (num_nodes, num_timepoints, num_icprm_combinations) along the num_icprm_combinations axis
            # This gives us the (num_nodes, num_timepoints*num_icprm_combinations) array
            time_series_array = jnp.concatenate(time_series_array, axis=0)
            # Creating a time array and stacking it with the time series values
            time_series_array = jnp.concatenate(
                [time_series_array, jnp.tile(jnp.array(ts), num_icprm).reshape(-1, 1)],
                axis=1,
            )
            # # Round the stacked array
            time_series_array = jnp.round(time_series_array, round)
            # Getting the initial condition and parameter numbers and concatenating them with the time series values
            time_series_array = jnp.column_stack(
                [
                    time_series_array,
                    jnp.repeat(inicond[icprm_chunk[:, 0], -1], len(ts)),
                    jnp.repeat(paramvals[icprm_chunk[:, 1], -1], len(ts)),
                ]
            )
            # Returning as a numpy array to save VRAM space
            return np.array(time_series_array)

    return solve_ode, icprm_comb, _format_solution_chunk


# Function to solve for a given replicate folder
def solve_replicate(
    ode_sys,
    repfl,
    topo_name,
    init_cond,
    param_vals,
    batch_size=100000,
    num_chunks=5,
    round=4,
    parequet=True,
    compress=None,
    solver=Tsit5(),
    t0=0,
    t1=250,
    dt0=0.1,
    ts=None,
    retol=1e-5,
    atol=1e-6,
    max_steps=None,
):
    print("Replicate Folder: ", repfl[:-1])
    # Get the initial conditions csv file
    # init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # Number of nodes
    num_nodes = len(init_cond.columns) - 1
    print(f"Number of nodes: {num_nodes}")
    # Get the parameter values csv file
    # param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    #  #Time the solving of the ODEs
    start = time.time()
    # Getting the solve ode function and the index combinations of paramters and initial conditions
    # Not Scan
    solve_ode_fn, icprm_comb, format_solchunk = parameterise_solveode(
        ode_sys,
        inicond=init_cond,
        paramvals=param_vals,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        ts=ts,
        retol=retol,
        atol=atol,
        max_steps=max_steps,
    )
    print(f"Number of actual combinations: {len(icprm_comb)}")
    # If number of chunks is greater than the number of combinations, terminate with an error
    if num_chunks > len(icprm_comb):
        raise ValueError(
            "Number of chunks is greater than the number of combinations. Reduce the number of chunks"
        )
    # Getting chunks of the icprm_comb
    icprm_chunks = jnp.array_split(icprm_comb, num_chunks)
    print(
        f"No. of chunks: {len(icprm_chunks)}\nNo. of combinations per chunk: {[i.shape[0] for i in icprm_chunks]}"
    )
    # Get the columns names of the initial conditions -> gives node names
    node_li = [i for i in init_cond.columns if "InitCondNum" not in i]
    # Saving the solutions
    sol_li = []
    # # Running only one parameter combination
    # chunk_sol_li = solve_ode_fn(icprm_chunks[0][0])
    # Runnning the solve_ode_fn by mapping over the icprm_chunks in for loop
    for i in range(len(icprm_chunks)):
        # Start time
        b_start = time.time()
        chunk_sol_li = lax.map(
            solve_ode_fn, icprm_chunks[i], batch_size=int(batch_size) - 1
        )
        # Start time for chunk formatting
        chunk_format_start = time.time()
        # Formatting the solution list
        chunk_sol_li = format_solchunk(
            chunk_sol_li, round=round, ts=ts, icprm_chunk=icprm_chunks[i]
        )
        print(f"Time taken to format chunk {i}: {time.time() - chunk_format_start}")
        # Appending the output to the solution list
        sol_li.append(chunk_sol_li)
        print(f"Time taken for chunk {i}: {time.time() - b_start}")
        # Clearing the cache
        clear_caches()
    # # Print the time taken to solve the ODEs
    print(f"Total time taken: {time.time() - start}")
    # If the simulation are steady states
    if ts is None:
        # Converting to a dataframe
        sol_li = pd.DataFrame(
            np.vstack(sol_li),
            columns=node_li + ["TimeEnd", "SteadyState", "InitCondNum", "ParamNum"],
        )
    else:
        sol_li = pd.DataFrame(
            np.vstack(sol_li),
            columns=node_li + ["TimePoint", "InitCondNum", "ParamNum"],
        )
    # Save the solution dataframe to the relevent replicate folder
    if parequet:
        print(f"Saving solution to parquet {compress} at {repfl}.\n")
        sol_li.to_parquet(
            f"{repfl}{topo_name}_sol_{repfl[:-1]}.parquet.gzip",
            compression=compress,
            index=False,
        )
    else:
        print(f"Saving solution to csv at {repfl}.\n")
        sol_li.to_csv(f"{repfl}{topo_name}_sol_{repfl[:-1]}.csv", index=False)
    # Return True on success
    return True


# Get the DIrectory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Change the directory to the script directory
os.chdir(script_dir)

# List all the folders with numeric names in the current folder
rep_folders = sorted(glob.glob("[0-9]/"))
print(rep_folders)

# Get the current folder name
topo_name = os.path.basename(os.getcwd())
print(topo_name)


# Loop through each of the replicate folders
for repfl in rep_folders[:1]:
    # # Get the initial conditions csv file
    init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # # Get the parameter values csv file
    param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    # # Subset 10 rows of the initial conditions and parameter values
    init_cond = init_cond[:1000]
    param_vals = param_vals[:10000]
    ## Solve the ODEs for the replicate folder
    solve_replicate(
        odesys_11,
        repfl,
        topo_name,
        init_cond=init_cond,
        param_vals=param_vals,
        round=4,
        num_chunks=2,
        # parequet=False,
        compress="gzip",
        solver=Tsit5(),
        ts=[1.0, 2.0, 5.0, 60.0, 150.0, 250.0],
    )
