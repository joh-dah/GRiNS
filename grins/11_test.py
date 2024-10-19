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
from jax import lax, clear_caches
import pandas as pd
import glob
import os
import time
from multiprocessing import Pool


# Positive Shifted Hill function
def psH(nod, fld, thr, hill):
    return (fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))) / fld


# Negative Shifted Hill function
def nsH(nod, fld, thr, hill):
    return fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))


# ODE system function for diffrax
# @jit
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


# Function to generate the combinations of the initial conditions and parameter values
def _gen_combinations(num_init_conds, num_params):
    # Generate the combinations of the initial conditions and parameter values
    i, p = jnp.meshgrid(
        jnp.arange(num_init_conds), jnp.arange(num_params), indexing="ij"
    )
    icprm_comb = jnp.vstack([i.flatten(), p.flatten()]).T
    return icprm_comb


# Function which takes in the ode solution and then returns the formatted version of the solution
def _format_sol(sol, init_cond_num, param_num):
    sol_vals = sol.ys
    # Convert sol_vals into a 2d array
    sol_vals = jnp.array(sol_vals).T
    # Get the time point values
    time_points = sol.ts
    # Add the time points to the sol_vals as a column
    sol_vals = jnp.hstack((sol_vals, jnp.array(time_points).reshape(-1, 1)))
    # Return the solution list of list, the initial condition number, the parameter nuumber and steady state event mask
    if len(sol_vals) == 1:
        return [
            sol_vals,  # An array of the steady state values
            init_cond_num,  # The initial condition number
            param_num,  # The parameter number
            jnp.astype(sol.event_mask, jnp.int32),  # The steady state event mask
        ]
    else:
        return (
            sol_vals,  # An array of the time series values
            init_cond_num,
            param_num,  # The initial condition number
        )


# A closure to parameterise all the non initial and paramter values in the ODE system
def parameterise_solveode(
    odesys,
    inicond,
    paramvals,
    solver=Tsit5(),
    t0=0,
    t1=200,
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
        # Make saaveat to be steps
        saveat = SaveAt(ts=ts)
    stepsize_controller = PIDController(rtol=retol, atol=atol)
    # Convert the initial conditions and parameter values to a jax array
    inicond = jnp.array(inicond, dtype=jnp.float32)
    paramvals = jnp.array(paramvals, dtype=jnp.float32)
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
            return _format_sol(sol, inicond[pi_row[0]][-1], paramvals[pi_row[1]][-1])
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
            return _format_sol(sol, inicond[pi_row[0]][-1], paramvals[pi_row[1]][-1])

    return solve_ode, icprm_comb


# Function to format the solution list into a dataframe
def _format_solchunk(sol_li, node_li, round):
    # Get the length of the node values list to get the number of time points
    num_time_points = len(sol_li[0][0])
    # If the number of time points is 1, then it is a steady state simulation
    if num_time_points == 1:
        # Creating the dataframe with the node steady state values
        sol_df = pd.DataFrame(np.concatenate(sol_li[0]), columns=node_li + ["TimeEnd"])
        # Add the initial condition number and parameter number columns
        sol_df["InitCondNum"] = sol_li[1]
        sol_df["ParaNum"] = sol_li[2]
        sol_df["SteadyState"] = sol_li[3]
    else:
        # Creating the initial condition list
        init_cond_li = np.concatenate(
            [[i] * num_time_points for i in sol_li[1].tolist()]
        )
        # Creating the parameter number list
        param_num_li = np.concatenate(
            [[i] * num_time_points for i in sol_li[2].tolist()]
        )
        # Concatenating the node values list and converting to a dataframe
        sol_df = pd.DataFrame(np.concatenate(sol_li[0]), columns=node_li + ["Time"])
        # Adding the initial condition number and parameter number columns
        sol_df["InitCondNum"] = init_cond_li
        sol_df["ParaNum"] = param_num_li
        # Convert the dataframe into a multi-index dataframe
        sol_df = sol_df.set_index(["ParaNum", "InitCondNum", "Time"])
    # Round values of the node values to number of decimal places
    sol_df[node_li] = sol_df[node_li].astype(float).round(round)
    return sol_df


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
    t1=200,
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
    solve_ode_fn, icprm_comb = parameterise_solveode(
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
    # Stacking into a 3D array
    # Number of chunks should give same shaped arrays - will give error otherwise
    # Can be fixed if its used as a list instead of a jax 3d-array
    icprm_chunks = jnp.stack(icprm_chunks)
    print(
        f"No. of chunks: {icprm_chunks.shape[0]}\nNo. of combinations per chunk: {icprm_chunks.shape[1]}"
    )
    # Get the columns names of the initial conditions -> gives node names
    node_li = [i for i in init_cond.columns if "InitCondNum" not in i]
    # Saving the solutions
    sol_li = []
    # Runnning the solve_ode_fn by mapping over the icprm_chunks in for loop
    for i in range(len(icprm_chunks)):
        # Start time
        b_start = time.time()
        chunk_sol_li = lax.map(
            solve_ode_fn, icprm_chunks[i], batch_size=int(batch_size) - 1
        )
        # Formatting the solution list
        chunk_sol_li = _format_solchunk(chunk_sol_li, node_li, round=round)
        # Appending the output to the solution list
        sol_li.append(chunk_sol_li)
        print(f"Time taken for chunk {i}: {time.time() - b_start}")
        # Clear the cache
        clear_caches()
    # Print the time taken to solve the ODEs
    print(f"Total time taken: {time.time() - start}")
    # Concatenate the list of dataframes into a single dataframe
    sol_li = pd.concat(sol_li)
    print(sol_li)
    # Save the solution dataframe to the relevent replicate folder
    if parequet:
        print(f"Saving solution to parquet ({compress}) at {repfl}.\n")
        sol_li.to_parquet(
            f"{repfl}{topo_name}_sol_{repfl[:-1]}.parquet.gzip",
            compression=compress,
            index=False,
        )
    else:
        print(f"Saving solution to csv at {repfl}.\n")
        sol_li.to_csv(f"{repfl}{topo_name}_sol_{repfl[:-1]}.csv", index=False)
    ## Return True on success
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
for repfl in rep_folders:
    # # Get the initial conditions csv file
    init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # # Get the parameter values csv file
    param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    # Subset 10 rows of the initial conditions and parameter values
    init_cond = init_cond[:100]
    param_vals = param_vals[:100]
    # Solve the ODEs for the replicate folder
    solve_replicate(
        odesys_11,
        repfl,
        topo_name,
        init_cond=init_cond,
        param_vals=param_vals,
        round=4,
        num_chunks=1,
        parequet=False,
        # compress="gzip",
        solver=Tsit5(),
        ts=[1.0, 2.0, 5.0, 60.0, 150.0, 250.0],
    )
