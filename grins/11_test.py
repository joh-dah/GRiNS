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
import subprocess


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
        t0 = ts[0]
        t1 = ts[-1]
        saveat = SaveAt(t1=True, ts=ts)
    stepsize_controller = PIDController(rtol=retol, atol=atol)
    # Convert the initial conditions and parameter values to a jax array
    inicond = jnp.array(inicond, dtype=jnp.float32)
    paramvals = jnp.array(paramvals, dtype=jnp.float32)
    # Generate the combinations of the indices of the initial conditions and parameter values
    icprm_comb = _gen_combinations(len(inicond), len(paramvals))

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
        return [s[-1] for s in sol.ys] + [
            inicond[pi_row[0]][-1],  # The initial condition number
            paramvals[pi_row[1]][-1],  # The parameter number
            sol.ts[-1],  # The end time
            jnp.astype(sol.event_mask, jnp.int32),  # The steady state event mask
        ]

    return solve_ode, icprm_comb


# Function to solve for a given replicate folder
def solve_replicate(
    ode_sys,
    repfl,
    topo_name,
    init_cond,
    param_vals,
    batch_size=100000,
    num_chunks=5,
    round=2,
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
    # Subset icprm_comb
    print(f"Number of actual combinations: {len(icprm_comb)}")
    # icprm_comb = icprm_comb[:100000000]
    # Getting chunks of the icprm_comb
    icprm_chunks = jnp.array_split(icprm_comb, num_chunks)
    # Stacking into a 3D array
    # Number of chunks should give same shaped arrays - will give error otherwise
    # Can be fixed if its used as a list instead of a jax 3d-array
    icprm_chunks = jnp.stack(icprm_chunks)
    print(
        f"No. of chunks: {icprm_chunks.shape[0]}\nNo. of combinations per chunk: {icprm_chunks.shape[1]}"
    )
    # Saving the solutions
    sol_li = []
    # Runnning the solve_ode_fn by mapping over the icprm_chunks in for loop
    for i in range(len(icprm_chunks)):
        # Start time
        b_start = time.time()
        batch_sol_li = lax.map(
            solve_ode_fn, icprm_chunks[i], batch_size=int(batch_size) - 1
        )
        # Converting the output to a numpy array and appending to the solution list
        sol_li.append(np.array(batch_sol_li))
        print(f"Time taken for chunk {i}: {time.time() - b_start}")
        # Clear the cache
        clear_caches()
    # Print the time taken to solve the ODEs
    print(f"Total time taken: {time.time() - start}")
    # Concatenate the solution list along the rows
    sol_li = np.concatenate(sol_li, axis=1).T
    # Get the columns names of the initial conditions -> gives node names
    node_li = [i for i in init_cond.columns if "InitCondNum" not in i]
    # Convert the solution list to a data frame
    sol_li = pd.DataFrame(
        np.round(sol_li, round),
        columns=node_li + ["InitCondNum", "ParaNum", "TimeEnd", "SteadyState"],
    )
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
        pd.DataFrame(
            np.round(sol_li, round),
            columns=node_li + ["InitCondNum", "ParaNum", "TimeEnd", "SteadyState"],
        ).to_csv(f"{repfl}{topo_name}_sol_{repfl[:-1]}.csv", index=False)
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
for repfl in rep_folders:
    # # Get the initial conditions csv file
    init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # # Get the parameter values csv file
    param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    # # Subset 10 rows of the initial conditions and parameter values
    # init_cond = init_cond[:10]
    # param_vals = param_vals[:10]
    # Solve the ODEs for the replicate folder
    solve_replicate(
        odesys_11,
        repfl,
        topo_name,
        init_cond=init_cond,
        param_vals=param_vals,
        round=4,
        num_chunks=4,
        parequet=True,
        compress="gzip",
        solver=Tsit5(),
    )
