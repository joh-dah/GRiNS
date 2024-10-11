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
from jax import lax, vmap, jit, clear_caches
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
            inicond[pi_row[0]][-1],
            paramvals[pi_row[1]][-1],
            sol.ts[-1],
        ]

    return solve_ode, icprm_comb


# Function to solve for a given replicate folder
def solve_replicate(
    repfl,
    topo_name,
    batch_size=100000,
    num_chunks=5,
    param_li=None,
    init_li=None,
    round=2,
    parequet=True,
    compress=None,
):
    print("Replicate Number: ", repfl[:-1])
    # Get the initial conditions csv file
    init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # Number of nodes
    num_nodes = len(init_cond.columns) - 1
    print(f"Number of nodes: {num_nodes}")
    # Get the parameter values csv file
    param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    # If the parameter list is provided, then subset the parameter values
    if param_li is not None:
        param_vals = param_vals[param_vals["ParaNum"].isin(param_li)]
    # If the initial condition list is provided, then subset the initial conditions
    if init_li is not None:
        init_cond = init_cond[init_cond["InitCondNum"].isin(init_li)]
    #  #Time the solving of the ODEs
    start = time.time()
    # Getting the solve ode function and the index combinations of paramters and initial conditions
    # Not Scan
    solve_ode_fn, icprm_comb = parameterise_solveode(
        odesys_11, inicond=init_cond, paramvals=param_vals
    )
    # Subset icprm_comb
    print(f"Number of actual combinations: {len(icprm_comb)}")
    # icprm_comb = icprm_comb[:100000000]
    # Getting chunks of the icprm_comb
    icprm_chunks = jnp.array_split(icprm_comb, num_chunks)
    # Stacking into a 3D array
    icprm_chunks = jnp.stack(icprm_chunks)
    print(icprm_chunks.shape)
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
        print(f"Time taken: {time.time() - b_start}")
        # Clear the cache
        clear_caches()
    # Print the time taken to solve the ODEs
    print(f"Time taken to solve the ODEs: {time.time() - start}")
    # Concatenate the solution list along the rows
    sol_li = np.concatenate(sol_li, axis=1).T
    # Get the columns names of the initial conditions -> gives node names
    node_li = [i for i in init_cond.columns if "InitCondNum" not in i]
    # Convert the solution list to a data frame
    # Save the solution dataframe to the relevent replicate folder
    if parequet:
        pd.DataFrame(
            np.round(sol_li, round),
            columns=node_li + ["InitCondNum", "ParaNum", "TimeEnd"],
        ).to_parquet(
            f"{repfl}{topo_name}_sol_{repfl[:-1]}.parquet.gzip",
            compression=compress,
            index=False,
        )
    else:
        pd.DataFrame(
            np.round(sol_li, round),
            columns=node_li + ["InitCondNum", "ParaNum", "TimeEnd"],
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
    # Solve the ODEs for the replicate folder
    solve_replicate(repfl, topo_name, round=4, parequet=True, compress="gzip")
    # print("Replicate Number: ", repfl[:-1])
    # # Get the initial conditions csv file
    # init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # # Number of nodes
    # num_nodes = len(init_cond.columns) - 1
    # print(f"Number of nodes: {num_nodes}")
    # # Get the parameter values csv file
    # param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    # #  #Time the solving of the ODEs
    # start = time.time()
    # # Getting the solve ode function and the index combinations of paramters and initial conditions
    # # Not Scan
    # solve_ode_fn, icprm_comb = parameterise_solveode(
    #     odesys_11, inicond=init_cond, paramvals=param_vals
    # )
    # # Subset icprm_comb
    # print(f"Number of actual combinations: {len(icprm_comb)}")
    # # icprm_comb = icprm_comb[:100000000]
    # # Getting chunks of the icprm_comb
    # icprm_chunks = jnp.array_split(icprm_comb, 10)
    # # Stacking into a 3D array
    # icprm_chunks = jnp.stack(icprm_chunks)
    # print(icprm_chunks.shape)
    # # print(len(icprm_chunks))
    # # print(icprm_chunks[0].shape)
    # print(f"Number of combinations: {len(icprm_comb)}")
    # # Specify batch size
    # batch_size = 10000 - 1
    # print(batch_size)
    # # Run the solve_ode function over the combinations array
    # # sol_li = lax.map(solve_ode_fn, icprm_comb, batch_size=batch_size - 1)
    # # sol_li = solve_all_odes(solve_ode_fn, icprm_comb)
    # # sol_li = vmap(solve_ode_fn)(icprm_chunks[0])
    # # Saving the solutions
    # sol_li = []
    # # Runnning the solve_ode_fn by mapping over the icprm_chunks in for loop
    # for i in range(len(icprm_chunks)):
    #     # Start time
    #     b_start = time.time()
    #     batch_sol_li = lax.map(solve_ode_fn, icprm_chunks[i], batch_size=batch_size)
    #     # Converting the output to a numpy array and appending to the solution list
    #     sol_li.append(np.array(batch_sol_li))
    #     print(f"Time taken: {time.time() - b_start}")
    #     # Clear the cache
    #     clear_caches()
    # # Print the time taken to solve the ODEs
    # print(f"Time taken to solve the ODEs: {time.time() - start}")
    # # Concatenate the solution list along the rows
    # sol_li = np.concatenate(sol_li, axis=1).T
    # # Get the columns names of the initial conditions -> gives node names
    # node_li = [i for i in init_cond.columns if "InitCondNum" not in i]
    # # Convert the solution list to a data frame
    # # Save the solution dataframe to the relevent replicate folder
    # pd.DataFrame(
    #     np.round(sol_li, 2),
    #     columns=node_li + ["InitCondNum", "ParaNum", "TimeEnd"],
    # ).to_parquet(
    #     f"{repfl}{topo_name}_sol_{repfl[:-1]}.parquet.gzip", compression="gzip"
    # )
    # # # Clear the cache of the solve_ode_fn
    # # solve_ode_fn._clear_cache()
