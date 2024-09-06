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
from jax import lax
import pandas as pd
import glob
import os
import time


# Positive Shifted Hill function
def psH(nod, fld, thr, hill):
    return (fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))) / fld


# Negative Shifted Hill function
def nsH(nod, fld, thr, hill):
    return fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))


# ODE system function for diffrax
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
    inicond = jnp.array(inicond)
    paramvals = jnp.array(paramvals)
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
        ]

    return solve_ode, icprm_comb


# List all the folders with numeric names in the current folder
rep_folders = sorted(glob.glob("[0-9]*/"))
print(rep_folders)

# Get the current folder name
topo_name = os.path.basename(os.getcwd())
print(topo_name)

# Loop through each of the replicate folders
for repfl in rep_folders:
    print("Replicate Number: ", repfl[:-1])
    # Get the initial conditions csv file
    init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # Get the parameter values csv file
    param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    #  #Time the solving of the ODEs
    start = time.time()
    # Getting the solve ode function and the index combinations of paramters and initial conditions
    solve_ode_closure, icprm_comb = parameterise_solveode(
        odesys_11, inicond=init_cond, paramvals=param_vals
    )
    # print(solve_ode_closure)
    # print(icprm_comb)
    # Specify batch size
    batch_size = int(len(icprm_comb) * 0.1 - 1)
    # Checking to solve_ode for a single combination
    # print(solve_ode_closure(icprm_comb[0]))
    # Run the solve_ode function over the combinations array
    sol_li = lax.map(solve_ode_closure, icprm_comb, batch_size=batch_size)
    # sol_li = vmap(solve_ode_closure)(comb_arr)
    # Print the time taken to solve the ODEs
    print(f"Time taken to solve the ODEs: {time.time() - start}")
    # Convert the solution list to a numpy array
    sol_li = jnp.array(sol_li).T
    # Get the columns names of the initial conditions -> gives node names
    node_li = [i for i in init_cond.columns if "InitCondNum" not in i]
    # Conver the solution list to a data frame
    sol_df = pd.DataFrame(
        sol_li,
        columns=node_li + ["InitCondNum", "ParaNum"],
    )
    print(sol_df)
