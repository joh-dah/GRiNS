from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, SteadyStateEvent

import jax.numpy as jnp
from jax import vmap
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


# ODE solver wrapper function
def solve_ode(term, solver, t0, t1, dt0, y0, args, init_cond_num, para_num):
    y0 = tuple(y0)
    args = tuple(args)
    sol = diffeqsolve(
        term,
        solver,
        t0,
        t1,
        dt0,
        y0,
        args,
        saveat=SaveAt(t1=True),
        stepsize_controller=PIDController(rtol=1e-5, atol=1e-6),
        max_steps=None,
        discrete_terminating_event=SteadyStateEvent(),
    )
    return [s[-1] for s in sol.ys] + [init_cond_num, para_num]


# Deine the ODESysytem as a ODETerm
term = ODETerm(odesys_11)

# List all the folders with numeric names in the current folder
rep_folders = sorted(glob.glob("[0-9]*/"))
print(rep_folders)

# Get the current folder name
topo_name = os.path.basename(os.getcwd())
print(topo_name)

# Get the node names form the paramters range csv file
# Filter only the rows of parameters with Prod_ in the name
# Then replace Prod with an empty string
node_li = pd.read_csv(f"{topo_name}_param_range.csv", sep=r"\s+")
node_li = list(
    node_li[node_li["Parameter"].str.contains("Prod_")]["Parameter"].str.replace(
        "Prod_", ""
    )
)

# Loop through each of the replicate folders
for repfl in rep_folders:
    print("Replicate Number: ", repfl[:-1])
    # Get the initial conditions csv file
    init_cond = pd.read_csv(f"{repfl}{topo_name}_init_conds_{repfl[:-1]}.csv")
    # Remove the InitCondNum column
    init_cond_nums = list(init_cond["InitCondNum"])
    # init_cond = init_cond.drop(columns="InitCondNum")
    # Get the parameter values csv file
    param_vals = pd.read_csv(f"{repfl}{topo_name}_params_{repfl[:-1]}.csv")
    # Remove ParaNum column
    param_vals_nums = list(param_vals["ParaNum"])
    # Merge the initial conditions and parameter values dataframes with combinations of inital conditions and parameter values
    comb_df = param_vals.merge(init_cond, how="cross")
    # Reorder the columns so that paramnum and initcondnum are the last two columns
    comb_df = comb_df[
        [col for col in comb_df if col not in ["ParaNum", "InitCondNum"]]
        + ["ParaNum", "InitCondNum"]
    ]
    # Convert the dataframe into a jax array
    comb_arr = jnp.array(comb_df)
    # Subset only the first 333400 rows
    comb_arr = comb_arr[:1000]
    # Time the solving of the ODEs
    start = time.time()
    # Run the solve_ode function over the combinations array
    sol_li = vmap(
        lambda i: solve_ode(
            term,
            Tsit5(),
            0,
            200,
            0.1,
            i[: len(node_li)],
            i[len(node_li) : -2],
            i[-1],
            i[-2],
        ),
    )(comb_arr)
    # Print the time taken to solve the ODEs
    print(f"Time taken to solve the ODEs: {time.time() - start}")
    # Convert the solution list to a numpy array
    sol_li = jnp.array(sol_li).T
    # Conver the solution list to a data frame
    sol_df = pd.DataFrame(
        sol_li,
        columns=node_li + ["InitCondNum", "ParaNum"],
    )
    print(sol_df)
