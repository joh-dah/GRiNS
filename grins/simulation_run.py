from glob import glob
import time
import numpy as np
import os
from gen_diffrax_ode import gen_diffrax_odesys
from generate_params_old import (
    _gen_sobol_seq,
    parse_topos,
    gen_param_names,
    gen_param_df,
    get_param_range_df,
)
import subprocess
from multiprocessing import Pool  # noqa: F401
import pandas as pd  # noqa: F401


# Function to generate the required directory structure
def gen_sim_dirstruct(topo_file, save_dir="."):
    """
    Generate directory structure for simulation run.

    Args:
        topo_file (str): Path to the topo file.
        save_dir (str, optional): Directory to save the generated structure. Defaults to ".".
    Returns:
        Directory structure is created with the topo file name and three folders for the replicates.
    """
    # Get the topo file name
    topo_name = topo_file.split("/")[-1].split(".")[0]
    # Check if the folder with the name of topo file exists
    if not os.path.exists(f"{save_dir}/{topo_name}"):
        os.makedirs(f"{save_dir}/{topo_name}")
    # Move the topo file to the created folder
    subprocess.run(
        [
            "cp",
            topo_file,
            f"{save_dir}/{topo_name}/{topo_file.split('/')[-1]}",
        ]
    )
    # Make directories for the replicates
    subprocess.run(
        [
            "mkdir",
            "-p",
            f"{save_dir}/{topo_name}/1",
            f"{save_dir}/{topo_name}/2",
            f"{save_dir}/{topo_name}/3",
        ]
    )
    return None


# Functiont to generate all the parameters related files with replicates
def gen_topo_param_files(
    topo_file,
    save_dir=".",
    num_replicates=3,
    num_params=2**10,
    num_init_conds=2**7,
):
    """
    Generate parameter files for simulation.

    Args:
        topo_file (str): The path to the topo file.
        save_dir (str, optional): The directory where the parameter files will be saved. Defaults to ".".
        num_params (int, optional): The number of parameter files to generate. Defaults to 2**10.
    Returns:
        The parameter files and initial conditions are generated and saved in the specified replicate directories.
    """
    # Get the name of the topo file
    topo_name = topo_file.split("/")[-1].split(".")[0]
    # Parse the topo file
    topo_df = parse_topos(topo_file)
    # Generate the parameter names
    param_names = gen_param_names(topo_df)
    # Get the unique nodes in the topo file
    unique_nodes = sorted(set(param_names[1] + param_names[2]))
    # Generate the required directory structure
    gen_sim_dirstruct(topo_file, save_dir)
    # Specify directory where all the generated julia ode system file will be saved
    sim_dir = f"{save_dir}/{topo_file.split('/')[-1].split('.')[0]}"
    # Generate the ODE system for diffrax
    gen_diffrax_odesys(topo_df, topo_name, sim_dir)
    # Generate the parameter range dataframe
    param_range_df = get_param_range_df(topo_df, num_params)
    # Save the parameter range dataframe
    param_range_df.to_csv(
        f"{sim_dir}/{topo_name}_param_range.csv", index=False, sep="\t"
    )
    # Generate the parameter dataframe and save in each of the replicate folders
    for rep in range(1, num_replicates + 1):
        # print(f"Replicate {rep}")
        param_df = gen_param_df(param_range_df, num_params)
        # Add a column for the parameter number
        param_df["ParaNum"] = param_df.index + 1
        # print(param_df)
        param_df.to_csv(f"{sim_dir}/{rep}/{topo_name}_params_{rep}.csv", index=False)
        # Generate the sobol sequence for the initial conditions
        initial_conds = _gen_sobol_seq(len(unique_nodes), num_init_conds)
        # Scale the initial conditions between 1 to 100
        initial_conds = 1 + initial_conds * (100 - 1)
        # Convert the initial conditions to a dataframe and save in the replicate folders
        initcond_df = pd.DataFrame(initial_conds, columns=unique_nodes)
        # Add a column for the initial condition number
        initcond_df["InitCondNum"] = initcond_df.index + 1
        initcond_df.to_csv(
            f"{sim_dir}/{rep}/{topo_name}_init_conds_{rep}.csv", index=False
        )
    return None


# Function to get the directories in which to call SimODESys
def get_simfile_directories(topo_files, save_dir):
    """
    Get the directories in which to call SimODESys for each topo file.

    Parameters:
    - topo_files (list): List of topo file paths.
    - save_dir (str): Directory path where the topo directory are created.

    Returns:
    - sim_directories (list): List of replicate directory paths.
    """
    # List to store the directories in which to call SimODESys (the replicate directories)
    # As SimODESys can look at the parent directory, it can automtically find the julia ode file
    # It also looks for the parameter files and initial conditions in the replicate directory
    sim_directories = []
    # For each topo file get the replicate directory path for running the SimODESys in
    for topo_file in topo_files:
        # Get the name of the topo file
        topo_name = topo_file.split("/")[-1].split(".")[0]
        # Get the paths of the
        topo_result_dir = [os.path.abspath(f"{save_dir}/{topo_name}/{topo_name}.jl")]
        # Append the replicate directories to the sim_directories list
        sim_directories.extend(topo_result_dir)

    return sim_directories


if __name__ == "__main__":
    # Specify the number of cores to use
    numCores = 10
    print(f"Number of cores: {numCores}")
    # Topo file directory
    topo_dir = "../TOPOS"
    # Specify the root folder where the generated parameter files and then the simulation files will be saved
    sim_save_dir = "SimResults"
    # Make the directories to store the results
    os.makedirs(sim_save_dir, exist_ok=True)
    # Get the list of all the topo files
    topo_files = sorted(glob(f"{topo_dir}/*.topo"))
    print(f"Number of topo files: {len(topo_files)}")
    # Specify the number of replicates required
    num_replicates = 3
    # Specify the number of parameters required
    num_params = 100000
    # Specify the number of initial conditions required
    num_init_conds = 1000
    # Print the number of replicates, parameters and initial conditions
    print(f"Number of replicates: {num_replicates}")
    print(f"Number of parameters: {num_params}")
    print(f"Number of initial conditions: {num_init_conds}\n")
    # Start the pool of worker processes
    pool = Pool(int(numCores))
    # Parllelise the generation of the parameter and inital condition files
    pool.starmap(
        gen_topo_param_files,
        [
            (
                topo_file,
                sim_save_dir,
                # sim_ode_dir,
                num_replicates,
                num_params,
                num_init_conds,
            )
            for topo_file in topo_files
        ],
    )
    # Close the pool of workers
    pool.close()
