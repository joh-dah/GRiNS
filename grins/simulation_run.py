from glob import glob
import time
import numpy as np
import os

# from generate_odesys import gen_julia_odesys
from gen_diffrax_ode import gen_diffrax_odesys
from generate_params import (
    _gen_sobol_seq,
    parse_topos,
    gen_param_names,
    gen_param_df,
    get_param_range_df,
)
import subprocess
from multiprocessing import Pool  # noqa: F401
import pandas as pd  # noqa: F401
from sys import stdout
from topoanalyser import (
    # create_groups,
    # calc_simple_pthdf,
    # get_coherence_matrix,
    # convert_topodf_netx,
    get_cohmat_groupcomp,
    convert_topo_netx,
)  # noqa: F401


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


# Function to read the teams composition file
def read_teams_comp_file(teams_comp_path):
    # Read the teams composition file
    with open(teams_comp_path, "r") as f:
        teams_comp = f.readlines()
    # Remove the newline character from the end of each line
    teams_comp = [t.rstrip("\n") for t in teams_comp]
    # SPlit each line at : to get the team name and the members
    teams_comp = [t.split(":") for t in teams_comp]
    # Create a dictionary to store the team composition
    teams_comp_dict = {}
    # Loop through the teams composition and add to the dictionary
    [teams_comp_dict.update({t[0]: t[1].strip().split(",")}) for t in teams_comp]
    return teams_comp_dict


# Functiont to generate all the parameters related files with replicates
def gen_topo_param_files(
    topo_file,
    save_dir=".",
    num_replicates=3,
    num_params=2**10,
    num_init_conds=2**7,
    team_init_req=False,
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
    # print(param_range_df)
    if not team_init_req:
        # Generate the parameter dataframe and save in each of the replicate folders
        for rep in range(1, num_replicates + 1):
            print(f"Replicate {rep}")
            param_df = gen_param_df(param_range_df, num_params)
            # Add a column for the parameter number
            param_df["ParaNum"] = param_df.index + 1
            # print(param_df)
            param_df.to_csv(
                f"{sim_dir}/{rep}/{topo_name}_params_{rep}.csv", index=False
            )
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
    else:
        # Get the team composition
        teams_comp_dict = read_teams_comp_file(
            f"CohResults/TeamComp/{topo_name}_teamcomp.txt"
        )
        # print(teams_comp_dict)
        for rep in range(1, 4):
            param_df = gen_param_df(param_range_df, num_params)
            # Add a column for the parameter number
            param_df["ParaNum"] = param_df.index + 1
            param_df.to_csv(
                f"{sim_dir}/{rep}/{topo_name}_params_{rep}.csv", index=False
            )
            # Create a dataframe filled with zeros
            initial_conds = pd.DataFrame(
                np.zeros((num_init_conds, len(unique_nodes))), columns=unique_nodes
            )
            # Iterate thorugh chunks of the dataframe and assign the initial conditions
            for chunk, team in zip(
                np.array_split(initial_conds.index, len(teams_comp_dict)),
                teams_comp_dict,
            ):
                # Generate the sobol sequence for the initial conditions of the team
                team_init_conds = _gen_sobol_seq(len(teams_comp_dict[team]), len(chunk))
                # Scale the initial conditions between 1 to 100
                team_init_conds = 1 + team_init_conds * (100 - 1)
                # Access the inital conditions dataframe and assign the team initial conditions
                initial_conds.loc[chunk, teams_comp_dict[team]] = team_init_conds
                # Add a column for the initial condition number
                initial_conds["InitCondNum"] = initial_conds.index + 1
                initial_conds.to_csv(
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


# Function to run the SimODESys command on the sim directories in parallel
def run_sim_odesys(sim_directories, numCores=0, sys_image_path=""):
    """
    Run the SimODESys command on the sim directories in parallel.

    Parameters:
    - sim_directories (list): List of replicate directory paths.
    - sim_type (str, optional): Type of simulation to run. Default is "init".
    - numCores (int, optional): Number of CPU cores to use for parallel generation. Default is 0, which uses all available cores - 2.

    Returns:
    - None
    """
    # If numCores is set to 0 (Default), use all the available cores - 2
    if numCores == 0:
        numCores = np.floor(os.cpu_count()) - 2
    print(f"Number of cores used for simluation: {numCores}")
    # # Print the command to be run
    # for sim_dir in sim_directories:
    #     print(f"julia -J {sys_image_path} -p {int(numCores)} {sim_dir}")
    if sys_image_path:
        # Call the julia fine in each topo file result directory to simulate the replicates
        for sim_dir in sim_directories:
            # Track the time taken to run the simulation and divide by 3 and then print
            # sys.stdout.write("##############################################\n")
            start_time = time.time()
            # Running the simualtion
            cmd = " ".join(
                [
                    "julia",
                    "-J " + sys_image_path,
                    "-p " + str(int(numCores)),
                    sim_dir,
                    "& wait",
                ]
            )
            subprocess.run(cmd, shell=True)
            # Get the time taken to run the simulation
            stdout.write(
                f"{sim_dir.split('/')[-1].split('.')[0]} Time taken: {(time.time() - start_time)/3}s per replicate\n"
            )
            # sys.stdout.write("##############################################\n")
    else:
        # Call the julia fine in each topo file result directory to simulate the replicates
        for sim_dir in sim_directories:
            # Track the time taken to run the simulation and divide by 3 and then print
            # sys.stdout.write("##############################################\n")
            start_time = time.time()
            # Running the simualtion
            cmd = " ".join(["julia", "-p " + str(int(numCores)), sim_dir, "& wait"])
            subprocess.run(cmd, shell=True)
            # Get the time taken to run the simulation
            stdout.write(
                f"Time taken to run the simulation: {(time.time() - start_time)/3}s per replicate\n"
            )
            # sys.stdout.write("##############################################\n")
    return None


if __name__ == "__main__":
    # Specify the number of cores to use
    numCores = 10
    print(f"Number of cores: {numCores}")
    # Specify the directory to work in
    # Create the directoreis to store pathdf, cohmats and team compositions
    if not os.path.exists("CohResults/PathDF"):
        os.makedirs("CohResults/PathDF")
    if not os.path.exists("CohResults/CohMat"):
        os.makedirs("CohResults/CohMat")
    if not os.path.exists("CohResults/TeamComp"):
        os.makedirs("CohResults/TeamComp")
    # Topo file directory
    topo_dir = "TOPOS"
    # Specify the root folder where the generated parameter files and then the simulation files will be saved
    sim_save_dir = "SimResults"
    # Make the directories to store the results
    os.makedirs(sim_save_dir, exist_ok=True)
    # Get the list of all the topo files
    topo_files = sorted(glob(f"{topo_dir}/*.topo"))
    print(f"Number of topo files: {len(topo_files)}")
    # Specify the number of replicates required
    num_replicates = 1
    # Specify if single replicate
    single_rep = True
    # Specify the number of parameters required
    num_params = 10000
    # Specify the number of initial conditions required
    num_init_conds = 100
    # Modify the num_params and num_init_conds
    if single_rep:
        param_remineder = num_replicates - (num_params % num_replicates)
        num_params = int((num_params + param_remineder) / num_replicates)
    # Print the number of replicates, parameters and initial conditions
    print(f"Number of replicates: {num_replicates}")
    print(f"Number of parameters: {num_params}")
    print(f"Number of initial conditions: {num_init_conds}\n")
    # # Generate the parameter files and directory structure
    # If numCores is set to 0 (Default), use all the available cores - 2
    if numCores == 0:
        for topo_file in topo_files:
            gen_topo_param_files(
                topo_file,
                sim_save_dir,
                # sim_ode_dir,
                num_params,
                num_init_conds,
                team_init_req=False,
            )
    else:
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
                    False,
                )
                for topo_file in topo_files[:1]
            ],
        )
        # Close the pool of workers
        pool.close()
    # # Get the directories in which to call SimODESys
    # sim_dir_list = get_simfile_directories(topo_files, sim_save_dir)
    # # Run the SimODESys for each replicate directory in sim_directories
    # run_sim_odesys(sim_dir_list, numCores=numCores, sys_image_path=sys_image_path)
