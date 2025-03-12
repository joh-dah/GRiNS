# Ising Boolean GRN Simulation Tutorial

## Overview

This tutorial explains the pipeline for running Boolean simulations on gene regulatory networks (GRNs). The script processes topology files and runs simulations using different modes (synchronous and asynchronous).

## Step 1: Define the Topology File Directory

We first specify the directory containing the topology files. These files define the network structure and interactions between genes.

```python
# Specify the path to the topo file
topo_folder = "TOPOS"
```

## Step 2: Retrieve Topology Files

We retrieve all topology files from the specified directory using the `glob` module. The files are sorted to ensure a consistent processing order - this sorting is not necessary.

```python
# Get the list of all the topo files
topo_files = sorted(glob.glob(f"{topo_folder}/*.topo"))
print(topo_files)
```

## Step 3: Define Simulation Parameters

Before running the simulations, we specify key parameters that control the simulation behavior:

- **Replacement Values**: The Boolean states (0 and 1) used in the simulation. Can also be (-1 and 1).
- **Number of Steps**: The total number of steps to simulate.
- **Number of Initial Conditions**: The number of different initial conditions to explore.
- **Batch Size**: The batch size for the simualtions, depends on the VRAM of the GPU, the function uses 'vmap' of jax to run each batch.

```python
# Specify the replacement values
replacement_values = jnp.array([0, 1])

# Specify the number of steps to simulate
max_steps = 100
print(f"Number of steps: {max_steps}")

# Specify the number of initial conditions to simulate
num_initial_conditions = 2**14
print(f"Number of initial conditions: {num_initial_conditions}")

# Specify the batch size for parallel evaluation
batch_size = 2**10
```

## Step 4: Process Topology Files and Run Simulations

We loop through each topology file, extract the adjacency matrix, and run simulations in both synchronous and asynchronous modes.

```python
# Loop over all the topo files
for topo_file in topo_files:
    # Get the adjacency matrix and node names
    topo_adj, node_names = parse_topo_to_matrix(topo_file)
    print(f"Topology: {topo_file}")
    
    # Run the simulation in synchronous mode
    run_simulations(
        topo_file=topo_file,
        num_initial_conditions=num_initial_conditions,
        batch_size=batch_size,
        replacement_values=replacement_values,
        mode="sync",
        packbits=True,
    )
    
    # Run the simulation in asynchronous mode
    run_simulations(
        topo_file=topo_file,
        num_initial_conditions=num_initial_conditions,
        batch_size=batch_size,
        replacement_values=replacement_values,
        mode="async",
        packbits=True,
    )
```

The simulation results will be stored in the output directory, with each topology file having its own dedicated folder within the simulation directory. This folder will contain the results saved as a Parquet file named '<topo_name>_<simulation_mode>_ising_results.parquet'. For further customization options, refer to the 'run_simulations' function.
