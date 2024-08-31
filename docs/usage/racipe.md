# RACIPE

Simulates ODEs for a given network topology for over different parameter sets and initial conditions and outputs the steady state values.

## Usage
The `racipe` command can be used with the following options:

### Default usage:
Generate parameters and initial conditions by randomly sampling from sobol sequences within the default range. Solve ODEs for each parameter set and initial condition, and output the steady state values.

```bash
racipe --topodir <topo_dir> --outdir <output_dir> --num_params <num_params> --num_inits <num_inits> --num_reps <num_reps> --num_cores <num_cores>
```
**Parameters:**
    - `topodir`: Directory containing the network topology files. Default is `./TOPOS`.
    - `outdir`: Directory to save the output files. Default is `./SimResults`.
    - `num_params`: Number of parameter sets to generate. Default is 10000.
    - `num_inits`: Number of initial conditions to generate. Default is 1000.
    - `num_reps`: Number of repetitions for each parameter set and initial condition. Default is 3.
    - `num_cores`: Number of cores to use for parallel processing. Default is all available cores. (Note: This option does nothing when a GPU is present and grins[cuda12] version is installed.)
