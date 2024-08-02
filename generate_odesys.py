# import pandas as pd
# import numpy as np
from glob import glob
from generate_params import parse_topos, gen_param_names
# import subprocess


# Internal function to generate a hills equation string of the edge
def _gen_edge_hills(edge):
    """
    Generate the edge hills function based on the edge type.

    Args:
        edge (dict): The edge dictionary (Series) containing the source, target, and type information.

    Returns:
        str: The generated edge hills function.

    """
    sn, tn, ia = edge["Source"], edge["Target"], edge["Type"]
    if ia == 1:
        # print(f"psHill(ActFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})")
        return f"psH({sn}, ActFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})"
    else:
        # print(f"nsHill(InhFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})")
        return f"nsH({sn}, InhFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})"


# Function to take in the target rwos and generate the ODE for a node
def gen_node_ode(target_edges, node_name):
    """
    Generate the ordinary differential equation (ODE) for a given node.

    Parameters:
    - target_edges: DataFrame containing the target edges of the node.
    - node_name: Name of the node.

    Returns:
    - ODE string representing the regulation terms for the node.

    """
    # Check if the target_edges is empty
    if not target_edges.empty:
        # Apply gen_edge_hills to each row and convert the values to a string joined by *
        return f"Prod_{node_name}*{'*'.join(target_edges.apply(_gen_edge_hills, axis=1))} - Deg_{node_name}*{node_name}"
    else:
        # Only Production term - degradation term
        return f"Prod_{node_name} - Deg_{node_name}*{node_name}"


# Function to the generate the ODE file in julia from a topo file
def gen_julia_odesys(topo_df, topo_name, save_dir=".", simodenet_dir="SimulateODENet"):
    """
    Generate the Julia ODE system code based on the given topology dataframe.

    Args:
        topo_df (pandas.DataFrame): The topology dataframe containing the edges information.
        topo_name (str): The name of the topology.
        save_dir (str, optional): The directory to save the generated code. Defaults to ".".

    Returns:
        None:  Saves the generated file in the driectory specified by save_dir.
    """
    # Get the list of parameters, target nodes and source nodes
    param_names_list, target_nodes, source_nodes = gen_param_names(topo_df)
    # List of unique nodes
    unique_nodes = sorted(set(target_nodes + source_nodes))
    # Inititalise a list to store the ODE strings
    ode_list = []
    # Loop through the target nodes
    for ni, nod in enumerate(unique_nodes):
        # Get the edges where n is the target node
        target_edges = topo_df[topo_df["Target"] == nod]
        # Plus 1 for index as julia starts index from 1
        ode_list.append("\t\t" + f"du[{ni+1}] = {gen_node_ode(target_edges, nod)}")
    # print("\n")
    # Define a list to hold the lines of the ODE system
    ode_file_lines = []
    # Add line to import the Distributed package
    ode_file_lines.append("using Distributed\n")
    # Add line to change directory to the current directory
    ode_file_lines.append("cd(@__DIR__)\n")
    # Add line to push the directory of the SimulateODENet to the LOAD_PATH
    ode_file_lines.append(f'@everywhere push!(LOAD_PATH, "{simodenet_dir}")\n')
    # Add line to use SimulateODENet
    ode_file_lines.append("@everywhere using SimulateODENets\n")
    # Add a line to start the everywhere block
    ode_file_lines.append("@everywhere begin\n")
    # Add the lines defining the postive hills functions
    ode_file_lines.append("\t# Positive Shifted Hill function")
    ode_file_lines.append(
        "\tfunction psH(nod, fld, thr, hill)\n\t\treturn (fld + (1 - fld) * (1 / (1 + ((nod/ thr) ^ hill))))/fld\n\tend\n"
    )
    # Add the lines defining the negetive hills functions
    ode_file_lines.append("\t# Negative Shifted Hill function")
    ode_file_lines.append(
        "\tfunction nsH(nod, fld, thr, hill)\n\t\treturn (fld + (1 - fld) * (1 / (1 + ((nod/ thr) ^ hill))))\n\tend\n"
    )
    # Append the ode systems related lines
    # Append the function definition line
    ode_file_lines.append("\t# ODE system defined in ModelingToolkit.jl macro")
    ode_file_lines.append(
        f"\tfunction odesys{topo_name.split('/')[-1].split('.')[0]}!(du, u, p, t)"
    )
    # Append the nodde name line
    ode_file_lines.append("\t\t" + ",".join(unique_nodes) + " = u")
    # Append the parameter names line
    ode_file_lines.append("\t\t" + ",".join(param_names_list) + " = p")
    # Append the ODE lines
    ode_file_lines.append("\n".join(ode_list))
    # Append the end line
    ode_file_lines.append("\tend\n")
    # End the everywhere block
    ode_file_lines.append("end\n")
    # Add the line to call the run_sim_replicate function
    ode_file_lines.append(
        f"run_sim_replicates(odesys{topo_name.split('/')[-1].split('.')[0]}!)\n"
    )
    # print("\n".join(ode_file_lines))
    # Write the lines to a file
    with open(f"{save_dir}/{topo_name.split('/')[-1].split('.')[0]}.jl", "w") as f:
        f.write("\n".join(ode_file_lines))


if __name__ == "__main__":
    # Specify the topo folder
    topo_folder = "TOPOS"
    # Find all the topo files in the topo folder
    topo_list = sorted(glob(f"{topo_folder}/*.topo"))
    # Loop thorugh the topo files
    for t in topo_list[:1]:
        # Parse the topo file
        topo_df = parse_topos(t)
        print(topo_df)
        # Generate the julia ODE system file
        gen_julia_odesys(topo_df, t)
        # Run the julia ODE system file to check for errors
        # subprocess.run(["julia", f"{t.split('/')[-1].split('.')[0]}.jl"])
