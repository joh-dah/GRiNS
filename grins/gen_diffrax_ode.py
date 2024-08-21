from diffrax import diffeqsolve, ODETerm, Dopri5  # type: ignore  # noqa: F401
import jax.numpy as jnp  # type: ignore  # noqa: F401
import pandas as pd  # type: ignore  # noqa: F401
from glob import glob
from grins.generate_params import parse_topos, gen_param_names  # noqa: F401
import os


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
        return f"rf.psH({sn}, ActFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})"
    else:
        # print(f"nsHill(InhFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})")
        return f"rf.nsH({sn}, InhFld_{sn}_{tn}, Thr_{sn}_{tn}, Hill_{sn}_{tn})"


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
def gen_diffrax_odesys(topo_df, topo_name, save_dir="."):
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
    ode_list = [
        f"import grins.reg_funcs as rf",
        f"def odesys(t,y,args):",
        f"\t({', '.join(unique_nodes)}) = y",
        f"\t({', '.join(param_names_list)}) = args",
    ]
    # Loop through the target nodes
    for ni, nod in enumerate(unique_nodes):
        # Get the edges where n is the target node
        target_edges = topo_df[topo_df["Target"] == nod]
        # The diffrax ODE for each node is d_<nod> = <ODE>
        ode_list.append("\t" + f"d_{nod} = {gen_node_ode(target_edges, nod)}")
    # Append the d_y line
    ode_list.append(
        f"\td_y = ({', '.join([f'd_{nod}' for nod in unique_nodes])})"
    )
    # Append the end line
    ode_list.append("\treturn d_y\n")
    # Write the lines to a file
    with open(f"{save_dir}/{topo_name.split('/')[-1].split('.')[0]}.py", "w") as f:
        f.write("\n".join(ode_list))