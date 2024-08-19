from diffrax import diffeqsolve, ODETerm, Dopri5  # type: ignore  # noqa: F401
import jax.numpy as jnp  # type: ignore  # noqa: F401
import pandas as pd  # type: ignore  # noqa: F401
from glob import glob
from generate_params import parse_topos, gen_param_names  # noqa: F401
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
    ode_list = []
    # Loop through the target nodes
    for ni, nod in enumerate(unique_nodes):
        # Get the edges where n is the target node
        target_edges = topo_df[topo_df["Target"] == nod]
        # The diffrax ODE for each node is d_<nod> = <ODE>
        ode_list.append("\t" + f"d_{nod} = {gen_node_ode(target_edges, nod)}")
    # print("\n")
    # Define a list to hold the lines of the ODE system
    ode_file_lines = []
    # Add line to import the diffrax package and its necessary functions
    ode_file_lines.append(
        "from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, PIDController, SteadyStateEvent\n"
    )
    # Add the jax andd jax numpy import lines
    ode_file_lines.append("import jax.numpy as jnp\nfrom jax import vmap, pmap")
    # Add the pandas import line
    ode_file_lines.append("import pandas as pd\n")
    # Add the lines defining the postive hills functions
    ode_file_lines.append("# Positive Shifted Hill function")
    ode_file_lines.append(
        "def psH(nod, fld, thr, hill):\n\treturn (fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))) / fld\n"
    )
    # Add the lines defining the negetive hills functions
    ode_file_lines.append("# Negative Shifted Hill function")
    ode_file_lines.append(
        "def nsH(nod, fld, thr, hill):\n\treturn fld + (1 - fld) * (1 / (1 + jnp.power(nod / thr, hill)))\n"
    )
    # Append the ode systems related lines
    # Append the function definition line
    ode_file_lines.append("# ODE system function for diffrax")
    ode_file_lines.append(
        f"def odesys_{topo_name.split('/')[-1].split('.')[0]}(t, y, args):"
    )
    # Append the nodde name line
    ode_file_lines.append(f"\t({', '.join(unique_nodes)}) = y")
    # Append the parameter names line
    ode_file_lines.append(f"\t({', '.join(param_names_list)}) = args")
    # Append the ODE lines
    ode_file_lines.append("\n".join(ode_list))
    # Append the d_y line
    ode_file_lines.append(
        f"\td_y = ({', '.join([f'd_{nod}' for nod in unique_nodes])})"
    )
    # Append the end line
    ode_file_lines.append("\treturn d_y\n")
    # Define the ode solver wrapper function
    ode_file_lines.append("# ODE solver wrapper function")
    ode_file_lines.append("def solve_ode(term, solver, t0, t1, dt0, y0, args):")
    # Add line to convert y0 to a tuple
    ode_file_lines.append("\ty0 = tuple(y0)")
    # Add the diffrax solve line
    ode_file_lines.append(
        "\tsol = diffeqsolve(ODETerm(term), solver, t0, t1, dt0, y0, args, saveat=SaveAt(t1=True), stepsize_controller=PIDController(reltol=1e-5, abstol=1e-6), max_steps=None, events=SteadyStateEvent())"
    )
    # Add the return line
    ode_file_lines.append("\treturn [s[-1] for s in sol.ys]\n")
    # Add the line to actually solve the ODE using vmap over the parameters
    ode_file_lines.append("# Solve the ODE using vmap over the parameters")
    # ode_file_lines.append(
    # Write the lines to a file
    with open(f"{save_dir}/{topo_name.split('/')[-1].split('.')[0]}.py", "w") as f:
        f.write("\n".join(ode_file_lines))


if __name__ == "__main__":
    # Specify the topo file folder
    topo_folder = "TOPOS"
    # Load the topology files
    topo_list = sorted(glob(f"{topo_folder}/*.topo"))
    print(topo_list)
    # Specifyu the simulation directory
    sim_dir = "SimResults"
    # Create directories pf each topo file in the simulation directory
    for tpfl in topo_list:
        # Get the topo name
        topo_name = tpfl.split("/")[-1].split(".")[0]
        # Create the directory
        os.makedirs(f"{sim_dir}/{topo_name}", exist_ok=True)
    # Loop through the topo files to generate the differential equations
    for tpfl in topo_list[:1]:
        # Parse the topology file
        topo_df = parse_topos(tpfl)
        print(topo_df)
        # Generate the parameter names
        param_names = gen_param_names(topo_df)
        # Generate the diffrax ode system
        gen_diffrax_odesys(
            topo_df, tpfl, save_dir=f"{sim_dir}/{tpfl.split('/')[-1].split('.')[0]}"
        )
