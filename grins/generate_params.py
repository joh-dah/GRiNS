import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import qmc
import itertools as it
import warnings
from grins.reg_funcs import nsH, psH

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message="The balance properties of Sobol' points require n to be a power of 2",
    category=UserWarning,
)


# Function to generate the dataframe from the topofile
def parse_topos(topofile):
    """
    Parse the given topofile and return the dataframe.

    Parameters:
        topofile (str): The path to the topofile.

    Returns:
        topo_df (pandas.DataFrame): The parsed dataframe.
    """
    topo_df = pd.read_csv(topofile, sep=r"\s+")
    # Check if the dataframe has the required columns
    if topo_df.shape[1] != 3:
        raise ValueError(
            "The topology file should have three columns: Source, Target, and Type."
        )
    # Rename the columns to Source, Target, and Type
    topo_df.columns = ["Source", "Target", "Type"]
    # Go through the source and target columns and replace non-alphanumeric characters with underscores
    topo_df["Source"] = topo_df["Source"].str.replace(r"\W", "_", regex=True)
    topo_df["Target"] = topo_df["Target"].str.replace(r"\W", "_", regex=True)
    # Append 'Node_' to the source and target columns if they do not start with an alphabet
    topo_df["Source"] = topo_df["Source"].apply(
        lambda x: f"Node_{x}" if not x[0].isalpha() else x
    )
    topo_df["Target"] = topo_df["Target"].apply(
        lambda x: f"Node_{x}" if not x[0].isalpha() else x
    )
    # Check if the Type column has any value other than 1 or 2 by getting the counts of unique values
    type_counts = topo_df["Type"].value_counts()
    # If there is a value other than 1 or 2 in the Type column, warn the user
    if len(type_counts) > 2:
        warnings.warn(
            "The Type column should only have values 1 or 2. Other values may not be compatible with the default framework."
        )
    return topo_df


# Internal function to get the type of regualtion of the edge required for fold change name and sampling
def _get_regtype(sn, tn, topo_df):
    """
    Get the type of regulation for a given source and target node.

    Parameters:
        sn (str): The source node.
        tn (str): The target node.
        topo_df (pandas.DataFrame): The DataFrame containing the topology information.

    Returns:
        str: The type of regulation, either "ActFld_{sn}_{tn}" for activation or "InhFld_{sn}_{tn}" for inhibition.
    """
    # Get the type of regulation
    # For Activation
    if (
        topo_df[(topo_df["Source"] == sn) & (topo_df["Target"] == tn)]["Type"].iloc[0]
        == 1
    ):
        return f"ActFld_{sn}_{tn}"
    # For Inhibition
    else:
        return f"InhFld_{sn}_{tn}"


# Function to get the paramter names to create the parameter configuration dataframe row names
def gen_param_names(topo_df):
    """
    Generate parameter names based on the given topology dataframe.

    Parameters:
        topo_df (pandas.DataFrame): The topology dataframe containing the information about the nodes and edges.

    Returns:
        tuple: A tuple containing the parameter names, unique target node names, and unique source node names.
    """
    # Get the list of unique target nodes to loop through
    target_nodes = sorted(topo_df["Target"].unique())
    # Get the list of unique source nodes to loop through (Necessary for threshold geenration)
    # Threshold ranges will be calculated for the source nodes as they have outgoing edges
    source_nodes = sorted(topo_df["Source"].unique())
    # Get the set of unique nodes
    unique_nodes = sorted(set(source_nodes + target_nodes))
    # Initialise a list for storing the paramter names
    param_names = []
    # Get the production and degradation parameter names
    param_names = (
        param_names
        + [f"Prod_{n}" for n in unique_nodes]
        + [f"Deg_{n}" for n in unique_nodes]
    )
    # Loop through each of the target nodes
    for tn in target_nodes:
        # Generate the combinations of the paramters and the source nodes
        param_comb_li = it.product(
            topo_df[topo_df["Target"] == tn]["Source"].sort_values(),
            ["Fld", "Thr", "Hill"],
        )
        # Add fold change paramter names (Need to add inhibition and activation for making sampling easier)
        # Append the parameter names for a node to the param_names list
        param_names = param_names + [
            f"{p}_{sn}_{tn}" if p != "Fld" else _get_regtype(sn, tn, topo_df)
            for sn, p in param_comb_li
        ]
    # Return the paramter names and the unique target node names (required for half hunctinal)
    return param_names, target_nodes, source_nodes


# Internal function to generate sobol sequences
def _gen_sobol_seq(num_points, ranges, optimise=False):
    """
    Generate Sobol sequence samples.

    Parameters:
        num_points (int): The number of samples to generate.
        ranges (dataframe): The dataframe containing the minimum and maximum values for each parameter.
        optimise (bool, optional): Whether to use optimization for generation. Optimisation leads to a significant slowdown in the generation. Defaults to False.

    Returns:
        numpy.ndarray: The generated Sobol sequence samples.
    """
    dimensions = len(ranges)
    optimise = "lloyd" if (optimise) and (dimensions > 1) else None
    # Generate the Sobol sequence samples
    samples = qmc.Sobol(d=dimensions, optimization=optimise, scramble=True).random(
        num_points
    )
    # Scale the samples by the minimum and maximum values
    samples = (
        samples * (ranges["Maximum"].values - ranges["Minimum"].values)
        + ranges["Minimum"].values
    )
    return samples


def _gen_uniform_seq(num_points, ranges):
    """
    Generate sampling of uniform random variables

    Parameters:
        num_points (int): The number of samples to generate.
        ranges (dataframe): The dataframe containing the minimum and maximum values for each parameter.

    Returns:
        numpy.ndarray: The generated uniform random samples.
    """
    samples = np.random.uniform(
        low=ranges["Minimum"].values,
        high=ranges["Maximum"].values,
        size=(num_points, len(ranges)),
    )
    return samples


def _gen_latin_hypercube(num_points, ranges, optimise=False):
    """
    Generate Latin Hypercube samples.

    Parameters:
        num_points (int): The number of samples to generate.
        ranges (dataframe): The dataframe containing the minimum and maximum values for each parameter.
        optimise (bool, optional): Whether to use optimization for generation. Optimisation leads to a significant slowdown in the generation. Defaults to False.

    Returns:
        numpy.ndarray: The generated Latin Hypercube samples.
    """
    dimensions = len(ranges)
    optimise = "lloyd" if (optimise) and (dimensions > 1) else None
    # Generate the Latin Hypercube samples
    samples = qmc.LatinHypercube(d=dimensions, optimization=optimise).random(num_points)
    # Scale the samples by the minimum and maximum values
    samples = (
        samples * (ranges["Maximum"].values - ranges["Minimum"].values)
        + ranges["Minimum"].values
    )
    return samples


def _gen_loguni_seq(num_points, ranges):
    """
    Generate sampling of log-uniform random variables

    Parameters:
        num_points (int): The number of samples to generate.
        ranges (dataframe): The dataframe containing the minimum and maximum values for each parameter.

    Returns:
        numpy.ndarray: The generated log-uniform random samples.
    """
    samples = np.exp(
        np.random.uniform(
            low=np.log(ranges["Minimum"].values),
            high=np.log(ranges["Maximum"].values),
            size=(num_points, len(ranges)),
        )
    )
    return samples


def _samp_func(sampling):
    """
    Get the sampling function based on the given sampling method.

    Parameters:
        sampling (str): The sampling method to use.
        Options:
            - "sobol": Sobol sequence sampling.
            - "uni": Uniform random sampling.
            - "latin_hc": Latin Hypercube sampling.
            - "loguni": Log-uniform random sampling.
    """
    samp_meth = {
        "sobol": _gen_sobol_seq,
        "uni": _gen_uniform_seq,
        "latin_hc": _gen_latin_hypercube,
        "loguni": _gen_loguni_seq,
    }
    if sampling not in samp_meth.keys():
        raise ValueError(
            f"Sampling method {sampling} not recognised. Choose from {samp_meth.keys()}"
        )
    return samp_meth[sampling]


# Function to define the ranges of the threshold values
def get_thr_range_node(
    source_node, topo_df, prange_df, num_params=2**12, sampling="Sobol"
):
    """
    Calculate the threshold ranges for a given source node based on the topology dataframe.

    Parameters:
        source_node (str): The source node for which to calculate the threshold ranges.
        topo_df (pandas.DataFrame): The topology dataframe containing the network information.
        num_params (int): The number of parameters to generate for the threshold calculation.
        sampling (str): The sampling method to use. Default is "sobol".

    Returns:
        gk.median (float): The median value of the threshold ranges for the source node.
    """
    prange_df = prange_df.copy()
    # Select the parameters for the interactions coming into the source node
    upstream_topo = topo_df[topo_df["Target"] == source_node]
    if upstream_topo.empty:
        # Choose only the production and degradation rates if the source node is isolated
        param_names = [f"Prod_{source_node}", f"Deg_{source_node}"]
    else:
        # Get the parameter names of the source node and the incoming edges
        param_names, *_ = gen_param_names(upstream_topo)
    # Subset the parameter range dataframe to only include the required parameters
    prange_df = prange_df.loc[param_names, :]
    # Generate the parameter dataframe
    param_df = gen_param_df(prange_df, num_params, sampling)
    # Get the median steady state of the isolted node i.e g/k value
    g = param_df.loc[:, f"Prod_{source_node}"].values
    k = param_df.loc[:, f"Deg_{source_node}"].values
    gk = g / k
    m0 = np.median(gk)
    # gk = param_df.loc[:, f"Prod_{source_node}"] / param_df.loc[:, f"Deg_{source_node}"]
    # m0 = gk.median()
    if not upstream_topo.empty:
        # Pre extract the source nodes and thier type values
        source_nodes = upstream_topo["Source"].values
        source_types = upstream_topo["Type"].values
        # Iterate over each incoming edge and calculate the g/k value
        for i, up_node in enumerate(source_nodes):
            g = param_df.loc[:, f"Prod_{up_node}"].values
            k = param_df.loc[:, f"Deg_{up_node}"].values
            n = param_df.loc[:, f"Hill_{up_node}_{source_node}"].values
            thr = param_df.loc[:, f"Thr_{up_node}_{source_node}"].values * m0
            if source_types[i] == 1:
                fld = param_df.loc[:, f"ActFld_{up_node}_{source_node}"].values
                gk *= psH(g / k, fld, thr, n)
            else:
                fld = param_df.loc[:, f"InhFld_{up_node}_{source_node}"].values
                gk *= nsH(g / k, fld, thr, n)
        # # Iterate over each incoming edge and calculate the g/k value
        # for idx in upstream_topo.index:
        #     up_node = upstream_topo.loc[idx, "Source"]
        #     # Get the parameter values for the incoming edge
        #     g = param_df.loc[:, f"Prod_{up_node}"]
        #     k = param_df.loc[:, f"Deg_{up_node}"]
        #     n = param_df.loc[:, f"Hill_{up_node}_{source_node}"]
        #     thr = param_df.loc[:, f"Thr_{up_node}_{source_node}"] * m0
        #     if topo_df.loc[idx, "Type"] == 1:
        #         fld = param_df.loc[:, f"ActFld_{up_node}_{source_node}"]
        #         gk *= psH(g / k, fld, thr, n)
        #     else:
        #         fld = param_df.loc[:, f"InhFld_{up_node}_{source_node}"]
        #         gk *= nsH(g / k, fld, thr, n)
    # return gk.median()
    return np.median(gk)


def get_thr_ranges(
    prange_df, topo_df, source_nodes, num_params=2**12, sampling="sobol"
):
    """
    Adjusts the threshold ranges and production rates for source nodes based on their median threshold values. The production rates are adjusted by the amplification factor which accounts for the bias in the production rates due to low threshold values as a result of many inhibitory edges, specifically when the minimum threshold value is below 0.01. The threshold values also get adjusted based on the amplification factor.

    Parameters:
        prange_df (pd.DataFrame): DataFrame containing parameter ranges. (Will be modified in place)
        topo_df (pd.DataFrame): DataFrame containing topology information.
        source_nodes (list): List of source nodes.
        num_params (int, optional): Number of parameters for sampling. Default is 4096.
        sampling (str, optional): Sampling method to use. Default is "sobol".
    Returns:
        None: The parameter range dataframe is modified in place.
    """
    prange_df_copy = prange_df.copy()
    for sn in set(source_nodes):
        # As if the node is source node, its amplification value and threshold values will change if the minimum threshold value is below 0.01, assign the ranges speprately after calcualting the threshold minimum
        # median_thr_val = get_thr_range_node_old(sn, topo_df, num_params)
        median_thr_val = get_thr_range_node(
            sn, topo_df, prange_df_copy, num_params, sampling
        )
        # Find the amplification factor if the value of the minimum is lower than 0.01
        if (median_thr_val * 0.02) < 0.010:
            # Get the exponent power of 10 from the scientific notation of the number
            exp_val = np.floor(np.log10(np.abs(median_thr_val * 0.02))).astype(int)
            # Substract -2 from the -1*exponent value (which is negetive). Will give the value of the amplification factor
            # Same as dividing 10**exp/0.01
            amplify_val = 10 ** (-1 * int(exp_val) - 2)
        else:
            # Define the default amplification values
            amplify_val = 1.0
        # Get all the threshold parameters with current source node as the source
        # Then repalce the values in the minimum and maximum column with the range values
        # Adjust the Threshold values
        prange_df.loc[prange_df.index.str.contains(f"Thr_{sn}"), :] *= (
            median_thr_val * amplify_val
        )
        # Set the Production Rates (Amplified) for the source node
        prange_df.loc[prange_df.index == f"Prod_{sn}", :] *= amplify_val


# Function to get the parameter range dataframe
def get_param_range_df(
    topo_df, num_params=2**10, sampling="sobol", threshold_calc=True
):
    """
    Generate a parameter range dataframe based on the given topology dataframe. The default values for the parameters are as follows:
    - Production Rate: (1.0, 100.0). Will be adjusted based on the amplification factor if the minimum threshold value i.e 0.02 * median threshold value is below 0.01. The amplification is multiplies with this default minimum and maximum values to get the final range in cases where the amiplification factor is not 1.0.
    - Degradation Rate: (0.1, 1.0)
    - Activation Fold Change: (1.0, 100.0)
    - Inhibition Fold Change: (0.01, 1.0)
    - Hill Coefficient: (1.0, 6.0)
    - Threshold: (0.02, 1.98) Values are in relation to the median threshold value. Will be adjusted based on the amplification factor if the minimum threshold value i.e 0.02 * median threshold value is below 0.01. The aplification factor is calculated based on the median threshold value and orders of magnitude needed to bring the minimum threshold value to 0.01. The amplification factor is calculated as 10**(-1*exp_val - 2) where exp_val is the exponent value of the median threshold value.

    Parameters:
        topo_df (DataFrame): The topology dataframe containing information about the network topology.
        num_params (int): The number of parameters to generate for threshold calculation. Default is 1024.
        sampling (str): The sampling method to use. Default is "sobol".
        threshold_calc (bool): Whether to calculate the threshold ranges. Default is True.

    Returns:
        prange_df (DataFrame): The parameter range dataframe with columns "Parameter", "Minimum", and "Maximum".
    """
    # Get the paramter and unique_target_node names
    param_names, target_nodes, source_nodes = gen_param_names(topo_df)
    # Create a parameter range dataframe
    prange_df = pd.DataFrame(
        columns=["Minimum", "Maximum"], index=param_names, dtype=float
    )
    prange_df.index.name = "Parameter"
    # Assign the minimum and maximum values of different parameters
    default_rates = {
        "Prod": (1.0, 100.0),
        "Deg": (0.1, 1.0),
        "ActFld": (1.0, 100.0),
        "InhFld": (0.01, 1.0),
        "Hill": (1.0, 6.0),
        "Thr": (0.02, 1.98),
    }
    # Assign the default values to the parameter range dataframe
    for key, val in default_rates.items():
        prange_df.loc[prange_df.index.str.startswith(key), :] = val
    # Get the threshold ranges for each of the source nodes
    if threshold_calc:
        get_thr_ranges(prange_df, topo_df, source_nodes, num_params, sampling)
    # Return the parameter range dataframe
    return prange_df


# Function to generate the parameter dataframe
def gen_param_df(prange_df, num_paras=2**10, sampling="sobol"):
    """
    Generate a parameter dataframe based on the given parameter range dataframe. A custom parameter range dataframe can be

    Parameters:
        prange_df (pd.DataFrame): The parameter range dataframe containing the minimum and maximum values for each parameter.
        num_paras (int): The number of parameters to generate. Default is 1024.
        sampling (str): The sampling method to use. Default is "sobol".

    Returns:
        param_mat (pd.DataFrame): The generated parameter dataframe with sampled values for each parameter.
    """
    # Set InhFld to be sampled as reciprocal
    prange_df = prange_df.copy()
    prange_df.loc[prange_df.index.str.startswith("InhFld"), :] = (
        1 / prange_df.loc[prange_df.index.str.startswith("InhFld"), :]
    )
    # Sample the values along the number of dimensions equal to the number of parameters
    param_df = _samp_func(sampling)(num_paras, prange_df)
    # Convert the parameter matrix to a dataframe
    param_df = pd.DataFrame(param_df, columns=prange_df.index, dtype=float)
    param_df.index.name = "ParaNum"
    # If the parameter is a Hill coefficient, round the value
    param_df.loc[:, prange_df.index.str.startswith("Hill")] = np.ceil(
        param_df.loc[:, prange_df.index.str.startswith("Hill")]
    )
    # If the parameter is InhFld then take the inverse of the value
    param_df.loc[:, prange_df.index.str.startswith("InhFld")] = np.reciprocal(
        param_df.loc[:, prange_df.index.str.startswith("InhFld")]
    )
    return param_df


def gen_init_cond(topo_df, num_init_conds=1000, sampling="sobol"):
    """
    Generate the initial conditions dataframe based on the given topology dataframe.

    Parameters:
        topo_df (pd.DataFrame): The topology dataframe containing the information about the network topology.
        num_init_conds (int): The number of initial conditions to generate. Default is 1000.
        sampling (str): The sampling method to use. Default is "sobol".

    Returns:
        initcond_df (pd.DataFrame): The initial conditions dataframe with the initial conditions for each node.
    """
    param_names, target_nodes, source_nodes = gen_param_names(topo_df)
    unique_nodes = sorted(set(source_nodes + target_nodes))
    # Generate the sobol sequence for the initial conditions
    i_range = pd.DataFrame(
        [(1, 100)] * len(unique_nodes),
        columns=["Minimum", "Maximum"],
        index=unique_nodes,
    )
    initial_conds = _samp_func(sampling)(num_init_conds, i_range)
    # Convert the initial conditions to a dataframe and save in the replicate folders
    initcond_df = pd.DataFrame(
        initial_conds,
        columns=unique_nodes,
        index=range(1, num_init_conds + 1),
        dtype=float,
    )
    initcond_df.index.name = "InitCondNum"
    return initcond_df
