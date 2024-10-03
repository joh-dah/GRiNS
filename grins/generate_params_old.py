import pandas as pd
import numpy as np
from glob import glob
from scipy.stats import qmc
import itertools as it
import warnings

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message="The balance properties of Sobol' points require n to be a power of 2",
    category=UserWarning,
)


# Function to generate the dataframe from the topofile
def parse_topos(topofile, save_cleaned=False):
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
    # Checking is Source, Target and Type are the column names
    if not all(col in topo_df.columns for col in ["Source", "Target", "Type"]):
        raise ValueError(
            "The topology file should have the columns: Source, Target, and Type."
        )
    # Reorder the columns if they are not in the correct order
    topo_df = topo_df[["Source", "Target", "Type"]]
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
    # If there is a value other than 1 or 2 in the Type column, print the topo file path and exit the function
    if len(type_counts) > 2:
        raise ValueError(f"Check the topo file: {topofile}")
    # If save cleaned is True, save the cleaned dataframe to a new topo file
    if save_cleaned:
        topo_df.to_csv(
            topofile.replace(".topo", "_cleaned.topo"), sep="\t", index=False
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
    target_nodes = list(topo_df["Target"].unique())
    # Get the list of unique source nodes to loop through (Necessary for threshold geenration)
    # Threshold ranges will be calculated for the source nodes as they have outgoing edges
    source_nodes = list(topo_df["Source"].unique())
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
    for tn in unique_nodes:
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
def _gen_sobol_seq(dimensions, num_points, optimise=False):
    """
    Generate Sobol sequence samples.

    Args:
        dimensions (int): The number of dimensions for the Sobol sequence.
        num_points (int): The number of samples to generate.
        optimise (bool, optional): Whether to use optimization for generation. Defaults to False.

    Returns:
        numpy.ndarray: The generated Sobol sequence samples.
    """
    if not optimise:
        samples = qmc.Sobol(d=dimensions).random(num_points)
    else:
        # Optimisation leads to a significant slowdown in the generation
        # Use only if needed
        samples = qmc.Sobol(d=dimensions, scramble=True, optimization="lloyd").random(
            num_points
        )
    return samples


# Internal function to generate uniform random distribution
def _gen_uniform_seq(dimension, num_points):
    """
    Generate sampling of uniform random variables

    Args:
        dimensions (int): The number of dimensions for the Uniform Random distribution.
    """
    samples = np.random.uniform(low=0, high=1, size=(num_points, dimension))
    return samples


# Internal function to generate log uniform random distribution
def _gen_loguniform_seq(dimension, num_points):
    """
    Generate sampling of log uniform random variables

    Args:
        dimensions (int): The number of dimensions for the Log Uniform Random distribution.
    """
    samples = np.random.uniform(low=0, high=1, size=(num_points, dimension))
    samples = np.exp(samples)
    return samples


# Internal function to generate latin hypercube sampling
def _gen_latin_hypercube(dimension, num_points, optimise=False):
    """
    Generate Latin Hypercube samples.

    Args:
        dimensions (int): The number of dimensions for the Latin Hypercube.
        num_points (int): The number of samples to generate.
        optimise (bool, optional): Whether to use optimization for generation. Defaults to False.
    """
    if not optimise:
        samples = qmc.LatinHypercube(d=dimension).random(num_points)
    else:
        # Optimisation leads to a significant slowdown in the generation
        # Use only if needed
        samples = qmc.LatinHypercube(d=dimension, optimization="lloyd").random(
            num_points
        )
    return samples


# Internal function to generate gaussian sampling
def _gen_normal(dimension, num_points, std_dev=1):
    """
    Generate sampling of Gaussian random variables

    Args:
        dimensions (int): The number of dimensions for the Gaussian distribution.
        std_dev (float): The standard deviation of the Gaussian distribution.
    """
    samples = np.random.normal(loc=0, scale=std_dev, size=(num_points, dimension))
    return samples


# Internal function to generate log normal sampling
def _gen_lognormal(dimension, num_points, std_dev=1):
    """
    Generate sampling of Log Normal random variables

    Args:
        dimensions (int): The number of dimensions for the Log Normal distribution.
        std_dev (float): The standard deviation of the Log Normal distribution.
    """
    # First sample from a normal distribution
    samples = np.random.normal(loc=0, scale=std_dev, size=(num_points, dimension))
    # Then exponentiate the samples
    samples = np.exp(samples)
    return samples


# Function which scales a given distribution to the required ranges
def _scale_distribution_column(
    sample,
    min_val,
    max_val,
    round_int=False,
):
    """
    Scale the given distribution to the required ranges.

    Args:
        sample (numpy.ndarray): The generated samples.
        min_val (float): The minimum value for scaling.
        max_val (float): The maximum value for scaling.
        round_int (bool, optional): Whether to round the values to integers. Defaults to False.
    """
    if round_int:
        min_val = min_val - 1
    # Scaling the values for the log distributions
    sample = min_val + (max_val - min_val) * (sample - min(sample)) / (
        max(sample) - min(sample)
    )
    # If round is required, return the scaled and rounded values
    if round_int:
        # Rounding the values to the next integer
        sample = np.ceil(sample)
        # If values are present below the minimum value, shift them to the minimum value
        sample[sample < min_val + 1] = min_val + 1
        # If values are present above the maximum value, shift them to the maximum value
        sample[sample > max_val] = max_val
        return sample
    else:
        return sample


# Function when given a subset of a parameter range dataframe, sampls the values in as dependent way as possible, scales them accordinf to min and max values and returns the dataframe
def sample_param_df(prange_df, num_params=2**10):
    # Save the original order of the parameters
    original_order = prange_df["Parameter"].values
    # Sort the rows according to the sampling method
    prange_df = prange_df.sort_values(by=["Sampling"])
    # Check if 'StdDev' column exists
    has_stddev = "StdDev" in prange_df.columns
    # If "Normal" or "LogNormal" appear more than twice, sort by "StdDev" as well
    if prange_df["Sampling"].isin(["Normal", "LogNormal"]).sum() >= 2:
        sort_columns = ["Sampling", "StdDev"] if has_stddev else ["Sampling"]
        prange_df = prange_df.sort_values(by=sort_columns)
    # Get unique rows for sampling methods (and optionally 'StdDev')
    unique_sampling_rows = (
        prange_df[["Sampling", "StdDev"]].drop_duplicates()
        if has_stddev
        else prange_df["Sampling"].drop_duplicates()
    )
    # Convert the unique rows to a dataframe - this is for cases where 'StdDev' is not present and a series is returned
    if isinstance(unique_sampling_rows, pd.Series):
        unique_sampling_rows = pd.DataFrame(unique_sampling_rows)
    # Get the order in which the parameters are listed
    param_order = prange_df["Parameter"].values
    # Placeholder list to store the sampled values
    sampled_values = []
    # Dictionary to map sampling methods to corresponding functions
    sampling_methods = {
        "Sobol": _gen_sobol_seq,
        "LHS": _gen_latin_hypercube,
        "Uniform": _gen_uniform_seq,
        "LogUniform": _gen_loguniform_seq,
        "Normal": _gen_normal,
        "LogNormal": _gen_lognormal,
    }
    # Loop through the unique sampling methods and sample values in chunks
    for _, row in unique_sampling_rows.iterrows():
        # Get the sampling method
        method = row["Sampling"]
        # Subset the dataframe for the current sampling method and 'StdDev'
        subset_df = prange_df[(prange_df["Sampling"] == method)]
        if has_stddev and "StdDev" in row:
            subset_df = subset_df[subset_df["StdDev"] == row["StdDev"]]
        # Get the number of dimensions
        num_dims = len(subset_df)
        # Get the corresponding sampling function
        sampling_func = sampling_methods[method]
        # If 'Normal' or 'LogNormal', pass 'std_dev' argument
        if method in ["Normal", "LogNormal"]:
            samples = sampling_func(num_dims, num_params, std_dev=row.get("StdDev"))
        else:
            samples = sampling_func(num_dims, num_params)
        # Add the samples to the sampled values list
        sampled_values.append(samples)
    # Concatenate the sampled values and convert them to a dataframe
    sampled_values = np.concatenate(sampled_values, axis=1)
    sampled_df = pd.DataFrame(sampled_values, columns=param_order)

    # Function to paramterise the scalainf of the columns
    def scale_column(col, param_name):
        # Find the corresponding row in the parameter range dataframe
        param_range_row = prange_df[prange_df["Parameter"] == param_name].iloc[0]
        # Get the sample column and pass it to the scaling function
        return _scale_distribution_column(
            sample=col.values,
            min_val=param_range_row["Minimum"],
            max_val=param_range_row["Maximum"],
            round_int=True if "Hill" in param_name else False,
        )

    # Apply the scaling function to each column
    sampled_df = sampled_df.apply(lambda col: scale_column(col, col.name), axis=0)
    # Reorder the columns to match the original order
    sampled_df = sampled_df[original_order]
    return sampled_df


# Internal function which calcualtes the hills equation values for a given source node gk values and the corresponding sampled in-coming edge paramters dataframe
def _get_updated_gkn_hills(gk_n, in_edge_params, in_edge_topo, num_params=2**10):
    """
    Update the g/k values for a given source node based on the incoming edge parameters.

    Parameters:
        gk_n (numpy.ndarray): The g/k values for the source node.
        in_edge_params (pandas.DataFrame): The incoming edge parameters dataframe.

    Returns:
        numpy.ndarray: The updated g/k values for the source node.
    """
    # Creatinf a dataframe to store teh gk_n values of source node to which the hills values will be added as a columns
    gk_hills_df = pd.DataFrame(gk_n, columns=["src_gk_n"])
    # Sample the incoming edge parameters
    inedg_param_samples = sample_param_df(in_edge_params, num_params)
    # Loop thorough each of the incoming nodes and calcualte the hills equation values
    for _, in_row in in_edge_topo.iterrows():
        # Get the insource and target nodes - here the target node is the source node for which the threshold values are being calculated in the function this is being called in
        insrc_nd, tgt_nd, type_reg = in_row["Source"], in_row["Target"], in_row["Type"]
        # Calcualting the shifted hill value depending on the type of regulation
        if type_reg == 1:
            fld = inedg_param_samples[f"ActFld_{insrc_nd}_{tgt_nd}"]
            g = inedg_param_samples[f"Prod_{insrc_nd}"]
            k = inedg_param_samples[f"Deg_{insrc_nd}"]
            n = inedg_param_samples[f"Hill_{insrc_nd}_{tgt_nd}"]
            thr = inedg_param_samples[f"Thr_{insrc_nd}_{tgt_nd}"]
            # Calculating the shifted hills value
            gk_hills_df[f"AsH_{insrc_nd}_{tgt_nd}"] = (
                fld + (1 - fld) * (1 / (1 + ((g / k) / thr) ** n))
            ) / fld
        if type_reg == 2:
            fld = inedg_param_samples[f"InhFld_{insrc_nd}_{tgt_nd}"]
            g = inedg_param_samples[f"Prod_{insrc_nd}"]
            k = inedg_param_samples[f"Deg_{insrc_nd}"]
            n = inedg_param_samples[f"Hill_{insrc_nd}_{tgt_nd}"]
            thr = inedg_param_samples[f"Thr_{insrc_nd}_{tgt_nd}"]
            # Calculating the shifted hills value
            gk_hills_df[f"InH_{insrc_nd}_{tgt_nd}"] = fld + (1 - fld) * (
                1 / (1 + ((g / k) / thr) ** n)
            )
    # Calcualting the final gk_n values by doing the product of the gk_n values and the hills values
    # gk_hills_df["prod"] = gk_hills_df.prod(axis=1)
    # Returning the meadian of the calculated gk_n values
    return np.median(gk_hills_df.prod(axis=1))


# Function to define the ranges of the threshold values
def get_thr_ranges(source_node, topo_df, prange_df, num_params=2**10, optimise=False):
    """
    Calculate the threshold ranges for a given source node based on the topology dataframe.

    Parameters:
        source_node (str): The source node for which to calculate the threshold ranges.
        topo_df (pandas.DataFrame): The topology dataframe containing the network information.
        num_params (int): The number of parameters to generate for the threshold calculation.

    Returns:
        float: The median value of the threshold ranges for the source node.
    """
    # Subset the Prod_ and Deg_ rows for the source node
    sn_params = prange_df[
        prange_df["Parameter"].str.contains(f"Prod_{source_node}|Deg_{source_node}")
    ]
    # Generate the sampled values for the source node parameters
    sn_gk = sample_param_df(sn_params, num_params)
    # # Get the in coming edges into source node and their types and counts
    # source_node_in_valcounts = topo_df[topo_df["Target"] == source_node][
    #     "Type"
    # ].value_counts()
    # Get the spubset of edges with soruce node as a target
    in_edge_topo = topo_df[topo_df["Target"] == source_node]
    # The g/k value list will be updated to give the final distribution from which median will be taken for the threshold
    # Generate the g/k values of the node
    sn_gk_n = sample_param_df(sn_params, num_params)
    sn_gk_n = np.array(sn_gk_n["Prod_" + source_node] / sn_gk_n["Deg_" + source_node])
    # If there are incoming edges the parameters need to be sampled to get the threshold values
    if not in_edge_topo.empty:
        # Get the | logic string for the incoming nodes
        isn = "|".join(in_edge_topo["Source"].values)
        # Getting the parameters for the incoming edges
        in_edge_params = prange_df[
            (
                prange_df["Parameter"].str.contains(
                    f"Fld_{isn}_{source_node}|Thr_{isn}_{source_node}|Hill_{isn}_{source_node}"
                )
            )
            | (prange_df["Parameter"].str.contains(f"Prod_{isn}|Deg_{isn}"))
        ]
        # Generating the G/k values for the incoming edge nodes to get the threshold values
        isn_gk = sample_param_df(
            in_edge_params[in_edge_params["Parameter"].str.contains("Prod_|Deg_")],
            num_params,
        )
        # Loop thorugh the incoming nodes, get the g/k values and thier median to update the Thr_isn_sn values
        for in_node in in_edge_topo["Source"].values:
            # Get the g/k values for the incoming node
            in_gk = isn_gk["Prod_" + in_node] / isn_gk["Deg_" + in_node]
            # Get the median of the g/k values
            in_gk_median = np.median(in_gk)
            # Get the corresponding row of the in_edge_params for the incoming node
            in_edge_params.loc[
                in_edge_params["Parameter"].str.contains(
                    f"Thr_{in_node}_{source_node}"
                ),
                ["Minimum", "Maximum"],
            ] = [0.02 * in_gk_median, (1.98 - 0.02) * in_gk_median]
        # Update the g/k values for the source node based on the incoming edge parameters and return the median
        return _get_updated_gkn_hills(sn_gk_n, in_edge_params, in_edge_topo, num_params)
    else:
        # # Get the median steady state of the isolted node i.e g/k value
        return np.median(sn_gk["Prod_" + source_node] / sn_gk["Deg_" + source_node])


# Function to get the parameter range dataframe
def get_param_range_df(
    topo_df, num_params=2**10, sampling_method="Sobol", thr_rows=True
):
    """
    Generate a parameter range DataFrame based on the topology DataFrame.

    Parameters:
    topo_df (pd.DataFrame): DataFrame containing the topology information.
    num_params (int, optional): Number of parameters to generate. Default is 1024.
    sampling_method (str or dict, optional): Sampling method to use. Can be one of "Sobol", "LHS", "Uniform", or "LogUniform". If a dictionary is provided, it should map parameter names to sampling methods. Default is "Sobol".
    thr_rows (bool, optional): Whether to add threshold-related rows to the DataFrame. Default is True.

    Returns:
    pd.DataFrame: DataFrame containing the parameter ranges with columns ["Parameter", "Minimum", "Maximum", "Sampling"].
    """
    # Get the paramter and unique_target_node names
    param_names, target_nodes, source_nodes = gen_param_names(topo_df)
    # print(param_names)
    # print(unique_target_nodes)
    # Create a parameter range dataframe
    prange_df = pd.DataFrame(columns=["Parameter", "Minimum", "Maximum"])
    prange_df["Parameter"] = param_names
    # Assign the minimum and maximum values of different parameters
    # Production Rate
    prange_df.loc[
        prange_df["Parameter"].str.contains("Prod_"), ["Minimum", "Maximum"]
    ] = [1.0, 100.0]
    # Degreadation Rate
    prange_df.loc[
        prange_df["Parameter"].str.contains("Deg_"), ["Minimum", "Maximum"]
    ] = [0.1, 1.0]
    # Fold Change Values
    # For Activation Fold changes
    prange_df.loc[
        prange_df["Parameter"].str.contains("ActFld_"), ["Minimum", "Maximum"]
    ] = [1.0, 100.0]
    # For Inhibition Fold changes
    prange_df.loc[
        prange_df["Parameter"].str.contains("InhFld_"), ["Minimum", "Maximum"]
    ] = [0.01, 1.0]
    # Hills Coefficient
    prange_df.loc[
        prange_df["Parameter"].str.contains("Hill"), ["Minimum", "Maximum"]
    ] = [1.0, 6.0]
    # Checkind if the sampling method is in the right format
    if not isinstance(sampling_method, str) and sampling_method not in [
        "Sobol",
        "LHS",
        "Uniform",
        "LogUniform",
    ]:
        raise ValueError(
            "The sampling method should be a string and one of Sobol, LHS, Uniform, or LogUniform."
        )
    # Cheking if the sampling method is a dict and all the unique values are in the list of sampling methods
    if not isinstance(sampling_method, str) and not all(
        [
            method in ["Sobol", "LHS", "Uniform", "LogUniform"]
            for method in set(sampling_method.values())
        ]
    ):
        raise ValueError(
            "The sampling method should be a dictionary with values as one of Sobol, LHS, Uniform, or LogUniform in the froamt {param: method}."
        )

    if isinstance(sampling_method, str):
        prange_df["Sampling"] = sampling_method
        # If the sampling method is either Normal or LogNormal add StdDev column
        if sampling_method in ["Normal", "LogNormal"]:
            prange_df["StdDev"] = 1.0
    else:
        # If the sampling method is a dictionary, then assign the sampling method to the specific parameters
        for param, method in sampling_method.items():
            prange_df.loc[prange_df["Parameter"].str.contains(param), "Sampling"] = (
                method
            )
        # If any of the paramters have not been assigned a sampling method, assign them the default method
        prange_df["Sampling"] = prange_df["Sampling"].fillna("Sobol")
        # If the sampling method is either Normal or LogNormal add StdDev column
        if any(prange_df["Sampling"].isin(["Normal", "LogNormal"])):
            prange_df["StdDev"] = 1.0
    # For the columns corresponding to
    if thr_rows:
        # Calling the function to add the threshold related columns
        prange_df = add_thr_rows(prange_df, topo_df, num_params)
    # Return the parameter range dataframe
    return prange_df


# Function to add the threshold related columns
def add_thr_rows(prange_df, topo_df, num_params=2**10):
    """
    Add the threshold related columns to the parameter dataframe.

    Parameters:
        param_df (pd.DataFrame): The parameter dataframe containing the parameter values.
        topo_df (pd.DataFrame): The topology dataframe containing the network information.

    Returns:
        pd.DataFrame: The parameter dataframe with the threshold columns added.
    """
    # Get the source target and parameter names
    param_names, target_nodes, source_nodes = gen_param_names(topo_df)
    # Find the threshold values of the the edges and calulate the amplification factor
    # As production rate also depeneds on amplification factor the range is assigned in this loop
    # Iterate through the source nodes and get their threshold ranges
    for sn in set(target_nodes + source_nodes):
        # As if the node is source node, its amplification value and threshold values will change if the minimum threshold value is below 0.01, assign the ranges speprately after calcualting the threshold minimum
        if sn in source_nodes:
            # print(sn)
            median_thr_val = get_thr_ranges(sn, topo_df, prange_df, num_params)
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
            # Replace Minimum values
            prange_df.loc[
                prange_df["Parameter"].str.contains(f"Thr_{sn}"), "Minimum"
            ] = median_thr_val * 0.02 * amplify_val
            # Replace the maximum values
            prange_df.loc[
                prange_df["Parameter"].str.contains(f"Thr_{sn}"), "Maximum"
            ] = median_thr_val * 1.98 * amplify_val
            # print(f"Threshold Range: {median_thr_val*0.02*amplify_val} {median_thr_val*1.98*amplify_val}")
            # Set the Production Rates (Amplified) for the source node
            prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Minimum"] = (
                prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Minimum"]
                * amplify_val
            )
            prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Maximum"] = (
                prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Maximum"]
                * amplify_val
            )
    # Return the parameter range dataframe
    return prange_df


# Function to generate the parameter dataframe
def gen_param_df(prange_df, num_paras=2**10):
    """
    Generate a parameter dataframe based on the given parameter range dataframe.

    Parameters:
        prange_df (pd.DataFrame): The parameter range dataframe containing the minimum and maximum values for each parameter.

    Returns:
        pd.DataFrame: The generated parameter dataframe with sampled values for each parameter.
    """
    # Get the parameter names list from the parameter range dataframe
    param_name_li = prange_df["Parameter"].values
    # Sample the values along the number of dimensions equal to the number of parameters
    param_mat = _gen_sobol_seq(len(param_name_li), num_paras)
    # Scale the parameter columns by the minimum and maximum value from parameter range dataframe
    for i, param in enumerate(param_name_li):
        # Scale the parameter column
        # If the parameter is a Hill coefficient, round the value
        if "Hill" in param:
            param_mat[:, i] = np.ceil(prange_df.loc[i, "Maximum"] * param_mat[:, i])
        # Else if the parameter is InhFld then take the inverse of the value
        elif "InhFld" in param:
            # Define the minimum and maximum values for the parameter (as they are reciprocal of what is given in the prange_df)
            min = 1 / prange_df.loc[i, "Maximum"]
            max = 1 / prange_df.loc[i, "Minimum"]
            param_mat[:, i] = np.reciprocal(min + (max - min) * param_mat[:, i])
        else:
            param_mat[:, i] = (
                prange_df.loc[i, "Minimum"]
                + (prange_df.loc[i, "Maximum"] - prange_df.loc[i, "Minimum"])
                * param_mat[:, i]
            )
    # Convert the matrix to a dataframe and return
    return pd.DataFrame(param_mat, columns=param_name_li)


def gen_init_cond(topo_df, num_init_conds=1000):
    """
    Generate the initial conditions dataframe based on the given topology dataframe.

    Parameters:
        topo_df (pd.DataFrame): The topology dataframe containing the information about the network topology.
        num_init_conds (int): The number of initial conditions to generate. Default is 1000.

    Returns:
        initcond_df (pd.DataFrame): The initial conditions dataframe with the initial conditions for each node.
    """
    param_names, target_nodes, source_nodes = gen_param_names(topo_df)
    unique_nodes = sorted(set(source_nodes + target_nodes))
    # Generate the sobol sequence for the initial conditions
    initial_conds = _gen_sobol_seq(len(unique_nodes), num_init_conds)
    # Scale the initial conditions between 1 to 100
    initial_conds = 1 + initial_conds * (100 - 1)
    # Convert the initial conditions to a dataframe and save in the replicate folders
    initcond_df = pd.DataFrame(
        initial_conds, columns=unique_nodes, index=range(1, num_init_conds + 1)
    )
    initcond_df.index.name = "InitCondNum"
    return initcond_df
