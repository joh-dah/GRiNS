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

# Plotting
# from matplotlib import pyplot as plt
# import seaborn as sns


# Function to generate the dataframe from the topofile
def parse_topos(topofile):
    """
    Parse the given topofile and return the dataframe.

    Parameters:
    topofile (str): The path to the topofile.

    Returns:
    pandas.DataFrame: The parsed dataframe.
    """
    return pd.read_csv(topofile, sep=r"\s+")


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
        # print(tn)
        # print(topo_df[topo_df["Target"] == tn])
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
    if dimensions == 1:
        samples = qmc.Sobol(d=dimensions).random(num_points)
    else:
        if not optimise:
            samples = qmc.Sobol(d=dimensions).random(num_points)
        else:
            # Optimisation leads to a significant slowdown in the generation
            # Use only if needed
            samples = qmc.Sobol(
                d=dimensions, scramble=True, optimization="lloyd"
            ).random(num_points)
    # return min_max_val[0] + (min_max_val[1] - min_max_val[0])*samples
    return samples


# Function to define the ranges of the threshold values
def get_thr_ranges(source_node, topo_df, num_params=2**12):
    """
    Calculate the threshold ranges for a given source node based on the topology dataframe.

    Parameters:
    - source_node (str): The source node for which to calculate the threshold ranges.
    - topo_df (pandas.DataFrame): The topology dataframe containing the network information.
    - num_params (int): The number of parameters to generate for the threshold calculation.

    Returns:
    - float: The median value of the threshold ranges for the source node.
    """
    # print(source_node)
    # Sample the median value of the node
    gk = _gen_sobol_seq(2, num_params)
    # Get the median steady state of the isolted node i.e g/k value
    m0 = np.median((1 + (100 - 1) * gk[:, 0]) / (0.1 + (1 - 0.1) * gk[:, 1]))
    # Get the in coming edges into source node and their types and counts
    source_node_in_valcounts = topo_df[topo_df["Target"] == source_node][
        "Type"
    ].value_counts()
    # print(source_node_in_valcounts)
    # The g/k value list will be updated to give the final distribution from which median will be taken for the threshold
    # Generate the g/k values of the node
    gk_n = _gen_sobol_seq(2, num_params)
    # Scale and divide the two columns
    gk_n = (1 + (100 - 1) * gk[:, 0]) / (0.1 + (1 - 0.1) * gk[:, 1])
    # If the source_node_value_counts does not have any edges (the source is not being regulated by anythong else) the for loop is skipped as there are no values in the list
    # The median of the gk_n distribution is be then returned
    # print(f"WayBefore: {gk_n[:5]}")
    # Loop thorugh the postive and negetive edges to add to the threshold list
    for i_a, count in source_node_in_valcounts.items():
        # If its activation go thorugh the activation threshold generation scheme
        if i_a == 1:
            # Generates a distribution for each of the incoming edge nodes
            for i in range(count):
                # Generate a sobol sequence for each of the hills parameters
                h_params = _gen_sobol_seq(5, num_params)
                # We need to scale the parameters according to the ranges
                g = 1 + (100 - 1) * h_params[:, 0]
                k = 0.1 + (1 - 0.1) * h_params[:, 1]
                n = np.ceil((6) * h_params[:, 2])
                fld = 1 + (100 - 1) * h_params[:, 3]
                thr = 0.02 * m0 + (1.98 - 0.02) * m0 * h_params[:, 4]
                # Get the shifted Hills equation value
                # print(f"Before: {gk_n[:5]}")
                # print(f"Mod: {(fld + (1-fld)*(1/(1+((g/k)/thr)**n)))[:5]}")
                # print(f"ModF: {((fld + (1-fld)*(1/(1+((g/k)/thr)**n)))/fld)[:5]}")
                gk_n = (
                    gk_n * (fld + (1 - fld) * (1 / (1 + ((g / k) / thr) ** n)))
                ) / fld
                # print(f"Before: {gk_n[:5]}")
                # print(f"After: {gk_n[:5]}")
                # source_nodes.scatterplot(x=thr, y=fld)
                # plt.show()
        else:
            # Generates a distribution for each of the incoming edge nodes
            for i in range(count):
                # Generate a sobol sequence for each of the hills parameters
                h_params = _gen_sobol_seq(5, num_params)
                # We need to scale the parameters according to the ranges
                g = 1 + (100 - 1) * h_params[:, 0]
                k = 0.1 + (1 - 0.1) * h_params[:, 1]
                n = np.ceil((6) * h_params[:, 2])
                fld = 1 / (1 + (100 - 1) * h_params[:, 3])
                # fld = 0.1 + (1 - 0.1)*h_params[:, 0]
                thr = 0.02 * m0 + (1.98 - 0.02) * m0 * h_params[:, 4]
                # Get the shifted Hills equation value
                # print(f"Before: {gk_n[:5]}")
                # print(f"Mod: {(fld + (1-fld)*(1/(1+((g/k)/thr)**n)))[:5]}")
                # print(f"ModF: {((fld + (1-fld)*(1/(1+(g/k)**n)))/fld)[:5]}")
                gk_n = gk_n * (fld + (1 - fld) * (1 / (1 + ((g / k) / thr) ** n)))
                # print(f"After: {gk_n[:5]}")
                # source_nodes.scatterplot(x=thr, y=fld)
                # plt.show()
    # print(gk_n[:10])
    # print(f"Median : {np.median(gk_n)}")
    # print(f"Median : {np.median(gk_n)*0.02} - {np.median(gk_n)*1.98}")
    # source_nodes.histplot(gk_n)
    # plt.show()
    return np.median(gk_n)


# Function to get the parameter range dataframe
def get_param_range_df(topo_df, num_params=2**10):
    """
    Generate a parameter range dataframe based on the given topology dataframe.

    Parameters:
    - topo_df (DataFrame): The topology dataframe containing information about the network topology.
    - num_params (int): The number of parameters to generate for threshold calculation. Default is 2**10.

    Returns:
    - prange_df (DataFrame): The parameter range dataframe with columns "Parameter", "Minimum", and "Maximum".
    """
    # Get the paramter and unique_target_node names
    param_names, target_nodes, source_nodes = gen_param_names(topo_df)
    # print(param_names)
    # print(unique_target_nodes)
    # Create a parameter range dataframe
    prange_df = pd.DataFrame(columns=["Parameter", "Minimum", "Maximum"])
    prange_df["Parameter"] = param_names
    # print(prange_df)
    # Assign the minimum and maximum values of different parameters
    # Degreadation Rate
    prange_df.loc[prange_df["Parameter"].str.contains("Deg_"), "Minimum"] = 0.1
    prange_df.loc[prange_df["Parameter"].str.contains("Deg_"), "Maximum"] = 1.0
    # Fold Change Values
    # For Activation Fold changes
    prange_df.loc[prange_df["Parameter"].str.contains("ActFld_"), "Minimum"] = 1.0
    prange_df.loc[prange_df["Parameter"].str.contains("ActFld_"), "Maximum"] = 100.0
    # For Inhibition Fold changes
    prange_df.loc[prange_df["Parameter"].str.contains("InhFld_"), "Minimum"] = 0.01
    prange_df.loc[prange_df["Parameter"].str.contains("InhFld_"), "Maximum"] = 1.0
    # Hills Coefficient
    prange_df.loc[prange_df["Parameter"].str.contains("Hill_"), "Minimum"] = 1.0
    prange_df.loc[prange_df["Parameter"].str.contains("Hill_"), "Maximum"] = 6.0
    # Find the threshold values of the the edges and calulate the amplification factor
    # As production rate also depeneds on amplification factor the range is assigned in this loop
    # Iterate through the source nodes and get their threshold ranges
    for sn in set(target_nodes + source_nodes):
        # As if the node is source node, its amplification value and threshold values will change if the minimum threshold value is below 0.01, assign the ranges speprately after calcualting the threshold minimum
        if sn in source_nodes:
            # print(sn)
            median_thr_val = get_thr_ranges(sn, topo_df, num_params)
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
                1.0 * amplify_val
            )
            prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Maximum"] = (
                100.0 * amplify_val
            )
        else:
            # Set the Production Rates (Amplify factor = 1) for the target node
            prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Minimum"] = 1.0
            prange_df.loc[prange_df["Parameter"] == f"Prod_{sn}", "Maximum"] = 100.0
    # print(prange_df)
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


if __name__ == "__main__":
    # Specify the topo file folder
    topo_folder = "TOPOS"
    # Get the list of all the topology files in the topo folder
    topo_list = sorted(glob(f"{topo_folder}/TSSA*.topo"))
    for t in topo_list[:1]:
        # Parse the topo file to get the dataframe
        topo_df = parse_topos(t)
        print("\n")
        print(t)
        # print(topo_df)
        # Generate parameter range dataframe
        prange_df = get_param_range_df(topo_df)
        print(prange_df)
        # Generate the parameter dataframe
        param_df = gen_param_df(prange_df)
        print(param_df)
        # Get the mean and standard deviation of the parameter dataframe columns
        print(param_df.mean())
        print(param_df["InhFld_A_B"].std())
        print(param_df["InhFld_A_B"].mean())
        print(param_df["InhFld_A_B"].median())
