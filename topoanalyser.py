import pandas as pd
import networkx as nx
import numpy as np
import glob
import os
import math
import rustworkx as rx
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)

# Function to convert a topo file to a networkx networkx
def convert_topo_netx(topo_path):
    """
    Convert a topoglofy file into a NetworkX graph.

    Args:
        topo_path (str): The path to the topoglofy file.

    Returns:
        net (networkx.DiGraph): The converted NetworkX graph.
    """
    # Read the topoglofy file has columns [Source, Target, Type]
    topo = pd.read_csv(topo_path, sep=r"\s+")
    # Replace 2 (inhibitions) with -1 values
    topo.replace({2: -1}, inplace=True)
    # Convert the dataframe into a networkx graph
    net = nx.from_pandas_edgelist(
        topo,
        source="Source",
        target="Target",
        edge_attr="Type",
        create_using=nx.DiGraph,
    )
    return net


# Function to convert the topodf to a networkx network
def convert_topodf_netx(topo_df):
    # Replace 2 (inhibitions) with -1 values
    topo_df.replace({2: -1}, inplace=True)
    # Convert the dataframe into a networkx graph
    net = nx.from_pandas_edgelist(
        topo_df,
        source="Source",
        target="Target",
        edge_attr="Type",
        create_using=nx.DiGraph,
    )
    return net


# Function to convert the networx network to a pandas adjacency matrix
def _to_pandas_adj(net, order=None):
    """
    Convert a networkx graph to a pandas adjacency matrix.

    Parameters:
        net (networkx.Graph): The input networkx graph.
        order (list): The order of nodes in the adjacency matrix. If None, the nodes will be sorted from the the nodes list output.

    Returns:
        pandas.DataFrame: The pandas adjacency matrix.
    """
    if order is None:
        node_list = sorted(net.nodes)
    else:
        node_list = order
    return nx.to_pandas_adjacency(net, nodelist=node_list, weight="Type")


# Function to convert networkx graph into an adjacency matrix (numpy) ans return the list of nodes
def _to_numadj(net, order=None):
    """
    Convert a networkx graph into a numpy 2D array representing the adjacency matrix.

    Parameters:
    net (networkx.Graph): The input graph.
    order (list, optional): The desired order of nodes in the adjacency matrix. If not provided, the nodes will be sorted in ascending order.

    Returns:
    tuple: A tuple containing the adjacency matrix as a numpy 2D array and the list of nodes in the same order as the adjacency matrix.
    """
    # Get the list of nodes in the graph and sort (for consistancy)
    if order is None:
        node_list = sorted(net.nodes)
    else:
        node_list = order
    # Convert into a numpy 2D array with the same node order as the node list
    net_adj = nx.to_numpy_array(net, weight="Type", nodelist=node_list)
    return net_adj, node_list


# Function to calculate the influence matrix from the pandas adjacency matrix
def calc_infl_matrix(net, pathlen=100):
    """
    Calculate the influence matrix of a network.

    Parameters:
    - net: NetworkX graph object
        The input network.
    - pathlen: int, optional
        The maximum path length to consider when calculating influence. Default is 100.

    Returns:
    - pandas.DataFrame
        The influence matrix as a pandas DataFrame, with nodes as both row and column indices.
    """
    # Get the numpy 2D array of the networkx network
    net_adj, node_order = _to_numadj(net)
    # The influence calculation algorithm
    # Shallow copy the numpy adj ,matrix
    inf_mat = net_adj.copy()
    max_mat = net_adj.copy()
    # Convert the non-zero elements in the matrix to 1 to get Max Matrix
    max_mat[max_mat != 0] = 1.0
    # path_length = 100
    # path_length = nx.diameter(tpG.to_undirected())
    # Take powers of the metix to get the influence matrix and update the Max Martix
    for i in range(2, pathlen + 1):
        infpow = np.linalg.matrix_power(net_adj, i).astype(float)
        maxpow = np.linalg.matrix_power(max_mat, i).astype(float)
        inf_mat = inf_mat + (
            np.divide(infpow, maxpow, out=np.zeros_like(infpow), where=maxpow != 0)
            / (i)
        )
    # Normalise by the path length
    inf_mat = inf_mat / sum(np.reciprocal(np.arange(1, pathlen + 1), dtype=float))
    # Convert into a padas dataframe and return
    return pd.DataFrame(inf_mat, index=node_order, columns=node_order)


# Check for self edge
def _get_self_edge(i, path_li_rw, get_edge_attr):
    """
    Get self edges for a given node and append to the path list.

    Parameters:
    i (int): The node index.
    path_li_rw (list): The list to store the self edges.
    get_edge_attr (dict): The dictionary containing edge attributes.

    Returns:
    None
    """
    # Check if self loops are present
    try:
        sloop_type = get_edge_attr[i, i]
        path_li_rw.append([i, i, f"{i}-{i}", sloop_type, 1])
    except KeyError:
        pass


# Find self simple paths (only the first element is repeated 1st-last position) from the already found paths to populate the diagonals (i.e. self loops)
def _get_self_infl(net, path_li, get_edge_attr):
    """
    Check if there is a self-influence path in the network.

    Args:
        net: The network object.
        path_li: A list of paths in the network.
        get_edge_attr: A function to get edge attributes.

    Returns:
        None
    """
    for pth in path_li:
        # Find out if there is an edge from the last node to the first node of the path
        if pth[-1] != pth[0]:
            try:
                get_edge_attr[pth[-1], pth[0]]
                # print(pth + [pth[0]])
                path_li.append(pth + [pth[0]])
            except KeyError:
                pass


# Function to calculate the simple path infulence matrix
def calc_simple_pthdf(net, cutoff_len=None, parallel=False):
    """
    Calculate the simple paths in a network and return the path information as a DataFrame i.e. path influence matrix.

    Parameters:
    net (NetworkX graph): The input network.
    cutoff_len (int, optional): The maximum length of paths to consider. Defaults to None.

    Returns:
    pandas.DataFrame: The DataFrame containing the path information with columns:
                      "Source", "Target", "Path", "NetInfl", "PathLen".
                      The DataFrame is sorted by "PathLen", "Source", and "Target".
    """

    # Get the node lists
    node_li = list(net.nodes)
    # Define a function to get the edge attributes of a list of edges (which are tuples)
    get_edge_attr = nx.get_edge_attributes(net, "Type")
    # A list of list to store the path infromation
    path_li_rw = []
    # Define the cutoff length if not given
    if cutoff_len is None:
        cutoff_len = len(node_li) + 1
    # Loop over the nodes to find all the paths from the node to every other node in the network
    for i in enumerate(node_li):
        # Make the list of other nodes
        other_nodes = node_li[: i[0]] + node_li[i[0] + 1 :]
        # Check is any of the nodes do not have a path coming from this node and remove them
        other_nodes = [n for n in other_nodes if nx.has_path(net, i[1], n)]
        # Get the path list of the simple paths
        # path_li = []
        # for n in other_nodes:
        #     print(n)
        #     p = nx.all_simple_paths(net, i[1], n)
        #     print(list(p))
        path_li = list(
            p for p in nx.all_simple_paths(net, i[1], other_nodes, cutoff=cutoff_len)
        )
        # Check for self loop
        _get_self_infl(net, path_li, get_edge_attr)
        # Process the paths so that paths of different lengths are grouped
        for pth in path_li:
            # Get the list of edge values from the path
            inf_val = [get_edge_attr[pth[e], pth[e + 1]] for e in range(len(pth) - 1)]
            # Append the values to the path information list
            path_li_rw.append(
                [pth[0], pth[-1], "-".join(pth), math.prod(inf_val), len(inf_val)]
            )
    # Find the self-loops
    for i in node_li:
        _get_self_edge(i, path_li_rw, get_edge_attr)
    # Return the dataframe having path information
    return (
        pd.DataFrame(
            data=path_li_rw, columns=["Source", "Target", "Path", "NetInfl", "PathLen"]
        )
        .sort_values(by=["PathLen", "Source", "Target"])
        .reset_index(drop=True)
    )


# Rust function to get all simple paths between all the pairs of nodes in a network
def calc_simple_pthdf_rs(net, cutoff_len=None):
    """
    Get the simple paths between pairs of nodes in a network. Add the line os.environ["RAYON_NUM_THREADS"] = "<numCores>" to the code to set the number of threads to use mutithreading.

    Args:
        net (networkx.Graph): The input network.
        cutoff_len (int, optional): The maximum length of paths to consider. Defaults to None.

    Returns:
        pandas.DataFrame: A dataframe containing the path information, including the source node,
        target node, path string, network influence, and path length.

    """
    # Get the dictionary mapping of the keys as node indices and values as the node names
    node_map = {i: n for i, n in enumerate(net.nodes)}
    # Function to convert the networkx graph to a rustworkx graph
    rsnet = rx.networkx_converter(net, keep_attributes=True)
    # Define a function to get the edge attributes of a list of edges (which are tuples)
    get_edge_attr = nx.get_edge_attributes(net, "Type")
    # A list of lists to store the path information
    path_li_rw = []

    # Get all the simple paths between pairs of nodes
    all_paths = rx.all_pairs_all_simple_paths(rsnet, cutoff=cutoff_len)
    for i in all_paths:
        if len(all_paths[i]) > 0:
            path_li = []
            for j in all_paths[i]:
                for k in all_paths[i][j]:
                    path_li.append([node_map[n] for n in k])
            # Check for self edge
            _get_self_infl(net, path_li, get_edge_attr)
            # Process the paths so that paths of different lengths are grouped
            for pth in path_li:
                # Get the list of edge values from the path
                inf_val = [
                    get_edge_attr[pth[e], pth[e + 1]] for e in range(len(pth) - 1)
                ]
                path_li_rw.append(
                    [pth[0], pth[-1], "-".join(pth), np.prod(inf_val), len(inf_val)]
                )

    # Find the self-loops
    for i in net.nodes:
        _get_self_edge(i, path_li_rw, get_edge_attr)

    # Return the dataframe having path information
    return (
        pd.DataFrame(
            data=path_li_rw, columns=["Source", "Target", "Path", "NetInfl", "PathLen"]
        )
        .sort_values(by=["PathLen", "Source", "Target"])
        .reset_index(drop=True)
    )


# Function to calculate max matrix
def _get_max_infmat(pdf):
    """
    Calculate the maximum influence matrix from a given pandas DataFrame.

    Parameters:
    pdf (pandas.DataFrame): The input DataFrame containing the influence data.

    Returns:
    pandas.DataFrame: The maximum influence matrix.
    """
    # Convert all the values in the NetInfl column to postive
    pdf["NetInfl"] = pdf["NetInfl"].abs()
    # Add all the paths to get effective infleunce value
    pth_infmat = (
        pdf.groupby(["Source", "Target"])["NetInfl"].apply(np.sum).reset_index()
    )
    # Placeholder matrix for the influence matrix
    pth_infmat = nx.from_pandas_edgelist(
        pth_infmat,
        source="Source",
        target="Target",
        edge_attr="NetInfl",
        create_using=nx.DiGraph,
    )
    # Create an influence matrix from the path dataframe ans return
    return nx.to_pandas_adjacency(pth_infmat, weight="NetInfl")


# Get the net influence when there are multiple paths present
# Taken from : https://stackoverflow.com/questions/36271413/pandas-merge-nearly-duplicate-rows-based-on-column-value
def _get_net_multipaths(pdf):
    """
    Calculate the influence matrix for multiple paths in a network.

    Parameters:
    pdf (pandas.DataFrame): The input dataframe containing the paths and their influence values.

    Returns:
    pandas.DataFrame: The influence matrix representing the cumulative influence of multiple paths.
    """

    # Add all the paths to get effective infleunce value
    pth_infmat = (
        pdf.groupby(["Source", "Target"])["NetInfl"].apply(np.sum).reset_index()
    )
    # Placeholder matrix for the influence matrix
    pth_infmat = nx.from_pandas_edgelist(
        pth_infmat,
        source="Source",
        target="Target",
        edge_attr="NetInfl",
        create_using=nx.DiGraph,
    )
    # Create an influence matrix from the path dataframe and return
    return nx.to_pandas_adjacency(pth_infmat, weight="NetInfl")


# Function to get the fraction of postive interactions
# Ranges from 0 - 1. 1 is only positive influence and 0 being only negetive influence. 0.5 means the positive and negetive influnce are the equivalent.
def _get_positive_influence(infli):
    """
    Calculate the percentage of positive influence in a given list.

    Parameters:
    infli (list): A list of influence values.

    Returns:
    float: The percentage of positive influence.
    """
    pos_influence = np.sum(np.array(infli) > 0) / len(infli)
    if pos_influence != 0.5:
        return pos_influence
    else:
        return pos_influence - 0.001


# Function to get the coherence matrix
def get_coherence_matrix(pthdf, net_nodes=None, centered=True):
    """
    Calculate the coherence matrix based on the given path dataframe.

    Parameters:
    pthdf (DataFrame): The path dataframe containing the source, target, and net influence columns.
    net_nodes (list, optional): The list of all nodes in the network. If not provided, it will be inferred from the path dataframe.
    centered (bool, optional): Whether to center the coherence values between -1 and 1. Default is True.

    Returns:
    DataFrame: The coherence matrix with nodes as rows and columns, and coherence values as the matrix elements.
    """

    if net_nodes is None:
        # Get a list of all the nodes
        net_nodes = list(set(pthdf["Source"]).union(set(pthdf["Target"])))
    # Apply groupby on the entire path dataframe to  calcculate the pairwise fraction of positive feedback loops
    coh_df = (
        pthdf.groupby(["Source", "Target"])["NetInfl"]
        .apply(_get_positive_influence)
        .reset_index()
    )
    if centered:
        # Substract 0.5 from the fractions to get if its negetive or postive influence dominant
        # 0 means balances positive and negetive influence
        # Then divide by 0.5 to make it between -1 to 1
        coh_df["NetInfl"] = ((coh_df["NetInfl"]) - 0.5) / (0.5)
        # Convert the dataframe into a matrix
        # Add combinatoins of source and target which are not present
        coh_df = (
            coh_df.set_index(["Source", "Target"])
            .reindex(pd.MultiIndex.from_product([net_nodes, net_nodes]))
            .reset_index()
            .rename(columns={"level_0": "Source", "level_1": "Target"})
        )
        coh_df = coh_df.pivot(
            index="Source", columns="Target", values="NetInfl"
        ).fillna(0.0)
    else:
        # Convert the dataframe into a matrix
        # Add combinatoins of source and target which are not present
        coh_df = (
            coh_df.set_index(["Source", "Target"])
            .reindex(pd.MultiIndex.from_product([net_nodes, net_nodes]))
            .reset_index()
            .rename(columns={"level_0": "Source", "level_1": "Target"})
        )
        coh_df = coh_df.pivot(
            index="Source", columns="Target", values="NetInfl"
        )  # .fillna(0.5)
    return coh_df


# Get the complete matrix (with nodes which do not appear in the pdf but are present in the network)
# Takes the infmat and adds rows and columns filled with zeroes to it
def _fill_infmat(infmat, net_nodes):
    """
    Fill the information matrix with missing nodes (from path dataframe) as rows and columns.

    Parameters:
    infmat (pd.DataFrame): The information matrix.
    net_nodes (list): The list of network nodes.

    Returns:
    pd.DataFrame: The filled information matrix.
    """
    if not infmat.empty:
        # Get the list of missing nodes
        missing_nodes = [n for n in net_nodes if n not in infmat.columns]
        # Add the nodes to the infmat as rows and columns
        if missing_nodes:
            for n in missing_nodes:
                infmat[n] = 0.0
                infmat.loc[n] = [0.0] * len(infmat.columns)
        return infmat
    else:
        # Create a dataframe with just zeroes
        return pd.DataFrame(
            np.zeros((len(net_nodes), len(net_nodes))),
            columns=net_nodes,
            index=net_nodes,
        )


# Make the simple influence matrix form the paths DataFrame
def get_pth_infmat(pthdf, pathlen, net_nodes):
    """
    Get the path influence matrix and the maximum influence matrix for a given path length.

    Args:
        pthdf (DataFrame): The dataframe containing the paths.
        pathlen (int): The desired path length.
        net_nodes (list): The list of network nodes.

    Returns:
        tuple: A tuple containing the path influence matrix and the maximum influence matrix.
    """
    # Subset the paths of the path length
    pdf = pthdf[pthdf["PathLen"] == pathlen][["Source", "Target", "NetInfl"]]
    # Check if the dataframe is empty
    if not pdf.empty:
        # Get the net effect of paths if multiple of them are present from one node to another
        p_infmat = _fill_infmat(_get_net_multipaths(pdf), net_nodes)
        p_infmat = p_infmat.reindex(columns=net_nodes, index=net_nodes)
        # Get the absolute path dataframe
        m_infmat = _fill_infmat(_get_max_infmat(pdf), net_nodes)
        m_infmat = m_infmat.reindex(columns=net_nodes, index=net_nodes)
        return p_infmat, m_infmat
    else:
        # Create a dataframe with just zeroes
        p_infmat = _fill_infmat(pdf, net_nodes)
        return p_infmat, p_infmat


# Function to plot the path influence matrix as a heatmap
def _plt_pth_infmat(pth_infmat, pthlen, net_name):
    """
    Plot the path influence matrix.

    Args:
        pth_infmat (numpy.ndarray): The path influence matrix.
        pthlen (int): The length of the path.
        net_name (str): The name of the network.

    Returns:
        None
    """
    # Set the figure size
    plt.figure(figsize=(10, 8))
    # Set the colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True, center="dark")
    # Plot the path influence matrix
    ax = sns.heatmap(pth_infmat, cmap=cmap, center=0, annot=True, fmt=".2f")
    # Set the title to show the network name and the path length
    ax.set_title(f"{net_name} - PathLen : {pthlen}")
    # Tight Layout
    plt.tight_layout()
    # Show the heatmap
    plt.show()


# Function to plot the path influence matrix as a heatmap
def _plt_infmat(infmat, net_name, teams_comp=None, mat_type="coh"):
    """
    Plot the influence matrix heatmap.

    Args:
        infmat (pandas.DataFrame): The influence matrix.
        net_name (str): The name of the network.
        teams_comp (dict, optional): Dictionary containing team compositions. Defaults to None.
        mat_type (str, optional): Type of matrix to plot. Can be "adj" (adjacency), "inf" (influence), or "coh" (coherence). Defaults to "coh".

    Returns:
        None
    """
    # Set the figure size
    plt.figure(figsize=(12, 8))
    # Set the colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    # Node order the same as the teams which have been identified
    if teams_comp is None:
        teams_comp_li = [o for v in create_groups(infmat).values() for o in v]
    else:
        teams_comp_li = [o for v in teams_comp.values() for o in v]
    # Reorder the influence matrix according to the node order
    infmat = infmat.reindex(index=teams_comp_li[::-1], columns=teams_comp_li)
    # Plot the heatmap
    ax = sns.heatmap(
        infmat,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        linewidth=0.7,
        linecolor="black",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 35 / np.sqrt(len(teams_comp_li))},
    )
    # Highlight sections based on team composition
    if teams_comp is None:
        # Define the highlight rectangles
        for i, team in enumerate(create_groups(infmat).values()):
            min_indx = teams_comp_li.index(team[0])
            max_indx = len(teams_comp_li) - teams_comp_li.index(team[-1]) - 1
            width = len(team)
            ax.add_patch(
                plt.Rectangle(
                    (min_indx, max_indx),
                    width,
                    width,
                    fill=False,
                    color="black",
                    alpha=0.6,
                    clip_on=False,
                    lw=2,
                )
            )
    else:
        # Define the highlight rectangles
        for i, team in enumerate(teams_comp.values()):
            min_indx = teams_comp_li.index(team[0])
            max_indx = len(teams_comp_li) - teams_comp_li.index(team[-1]) - 1
            width = len(team)
            ax.add_patch(
                plt.Rectangle(
                    (min_indx, max_indx),
                    width,
                    width,
                    fill=False,
                    color="black",
                    alpha=0.6,
                    clip_on=False,
                    lw=3,
                )
            )
    # Change title to show adjacency or influence matrix
    if mat_type == "adj":
        ax.set_title(f"Adjacency: {net_name}")
    elif mat_type == "inf":
        ax.set_title(f"Influence: {net_name}")
    else:
        ax.set_title(f"Coherence: {net_name}")
    # Tight Layout
    plt.tight_layout()
    # Show/Save figure the heatmap
    if mat_type == "adj":
        plt.savefig(f"Plots/AdjPlots/{net_name}_Adj.png", dpi=300)
    elif mat_type == "inf":
        plt.savefig(f"Plots/InflPlots/{net_name}_Infl.png", dpi=300)
    else:
        plt.savefig(f"Plots/CohPlots/{net_name}_Coh.png", dpi=300)
    plt.close()


# Function to calculate team strength of the individual teams
def calc_team_strn(inf_mat, teams_comp, zeroes=False):
    """
    Calculate the team strength values for individual teams based on the subset influence matrix.

    Parameters:
    inf_mat (pandas.DataFrame): The subset influence matrix of the team.
    teams_comp (dict): A dictionary containing the team compositions.

    Returns:
    dict: A dictionary containing the team strength values for individual teams.
    """
    if not zeroes:
        # Create a empty dictionary to store the team strength values of the individual teams
        indv_team_strn = {}
        # Iterate through the subset Influence matrix of the team and get the individual team strength
        for i in teams_comp.keys():
            indv_team_strn[i] = (
                inf_mat.loc[teams_comp[i], teams_comp[i]].to_numpy().mean()
            )
        return indv_team_strn
    else:
        # Create a empty dictionary to store the team strength values of the individual teams
        indv_team_strn = {}
        # Iterate through the subset Influence matrix of the team and get the individual team strength
        for i in teams_comp.keys():
            # Get the subset influence matrix of the team
            sub_infmat = inf_mat.loc[teams_comp[i], teams_comp[i]]
            # Get the mean of the values in the subset influence matrix excluding the zeroes
            indv_team_strn[i] = sub_infmat[sub_infmat != 0].to_numpy().mean()
        return indv_team_strn


# Function to calcluate the team strength of the entire team
def calc_inf_teamstrn(inf_mat, zeroes=False):
    """
    Calculate the average absolute value of the elements in the influence matrix i.e. team strength.

    Parameters:
    inf_mat (pandas.DataFrame): The input matrix.

    Returns:
    float: The average absolute value of the elements in the matrix.
    """
    if not zeroes:
        # Return the mean of the absolute values of the influence matrix
        return inf_mat.abs().mean().mean()
    else:
        # Return the mean of the absolute values of the influence matrix excluding the zeroes
        return inf_mat[inf_mat != 0].abs().mean().mean()


# Find teams, but better
def create_groups(df):
    """
    Create groups of node based on a condition that no one in the group inhibits or is inhibited by another node.

    Args:
        df (DataFrame): The input DataFrame having the relationships between node (Matrix - Influence or Coherence).

    Returns:
        dict: A dictionary where the keys are group numbers and the values are lists of node in each group.
    """
    # Initialize an empty list to store the groups
    groups = []
    # Iterate through each node
    for i in df.columns:
        # Flag to indicate if the node is added to any existing group
        found_group = False
        # Iterate through each existing group
        for group in groups:
            # Check if adding the node to this group maintains the condition
            # that no one in the group dislikes or is disliked by the node
            if (df.loc[group + [i], group + [i]].values >= 0).all():
                # If conditions are met, add the node to the group
                group.append(i)
                # Set the flag to indicate the node has been added to a group
                found_group = True
                break
        # If the node couldn't be added to any existing group, create a new group for it
        if not found_group:
            groups.append([i])
        # Sort the groups according to thier length to favour the growth of larger groups
        groups = sorted(groups, key=len, reverse=True)
    # Check is any nodes are stuck in suboptimal groups
    for l in range(len(groups)):
        current_group = groups.copy()
        for r in groups[1::-1]:
            for n in r:
                for g in groups[: groups.index(r)]:
                    # Check if the node is suitable to go to this group
                    if (df.loc[g + [n], g + [n]].values >= 0).all():
                        # Append the node to the group
                        g.append(n)
                        groups[groups.index(r)].remove(n)
                        # print([g, n, r])
                        break
        if groups == current_group:
            break
    # Convert list of lists to dictionary with group number as keys
    groups_dict = {}
    # for i, group in enumerate(sorted(rel_groups + [i for i in groups if len(i) == 1]), start=1):
    # for i, group in enumerate(sorted(rel_groups), start=1):
    for i, group in enumerate(sorted(groups, key=len, reverse=True), start=1):
        groups_dict[str(i)] = group
    # Return the list of groups
    return groups_dict


# Function to create the path_df, coherence mat and group composition disctionary of a network
def get_cohmat_groupcomp(
    net,
    net_name,
    cutoff_len=None,
    ptdf_path=None,
    coh_path=None,
    tms_path=None,
):
    # Check if any of the paths are None
    if ptdf_path is None or coh_path is None or tms_path is None:
        # Terminate the function and report that paths are not provided
        raise ValueError(
            "Path dataframe, coherence matrix, and team composition paths must be provided."
        )
    else:
        # get the path dataframe
        path_df = calc_simple_pthdf_rs(net, cutoff_len=cutoff_len)
        # path_df = calc_simple_pthdf(net, cutoff_len=cutoff_len, parallel=parallel)
        # Save the path dataframe
        path_df.to_csv(f"{ptdf_path}/{net_name}_path.csv", index=False)
        # get the coherence matrix
        coh_mat = get_coherence_matrix(path_df)
        # Save the coherence matrix
        coh_mat.to_csv(f"{coh_path}/{net_name}_coh.csv")
        # Get the team composition
        tms_comp = create_groups(coh_mat)
        # Save the team composition
        with open(f"{tms_path}/{net_name}_teamcomp.txt", "w") as f:
            for k in tms_comp.keys():
                f.write(f"{k}: {','.join(tms_comp[k])}\n")
        return None


if __name__ == "__main__":
    # Get the number of cores to use
    numC = np.floor(os.cpu_count() - 3)
    # Make a folder to save adjacency plots
    if not os.path.exists("Plots/AdjPlots"):
        os.makedirs("Plots/AdjPlots")
    # Make a folder to save coherence plots
    if not os.path.exists("Plots/CohPlots"):
        os.makedirs("Plots/CohPlots")
    # Make a folder to save influence plots
    if not os.path.exists("Plots/InflPlots"):
        os.makedirs("Plots/InflPlots")
    # Make a folder to save the COhenrence matrix
    if not os.path.exists("CohResults/CohMats"):
        os.makedirs("CohResults/CohMats")
    # Make a folder to save the Path dataframes
    if not os.path.exists("CohResults/PathDFs"):
        os.makedirs("CohResults/PathDFs")
    # Make a folder to save the teams compositions
    if not os.path.exists("CohResults/TeamComps"):
        os.makedirs("CohResults/TeamComps")
    # Iterate through the topo files
    for i in sorted(glob.glob("ArtCohNet/*.topo")[:3]):
        print(i)
        net = convert_topo_netx(i)
        print(net.nodes)
        print(len(net.nodes))
        # Plot the in vs oput degreee matrix of the network
        # indegli = [i[1] for i in net.in_degree(net.nodes)]
        # outdegli = [i[1] for i in net.out_degree(net.nodes)]
        # plt.scatter(indegli, outdegli)
        # plt.show()
        # Get the path dataframe, coherence matrix and team composition
        # path_df, coh_mat, tms_comp = get_cohmat_groupcomp(
        #     net,
        #     cutoff_len=len(net.nodes),
        #     net_name=i.split("/")[-1].replace(".topo", ""),
        #     parallel=False,
        #     ptdf_path="CohResults/PathDFs",
        #     coh_path="CohResults/CohMats",
        #     tms_path="CohResults/TeamComps",
        # )
        # # Save the path dataframe
        # path_df.to_csv(
        #     f"PathDFs/{i.split('/')[-1].replace('.topo', '_path.csv')}", index=False
        # )
        # # Save the coherence matrix
        # coh_mat.to_csv(f"CohMats/{i.split('/')[-1].replace('.topo', '_coh.csv')}")
        # # Save the team composition
        # with open(
        #     f"TeamComps/{i.split('/')[-1].replace('.topo', '_teamcomp.txt')}", "w"
        # ) as f:
        #     for k in tms_comp.keys():
        #         f.write(f"{k}: {','.join(tms_comp[k])}\n")
        # for pt_len in range(1, len(net.nodes)+1):
        #     print(f"# PathLen = {pt_len}")
        #     p_df = pth_df[pth_df["PathLen"] <= pt_len]
        #     # print(p_df)
        #     cdf = get_coherence_matrix(p_df, net.nodes)
        #     # print(cdf)
        #     # print(create_groups(cdf))
        #     # print(calc_inf_teamstrn(cdf))
        #     # if pt_len == len(net.nodes):
        #     # print(calc_team_strn(cdf, tms_comp))
        #     # _plt_infmat(cdf, i)
        # # print(get_coherence_matrix(pth_df))
        # # Plot the adjacency
        # _plt_infmat(_to_pandas_adj(net), i.split("/")[-1].replace(".topo",""), tms_comp, mat_type="adj")
        # # Plot the influence matrix
        # _plt_infmat(get_coherence_matrix(pth_df), i.split("/")[-1].replace(".topo", ""), tms_comp, mat_type="coh")
        # # Plot the influence matrix
        # _plt_infmat(calc_infl_matrix(net, pathlen=10), i.split("/")[-1].replace(".topo", ""), mat_type="inf")
        # # Show Plot
        # print("\n")
# print(net)
