import pandas as pd
import numpy as np
import glob
from itertools import cycle


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


# Function to convert the dataframe into a numpy matrix
def edgelist_to_matrix(df):
    """
    Convert the given dataframe to a numpy matrix.

    Parameters:
    df (pandas.DataFrame): The dataframe to be converted. Has coulmns of "Source", "Target" and "Type".

    Returns:
    numpy.ndarray: The converted numpy matrix.
    list: The list of nodes in the network.
    """
    # Replace all the 2s in the Type column with -1
    df["Type"] = df["Type"].replace(2, -1)
    # Make the edgelist into a matrix
    topo_matrix = df.pivot(index="Source", columns="Target", values="Type")
    # This pivoted matrix will have NaNs for missing edges, make them 0
    topo_matrix = topo_matrix.fillna(0)
    # print(topo_matrix)
    # Get the list of the nodes pivoted dataframe
    node_list = topo_matrix.index.to_list()
    # Convert the dataframe to a numpy matrix
    topo_matrix = topo_matrix.to_numpy()
    # Return the node list and the topo matrix
    return topo_matrix, node_list


# Functions to decide the nodes are on or off based on the value
# Function for 0/1 flip, if values is less than 0, return 0, if values are more than 0 return 1, if 0 return the value
def on_off_flip_onezero(value):
    """
    Function to decide the nodes are on or off based on the value.

    Parameters:
    value (int): The value to be checked.

    Returns:
    int: The value after the check. If the value is less than 0, return 0, if values are more than 0 return 1, if 0 return the value.
    """
    return 0 if value < 0 else 1 if value > 0 else value


# Function for -1/1 flip, if values is less than 0, return -1, if more than 0 return 1, if 0 return the value
def on_off_flip_minusplus(value):
    """
    Function to decide the nodes are on or off based on the value.

    Parameters:
    value (int): The value to be checked.

    Returns:
    int: The value after the check. If the value is less than 0, return -1, if values are more than 0 return 1, if 0 return the value.
    """
    return -1 if value < 0 else 1 if value > 0 else value


# Function to flip the states
def flip_states(new_state, prev_state, flip_function):
    """
    Flips the states of a given new state array based on a specified flip function.

    Parameters:
    new_state (ndarray): The new state array.
    prev_state (ndarray): The previous state array.
    flip_function (str): The flip function to be applied. Valid options are "onezero", "minusplus", and "sigmoid".

    Returns:
    ndarray: The flipped state array.

    Raises:
    ValueError: If an invalid flip function is provided.

    """
    # Get indices with 0 in the new state and replace them with the previous state values
    zero_indices = np.where(new_state == 0)
    # Run the flip function on the new state
    new_state = np.vectorize(flip_function)(new_state)
    # Update the new state with the previous state values where the new state is 0
    new_state[zero_indices] = prev_state[zero_indices]
    # Return the flipped state
    return new_state


# Function to run the ising model in synchronous mode
def update_ising_sync(topo_matrix, prev_state, flip_function):
    """
    Update the Ising model in synchronous mode.

    Parameters:
    topo_matrix (ndarray): The numpy matrix of the network.
    prev_state (ndarray): The previous state of the network.
    flip_function (str): The flip function to be applied. Valid options are "onezero", "minusplus", and "sigmoid".

    Returns:
    ndarray: The updated state of the network.
    """
    # Calculate the new state
    new_state = np.dot(topo_matrix, prev_state)
    # Run the flip function on the new state
    new_state = flip_states(new_state, prev_state, flip_function)
    # Return the new state
    return new_state


# Function to run the ising model in asynchronous mode
def update_ising_async(topo_matrix, prev_state, node_to_update, flip_function):
    """
    Update the Ising model in asynchronous mode.

    Parameters:
    topo_matrix (ndarray): The numpy matrix of the network.
    prev_state (ndarray): The previous state of the network.
    node_to_update (int): The node to be updated.
    flip_function (str): The flip function to be applied. Valid options are "onezero", "minusplus", and "sigmoid".

    Returns:
    ndarray: The updated state of the network.
    """
    # Calculate the new state by multiplying the row of the node to be updated with the previous state
    updated_node_state = np.dot(topo_matrix[node_to_update], prev_state)
    # Replace the node to be updated in the previous state with the new state
    new_state = np.copy(prev_state)
    new_state[node_to_update] = updated_node_state
    # Run the flip function on the new state
    new_state = flip_states(new_state, prev_state, flip_function)
    return new_state


# Function to check if the state has converged and return break the loop
def check_convergence(prev_state, new_state, step, flip_function, simulation_states):
    """
    Check if the simulation has converged by comparing the previous state with the new state.

    Parameters:
    prev_state (numpy.ndarray): The state of the system at the previous step.
    new_state (numpy.ndarray): The current state of the system.
    step (int): The current step number in the simulation.
    flip_function (str): The name of the flip function used in the simulation.
    simulation_states (list): A list to store the states of the system at each step.

    Returns:
    bool: True if the simulation has converged, False otherwise.
    """
    if np.array_equal(prev_state, new_state) and step > 0:
        # print(f"Converged at step {step}")
        # For cases like [-1, -1] output would be the same as the input state and they need to be converted to [0,0] (as they will be updated to previous state and become [-1, -1] again)
        # This would be the case for onezero flip function
        if flip_function == "onezero":
            new_state = np.vectorize(on_off_flip_onezero)(new_state)
            # # Add the state to the simulation states
            # # Add step as the first element to the new_state and print
            # simulation_states.append(np.insert(new_state, 0, step))
        return True


# Function to convert the simulation states to a pandas dataframe
def generate_solution_df(simulation_states, node_list):
    """
    Convert the simulation states to a pandas dataframe.

    Parameters:
    simulation_states (list): The list of states of the nodes.
    node_list (list): The list of nodes in the network.

    Returns:
    pandas.DataFrame: The converted dataframe.
    """
    # Convert the simulation states to a pandas dataframe
    simulation_states = pd.DataFrame(simulation_states, columns=["Step"] + node_list)
    # Replace step -1 on the first row with "Initial" and the last step with "Steady"
    simulation_states["Step"] = simulation_states["Step"].replace(-1, "InitialState")
    # Access the last row of the dataframe and check if steady state is reached (-2) else replace with Unsteady
    if simulation_states.iloc[-1, 0] == -2:
        simulation_states.iloc[-1, 0] = "SteadyState"
    else:
        simulation_states.iloc[-1, 0] = "Unsteady"
    # Return the simulation states dataframe
    return simulation_states


# Function to generate the final state dictionary
def generate_final_state_dict(simulation_states, node_list):
    """
    Generate the final state dictionary from the simulation states.

    Parameters:
    simulation_states (list): The list of states of the nodes.
    all_boolnodes (dict): A dictionary mapping state variable names to their indices.

    Returns:
    dict: The final state dictionary.
    """
    # Get the last simulation state
    # astype(int) is used to convert the numpy array to int
    # tolist() is used to convert the numpy values to python values
    last_state = simulation_states[-1][1:].astype(int).tolist()
    # Make a dictionary of the final state
    last_state = dict(zip(node_list, last_state))
    # Check if the last step is the steady state if not make it as UnSteady
    if simulation_states[-1][0] != -2:
        last_state["StateType"] = "UnSteady"
    else:
        last_state["StateType"] = "SteadyState"
    # Add time step to the dictionary
    last_state["Step"] = len(simulation_states) - 1
    # return the dictionary
    return last_state


def run_ising_sync(
    topo_file,
    max_steps=None,
    initial_state=None,
    flip_function="onezero",
    time_series=False,
):
    # Setting the flip function based on the input
    if flip_function == "onezero":
        flip_function = on_off_flip_onezero
    elif flip_function == "minusplus":
        flip_function = on_off_flip_minusplus
    else:
        print("Using User Provided Custom Flip Function")
    # Parse the topo file
    tpdf = parse_topos(topo_file)
    # Convert the dataframe to a numpy matrix and get the node list and the matrix
    topo_matrix, node_list = edgelist_to_matrix(tpdf)
    # If the max steps is not provided, set it to 100*number of nodes
    if max_steps is None:
        max_steps = 100 * len(node_list)
    # else:
    #     max_steps = max_steps * len(node_list)
    # Store the states of the nodes and their step numbers in a list of list
    simulation_states = []
    # If the initial state is not provided, generate a random initial state
    # -1 will be the placeholder for initial state flag
    if initial_state is None:
        # Generate a random initial state of 1s and -1s
        initial_state = np.random.choice([-1, 1], size=(len(node_list)))
    # Add the initial state to the simulation states
    simulation_states.append(np.insert(initial_state, 0, -1))
    # Intialising the previous state variable
    prev_state = initial_state
    # Updating the state in sync mode until max_steps or convergence
    for step in range(1, max_steps + 1):
        # Update the state of the network
        new_state = update_ising_sync(topo_matrix, prev_state, flip_function)
        # Check if the state has converged
        if check_convergence(
            prev_state, new_state, step, flip_function, simulation_states
        ):
            # Add the state to the simulation states
            # -2 will be the placeholder for steady state flag which will be the first element in the simulation states row
            simulation_states.append(np.insert(new_state, 0, -2))
            break
        # Update the previous state
        prev_state = new_state
        # Add the state to the simulation states
        # Add step as the first element to the new_state and print
        simulation_states.append(np.insert(new_state, 0, step))
    # If time_series is True, return the simulation states dataframe
    if time_series:
        # Convert the simulation states to a pandas dataframe
        # Return the simulation states dataframe
        return generate_solution_df(simulation_states, node_list)
    else:
        return generate_final_state_dict(simulation_states, node_list)


def run_ising_async(
    topo_file,
    max_steps=None,
    initial_state=None,
    flip_function="onezero",
    update_order=None,
    time_series=False,
):
    # Setting the flip function based on the input
    if flip_function == "onezero":
        flip_function = on_off_flip_onezero
    elif flip_function == "minusplus":
        flip_function = on_off_flip_minusplus
    else:
        print("Using User Provided Custom Flip Function")
    # Parse the topo file
    tpdf = parse_topos(topo_file)
    # Convert the dataframe to a numpy matrix and get the node list and the matrix
    topo_matrix, node_list = edgelist_to_matrix(tpdf)
    # If the max steps is not provided, set it to 100*number of nodes
    if max_steps is None:
        max_steps = 100 * len(node_list)
    # else:
    #     max_steps = max_steps * len(node_list)
    # Store the states of the nodes and their step numbers in a list of list
    simulation_states = []
    # If the initial state is not provided, generate a random initial state
    if initial_state is None:
        # Generate a random initial state of 1s and -1s
        initial_state = np.random.choice([-1, 1], size=(len(node_list)))
    # Add the initial state to the simulation states
    simulation_states.append(np.insert(initial_state, 0, -1))
    # Intialising the previous state variable
    prev_state = initial_state
    # IF node cycle is none, randomly sample node from the node list every step
    if update_order is None:
        update_order = np.random.choice(len(node_list), size=max_steps)
    elif update_order == "generate":
        # Shuffle the node list
        update_order = list(np.random.permutation(len(node_list)))
    else:
        update_order = [node_list.index(node) for node in update_order]
    # Make the node cycle into a itertools cycle
    update_order = cycle(update_order)
    # Intialising the step variable as 0
    step = 1
    # Update the state in async mode until max_steps or convergence
    for node_to_update in update_order:
        # Check if the step is greater than max_steps
        if step >= max_steps:
            break
        # Update the state of the network
        new_state = update_ising_async(
            topo_matrix, prev_state, node_to_update, flip_function
        )
        # Check if the state has converged
        if check_convergence(
            prev_state, new_state, step, flip_function, simulation_states
        ):
            # Add the state to the simulation states
            # -2 will be the placeholder for steady state flag which will be the first element in the simulation states row
            simulation_states.append(np.insert(new_state, 0, -2))
            break
        # Update the previous state
        prev_state = new_state
        # Add the state to the simulation states
        # Add step as the first element to the new_state and print
        simulation_states.append(np.insert(new_state, 0, step))
        # Increment the step
        step += 1
    # If time_series is True, return the simulation states dataframe
    if time_series:
        # Convert the simulation states to a pandas dataframe
        # Return the simulation states dataframe
        return generate_solution_df(simulation_states, node_list)
    else:
        return generate_final_state_dict(simulation_states, node_list)


if __name__ == "__main__":
    # Name of the topo folder
    topo_folder = "../TOPOS"
    # Get the list of all the topo files in the folder
    topo_files = sorted(glob.glob(f"{topo_folder}/T*.topo"))
    # Looping thorough the files in the topo folder
    for topo_file in topo_files:
        print(f"Running Ising model on {topo_file}")
        for i in range(3):
            # print("#" * 10)
            print(f"Run {i}")
            # Running the ising model in synchronous mode
            soldf = run_ising_async(
                topo_file,
                flip_function="onezero",
                # node_cycle="generate",
                time_series=True,
            )
            print(soldf)
            print("")
        # for i in [[-1, 1], [1, -1], [1, 1], [-1, -1]]:
        #     print(f"Run {i}")
        #     # Running the ising model in synchronous mode
        #     soldf = run_ising_async(
        #         topo_file,
        #         initial_state=np.array(i),
        #         flip_function="minusplus",
        #         # node_cycle="generate",
        #         time_series=True,
        #     )
        #     print(soldf)
        # print("#" * 10)
