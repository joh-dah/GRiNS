# File to clean the topo files and standardise the format
import pandas as pd
import glob
import os
import subprocess


# Function to clean the topo files - renames node names which have anything other than alphanumeric characters and replace those characters with underscores
# Also removes non standard delimiters and replaces them with spaces. So that its in the right format to to be read by parse_topo code
def clean_topo(topo_file):
    """
    Clean the topo file.

    Args:
        topo_file (str): Path to the topo file.
    Returns:
        The topo file is cleaned and saved in the same directory with the name cleaned_topo_file.
    """
    # Read the topo file depending on the delimiter
    # Read teh first line of the file to determine the delimiter
    with open(topo_file, "r") as f:
        first_line = f.readline()
        # Check if the delimiter is tab
        if "\t" in first_line:
            topo_df = pd.read_csv(topo_file, delim_whitespace=True)
        # Check if the delimiter is comma
        elif "," in first_line:
            topo_df = pd.read_csv(topo_file, sep=",")
        # Check if the delimiter is space
        elif " " in first_line:
            topo_df = pd.read_csv(topo_file, delim_whitespace=True)
        # Go through the source and target columns and replace non-alphanumeric characters with underscores
        topo_df["Source"] = topo_df["Source"].str.replace(r"\W", "_", regex=True)
        topo_df["Target"] = topo_df["Target"].str.replace(r"\W", "_", regex=True)
        # Check if the Type column has any value other than 1 or 2 by getting the counts of unique values
        type_counts = topo_df["Type"].value_counts()
        # If there is a value other than 1 or 2 in the Type column, print the topo file path and exit the function
        if len(type_counts) > 2:
            print(f"Check the topo file: {topo_file}")
            return None
        # Otherwise save the topo_df in a new file with the name cleaned_topo_file name with space as the delimiter
        else:
            topo_df.to_csv(
                topo_file.replace(".topo", "_cleaned.topo"), index=False, sep=" "
            )
    return None


# Specify the topo file directory
topo_dir = "TOPOS"

# Get the list of topo file paths ignoring files with "_cleaned" in the name
topo_files = sorted([t for t in glob.glob(f"{topo_dir}/*.topo") if "_cleaned" not in t])
# Print the list of topo files
print(topo_files)

# Clean the topo files
for topo_file in topo_files:
    # Print the topo file being cleaned
    print(f"Cleaning the topo file: {topo_file}")
    # Clean the topo file
    clean_topo(topo_file)

# Create a Directory call CleanedTOPOS if it does not exist
if not os.path.exists("CleanedTOPOS"):
    os.makedirs("CleanedTOPOS")

# Print the current directory
# print(os.getcwd())

# Move the cleaned topo files to the CleanedTOPOS directory
# subprocess.run(["ls"])
# subprocess.run(["mv", f"{topo_dir}/*cleaned.topo", "CleanedTOPOS/"])

# Rename the topo files in the CleanedTOPOS directory to remove the _cleaned suffix
# subprocess.run(["zmv -w TOPOS/*_cleaned.topo CleanedTOPOS/$1.topo"])
