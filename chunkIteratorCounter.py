import pandas as pd

# Define count_entries()
def count_entries(csv_file, c_size, colname, sep):
    """Return a dictionary with counts of
    occurrences as value for each key."""

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size, sep=sep):

        # Iterate over the column in dataframe
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict


# Call count_entries(): result_counts
result_counts = count_entries('Data\winequality-red.csv', 100, 'quality', ';')

# Print result_counts
print('Number of wine brands by quality')
print(result_counts)