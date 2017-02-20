import pandas as pd
import numpy as np

# Import the data
df = pd.read_csv('Data\\WDI\\WDI_Data.csv', sep=',', decimal='.', header=0,
                 quotechar='"', encoding='cp1252', nrows=49)

# Check header of dataframe
# print(df.head())

id_vars = list(df.columns.values[:4])
value_vars = list(df.columns.values[4:])

new_df = pd.melt(df, id_vars=id_vars, var_name='Year',
                 value_vars=value_vars, value_name='Value')


# Define lists2dicts()
def lists2dict(key_list, value_list):
    """Return a dictionary where list1 provides
    the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(key_list, value_list)

    # Create a dictionary
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict

feature_names = list(df.columns.values[:4]) + ['Year', 'Value']
row_vals = list(df.iloc[48, :5])


print(lists2dict(feature_names, row_vals))

# Use list comprehension for multiple lists
row_list = new_df.iloc[0:50, :].values.tolist()
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_list]

# Convert list into yet another Df
o_df = pd.DataFrame(list_of_dicts)
print(o_df.head())