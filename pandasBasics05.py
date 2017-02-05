import pandas as pd

# Read in the file
df1 = pd.read_csv("Data/world_population.csv")

# Create a list of the new column labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv('Data/world_population.csv', header=0, names=new_labels)

# Print both
print(df1)
print(df2)
