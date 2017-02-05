import pandas as pd

# Read in the file
df1 = pd.read_csv("Data/messy_yahoo_finance.csv", sep=' ')

# Print the output of df1.head()
print(df1.head())

# Read in with correct parameters
df2 = pd.read_csv("Data/messy_yahoo_finance.csv", sep=',', header=3, comment='#')

# Read in the file, specifying the header and names parameters: df2
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv('Data/clean_yahoo_finance.csv', index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('Data/clean_yahoo_finance.xlsx', index=False)

