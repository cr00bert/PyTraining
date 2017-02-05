import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
dfT = pd.read_csv('Data/clean_yahoo_finance.csv', index_col=0)

df = pd.DataFrame(data=np.array(dfT).T, columns=dfT.index.tolist())
df['Month'] = dfT.columns.values
df = df[['Month', 'ibm', 'msft', 'googl', 'aapl']]

print(df)

# Create a list of y-axis column names
y_columns = ['aapl', 'ibm']

# Generate a line plot
df.plot(x=df['Month'], y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()
