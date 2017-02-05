import pandas as pd
import matplotlib.pyplot as plt

# Read in the file
df = pd.read_csv("Data/temperatureDew.csv")

# Create a plot with color='red'
df.plot(color='red')
plt.show()

# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ["Dew Point (deg F)"]
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)', 'Dew Point (deg F)']
df[column_list2].plot()
plt.show()

