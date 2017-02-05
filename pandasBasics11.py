import pandas as pd
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv('Data/auto-mpg.data.txt', delim_whitespace=True, header=None,
                 decimal='.',
                 quotechar='"',
                 names=['mpg', 'cylinders', 'displacement', 'hp', 'weight',
                        'acceleration', 'model year', 'origin', 'car name'])

# Convert objects to float64
df = df.convert_objects(convert_numeric=True)

# Make a list of the column names to be plotted: cols
cols = ['weight', 'mpg']

# Generate the box plots
df[cols].plot(kind='box', subplots=True)

# Display plot
plt.show()
