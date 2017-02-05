import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import data
df = pd.read_csv('Data/auto-mpg.data.txt', delim_whitespace=True, header=None,
                 decimal='.',
                 quotechar='"',
                 names=['mpg', 'cylinders', 'displacement', 'hp', 'weight',
                        'acceleration', 'model year', 'origin', 'car name'])

# Convert objects to float64
df = df.convert_objects(convert_numeric=True)

# Normalise weights for scatter plot data point size
sizes = np.array(df['weight'])
sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 100

# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

plt.show()
if False:
    # Add the title
    plt.title('Fuel efficiency vs Horse-power')

    # Add the x-axis label
    plt.xlabel('Horse-power')

    # Add the y-axis label
    plt.ylabel('Fuel efficiency (mpg)')

    # Display the plot
    plt.show()
