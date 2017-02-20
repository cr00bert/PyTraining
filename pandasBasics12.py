import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('Data\\tips.csv')
df['fraction'] = df['tip']/df['total_bill']

fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df['fraction'].plot(ax=axes[0], kind='hist', normed=True, bins=30, range=(0, 0.3))


# Plot the CDF
df['fraction'].plot(ax=axes[1], kind='hist', normed=True,
                    cumulative=True, bins=30, range=(0, 0.3))
plt.show()
