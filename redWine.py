# Import Package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd
import matplotlib.pyplot as plt

# Assign url of file: url
url = "https://s3.amazonaws.com/assets.datacamp.com/production/course_1606/datasets/winequality-red.csv"

#urlretrieve(url, "winequality-red.csv")

# Read file into a DataFrame and print its head
#df = pd.read_csv("winequality-red.csv", sep = ";")
df = pd.read_csv(url, sep = ";")

print(df.head())

pd.DataFrame.hist(df.ix[:, 0:1])
plt.xlabel("fixed acidity (g(tartaric acid)/dm$^3$)")
plt.ylabel("count")
plt.show()

