import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

# Every algorithm in scikit-learn is exposed via Estimator object
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
print(model.normalize)
print(model)

# Simulate some data for a Regression exercise
x = np.arange(10)
y = 2 * x + 1

# Represent the data
print(x)
print(y)
plt.plot(x, y, 'r*')
plt.show()

# The input data for sklearn is 2D: (samples == 10 x features == 1)
# X = np.array(x).reshape(len(x), 1)
X = x[:, np.newaxis]
print(X)
print(y)

# fit the model on the data
model.fit(X, y)

# underscore at the end indicate a fit parameter
print(model.coef_)
print(model.intercept_)
print(model.residues_)


