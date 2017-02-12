####################################################################################################
# Model Validation
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np


# Load the iris data
iris = load_iris()
X, y = iris.data, iris.target

# Fit the KNN model to the data
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X, y)

# Produce model predictions
y_pred = clf.predict(X)
print(np.all(y == y_pred))

# Display the confusion matrix
print(confusion_matrix(y, y_pred))

# If the model is used on new data, it has rather poor results:
from sklearn.model_selection import train_test_split

# Split the data into train / test and fit to model
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf.fit(X_train, y_train)
y_pred  = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))