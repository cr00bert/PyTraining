# Load the breast cancer data from scikit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import decomposition

cancer = load_breast_cancer()

# Inspect data
print(cancer.keys())
print(cancer['DESCR'])

n_samples, n_features = cancer['data'].shape
print("Number of samples: " + str(n_samples))
print("Number of features: " + str(n_features))
print("Unique labels: " + str(len(np.unique(cancer['target']))))

cancer_data = cancer['data']
target = cancer['target']

# Principal Component Analysis

data = scale(cancer_data)
rand_PCA = decomposition.PCA(n_components=2, random_state=55, svd_solver='randomized')

data_rpca = rand_PCA.fit_transform(data)

colors = ['red', 'green']
for label in np.unique(target):
    x = data_rpca[:, 0][target == label]
    y = data_rpca[:, 1][target == label]
    plt.scatter(x, y, c=colors[label])
#plt.show()

# Data split
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=7, test_size=0.25)

# Model fit
mpl = MLPClassifier( hidden_layer_sizes=(15), activation='relu',
                      solver='adam', max_iter=500, alpha=0.001, tol=0.0001, verbose=True)
mpl.fit(X_train, y_train)

# Predictions and fit metrics
predictions = mpl.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

