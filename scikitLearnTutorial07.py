# This script covers supervised learning (Classification and Regeression)

# Data has no labels, and we are interested in finding similarities between the objects
# UL is as a means of discovering labels from the data itself
# UL tasks: dimensionality reduction, clustering, and density estimation


import numpy as np
import pylab as plt
from sklearn.datasets import load_iris

###################################################################################################
# Dimensionality reduction
from sklearn.decomposition import PCA

# Fit the PCA model with 0.95 conf interval
X, y = load_iris().data, load_iris().target
pca = PCA(n_components=0.95)
pca.fit(X)
X_reduced = pca.transform(X)

print("Reduced dataset shape: ", X_reduced.shape)
# Graphical representation

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='RdYlBu')

print("Meaning of the 2 components:")
for component in pca.components_:
    print(" + ".join("%.3f x %s" % (value, name)
                     for value, name in zip(component,
                                            load_iris().feature_names)))

plt.show()


###################################################################################################
# Clustering: K-means
from sklearn.cluster import KMeans

# Fit the K-Means model
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X)
y_pred = k_means.predict(X)

# Graphical representation
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred, cmap='RdYlBu')
plt.show()