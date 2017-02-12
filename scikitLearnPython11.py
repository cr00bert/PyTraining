# In-depth Supervised Learning
# Random forests

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.datasets import make_blobs
from figures import visualize_tree
from sklearn.tree import DecisionTreeClassifier
from ipywidgets import interact

sns.set()

# Generate some data blobs for the Random forest model
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=1)
print('Blob samples shape: ', X.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow', edgecolors='black')
#plt.show()
plt.close()

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

#visualize_tree(clf, X[:200], y[:200], boundaries=False)
#visualize_tree(clf, X[-200:], y[-200:], boundaries=False)
#plt.show()
plt.figure()
plt.close()

# It is extremely easy to over-fit data with Decision Trees.
# The model will response more to the noise than the signal
# That is where Random Trees come in

def fit_randomized_tree(random_state=0):
    X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=2)
    clf = DecisionTreeClassifier(max_depth=15)

    rng = np.random.RandomState(random_state)
    i = np.arange(len(y))
    rng.shuffle(i)
    visualize_tree(clf, X[i[:250]], y[i[:250]], boundaries=False,
                   xlim=(X[:, 0].min(), X[:, 1].max()),
                   ylim=(X[:, 1].min(), X[:, 1].max()))

#interact(fit_randomized_tree(), random_state=[50, 100])
#plt.show()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000, random_state=0, max_features=2, bootstrap=True)
visualize_tree(clf, X, y, boundaries=False)
plt.show()
