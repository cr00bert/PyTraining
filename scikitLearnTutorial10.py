# In-depth Supervised Learning
# Support Vector Machines

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.svm import SVC
from ipywidgets import interact

# Use seaborn plotting defaults
sns.set()

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='spring', marker='^')
#plt.show()
plt.close()


# Problem with discriminative classifiers, they allow for an infinite number of solutions
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='spring', marker='^')

for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 3.5)
#plt.show()
plt.close()

# That's where SVM comes in, who maximise the MARGIN
# It considers a whole region around the line, and tries to maximise it's area
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='spring', marker='^')

for m, b, d in [(1, 0.65, 0.35), (0.5, 1.6, 0.6), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)
#plt.show()
plt.close()

# In this case the line with the 0.65 intercept has the best fit
# Let's fit the actual model
clf = SVC(kernel='linear')
clf.fit(X, y)

# Let's write a function that depicts the boundaries to us
def plot_svc_decision_function(clf, ax=None):
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            P[i, j] = clf.decision_function([[xi, yj]])
    # plot the margins
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='spring', marker='^')
plt.xlim(-1, 3.5)
plot_svc_decision_function(clf)

# Notice that the line crosses some of the points. These are support vectors
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=200, facecolors='none')
#plt.show()
plt.close()

# SVM is even more interesting when used with kernels
X, y = make_circles(100, factor=0.1, noise=0.1, random_state=0)

clf = SVC(kernel='linear').fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
#plt.show()
plt.close()

# Linear discrimination does not work in this instance
# Let us apply a radial basis function
r = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2))
from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='spring')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

#interact(plot_3D(), elev=[-180, 180], azip=(-180, 180))
#plt.show()
plt.close()

# Now it becomes trivial to discriminate the data
# Let's use RBF for this (radial base function)
clf = SVC(kernel='rbf')
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200, facecolors='none', lw=1, edgecolor='0', alpha=0.4)
plt.show()