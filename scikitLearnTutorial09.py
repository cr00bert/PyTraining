from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Import digits data
digits = load_digits()

# Plot several images
fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# How does the image look like
print(digits.images.shape)
print(digits.images[0])

# The data for use in our algorithm
print(digits.data.shape)
print(digits.data[0])

# The target labels:
print(digits.target)

# Let us visualize the data using dimensionality reduction
iso = Isomap(n_components=2)
data_projected = iso.fit_transform(digits.data)

# New data shape
print("The shape of reduced dim data:", data_projected.shape)

# Graphical representation of the principal components and targets
plt.figure()
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolors='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

plt.show()
# The digits seem to be fairly well-separated. That is why a SLA should perform well
# Split the data
Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)

clf = LogisticRegression(penalty='l2')
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

print('\n Accuracy of LinearRegression classifier: ', accuracy_score(ytest, ypred))
print(confusion_matrix(ytest, ypred))

plt.imshow(np.log(confusion_matrix(ytest, ypred)),
           cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.05, 0.05, str(ypred[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == ypred[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

clf = SVC(C=100, kernel='linear', gamma=0.01)
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

print('\n Accuracy of SVC classifier: ', accuracy_score(ytest, ypred))
print(confusion_matrix(ytest, ypred))

plt.imshow(np.log(confusion_matrix(ytest, ypred)),
           cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8, 8), cmap='binary')
    ax.text(0.05, 0.05, str(ypred[i]),
            transform=ax.transAxes,
            color='green' if (ytest[i] == ypred[i]) else 'red')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()