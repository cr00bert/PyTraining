# import 'datasets' from 'sklearn'
from sklearn import datasets, decomposition
import numpy as np
import matplotlib.pyplot as plt

# Load the 'digits' data
digits = datasets.load_digits()

# import pandas as pd

# digits = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/'
#                     '/optdigits/optdigits.tra', header=None)

# Get the keys of the 'digits' data
print(digits.keys())

# Get the keys of the 'digits' data
print(digits.data)

# Print out the target values
print(digits.target)

# Print out the description of the `digits` data
print(digits.DESCR)

# Isolate the 'digits' data
digits_data = digits.data

# Inspect the shape
print(digits_data.shape)

# Isolate the target values with 'target'
digits_target = digits.target

# Inspect the shape
print(digits_target.shape)

# Print the number of unique labels
target_names = np.unique(digits.target)
number_digits = len(np.unique(digits.target))

# Isolate the 'images'
digits_images = digits.images

# Inspect the shape
print(digits_images.shape)

# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))

# Adjust the subplots
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid, at the ith position
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits_images[i], cmap=plt.cm.binary, interpolation='lanczos')
    # Label the image with the target value
    ax.text(0, 7, str(digits_target[i]))

# Show the plot
plt.show()

# Create a Randomized PCA and ordinary PCA model that takes two components
randomized_pca = decomposition.PCA(n_components=2, svd_solver='randomized')
pca = decomposition.PCA(n_components=2)

# Fit and transform the data to the mode
reduced_data_rpca = randomized_pca.fit_transform(digits_data)
reduced_data_pca = pca.fit_transform(digits_data)

# Inspect the shape
print(reduced_data_pca.shape)

# Print out data
print(reduced_data_rpca)
print(reduced_data_pca)

# Scatterplot building
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    x = reduced_data_rpca[:, 0][digits_target == i]
    y = reduced_data_rpca[:, 1][digits_target == i]
    plt.scatter(x, y, c=colors[i])
plt.legend(target_names, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA Sactter Plot')
plt.show()

# Use K-Means according to http://scikit-learn.org/stable/tutorial/machine_learning_map/
# Import
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import cluster

# Apply 'scale()' to the digits data
data = scale(digits_data)

# Split the digits data into training and test data
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data,
                                                                               digits_target,
                                                                               digits_images,
                                                                               test_size=0.25,
                                                                               random_state=42)

# Number of training features
n_samples, n_features = X_train.shape

# n_samples, n_features, n_training labels
print("Training samples: " + str(n_samples))
print("Training features: " + str(n_features))
print("Training labels: " + str(len(np.unique(y_train))))
print("Labels: " + str(len(y_train)))

# Create the KMeans model
clf = cluster.KMeans(init='k-means++', n_clusters=10, random_state=42)

# Fit the training data 'X_train' to the model
clf.fit(X_train)

# Figure size in inches
fig = plt.figure(figsize=(8, 3))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels (0-9):
for i in range(10):
    # Initialize subplot in a grid of 2x5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)
    # Display images
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    # Don't show the axes
    plt.axis('off')

# Show the plot
plt.show()

# Predict labels for X_test
y_pred = clf.predict(X_test)

# Print out the first 100 y_pred
print(y_pred[:100])

# Print out the first 100 y_test
print(y_pred[:100])

# Study the shape of the cluster centers
clf.cluster_centers_.shape

# Import `Isomap()`
from sklearn.manifold import Isomap

# Create an isomap and fit the digits data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Computer cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_train)

# Create a plot with subplot in a grid of 1x2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()

# Create a PCA analysis and fit the digits data to it
X_pca = decomposition.PCA(n_components=2).fit_transform(X_train)

# Computer cluster centers and predict cluster index for each sample
clusters = clf.fit_predict(X_pca)

# Create a plot with subplot in a grid of 1x2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust layout
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')

# Show the plots
plt.show()

# Import metrics from sklearn
from sklearn import metrics

# Print out the confusion matrix
confMx = metrics.confusion_matrix(y_test, y_pred)

rowSum = np.zeros((10, 1))
for i in range(10):
    rowSum[i] = 1 / np.array(confMx[i, :]).sum()

print(confMx)
print((confMx * rowSum).round(2))

# More metrics
from sklearn.metrics import homogeneity_score, \
    completeness_score, \
    v_measure_score, \
    adjusted_rand_score, \
    adjusted_mutual_info_score, \
    silhouette_score

print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
      % (clf.inertia_,
         homogeneity_score(y_test, y_pred),
         completeness_score(y_test, y_pred),
         v_measure_score(y_test, y_pred),
         adjusted_rand_score(y_test, y_pred),
         adjusted_mutual_info_score(y_test, y_pred),
         silhouette_score(X_test, y_pred, metric='euclidean')))

##############################################
#####SVC MODEL

# Split the data into training and test sets
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data,
                                                                               digits.target,
                                                                               digits.images,
                                                                               test_size=0.5,
                                                                               random_state=0)

# Import the `svm` model
from sklearn import svm

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Create the SVC model
svc_model = svm.SVC(gamma=0.001, C=100, kernel='linear')

# Set the parameters candidates
parameter_candidates = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
]

# Create a classifier with the parameter candidates
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

# Train the classifier on training data
clf.fit(X_train, y_train)

# Print out the results
print('Best score for training data:', clf.best_score_)
print('Best `C`:', clf.best_estimator_.C)
print('Best kernel:', clf.best_estimator_.kernel)
print('Best `gamma`:', clf.best_estimator_.gamma)

# Apply the classifier to the test data, and view the accuracy score
clf.score(X_test, y_test)

# Train and score a new classifier with the grid search parameters
svm.SVC(C=1000, kernel='rbf', gamma=0.001).fit(X_train, y_train).score(X_test, y_test)
svc_model = svm.SVC(C=1000, kernel='rbf', gamma=0.001).fit(X_train, y_train)


# Assign the predicted values to `predicted`
predicted = svc_model.predict(X_test)

# Zip together the `images_test` and `predicted` values in `images_and_predictions`
images_and_predictions = list(zip(images_test, predicted))

# For the first 4 elements in `images_and_predictions`
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    # Initialize subplots in a grid of 1 by 4 at positions i+1
    plt.subplot(1, 4, index + 1)
    # Don't show axes
    plt.axis('off')
    # Display images in all subplots in the grid
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    # Add a title to the plot
    plt.title('Predicted: ' + str(prediction))

# Show the plot
plt.show()


# Print the classification report of `y_test` and `predicted`
print(metrics.classification_report(y_test, predicted))

# Print the confusion matrix of `y_test` and `predicted`
print(metrics.confusion_matrix(y_test, predicted))

# Create an isomap and fit the `digits` data to it
X_iso = Isomap(n_neighbors=10).fit_transform(X_train)

# Compute cluster centers and predict cluster index for each sample
predicted = svc_model.predict(X_train)

# Create a plot with subplots in a grid of 1X2
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Adjust the layout
fig.subplots_adjust(top=0.85)

# Add scatterplots to the subplots
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')


# Add title
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')

# Show the plot
plt.show()