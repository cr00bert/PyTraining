# Import scikit learn
from sklearn import decomposition, cluster
from sklearn.model_selection import train_test_split
from json import loads
from base64 import b64decode
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np

# This file is a JSON Lines format file. Has to be read line by line
json_file = "Data\\digits.base64.json"

data = []
label = []
image = []
with open(json_file) as data_file:
    for line in data_file:
        s = b64decode(loads(line)["data"])
        img = np.fromstring(s, dtype=np.ubyte)
        image.append(img.reshape(28, 28))
        digit_vector = img.astype(np.float64)
        data.append(digit_vector)
        label.append(loads(line)["label"])

#digits = list(zip(data, label, image))


# Define a graphic function for 28x28 image
def display_digit(digit, title=""):
    """
    graphically displays 28x28 vector, representing a digit
    :param digit: image of digit
    :param title: graph title
    """
    plt.figure()
    fig = plt.imshow(digit, cmap=plt.cm.binary)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if title != "":
        fig.title("Inferred label: " + str(title))
    plt.show()

# Define a graphic function for PCA
def display_data_PCA(dataPCA, labelPCA):
    """
    graphically displays 28x28 vector, representing a digit
    :param dataPCA: principal components data of image
    :param labelPCA: digit label
    """
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        x = dataPCA[:, 0][np.array(labelPCA) == i]
        y = dataPCA[:, 1][np.array(labelPCA) == i]
        plt.scatter(x, y, c=colors[i])
    plt.legend(np.unique(label), bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Sactter Plot')
    plt.show()

# Apply scaling to data



# Create a Randomized PCA and ordinary PCA model that takes two components
randomized_PCA = decomposition.PCA(n_components=2, svd_solver="randomized")

# Fit model
reduced_data = randomized_PCA.fit_transform(data)

# Inspect shape and content
print(reduced_data.shape)
print(reduced_data[0])

# Get unique label from data_set
digits = np.unique(label)
print(len(digits))

# Represent PCA of digits data
display_data_PCA(reduced_data, label)

# Create a k-means model with 10 clusters

X_train, X_test, y_train, y_test, image_train, image_test = train_test_split(data,
                                                                             label,
                                                                             image,
                                                                             test_size=0.25,
                                                                             random_state=0
                                                                             )

# Number of training features
n_samples, n_features = np.array(X_train).shape

# n_samples, n_features, n_training labels
print("Training samples: " + str(n_samples))
print("Training features: " + str(n_features))
print("Training labels: " + str(len(np.unique(y_train))))
print("Labels: " + str(len(y_train)))

# Create the KMeans model
clf = cluster.KMeans(n_clusters=16, init='random', n_init=10, random_state=0)

# Fit the training sample to model
clf.fit(X_train)
y_pred = clf.predict(X_train)

# Figure size in inches
fig = plt.figure(figsize=(10, 10))

# Add title
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

# For all labels
for i in range(16):
    # Initialize subplot in a grid of 4x4, at i+1th position
    ax = fig.add_subplot(4, 4, 1 + i)
    # Display the images
    ax.imshow(clf.cluster_centers_[i].reshape(28, 28), cmap=plt.cm.binary)

# Show the plot
plt.show()
