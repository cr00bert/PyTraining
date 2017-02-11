# Import iris data and other relevant packages
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()

# Inspect data set
print(iris.keys())
print(iris['DESCR'])
print(iris['feature_names'])
print(iris['target_names'])

# Save the data
data = iris['data']
target = iris['target']
target_names = iris['target_names']

# Save the number of samples and features
n_samples, n_features = data.shape

print("Number of samples: " + str(n_samples))
print("Number of features: " + str(n_features))

# Normalize the data
data = scale(data)


# Plot 2 features versus their labels
def plot_iris_projection(x_index, y_index):
    '''
    This function plots the two selected should be used to determine the princilap components of this dataset
    :param x_index: feature of the x-axis
    :param y_index: feature of the y-axis
    '''
    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])

    plt.scatter(data[:, x_index], data[:, y_index],
                c=target, cmap=plt.cm.get_cmap('RdYlBu', 3))
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.clim(-0.5, 2.5)
    plt.xlabel(iris['feature_names'][x_index])
    plt.ylabel(iris['feature_names'][y_index])
    plt.show()


plot_iris_projection(0, 2)


# Produce RPCA data for graphical representation
rPCA = PCA(n_components=2, random_state=7, svd_solver='randomized')
data_rpca = rPCA.fit_transform(data)

# Display the data
colors = ['red', 'green', 'white']
for label in np.unique(target):
    x = data_rpca[:, 0][target == label]
    y = data_rpca[:, 1][target == label]
    plt.scatter(x, y, c=colors[label])
plt.show()
