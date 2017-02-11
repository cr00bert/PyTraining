# This script covers supervised learning (Classification and Regeression)
# Classification gives out a discrete label, white regression - continuous

############################################################################################
# Classification examples, the kNN(k Nearest Neighbours) model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


# Define a plot drawer for the estimators
def plot_data_model(model, data, target, feature1_index, feature2_index, xlabel='', ylabel=''):
    X = data[:, [feature1_index, feature2_index]]
    y = target

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    model.fit(X, y)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Put the result into a color plot
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('tight')
    plt.show()


iris = datasets.load_iris()
X, y = iris['data'], iris['target']

# Create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X, y)

# What kind of iris has 3cm x 5cm sepal and 4cm x 2cm petal?
# Call the "predict" method:
flower = [[3, 5, 4, 2]]
print(flower)

result = knn.predict(flower)
print(iris.target_names[result])

prob = knn.predict_proba(flower)
print(prob)

# Draw the model results for 2 features
print('KNN Model estimate plot')
if False:
    plot_data_model(neighbors.KNeighborsClassifier(n_neighbors=3),
                X, y, 0, 1,
                xlabel=iris['feature_names'][0],
                ylabel=iris['feature_names'][1])

# Classification examples, svm (support vector machine)
svc = SVC(C=1000, kernel='linear', random_state=7, probability=True)

# Fit the model
svc.fit(X, y)

print('\n')
print(flower)

resultSVC = svc.predict(flower)
print(iris.target_names[result])

prob = svc.predict_proba(flower)
print(prob)

print('SVC Model estimate plot')
if False:
    plot_data_model(SVC(C=1000, kernel='linear', random_state=7, probability=True),
                        X, y, 0, 1,
                        xlabel=iris['feature_names'][0],
                        ylabel=iris['feature_names'][1])


############################################################################################
# Regression examples
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Linear Regression examples
# Generate random data
np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.normal(0, 1, 20)

# Plot the data
plt.plot(X, y, '^')
plt.show()

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Plot the data and the model prediction
X_fit = np.linspace(0, 1, 100).reshape(100, 1)
y_fit = model.predict(X_fit)

print('\nLinear Regression coefficients:')
print('Model coefficient: ' + str(model.coef_))
print('Model intercept: ' + str(model.intercept_))

plt.plot(X, y, '^')
plt.plot(X_fit, y_fit)
plt.show()

# Fit a Random Forest
model = RandomForestRegressor()
model.fit(X, y)

# Plot the data and the model prediction
y_fit = model.predict(X_fit)

plt.plot(X, y, 'o')
plt.plot(X_fit, y_fit)
plt.show()

