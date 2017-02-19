# import pandas dataset
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the titanic training set
df = pd.read_csv('data\\titanic\\train.csv')


# Check out the data
# print(df.head())


# Because we are looking to apply the Random Forest model, we should clean up the data, and make it \
# consist of numerical values only
def clean_titanic(titanic):
    # fill in the missing age
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    # make male = 0, female = 1
    titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
    titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1
    # turn embarked into numerical classes
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
    titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
    titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
    clean_data = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    return titanic[clean_data]


data = clean_titanic(df)

X, X_test, y, y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], random_state=7, test_size=0.25)

# Create the random forest model that builds  a 100 trees
forest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
X = data.ix[:, 1:]
y = data.ix[:, 0]
forest.fit(X, y)

y_pred = forest.predict(X_test)

print(forest.predict_proba([[2, 1, 40, 1, 0, 25, 1], ]))


print('\nAccuracy score: ', accuracy_score(y_test, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))

vars = []
# from panda to list:
for column in X:
    vars.append(column)
imps = []
for imp in forest.feature_importances_:
    imps.append(imp)
y_pos = np.arange(len(vars))

plt.barh(y_pos, imps, align='center', alpha=0.5)
plt.yticks(y_pos, vars)
plt.ylabel('Feature')
plt.xlabel('Importance')
plt.title('Feature Importance')
#plt.show()
