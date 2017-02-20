from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']

# Create labels dictionary
labels = dict((k, v) for (k, v) in enumerate(target_names))

# Rename labels in target
target = [labels[x] for x in target]

# Convert numpyArr to pandaDf
iris = pd.DataFrame(data['data'])

old = list(range(0, 4))
feature_names = [x.replace(' (cm)', '').replace(' ', '_') for x in feature_names]
name_dict = dict(zip(old, feature_names))
iris.rename(columns=name_dict, inplace=True)
iris['species'] = target



