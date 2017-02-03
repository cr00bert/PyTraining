import pandas as pd

#Initialise dictionary
data = {'a': {0: '1980', 1: '1981', 2: '1982'},
        'b': {0: 'Blondie', 1: 'Chistorpher Cross', 2: 'Joan Jett'},
        'c': {0: 'Call Me', 1: 'Arthurs Theme', 2: 'I Love Rock and Roll'},
        'd': {0: '6', 1: '3', 2: '7'}}

#convert data to dataframe
df = pd.DataFrame(data)
#print(df)

# Build a list of labels: list_labels
list_labels = ['year', 'artist', 'song', 'chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels


print(df)