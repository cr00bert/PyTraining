import pandas as pd

# populate list
cities = ['Manheim',
          'Preston park',
          'Biglerville',
          'Indiana',
          'Curwensville',
          'Crown',
          'Harveys lake',
          'Mineral springs',
          'Cassville',
          'Hannastown',
          'Saltsburg',
          'Tunkhannock',
          'Pittsburgh',
          'Lemasters',
          'Great bend']

#Make a string with the value 'PA': state
state = 'PA'

#Construct a dictionary: data
data = {'state':state, 'city':cities}

#Construct a DataFrame from dicionary data: df
df = pd.DataFrame(data)

#Print the DataFrame
print(df)
