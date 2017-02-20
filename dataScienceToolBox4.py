# Repeat the previous excercise with the pandas chunk functionality
import matplotlib.pyplot as plt
import pandas as pd

# Define country plotter function
def plot_pop(filename, country_code):

    # Initialize the reader object: df_reader
    df_reader = pd.read_csv(filename, chunksize=1000, encoding='cp1252')

    data1 = pd.DataFrame()
    data2 = pd.DataFrame()

    for df in df_reader:

        # Transform into long data set
        id_vars = list(df.columns.values[:4])
        value_vars = list(df.columns.values[4:])
        df_prep = pd.melt(df[df['Indicator Name'].isin(['Population, total', 'Urban population (% of total)'])],
                         id_vars=id_vars, value_vars=value_vars,
                            value_name='Value', var_name='Year')
        df_prep = df_prep[df_prep['Country Code'] == country_code]
        id_vars.append('Year')
        id_vars = [elem for elem in id_vars if elem not in ['Indicator Name', 'Indicator Code']]
        df_urb_pop = df_prep.pivot_table(values='Value', index=id_vars, columns=['Indicator Name'])
        df_urb_pop = df_urb_pop.reset_index()

        if not all(c1 in ['Population, total', 'Urban population (% of total)'] for c1 in df_urb_pop.columns.values):
            if 'Population, total' in df_urb_pop.columns.values:
                data1 = data1.append(df_urb_pop)
            elif 'Urban population (% of total)' in df_urb_pop.columns.values:
                data2 = data2.append(df_urb_pop)
        else:
             continue

    df_merged = pd.merge(data1, data2, on=['Country Name', 'Country Code', 'Year'], how='inner')

    pops = zip(df_merged['Population, total'], df_merged['Urban population (% of total)'])
    pops_list = list(pops)

    df_merged['Urban Population'] = [float(tup[0]*tup[1]/100) for tup in pops_list]
    df_merged['Year'] = df_merged['Year'].astype('int')


    plt.scatter(x=df_merged['Year'], y=df_merged['Urban Population'], marker='o')
    title = 'Urban Population of ' + country_code
    plt.title(title)
    plt.show()



plot_pop('Data\\WDI\\WDI_data.csv', 'BRN')