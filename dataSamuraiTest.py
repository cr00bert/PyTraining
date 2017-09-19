# Data Samurai test
# 2017/02/26 12:00
# Create By Robert Bilkevic

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like data to resample
    n : int, length of resampled array, len(X) by default
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)

    resample_i = np.floor(np.random.rand(n) * len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample


# Import and visually inspect data
calls = pd.read_csv('Data\\DataSamurai\\calls.csv')
leads = pd.read_csv('Data\\DataSamurai\\leads.csv')
signups = pd.read_csv('Data\\DataSamurai\\signups.csv')

#print(calls.head())
#print(leads.head())
#print(signups.head())

# 1 ##################################################################
# Group agents by calls made
calls_per_agent = calls['Agent'].groupby(calls['Agent']).count()
print(calls_per_agent)

# 2 ##################################################################
# Group leads by calls received, and determine average
# Make sure that each lead has only one number
print(leads.groupby([leads['Name']]).count().max())
calls_lead = pd.DataFrame({'count': calls['Phone Number'].groupby(calls['Phone Number']).count()}).reset_index()
print('\nAverage calls per lead: ', calls_lead['count'].mean())


# 3 ##################################################################
# Find unique list of names and numbers for leads who signed up
signup_names = pd.DataFrame(signups['Lead'])

# Complement with phone numbers
signup_enr = signup_names.merge(leads.loc[:, ['Name', 'Phone Number']], how='left', left_on='Lead',
                                right_on='Name').loc[:, ['Name', 'Phone Number']]

# Inner join with calls made
signup_calls = signup_enr.merge(calls_lead, how='inner', on='Phone Number')
print('\nCalls per signup: ', signup_calls['count'].mean())


# 4 ##################################################################
# Filter out only the last calls made to a lead
calls_last = pd.DataFrame(calls.groupby('Phone Number').last()).reset_index()

# Filter out only the 'INTERESTED' ones and merge with signup list
calls_last_potential = calls_last.loc[calls_last['Call Outcome'] == 'INTERESTED']
calls_signed = calls_last_potential.merge(signup_enr, how='inner', on='Phone Number')

# Signups per agent
signups_per_agent = calls_signed['Agent'].groupby(calls_signed['Agent']).count()
print(signups_per_agent)

# 5 ##################################################################
# Signups per call per agent
print(signups_per_agent/calls_per_agent)

# 6 ##################################################################
xs = pd.DataFrame({'conversion': signups_per_agent/calls_per_agent})
#xs.plot(kind='hist', bins=3)
#plt.show()


X1 = []
for i in range(200000):
    X1.append(bootstrap_resample(xs['conversion'].values, len(xs['conversion'].values)-1).mean())

print('\nBootstrap Mean: ', np.array(X1).mean())
print('Sample Mean: ', xs['conversion'].mean())


X2 = []
for i in range(200000):
    X2.append(bootstrap_resample(xs['conversion'].values, len(xs['conversion'].values)-1).std())

print('\nBootstrap SD: ', np.array(X2).mean())
print('Sample SD: ', xs['conversion'].std())


conf_int95 = [np.array(X1).mean() - np.array(X2).mean() * 2, np.array(X1).mean() + np.array(X2).mean() * 2]
conf_int68 = [np.array(X1).mean() - np.array(X2).mean() * 1, np.array(X1).mean() + np.array(X2).mean() * 1]

print(conf_int95)
print(conf_int68)


# Agent 'black' performance statistically different from mean
print('\nAgent ''black'' performance is statistically difference from mean (a=0.05): ',
      conf_int95[0] > xs.loc['black'].values or conf_int95[1] < xs.loc['black'].values)

# Agent 'blue' performance statistically different from mean
print('\nAgent ''blue'' performance is statistically difference from mean (a=0.05): ',
      conf_int95[0] > xs.loc['blue'].values or conf_int95[1] < xs.loc['blue'].values)

# 7 ##################################################################

# Extract the calls that resulted in 'INTERESTED' leads
calls_potential = calls.loc[calls['Call Outcome'] == 'INTERESTED']

# Merge with the leads data
leads_interested = leads.merge(calls_potential, how='inner', on='Phone Number')

# Interested leads by region
leads_interested_reg = leads_interested['Region'].groupby(leads_interested['Region']).count()
print(leads_interested_reg.sort_values(ascending=False))


# 8 ##################################################################
# Interested leads by sector
leads_interested_sec = leads_interested['Sector'].groupby(leads_interested['Sector']).count()
print(leads_interested_sec.sort_values(ascending=False))


# 9 ##################################################################
# Enrich approved leads with phone numbers
signup_apr = pd.DataFrame(signups.loc[signups['Approval Decision'] == 'APPROVED', 'Lead'])
signup_apr_enr = signup_apr.merge(leads.loc[:, ['Name', 'Phone Number']], how='left', left_on='Lead',
                             right_on='Name').loc[:, ['Name', 'Phone Number']]

# Merge with region
signup_apr_enr_reg = signup_apr_enr.merge(leads, how='inner', on=['Phone Number', 'Name'])

# Display sorted data
signup_apr_reg = signup_apr_enr_reg['Region'].groupby(signup_apr_enr_reg['Region']).count().sort_values(ascending=False)
print(signup_apr_reg)

print(stats.jarque_bera(signup_apr_reg[signup_apr_reg < 95]))
print(stats.ttest_1samp(signup_apr_reg, 95))

signup_apr_reg.plot(kind='hist')


X1 = []
for i in range(10000):
    X1.append(bootstrap_resample(signup_apr_reg).mean())

print('\nBootstrap Mean: ', np.array(X1).mean())
print('Sample Mean: ', signup_apr_reg.mean())


X2 = []
for i in range(10000):
    X2.append(bootstrap_resample(signup_apr_reg).std())

print('\nBootstrap SD: ', np.array(X2).mean())
print('Sample SD: ', signup_apr_reg.std())


conf_int95 = [np.array(X1).mean() - np.array(X2).mean() * 2, np.array(X1).mean() + np.array(X2).mean() * 2]
print(conf_int95)

# Northwest signups are statistically different from mean
print('\nNorthwest signups are statistically different from mean (a=0.05): ',
      conf_int95[0] > signup_apr_reg.loc['north-west'] or conf_int95[1] < signup_apr_reg.loc['north-west'])


# 10 ##################################################################
# Sign-ups dataset
signups_pop = signups.merge(leads, how='inner', left_on='Lead', right_on='Name')\
    .drop(['Approval Decision', 'Lead'], axis=1)


# Data set of leads that did not sign-up
tmp = pd.merge(leads, signups, left_on='Name', right_on='Lead', how="outer", indicator=True)
leads_rem = tmp[tmp['_merge'] == 'left_only'].drop(['Lead', 'Approval Decision', '_merge'], axis=1)

# Visual inspection of selection criterion effect
signups_pop['Region'].groupby(signups_pop['Region']).count().sort_values(ascending=False)
leads_rem['Region'].groupby(leads_rem['Region']).count().sort_values(ascending=False)

signups_pop['Sector'].groupby(signups_pop['Sector']).count().sort_values(ascending=False)
leads_rem['Sector'].groupby(leads_rem['Sector']).count().sort_values(ascending=False)

signups_pop['Age'].groupby(pd.cut(signups_pop['Age'], 7)).count()
leads_rem['Age'].groupby(pd.cut(leads_rem['Age'], 7)).count()


# Cherry-picking from the sign-ups data-set
newdf = signups_pop.loc[leads_rem['Age'].between(17.855, 46.429) & signups_pop['Region'].isin(['north-west', 'south-west'])
                & signups_pop['Sector'].isin(['retail', 'food'])]


# Merging with successful agents
calls_signed = calls_last_potential.merge(newdf, how='inner', on='Phone Number')

# Cherry-picked signups per agent
signups_per_agent = calls_signed['Agent'].groupby(calls_signed['Agent']).count()

# All calls of the agent to the cherry-picked leads
best_calls = calls.merge(calls_signed, how='inner', on=['Phone Number', 'Agent'])
calls_per_agent = best_calls['Agent'].groupby(best_calls['Agent']).count()

# Signups per call per agent
print('\nAverage sign-up to call on optimal set of leads:', (signups_per_agent/calls_per_agent).mean())

