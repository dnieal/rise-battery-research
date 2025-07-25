import pandas as pd
import numpy as np

df = pd.read_csv("Data/Intermediate Data/rawInfo2.csv", skiprows = range(1, 97))

df = df.rename(columns={'Cycle Number': 'RPT Number'})

df = df.sort_values(axis = 0, by = ["RPT Number", "Battery Name"])

df['Battery Name'] = df['Battery Name'].replace("w1", "w10")

duty = {'g1': 'Commercial', 'v4': 'Residential', 'v5': 'Commercial', 'w10': 'Residential', 'w8': 'Commercial', 'w9': 'Residential'}
df["Duty"] = df["Battery Name"].map(duty)
df["Duty"] = pd.Categorical(df["Duty"])

initC = {'g1': '3C', 'v4': 'C/4', 'v5': '1C', 'w10': '3C', 'w8': 'C/2', 'w9': '1C'}
df["First Life Charge Rate"] = df["Battery Name"].map(initC)
df["First Life Charge Rate"] = pd.Categorical(df["First Life Charge Rate"])

df['Current Cycles'] = np.where(df['RPT Number'] >= 4, 10, 5)
df['Current Cycles'] = np.where(df['RPT Number'] == 16, 0, df['Current Cycles'])


df['Calendar'] = np.where((df['RPT Number'] == 11) | (df['RPT Number'] == 13) | (df['RPT Number'] == 15) | (df['RPT Number'] == 16), "Calendar", "No Calendar")
df['Temperature'] = np.where(df['RPT Number'] % 2 == 0, 20, 35)

pastCycle = {'g1': 354, 'v4': 386, 'v5': 171, 'w10': 492, 'w8': 489, 'w9': 483}
df["Past Cycles"] = df["Battery Name"].map(pastCycle)
new_values = []
cyc = df['Current Cycles'].to_numpy()

for i in range(len(df)):
    if i >= 5:
        new_values = np.append(new_values, cyc[i] + new_values[i - 5])
    else:
        new_values = np.append(new_values, cyc[i] + df.loc[i, "Past Cycles"])
df["Past Cycles"] = new_values

df['Past Discharge Capacity'] = df['Discharge Capacity'].shift(5)
df.loc[0, 'Past Discharge Capacity'] = 4.47613382339478
df.loc[1, 'Past Discharge Capacity'] = 4.43612957000732
df.loc[2, 'Past Discharge Capacity'] = 4.51626205444336
df.loc[3, 'Past Discharge Capacity'] = 4.32101917266846
df.loc[4, 'Past Discharge Capacity'] = 4.34257030487061

df = pd.get_dummies(df, columns=['Duty'], prefix='', prefix_sep='')

df = pd.get_dummies(df, columns=['Calendar'], prefix='', prefix_sep='')

# df.to_csv('info.csv', index=False)
print(df)

dat = pd.read_csv("info.csv")
info2 = pd.concat([dat, df])
info2['Calendar'] = info2['Calendar'].fillna(0).astype('bool')
info2 = info2.rename(columns={'Capacity': 'Charge Capacity'})
info2 = info2.drop('Charge Capacity', axis=1) # axis=1 or axis='columns' specifies dropping a column

print(info2)
info2.to_csv('Data/Analysis Data/info2.csv', index=False)

# This code was written to format the rawData from cycles 17-19 and place the processed data in info2.csv
# The code was written this way because the formats for cycles 1-16 and cycles 17-19 are slightly different. 