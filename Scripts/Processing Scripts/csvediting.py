import pandas as pd
import numpy as np

df = pd.read_csv("Data/Intermediate Data/rawInfo.csv")

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
df['Current Cycles'] = np.where(df['RPT Number'] == 1, 8, df['Current Cycles'])
df['Current Cycles'] = np.where(df['RPT Number'] == 2, 5, df['Current Cycles'])
df['Current Cycles'] = np.where(df['RPT Number'] == 3, 9, df['Current Cycles'])
df['Current Cycles'] = np.where(df['RPT Number'] == 16, 0, df['Current Cycles'])


df['Calendar'] = np.where((df['RPT Number'] == 11) | (df['RPT Number'] == 13) | (df['RPT Number'] == 15) | (df['RPT Number'] == 16), "Calendar", "No Calendar")
df['Temperature'] = np.where(df['RPT Number'] % 2 == 0, 20, 35)

pastCycle = {'g1': 212, 'v4': 244, 'v5': 29, 'w10': 350, 'w8': 347, 'w9': 341}
df["Past Cycles"] = df["Battery Name"].map(pastCycle)
new_values = []
cyc = df['Current Cycles'].to_numpy()

for i in range(len(df)):
    if i >= 6:
        new_values = np.append(new_values, cyc[i] + new_values[i - 6])
    else:
        new_values = np.append(new_values, cyc[i] + df.loc[i, "Past Cycles"])
df["Past Cycles"] = new_values

df['Past Discharge Capacity'] = df['Discharge Capacity'].shift(6)
df.loc[0, 'Past Discharge Capacity'] = 4.6000
df.loc[1, 'Past Discharge Capacity'] = 4.6024
df.loc[2, 'Past Discharge Capacity'] = 4.7055
df.loc[3, 'Past Discharge Capacity'] = 4.4591
df.loc[4, 'Past Discharge Capacity'] = 4.4568
df.loc[5, 'Past Discharge Capacity'] = 4.4636

df = pd.get_dummies(df, columns=['Duty'], prefix='', prefix_sep='')

df = pd.get_dummies(df, columns=['Calendar'], prefix='', prefix_sep='')

df.to_csv('Data/Intermediate Data/info.csv', index=False)
print(df)

# This code was written to format the rawData from cycles 1-16 and place the processed data in info.csv