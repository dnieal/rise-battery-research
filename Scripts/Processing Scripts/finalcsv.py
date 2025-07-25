import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Data\Analysis Data\info2.csv")
# df = pd.get_dummies(df, columns=['First Life Charge Rate'], prefix='', prefix_sep='')

df["Percent Capacity Decrease"] = (df["Past Discharge Capacity"] - df["Discharge Capacity"])/df["Past Discharge Capacity"] * 100
median = df['Percent Capacity Decrease'].quantile(0.5)
df['Category'] = df['Percent Capacity Decrease'].apply(lambda x: True if x > median else False)

print(df)

plt.hist(df['Percent Capacity Decrease'], bins=12, range = [-1, 2])
plt.show()

# df.to_csv("Data/Analysis Data/info2.csv", index = False)
