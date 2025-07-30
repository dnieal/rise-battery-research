import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("rise-battery-research/Data\Analysis Data\info2.csv")
df = pd.get_dummies(df, columns=['Battery Name'], prefix='', prefix_sep='')

print(df)

df.to_csv("rise-battery-research/Data/Analysis Data/infoBatName.csv", index = False)

# Code to one hot encode battery names