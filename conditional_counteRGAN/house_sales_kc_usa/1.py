#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('kc_house_data.csv')
print(df['bathrooms'].value_counts().sort_index())
print(df['bedrooms'].value_counts().sort_index())


