#%% import libs
import os
import sys
import pandas as pd
import numpy as np

print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print('* loaded libraries successfully')

#%% load data
grants_path = './nih-reporter-grants.tsv'
df = pd.read_csv(grants_path, sep='\t')
print(f"Data shape: {df.shape}")
print(df.head(2))


#%% check agency_ic_admin distrubution
freq = df['agency_ic_admin'].value_counts()
freq = freq.reset_index()
freq.columns = ['agency_ic_admin', 'count']
freq.head(20)