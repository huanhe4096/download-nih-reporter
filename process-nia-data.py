#%% load all libs
import pandas as pd

print("* loaded libraries successfully")



#%% load all data
df = pd.read_csv('nih-all-grants.tsv', sep='\t')
print(f"Data shape: {df.shape}")

df.head()


#%% load all NIA grants
print("* loading all NIA grants")
df_nia = df[df['agency_ic_admin'] == 'NIA']
print(f"NIA grants shape: {df_nia.shape}")

df_nia.head()


#%% save all NIA grants
df_nia.to_csv('nih-nia-grants.tsv', sep='\t', index=False)
print('* saved all NIA grants successfully')

#%%