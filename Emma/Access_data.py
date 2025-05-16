import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
import os
import pandas as pd

# Folder containing the data files
data_folder = 'Data'

# List of all .txt files in the folder
files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]

# Function to process a single file
def process_file(filepath):
    skiprows = 0
    with open(filepath, 'r', encoding='latin1') as f:
        for i, line in enumerate(f):
            if line.strip().startswith("Date") and "Time" in line:
                skiprows = i
                break

    df = pd.read_csv(
        filepath,
        sep='\s+',
        skiprows=skiprows,
        encoding='latin1'
    )

    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    return df

# Process all files and collect DataFrames in a list
all_dfs = []
for filename in files:
    filepath = os.path.join(data_folder, filename)
    df = process_file(filepath)
    all_dfs.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(all_dfs, ignore_index=True)

df_may_8 = combined_df[combined_df['Date'] == '2025.05.08']
df_may_7 = combined_df[combined_df['Date'] == '2025.05.07']
df_may_6 = combined_df[combined_df['Date'] == '2025.05.06']
