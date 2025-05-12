import Access_data as data

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gsw
import os
import pandas as pd


df_may_8 = data.df_may_8 # Skagen to GBG
df_may_6 = data.df_may_6 # GBG to Skagen


df_combined = pd.concat([df_may_6, df_may_8], ignore_index=True)
df_combined = df_combined.sort_values(by='Time').reset_index(drop=True)

df_combined['Time'] = df_combined['Time'] + pd.Timedelta(hours=2)

p_dbar = df_combined['pressure'] / 100 
depth = gsw.z_from_p(p_dbar, df_combined['Latitude'])
df_combined['depth_m'] = -depth  # z är negativ i gsw




# Plot:

lon = df_combined['Longitude']
lat = df_combined['Latitude']

variables = ['Temp_SBE45', 'Salinity_SBE45']
titles = ['Temperature (°C)', 'Salinity (psu)']
cmaps = ['coolwarm', 'viridis']

# Area
extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]
'''
fig, axs = plt.subplots(1, 2, figsize=(16, 12), subplot_kw={'projection': ccrs.Mercator()})
axs = axs.flatten()

for i, ax in enumerate(axs):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = gl.right_labels = False
    if i % 2 != 0:
        gl.left_labels = False
    if i < 2:
        gl.bottom_labels = False

    # Plot variables as scatter
    sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                    s=30, transform=ccrs.PlateCarree())

    # Colormap
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(titles[i])

    ax.set_title(titles[i])

plt.tight_layout()
plt.show()
'''

# Calculate boutancy gardient (b_x)

#density = df_combined['']

g = 9.81  

depth_target = 10
idx_depth = np.argmin(np.abs(depth - depth_target))

rho_10m = density[idx_depth, :]

rho_0 = np.nanmean(rho_10m)

buoyancy = -g * (rho_10m - rho_0) / rho_0

dx = 1.5e3  

db_dx = np.gradient(buoyancy, dx)
