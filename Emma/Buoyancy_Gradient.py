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

variables = ['Temp_in_SBE38', 'Salinity_SBE45'] 
titles = ['Temperature (°C)', 'Salinity (psu)']
cmaps = ['coolwarm', 'viridis']

extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]

temp_range = [10, 12]  
salinity_range = [22, 30]  

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

    if i == 0:  # Temperature plot
        sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=temp_range[0], vmax=temp_range[1])
    elif i == 1:  # Salinity plot
        sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=salinity_range[0], vmax=salinity_range[1])

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(titles[i])

    ax.set_title(titles[i])

plt.tight_layout()
plt.show()
'''


'''------------------------------------------------------------------------------------------------------------------------------------'''


# Calculate boutancy gardient (b_x)

temp = df_combined['Temp_in_SBE38'] 
salinity = df_combined['Salinity_SBE45']

g = 9.81  

depth_target = 10
idx_depth = np.argmin(np.abs(depth - depth_target))

# Omvandla praktisk salinitet (PSU) till absolut salinitet (SA)
SA = gsw.SA_from_SP(df_combined['Salinity_SBE45'], p_dbar, df_combined['Longitude'], df_combined['Latitude'])

# Omvandla temperatur till konservativ temperatur (CT)
CT = gsw.CT_from_t(SA, df_combined['Temp_in_SBE38'], p_dbar)

rho = gsw.rho(SA, CT, p_dbar)

df_combined['density'] = rho


# Calculate b_x:

rho = df_combined['density'].values

rho_0 = np.nanmean(rho)
g = 9.81
buoyancy = -g * (rho - rho_0) / rho_0

# 1. Geodetiskt avstånd mellan punkter (N-1)
dx = gsw.distance(lon, lat)  # meter mellan punkter

# 2. Kumulativ distans för varje punkt (N)
x = np.insert(np.cumsum(dx), 0, 0)  # avstånd från start

# 3. Gradient av buoyancy mht position
b_x = np.gradient(buoyancy, x)

df_combined['b_x'] = b_x

# Rolling mean:
df_combined['b_x_smooth'] = pd.Series(b_x).rolling(window=5, center=True, min_periods=1).mean()

b_x_smooth = df_combined['b_x_smooth']

# Plot b_x:
'''
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})

extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]
ax.set_extent(extent, crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = gl.right_labels = False
gl.left_labels = True
gl.bottom_labels = True

vmin, vmax = -1e-7, 1e-7 

sc = ax.scatter(lon, lat, c=df_combined['b_x'], cmap='coolwarm', 
                vmin=vmin, vmax=vmax, s=30, transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('bₓ (s⁻²)')

ax.set_title('Horisontell buoyancy-gradient bₓ')

plt.show()
'''