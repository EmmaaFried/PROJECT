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
from pyproj import Geod
from scipy.interpolate import interp1d


df_may_8 = data.df_may_8 # Skagen to GBG
df_may_6 = data.df_may_6 # GBG to Skagen


df_combined = pd.concat([df_may_6, df_may_8], ignore_index=True)
df_combined = df_combined.sort_values(by='Time').reset_index(drop=True)

df_combined['Time'] = df_combined['Time'] + pd.Timedelta(hours=2)

p_dbar = df_combined['pressure'] / 100 
depth = gsw.z_from_p(p_dbar, df_combined['Latitude'])
df_combined['depth_m'] = -depth  # z är negativ i gsw




# Plot:

bad_val = 1652.141541
df_combined.loc[df_combined['Temp_in_SBE38'] == bad_val, 'Temp_in_SBE38'] = 11

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
'''
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
'''

# Omvandla praktisk salinitet (PSU) till absolut salinitet (SA)
SA = gsw.SA_from_SP(df_combined['Salinity_SBE45'], p_dbar, df_combined['Longitude'], df_combined['Latitude'])

# Omvandla temperatur till konservativ temperatur (CT)
CT = gsw.CT_from_t(SA, df_combined['Temp_in_SBE38'], p_dbar)

df_combined['sigma0'] = gsw.sigma0(SA, CT) 

geod = Geod(ellps="WGS84")
lon  = df_combined["Longitude"].to_numpy()
lat  = df_combined["Latitude"].to_numpy()
_, _, step = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
dist = np.concatenate(([0.0], np.cumsum(step))) 

rho0 = 1025.0
g =  9.81

# buoyancy 
rho = df_combined["sigma0"].to_numpy() + 1000.0
b   = g * (1 - rho / rho0)

# regular 500 m grid
d_reg = np.arange(0, dist[-1], 500.0)
b_reg = interp1d(dist, b, bounds_error=False, fill_value=np.nan)(d_reg)

# finite-difference gradient then rolling mean 
dbdx_reg = np.gradient(b_reg, 500.0)                 # raw gradient
dbdx_smooth = (pd.Series(dbdx_reg)
               .rolling(5, center=True)              # 5-point window (≈2 km)
               .mean()
               .to_numpy())

#  map back to original rows
df_combined["b_x"] = np.interp(dist, d_reg, dbdx_smooth,
                                left=np.nan, right=np.nan)


b_x = df_combined['b_x']

# PLOT

# Mask to exclude the first 10 meters
mask = dist > 1500

'''
# Plot only data after 10 meters
plt.figure(figsize=(10, 5))
plt.plot(dist[mask], df_combined['b_x'][mask], label='db/dx')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Distance (m)')
plt.ylabel('Buoyancy Gradient (1/s²)')
plt.title('Horizontal Buoyancy Gradient')
plt.legend()
plt.tight_layout()
plt.show()

'''

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