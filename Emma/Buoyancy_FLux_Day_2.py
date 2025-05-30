import Access_data
import Weather_data

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

df_day_2_ferry_box = Access_data.df_may_7
df_day_2_weather = Weather_data.df_may_7

p_dbar = df_day_2_ferry_box['pressure'] / 100 
depth = gsw.z_from_p(p_dbar, df_day_2_ferry_box['Latitude'])
df_day_2_ferry_box['depth_m'] = -depth

lon = df_day_2_ferry_box['Longitude']
lat = df_day_2_ferry_box['Latitude']

extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]



# Calculate boutancy gardient (b_x)

temp = df_day_2_ferry_box['Temp_in_SBE38'] 
salinity = df_day_2_ferry_box['Salinity_SBE45']

g = 9.81  

depth_target = 10
idx_depth = np.argmin(np.abs(depth - depth_target))

# Omvandla praktisk salinitet (PSU) till absolut salinitet (SA)
SA = gsw.SA_from_SP(df_day_2_ferry_box['Salinity_SBE45'], p_dbar, df_day_2_ferry_box['Longitude'], df_day_2_ferry_box['Latitude'])

# Omvandla temperatur till konservativ temperatur (CT)
CT = gsw.CT_from_t(SA, df_day_2_ferry_box['Temp_in_SBE38'], p_dbar)

rho = gsw.rho(SA, CT, p_dbar)

df_day_2_ferry_box['density'] = rho


# Plot temp and salinity (day 2)


lon = df_day_2_ferry_box['Longitude']
lat = df_day_2_ferry_box['Latitude']

df_day_2_ferry_box['rho'] = rho
df_day_2_ferry_box['temp'] = CT     
df_day_2_ferry_box['salt'] = SA

variables = ['temp', 'salt', 'rho'] 
titles = ['Temperature (°C)', r'Salinity (g kg$^{-1}$)', r'Density (kg m$^{-3}$)']
cmaps = ['coolwarm', 'viridis', 'ocean_r'] 


extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]

temp_range = [9.5, 11.5]  
salinity_range = [27, 31]  
density_range = [1020,1027] 


fig, axs = plt.subplots(1, 3, figsize=(16, 12), subplot_kw={'projection': ccrs.Mercator()})
axs = axs.flatten()

for i, ax in enumerate(axs):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = gl.right_labels = False
    if i % 2 != 0:
        gl.left_labels = True
    if i < 2:
        gl.bottom_labels = True

    if i == 0:  # Temperature plot
        sc = ax.scatter(lon, lat, c=df_day_2_ferry_box[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=temp_range[0], vmax=temp_range[1])
    elif i == 1:  # Salinity plot
        sc = ax.scatter(lon, lat, c=df_day_2_ferry_box[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=salinity_range[0], vmax=salinity_range[1])
    elif i == 2:  # Density plot
        sc = ax.scatter(lon, lat, c=df_day_2_ferry_box[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=density_range[0], vmax=density_range[1])

    # Adjust colorbar size and font
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.08, pad=0.12)
    cbar.set_label(titles[i], fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    ax.set_title(titles[i], fontsize=14)
    ax.tick_params(labelsize=14)

plt.show()



# Calculate b_x:

SA = gsw.SA_from_SP(df_day_2_ferry_box['Salinity_SBE45'], p_dbar, df_day_2_ferry_box['Longitude'], df_day_2_ferry_box['Latitude'])

CT = gsw.CT_from_t(SA, df_day_2_ferry_box['Temp_in_SBE38'], p_dbar)

df_day_2_ferry_box['sigma0'] = gsw.sigma0(SA, CT) 

geod = Geod(ellps="WGS84")
lon  = df_day_2_ferry_box["Longitude"].to_numpy()
lat  = df_day_2_ferry_box["Latitude"].to_numpy()
_, _, step = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
dist = np.concatenate(([0.0], np.cumsum(step))) 

rho0 = 1025.0
g =  9.81

# buoyancy 
rho = df_day_2_ferry_box["sigma0"].to_numpy() + 1000.0
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
df_day_2_ferry_box["b_x"] = np.interp(dist, d_reg, dbdx_smooth,
                                left=np.nan, right=np.nan)


b_x = df_day_2_ferry_box['b_x']

split_pos = lat.argmax() 

lat_up = lat[:split_pos]
lon_up = lon[:split_pos]
b_x_up = df_day_2_ferry_box['b_x'][:split_pos]

lat_down = lat[split_pos:]
lon_down = lon[split_pos:]
b_x_down = df_day_2_ferry_box['b_x'][split_pos:]
'''
vmin, vmax = -1e-6, 1e-6

fig, axes = plt.subplots(1, 2, figsize=(10, 16),
                         subplot_kw={'projection': ccrs.Mercator()})

for ax, b_x_part, lat_part, lon_part, title in zip(
    axes,
    [b_x_up, b_x_down],
    [lat_up, lat_down],
    [lon_up, lon_down],
    ['(up)', '(down)']
):
    ax.set_extent([lon.min()-0.05, lon.max()+0.05,
                   lat.min()-0.05, lat.max()+0.05], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    sc = ax.scatter(lon_part, lat_part, c=b_x_part, cmap='coolwarm',
                    vmin=vmin, vmax=vmax, s=30, transform=ccrs.PlateCarree())

    ax.set_title(f'Horisontell buoyancy-gradient {title}')

cbar = plt.colorbar(sc, ax=axes, orientation='vertical', fraction=0.046, pad=0.14)
cbar.set_label('bₓ (s⁻²)')

plt.show()

'''

###########################################
###########    Calculate EBF    ###########

Cp = 4000    # J/kg/K
alpha = 1e-4 # 1/K
g = 9.81      # m/s^2
f_fast = 1.22e-4

weather_data = Weather_data.df_may_7

weather_data['ts'] = pd.to_datetime(weather_data['ts'])

df_weather_matched = weather_data[weather_data['ts'].isin(df_day_2_ferry_box['datetime'])].copy()

tau = df_weather_matched['tau']

tau_up = tau[:split_pos]
tau_down = tau[split_pos:]

Q_ekman_up = (b_x_up.values * tau_up.values * Cp) / (f_fast * alpha * g)
Q_ekman_down = (b_x_down.values * tau_down.values * Cp) / (f_fast * alpha * g)


'''

vmin, vmax = -1e-6, 1e-6

fig, axes = plt.subplots(1, 2, figsize=(10, 16),
                         subplot_kw={'projection': ccrs.Mercator()})

for ax, EBF_part, lat_part, lon_part, title in zip(
    axes,
    [Q_ekman_up, Q_ekman_down],
    [lat_up, lat_down],
    [lon_up, lon_down],
    ['(up)', '(down)']
):
    ax.set_extent([lon.min()-0.05, lon.max()+0.05,
                   lat.min()-0.05, lat.max()+0.05], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    sc = ax.scatter(lon_part, lat_part, c=EBF_part, cmap='coolwarm', vmin = -4000, vmax = 4000,
                    s=30, transform=ccrs.PlateCarree())

    ax.set_title(f'Q_ekman {title}')

cbar = plt.colorbar(sc, ax=axes, orientation='vertical', fraction=0.046, pad=0.14)
cbar.set_label('W/m^2')

plt.show()'''