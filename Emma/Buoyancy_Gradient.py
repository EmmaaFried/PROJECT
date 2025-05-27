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
import matplotlib.gridspec as gridspec


df_may_8 = data.df_may_8 # Skagen to GBG
df_may_6 = data.df_may_6 # GBG to Skagen


df_combined = pd.concat([df_may_6, df_may_8], ignore_index=True)
df_combined = df_combined.sort_values(by='Time').reset_index(drop=True)

df_combined['Time'] = df_combined['Time'] + pd.Timedelta(hours=2)

p_dbar = df_combined['pressure'] / 100 
depth = gsw.z_from_p(p_dbar, df_combined['Latitude'])
df_combined['depth_m'] = -depth  # z är negativ i gsw

bad_val = 1652.141541
df_combined.loc[df_combined['Temp_in_SBE38'] == bad_val, 'Temp_in_SBE38'] = 11

#######

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

#######
# Plot:

lon = df_combined['Longitude']
lat = df_combined['Latitude']

df_combined['rho'] = rho
df_combined['temp'] = CT     
df_combined['salt'] = SA

variables = ['temp', 'salt', 'rho'] 
titles = ['Temperature (°C)', 'Salinity (g/kg)', 'Density (kg/m³)']
cmaps = ['coolwarm', 'viridis', 'ocean_r'] 

extent = [lon.min() - 0.05, lon.max() + 0.05, lat.min() - 0.05, lat.max() + 0.05]

temp_range = [10, 12]  
salinity_range = [22, 30]
density_range = [1016,1024] 
ranges = [[10, 12], [22, 30], [1016, 1026]]

plt.rcParams['font.size'] = 14

fig = plt.figure(figsize=(12, 8))

bottom_left = 0.2
bottom_right = 0.8

top_width = 0.3
top_left = bottom_left + (bottom_right - bottom_left) / 2 - top_width / 2
ax_top = fig.add_axes([top_left, 0.55, top_width, 0.35], projection=ccrs.Mercator())

cbar_ax_top = fig.add_axes([top_left + top_width - 0.03, 0.55, 0.015, 0.35])

gs_bottom = gridspec.GridSpec(1, 2, left=bottom_left, right=bottom_right,
                              bottom=0.1, top=0.45, wspace=0.5)

ax_bottom_left = fig.add_subplot(gs_bottom[0, 0], projection=ccrs.Mercator())
ax_bottom_right = fig.add_subplot(gs_bottom[0, 1], projection=ccrs.Mercator())

axes = [ax_top, ax_bottom_left, ax_bottom_right]

# Add colorbar axes: one for each plot, positioned right to each plot
cbar_axes = [
    cbar_ax_top, 
    fig.add_axes([bottom_left + (bottom_right - bottom_left)/2 - 0.065 , 0.1, 0.015, 0.35]),  # salinity (middle plot)
    fig.add_axes([bottom_right + 0.001, 0.1, 0.015, 0.35])  # bottom right plot
]

for i, (ax, var) in enumerate(zip(axes, variables)):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = gl.right_labels = False

    if i == 1:
        gl.left_labels = True
    if i < 2:
        gl.bottom_labels = True
    else:
        gl.bottom_labels = True

    sc = ax.scatter(lon, lat, c=df_combined[var], cmap=cmaps[i], s=30,
                    transform=ccrs.PlateCarree(),
                    vmin=ranges[i][0], vmax=ranges[i][1])

    # Each plot gets its own colorbar now
    cbar = plt.colorbar(sc, cax=cbar_axes[i])
    cbar.set_label(titles[i], fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    ax.set_title(titles[i], fontsize=14)

ax_bottom_right.tick_params(axis='y', labelleft=False)
ax_bottom_right.set_ylabel('')

plt.show()







'''
fig = plt.figure(figsize=(18, 14), constrained_layout=True)

gs = gridspec.GridSpec(4, 4, figure=fig)

# Axes for plots
ax1 = fig.add_subplot(gs[:2, :2], projection=ccrs.Mercator())
ax2 = fig.add_subplot(gs[:2, 2:], projection=ccrs.Mercator())
ax3 = fig.add_subplot(gs[2:, 1:3], projection=ccrs.Mercator())

# Axes for colorbars
cax1 = fig.add_axes([0.4, 0.535, 0.015, 0.40])  # [left, bottom, width, height]
cax2 = fig.add_axes([0.9, 0.535, 0.015, 0.40])
cax3 = fig.add_axes([0.65, 0.055, 0.015, 0.40])

axs = [ax1, ax2, ax3]
caxs = [cax1, cax2, cax3]

variables = ['temp', 'salt', 'rho']
titles = ['Temperature (°C)', 'Salinity (g/kg)', 'Density (kg/m³)']
cmaps = ['coolwarm', 'viridis', 'ocean_r']
ranges = [[10, 12], [22, 30], [1016, 1024]]

for i, (ax, cax) in enumerate(zip(axs, caxs)):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
    gl.top_labels = gl.right_labels = False
    if i == 1:
        gl.left_labels = False
    if i < 2:
        gl.bottom_labels = False
    else:
        gl.bottom_labels = True

    sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                    s=30, transform=ccrs.PlateCarree(),
                    vmin=ranges[i][0], vmax=ranges[i][1])

    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label(titles[i], fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    ax.set_title(titles[i], fontsize=14)

plt.show()
'''




'''
fig, axs = plt.subplots(2, 2, figsize=(18, 14), subplot_kw={'projection': ccrs.Mercator()}, constrained_layout=True)
axs = axs.flatten()

for i in range(3): 
    ax = axs[i]
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

    vmin, vmax = {
        0: temp_range,
        1: salinity_range,
        2: density_range
    }[i]

    sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                    s=30, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.15)
    cbar.set_label(titles[i])
    ax.set_title(titles[i])

# Fourth plot: Track with hourly markers
ax = axs[3]

pos = ax.get_position()  # get current position: Bbox(x0, y0, x1, y1)
new_pos = [pos.x0 + 0.05, pos.y0, pos.width, pos.height]  # move right by 0.1 in figure coords
ax.set_position(new_pos)

ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True

sc = ax.scatter(lon, lat, c=df_combined['Turbidity'], cmap='Pastel2',
                    s=5, transform=ccrs.PlateCarree(),vmin=0,vmax=1)

hourly_df_1 = df_may_6[df_may_6['datetime'].dt.minute == 0]
hourly_df_1['datetime'] = hourly_df_1['datetime'] + pd.Timedelta(hours=2)

ax.scatter(hourly_df_1['Longitude'], hourly_df_1['Latitude'], color='red', s=5, zorder=3, transform=ccrs.PlateCarree(), label = 'May 6')

for idx, row in hourly_df_1.iterrows():
    time_str = row['datetime'].strftime('%H:%M')
    ax.text(row['Longitude'] + 0.01, row['Latitude'] + 0.001, time_str,
            transform=ccrs.PlateCarree(), fontsize=6.5, color='black')
    

hourly_df_2 = df_may_8[df_may_8['datetime'].dt.minute == 0]
hourly_df_2['datetime'] = hourly_df_2['datetime'] + pd.Timedelta(hours=2)

ax.scatter(hourly_df_2['Longitude'], hourly_df_2['Latitude'], color='blue', s=5, zorder=3, transform=ccrs.PlateCarree(), label = 'May 8')

for idx, row in hourly_df_2.iterrows():
    time_str = row['datetime'].strftime('%H:%M')
    ax.text(row['Longitude'] + 0.01, row['Latitude'] + 0.001, time_str,
            transform=ccrs.PlateCarree(), fontsize=6.5, color='black')

ax.set_title("Transect with Hourly Timestamps")

plt.legend()
plt.show()
'''


# Timestaps for transect:
'''
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})

ax.set_extent(extent, crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = gl.right_labels = False
gl.bottom_labels = True
gl.left_labels = True

sc = ax.scatter(lon, lat, c=df_combined['Turbidity'], cmap='Pastel2',
                s=10, transform=ccrs.PlateCarree(), vmin=0, vmax=1)

hourly_df_1 = df_may_6[df_may_6['datetime'].dt.minute == 0].copy()
hourly_df_1['datetime'] = hourly_df_1['datetime'] + pd.Timedelta(hours=2)

ax.scatter(hourly_df_1['Longitude'], hourly_df_1['Latitude'], color='red', s=10, zorder=3,
           transform=ccrs.PlateCarree(), label='May 6')

for idx, row in hourly_df_1.iterrows():
    time_str = row['datetime'].strftime('%H:%M')
    ax.text(row['Longitude'] + 0.01, row['Latitude'] + 0.001, time_str,
            transform=ccrs.PlateCarree(), fontsize=9.5, color='black')

hourly_df_2 = df_may_8[df_may_8['datetime'].dt.minute == 0].copy()
hourly_df_2['datetime'] = hourly_df_2['datetime'] + pd.Timedelta(hours=2)

ax.scatter(hourly_df_2['Longitude'], hourly_df_2['Latitude'], color='blue', s=10, zorder=3,
           transform=ccrs.PlateCarree(), label='May 8')

for idx, row in hourly_df_2.iterrows():
    time_str = row['datetime'].strftime('%H:%M')
    ax.text(row['Longitude'] + 0.01, row['Latitude'] + 0.001, time_str,
            transform=ccrs.PlateCarree(), fontsize=9.5, color='black')

ax.set_title("Transect with Hourly Timestamps")
ax.legend()

plt.show()
'''
######



'''
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
        gl.left_labels = False
    if i < 2:
        gl.bottom_labels = False

    if i == 0:  # Temperature plot
        sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=temp_range[0], vmax=temp_range[1])
    elif i == 1:  # Salinity plot
        sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin=salinity_range[0], vmax=salinity_range[1])
    elif i == 2:  # Density plot
        sc = ax.scatter(lon, lat, c=df_combined[variables[i]], cmap=cmaps[i],
                        s=30, transform=ccrs.PlateCarree(), vmin = density_range[0], vmax = density_range[1])

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.15)
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


# PLOT

# Mask to exclude the first meters
#mask = dist > 1500

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