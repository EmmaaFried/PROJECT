import ADCP_data_day_1
import ADCP_data_day_3

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import pandas as pd
import gsw_xarray as gsw
from cmocean import cm as cmo  
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
from pyproj import Geod


dtheta_6e = ADCP_data_day_1.dtheta
dtheta_8e = ADCP_data_day_3.dtheta

lon_6e = ADCP_data_day_1.lon_wind
lat_6e = ADCP_data_day_1.lat_wind

lon_8e = ADCP_data_day_3.lon_wind
lat_8e = ADCP_data_day_3.lat_wind

tolerance = 10
around_90_mask_6 = (dtheta_6e >= (90 - tolerance)) & (dtheta_6e <= (90 + tolerance))
around_90_mask_8 = (dtheta_8e >= (90 - tolerance)) & (dtheta_8e <= (90 + tolerance))

nan_mask_6 = np.isnan(dtheta_6e)
valid_mask_6 = ~nan_mask_6

nan_mask_8 = np.isnan(dtheta_8e)
valid_mask_8 = ~nan_mask_8


low_wind_mask_6 = ADCP_data_day_1.low_wind_mask
#low_wind_mask_6 = ~low_wind_mask_6

low_wind_mask_8 = ADCP_data_day_3.low_wind_mask
#low_wind_mask_8 = ~low_wind_mask_8


ds_6 = ADCP_data_day_1.ds
ds_8 = ADCP_data_day_3.ds

wind_6 = ADCP_data_day_1.subset_wind_df
wind_8 = ADCP_data_day_3.subset_wind_df

ds_combined = xr.concat([ds_6, ds_8], dim='time')  


### Vertical plots along transict
'''
variables = ["north_velocity", "east_velocity"]
titles = ["North Velocity", "East Velocity"]

fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

for (j, var), title in zip(enumerate(variables), titles):
    ds_6[var].T.plot(
        ax=axs[0, j],
        cmap=cmo.balance,
        yincrease=False,
        vmax=0.25,
        vmin=-0.25,
        cbar_kwargs={'label': f'{title} (m/s)'}
    )
    axs[0, j].set_title(f"May 6 - {title}")

for (j, var), title in zip(enumerate(variables), titles):
    ds_8[var].T.plot(
        ax=axs[1, j],
        cmap=cmo.balance,
        yincrease=False,
        vmax=0.25,
        vmin=-0.25,
        cbar_kwargs={'label': f'{title} (m/s)'}
    )
    axs[1, j].set_title(f"May 8 - {title}")

plt.show()
'''



### Speed and direction




lon_6 = ds_6['lon'].values
lat_6 = ds_6['lat'].values

geod = Geod(ellps="WGS84")

_, _, distances_6 = geod.inv(lon_6[:-1], lat_6[:-1], lon_6[1:], lat_6[1:])

dist_m_6 = np.concatenate(([0], np.cumsum(distances_6)))


lon_8 = ds_8['lon'].values
lat_8 = ds_8['lat'].values

geod = Geod(ellps="WGS84")

_, _, distances_8 = geod.inv(lon_8[:-1], lat_8[:-1], lon_8[1:], lat_8[1:])

dist_m_8 = np.concatenate(([0], np.cumsum(distances_8)))



ds_6 = ADCP_data_day_1.ds
ds_6['speed'] = np.sqrt(ds_6['east_velocity']**2 + ds_6['north_velocity']**2)
ds_6['direction'] = (np.arctan2(ds_6['east_velocity'], ds_6['north_velocity']) * 180 / np.pi) % 360

ds_8 = ADCP_data_day_3.ds
ds_8['speed'] = np.sqrt(ds_8['east_velocity']**2 + ds_8['north_velocity']**2)
ds_8['direction'] = (np.arctan2(ds_8['east_velocity'], ds_8['north_velocity']) * 180 / np.pi) % 360

wind_6 = ADCP_data_day_1.subset_wind_df
wind_8 = ADCP_data_day_3.subset_wind_df

variables = ["speed", "direction"]
titles = ["Current Speed", "Current Direction"]
cmaps = ['viridis', 'twilight']
units = ["(m/s)", "(°)"]


'''
fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)

for i, (ds, dist_m, label) in enumerate(zip([ds_6, ds_8], [dist_m_6, dist_m_8], ['May 6', 'May 8'])):
    for j, (var, title, cmap, unit) in enumerate(zip(variables, titles, cmaps, units)):
        data = ds[var].values.T  # Shape: (depth, profile)
        depth = ds.depth.values
        X, Y = np.meshgrid(dist_m/1000, depth)

        pcm = axs[i, j].pcolormesh(X, Y, data, cmap=cmap, shading='auto')
        cbar = fig.colorbar(pcm, ax=axs[i, j])
        cbar.set_label(f"{title} {unit}", fontsize=14)
        
        axs[i, j].invert_yaxis()
        axs[i, j].set_xlabel("Distance (km)", fontsize=14)
        axs[i, j].set_ylabel("Depth (m)", fontsize=14)
        axs[i, j].set_title(f"{label} - {title}", fontsize=14)
        axs[i, j].tick_params(labelsize=14)

plt.show()
'''







### PLOT DIFF
'''
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.Mercator())

ax.set_extent([lon_6e.min()-0.1, lon_8e.max()+0.1,
               lat_6e.min()-0.1, lat_8e.max()+0.1],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                  color="gray", linestyle="--")
gl.top_labels = gl.right_labels = False

sc = ax.scatter(lon_6e[valid_mask_6], lat_6e[valid_mask_6],
                c=dtheta_6e[valid_mask_6],
                cmap="seismic",  
                vmin=0, vmax=180,
                s=35,
                transform=ccrs.PlateCarree())

sc = ax.scatter(lon_8e[valid_mask_8], lat_8e[valid_mask_8],
                c=dtheta_8e[valid_mask_8],
                cmap="seismic",  
                vmin=0, vmax=180,
                s=35,
                transform=ccrs.PlateCarree())

ax.scatter(lon_6e[nan_mask_6], lat_6e[nan_mask_6],
           c='white', edgecolors='black',
           s=35, linewidth=0.5,
           transform=ccrs.PlateCarree(), label='Missing Data')

ax.scatter(lon_8e[nan_mask_8], lat_8e[nan_mask_8],
           c='white', edgecolors='black',
           s=35, linewidth=0.5,
           transform=ccrs.PlateCarree())      

ax.set_title("Current–wind direction difference for 8 m depth")

ax.scatter(lon_6e[around_90_mask_6], lat_6e[around_90_mask_6],
           marker='*', color='black', s=40, linewidth=1.2,
           transform=ccrs.PlateCarree(), label='90°±10° difference')

ax.scatter(lon_8e[around_90_mask_8], lat_8e[around_90_mask_8],
           marker='*', color='black', s=40, linewidth=1.2,
           transform=ccrs.PlateCarree())



ax.scatter(lon_6e[low_wind_mask_6], lat_6e[low_wind_mask_6],
           marker='x', color='black', s=40, linewidth=1.5,
           transform=ccrs.PlateCarree(), label='Wind Speed Below 1 m/s')

ax.scatter(lon_8e[low_wind_mask_8], lat_8e[low_wind_mask_8],
           marker='x', color='black', s=40, linewidth=1.5,
           transform=ccrs.PlateCarree())



cbar = plt.colorbar(sc, orientation="vertical", pad=0.04)
cbar.set_label("Absolute angular difference (°)")

plt.legend()
plt.tight_layout()
plt.show()
'''



