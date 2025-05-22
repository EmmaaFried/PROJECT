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
           marker='x', color='black', s=40, linewidth=1.2,
           transform=ccrs.PlateCarree(), label='90°±10° difference')

ax.scatter(lon_8e[around_90_mask_8], lat_8e[around_90_mask_8],
           marker='x', color='black', s=40, linewidth=1.2,
           transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, orientation="vertical", pad=0.04)
cbar.set_label("Absolute angular difference (°)")

plt.legend()
plt.tight_layout()
plt.show()




