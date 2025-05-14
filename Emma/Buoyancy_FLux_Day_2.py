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


# Calculate b_x:

rho = df_day_2_ferry_box['density'].values

rho_0 = np.nanmean(rho)
g = 9.81
buoyancy = -g * (rho - rho_0) / rho_0

# 1. Geodetiskt avstånd mellan punkter (N-1)
dx = gsw.distance(lon, lat)  # meter mellan punkter

# 2. Kumulativ distans för varje punkt (N)
x = np.insert(np.cumsum(dx), 0, 0)  # avstånd från start

# 3. Gradient av buoyancy med hänsyn till position
b_x = np.gradient(buoyancy, x)

df_day_2_ferry_box['b_x'] = b_x

#max_lat_index = lat.idxmax()

#split_pos = df_day_2_ferry_box.index.get_loc(max_lat_index)

#b_x_up = b_x[:split_pos]  
#b_x_down = b_x[split_pos:]


# PLOT:
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

sc = ax.scatter(lon, lat, c=df_day_2_ferry_box['b_x'], cmap='coolwarm', 
                vmin=vmin, vmax=vmax, s=30, transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('bₓ (s⁻²)')

ax.set_title('Horisontell buoyancy-gradient bₓ')

plt.show()
'''
# Dela upp i två plotar, en för vägen upp och en för vägen ner. Splitta vid det mest nordliga datavärdet. 