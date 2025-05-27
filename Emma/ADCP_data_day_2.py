import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import pandas as pd
import gsw_xarray as gsw
from cmocean import cm as cmo  
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
from pyproj import Geod

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import Weather_data

############## DAG 1 ##################
file_path = r"C:/Users/emmfr/Fysisk oceanografi/OC4920/PROJECT/Data/adcp0507_postpro20may.txt"

# 1) read everything after the 12‑line file header
raw = pd.read_csv(file_path, sep="\t", skiprows=12, engine="python")

# 2) drop the first two in‑file rows (units + bin numbers)
raw = raw.iloc[2:].reset_index(drop=True)

# 3) convert european decimals (comma→dot) *only* where needed
raw = raw.map(lambda x: str(x).replace(",", "."))
numeric_cols = raw.columns.drop(['    "FLat"', '    "FLon"', '    "LLat"', '    "LLon"'])
raw[numeric_cols] = raw[numeric_cols].astype(float)

# 4) build east & north matrices (time × depth 10)
east  = raw[[f"Eas{'.'+str(i) if i else ''}" for i in range(10)]].to_numpy()
north = raw[[f"Nor{'.'+str(i) if i else ''}" for i in range(10)]].to_numpy()

# 5) time index
t = [dt.datetime(2000+int(y), int(m), int(d), int(HH), int(MM), int(SS))
     for y, m, d, HH, MM, SS in raw[['YR','MO','DA','HH','MM','SS']].to_numpy()]

depth = np.arange(8, 8+4*10, 4)        # 8,12,…,44 m

east_da  = xr.DataArray(east,  dims=["time","depth"],
                        coords={"time":t,"depth":depth}, name="east_velocity")
north_da = xr.DataArray(north, dims=["time","depth"],
                        coords={"time":t,"depth":depth}, name="north_velocity")

# 6) lat/lon as float
lat = raw['    "FLat"'].astype(float)
lon = raw['    "FLon"'].astype(float)

# 7) merge & clean
ds = xr.merge([east_da, north_da])
ds = ds.where(np.abs(ds) < 500)          # spike filter
ds = ds/1000                             # mm/s → m/s
ds = ds.assign_coords(lat=("time", lat), lon=("time", lon))


# Split dataset:
split_pos = lat.argmax()

lat_up = lat[:split_pos]
lon_up = lon[:split_pos]

lat_down = lat[split_pos:]
lon_down = lon[split_pos:]

ds_up = ds.isel(time=slice(0, split_pos))
ds_down = ds.isel(time=slice(split_pos, None))





# Calculate speed and direction:

lon_up = ds_up['lon'].values
lat_up = ds_up['lat'].values

geod = Geod(ellps="WGS84")

_, _, distances_up = geod.inv(lon_up[:-1], lat_up[:-1], lon_up[1:], lat_up[1:])

dist_m_up = np.concatenate(([0], np.cumsum(distances_up)))


lon_down = ds_down['lon'].values
lat_down = ds_down['lat'].values

geod = Geod(ellps="WGS84")

_, _, distances_down = geod.inv(lon_down[:-1], lat_down[:-1], lon_down[1:], lat_down[1:])

dist_m_down = np.concatenate(([0], np.cumsum(distances_down)))


ds_up['speed'] = np.sqrt(ds_up['east_velocity']**2 + ds_up['north_velocity']**2)
ds_up['direction'] = (np.arctan2(ds_up['east_velocity'], ds_up['north_velocity']) * 180 / np.pi) % 360

ds_down['speed'] = np.sqrt(ds_down['east_velocity']**2 + ds_down['north_velocity']**2)
ds_down['direction'] = (np.arctan2(ds_down['east_velocity'], ds_down['north_velocity']) * 180 / np.pi) % 360


variables = ["speed", "direction"]
titles = ["Current Speed", "Current Direction"]
cmaps = ['viridis', 'twilight']
units = ["(m/s)", "(°)"]


# Plot:

'''
fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)

for i, (ds, dist_m, label) in enumerate(zip([ds_up, ds_down], [dist_m_up, dist_m_down], ['Up', 'Down'])):
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


'''
fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)

for j, (var, title, cmap, unit) in enumerate(zip(variables, titles, cmaps, units)):
    ds_up[var].T.plot(
        ax=axs[0, j],
        cmap=cmap,
        yincrease=False,
        cbar_kwargs={'label': f'{title} {unit}'}
    )
    axs[0, j].set_title(f"Up - {title}")


for j, (var, title, cmap, unit) in enumerate(zip(variables, titles, cmaps, units)):
    ds_down[var].T.plot(
        ax=axs[1, j],
        cmap=cmap,
        yincrease=False,
        cbar_kwargs={'label': f'{title} {unit}'}
    )
    axs[1, j].set_title(f"Down - {title}")

plt.show()'''