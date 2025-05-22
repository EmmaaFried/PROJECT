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

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import Weather_data

############## DAG 1 ##################
file_path = r"C:/Users/emmfr/Fysisk oceanografi/OC4920/PROJECT/Data/adcp_data_6_maj.txt"

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


# PLOT
'''
fig,axs = plt.subplots(2,1,figsize=(16,8),constrained_layout=True)

variables = ["north_velocity","east_velocity"]

for ax,v in zip(axs,variables):
    ds[v].T.plot(ax=ax,cmap=cmo.balance,yincrease=False,vmax=0.25,vmin=-0.25,cbar_kwargs={'label': '{} (m/s)'.format(v)})

plt.show()
'''

# Plot in map at 8 m depth

u_8m = ds.east_velocity.sel(depth=8, drop=True)    # m s‑1
v_8m = ds.north_velocity.sel(depth=8, drop=True)

direction = (270 - np.degrees(np.arctan2(v_8m, u_8m))) % 360

# 0 = North, 90 = East, 180 = South, 270 = West

lat = ds.lat.values
lon = ds.lon.values

lat_wind = ds.lat.values
lon_wind = ds.lon.values

'''
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.Mercator())

ax.set_extent(
    [lon.min() - 0.05, lon.max() + 0.05,
     lat.min() - 0.05, lat.max() + 0.05],
    crs=ccrs.PlateCarree()
)

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                  color="gray", linestyle="--")
gl.top_labels = gl.right_labels = False

sc = ax.scatter(
    lon, lat,
    c=direction,
    cmap="hsv",          # full 0‑360 hue circle
    vmin=0, vmax=360,
    s=35,
    transform=ccrs.PlateCarree()
)

ax.set_title("Current direction at 8 m depth")

cbar = plt.colorbar(sc, orientation="vertical", pad=0.04)
cbar.set_label("Direction (° from North)")

plt.tight_layout()
plt.show()
'''


# Compare with wind direction

weather_data_6_maj = Weather_data.df_may_6
wind_dir_6_maj = weather_data_6_maj['winddir']

# -------------------------------------------------

curr_da = xr.DataArray(
    (270 - np.degrees(np.arctan2(v_8m, u_8m))) % 360,
    coords={"time": ds.time},  # 87 values
    name="curr_dir",
)

# Round to nearest minute:
times_pd = pd.to_datetime(curr_da['time'].values)
times_rounded = times_pd.round('min')
curr_da['time'] = xr.DataArray(times_rounded.values.astype('datetime64[ns]'), dims=curr_da['time'].dims)

curr_da['time'] = pd.to_datetime(curr_da['time'].values).round('min')

curr_times = pd.to_datetime(curr_da.time.values)     
wind_t  = pd.to_datetime(weather_data_6_maj['ts'])

weather_data_6_maj['ts'] = pd.to_datetime(weather_data_6_maj['ts'])

weather_data_6_maj['ts'] = weather_data_6_maj['ts'] + pd.Timedelta(hours=2) # Ändra tidszon för att matcha ADCP

df = weather_data_6_maj.set_index('ts')
start_time = pd.Timestamp('2025-05-06 09:53:00')
end_time = pd.Timestamp('2025-05-06 17:03:00')
time_index = pd.date_range(start=start_time, end=end_time, freq='5T')
subset_wind_df = df.loc[df.index.isin(time_index)].reset_index()

dtheta = np.abs(((curr_da - subset_wind_df['winddir'] + 180) % 360) - 180) # Diff in wind and current direction


# 90 deg difference: 
tolerance = 10
around_90_mask = (dtheta >= (90 - tolerance)) & (dtheta <= (90 + tolerance))


### PLOT DIFFERENCE ###
'''
nan_mask = np.isnan(dtheta)
valid_mask = ~nan_mask

fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.Mercator())

ax.set_extent([lon.min()-0.1, lon.max()+0.1,
               lat.min()-0.1, lat.max()+0.1],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                  color="gray", linestyle="--")
gl.top_labels = gl.right_labels = False

sc = ax.scatter(lon[valid_mask], lat[valid_mask],
                c=dtheta[valid_mask],
                cmap="seismic",  
                vmin=0, vmax=180,
                s=35,
                transform=ccrs.PlateCarree())

ax.scatter(lon[nan_mask], lat[nan_mask],
           c='white', edgecolors='black',
           s=35, linewidth=0.5,
           transform=ccrs.PlateCarree(), label='Missing Data')   

ax.set_title("Current–wind direction difference for 8 m depth")

ax.scatter(lon[around_90_mask], lat[around_90_mask],
           marker='x', color='black', s=40, linewidth=1.2,
           transform=ccrs.PlateCarree(), label='90°±10° difference')

cbar = plt.colorbar(sc, orientation="vertical", pad=0.04)
cbar.set_label("Absolute angular difference (°)")

plt.legend()
plt.tight_layout()
plt.show()
'''


'''
fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.Mercator())

ax.set_extent([lon.min()-0.05, lon.max()+0.05,
               lat.min()-0.05, lat.max()+0.05],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                  color="gray", linestyle="--")
gl.top_labels = gl.right_labels = False

sc = ax.scatter(
    lon, lat,
    c=wind_da_matched['winddir'],
    cmap="hsv",       
    vmin=0, vmax=360,
    s=35,
    transform=ccrs.PlateCarree()
)


ax.set_title("Wind direction")

cbar = plt.colorbar(sc, orientation="vertical", pad=0.04)
cbar.set_label("Direction (° from North)")

plt.tight_layout()
plt.show()
'''


# Arrows in the direction:

'''
angles_deg = subset_wind_df['winddir'].values
#angles_deg = weather_data_6_maj['winddir'].values
angles_rad = np.deg2rad(angles_deg)
u = np.cos(angles_rad)   
v = np.sin(angles_rad)

lon = subset_wind_df['longitude']
lat = subset_wind_df['latitude']

#lon = weather_data_6_maj['longitude']
#lat = weather_data_6_maj['latitude']

fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.Mercator())

ax.set_extent([lon.min()-0.1, lon.max()+0.1,
               lat.min()-0.1, lat.max()+0.1],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                  color="gray", linestyle="--")
gl.top_labels = gl.right_labels = False

q = ax.quiver(
    lon, lat, u, v,
    angles_deg,            
    cmap="hsv",
    scale=20,              
    width=0.003,
    transform=ccrs.PlateCarree()
)

ax.set_title("Wind direction")
cbar = plt.colorbar(q, orientation="vertical", pad=0.04)
cbar.set_label("Direction (° from North)")

plt.tight_layout()
plt.show()
'''

'''
angles_deg = curr_da.values
angles_rad = np.deg2rad(angles_deg)
u = np.sin(angles_rad)   
v = np.cos(angles_rad)

lon = ds['lon']
lat = ds['lat']

fig = plt.figure(figsize=(8, 10))
ax = plt.axes(projection=ccrs.Mercator())

ax.set_extent([lon.min()-0.1, lon.max()+0.1,
               lat.min()-0.1, lat.max()+0.1],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.5)

gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                  color="gray", linestyle="--")
gl.top_labels = gl.right_labels = False

q = ax.quiver(
    lon, lat, u, v,
    angles_deg,            
    cmap="hsv",
    scale=20,              
    width=0.003,
    transform=ccrs.PlateCarree()
)

ax.set_title("Current direction")
cbar = plt.colorbar(q, orientation="vertical", pad=0.04)
cbar.set_label("Direction (° from North)")

plt.tight_layout()
plt.show()
'''

############################################