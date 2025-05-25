import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
from pathlib import Path
import cmocean as cmo
import numpy as np
import gsw
from pathlib import Path

import Buoyancy_Gradient

# New way thats working
# There is one outlier in the temp data, I will flag it and then interpolate inbetween

# # flag outliers
# bad = df_all['Temp_in_SBE38'] > 100            # any value > 100 °C is impossible  
# df_all.loc[bad, 'Temp_in_SBE38'] = np.nan      # set those outliers to NaN

df_all = Buoyancy_Gradient.df_combined

df_all['Temp_in_SBE38'] = df_all['Temp_in_SBE38'].mask(df_all['Temp_in_SBE38'] > 100)

# # interpolate the gaps
# df_all = df_all.set_index('datetime')              # make time the index  
# df_all['Temp_in_SBE38'] = df_all['Temp_in_SBE38'].interpolate('time')  # linear fill in time  
# df_all = df_all.reset_index()                      # restore datetime as a column

# Convert everything to TEOS-10
SP  = df_all['Salinity_SBE45'].to_numpy()   # practical salinity is what we have in the dataset, will convert to Absolute sal
T   = df_all['Temp_in_SBE38'].to_numpy()    # in-situ temp (°C, ITS-90), will convert to Conservative T
p   = df_all['pressure'].to_numpy() / 100.0    # mbar → dbar  (1 dbar = 100 mbar)
lon = df_all['Longitude'].to_numpy()
lat = df_all['Latitude'].to_numpy()

SA = gsw.SA_from_SP(SP, p, lon, lat)    # absolute salinity
CT = gsw.CT_from_t(SA, T, p)            # conservative temp

# Store everything in the the dataset
df_all['SA'] = SA
df_all['CT'] = CT
df_all['press_db'] =p
df_all['sigma0'] = gsw.sigma0(SA, CT)       # potential density anomaly σθ


import numpy as np
from pyproj import Geod
from scipy.interpolate import interp1d
import gsw

# Constants
g = 9.81                 # gravity (m/s^2)
target_depth = 2.0       # target depth (meters)
depth_window = 0.5       # allowable depth variation (± meters)
dx = 100.0              # horizontal resolution (meters)
geod = Geod(ellps="WGS84")

# Step 1: Convert pressure to depth (m)
df_all['Depth'] = -gsw.z_from_p(df_all['pressure'] / 100, df_all['Latitude'])
depth = df_all["Depth"].to_numpy()

# Step 2: Filter to ~2 m depth
mask = np.abs(depth - target_depth) <= depth_window


if np.sum(mask) < 10:
    raise ValueError("Not enough points near the target depth for reliable gradient computation.")

# Subset the filtered data
df_near_depth = df_all[mask].copy()
lon = df_near_depth["Longitude"].to_numpy()
lat = df_near_depth["Latitude"].to_numpy()
sigma0 = df_near_depth["sigma0"].to_numpy()

# Step 3: Compute density and buoyancy
rho = sigma0 + 1000.0
rho0 = 1025
rho0 = np.nanmean(rho)
b = g * (1 - (rho / rho0))

# Step 4: Compute cumulative distance along track



# regular 500 m grid
d_reg = np.arange(0, Buoyancy_Gradient.dist[-1], 500.0)
b_reg = interp1d(Buoyancy_Gradient.dist, b, bounds_error=False, fill_value=np.nan)(d_reg)

# finite-difference gradient then rolling mean 
dbdx_reg = np.gradient(b_reg, 500.0)                 # raw gradient
dbdx_smooth = (pd.Series(dbdx_reg)
               .rolling(5, center=True)              # 5-point window (≈2 km)
               .mean()
               .to_numpy())

#  map back to original rows
b_x = np.interp(Buoyancy_Gradient.dist, d_reg, dbdx_smooth,
                                left=np.nan, right=np.nan)



'''
# Calculate distances
_, _, step = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
dist = np.concatenate(([0.0], np.cumsum(step)))

# Filter out steps < threshold (e.g., 50 m)
valid = np.concatenate(([True], step > 50))
lon = lon[valid]
lat = lat[valid]
b = b[valid]
dist = np.concatenate(([0.0], np.cumsum(geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])[2])))


# Recalculate step after filtering
_, _, step_filtered = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])


# Step 5: Interpolate buoyancy onto regular grid
d_reg = np.arange(0, dist[-1], dx)
b_reg = interp1d(dist, b, bounds_error=False, fill_value=np.nan)(d_reg)


# Step 6: Compute buoyancy gradient
dbdx_reg = np.gradient(b_reg, dx)



# Step 7: Map gradient back to original points (irregular spacing)
buoy_grad = np.interp(dist, d_reg, dbdx_reg, left=np.nan, right=np.nan)
# Create a filtered dataframe that matches filtered lon/lat/b
df_clean = df_near_depth.iloc[valid].copy()
df_clean["buoy_grad"] = buoy_grad
'''