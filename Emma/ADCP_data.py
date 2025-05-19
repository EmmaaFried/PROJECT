import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
import pandas as pd
import gsw_xarray as gsw
from cmocean import cm as cmo  
import cartopy.crs as ccrs
import cartopy.feature as cfeature


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




fig,axs = plt.subplots(2,1,figsize=(16,8),constrained_layout=True)

variables = ["north_velocity","east_velocity"]

for ax,v in zip(axs,variables):
    ds[v].T.plot(ax=ax,cmap=cmo.balance,yincrease=False,vmax=0.25,vmin=-0.25,cbar_kwargs={'label': '{} (m/s)'.format(v)})

plt.show()