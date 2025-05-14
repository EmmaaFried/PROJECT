import Buoyancy_Gradient 
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

df_combined = Buoyancy_Gradient.df_combined
b_x = Buoyancy_Gradient.b_x
b_x_smooth = Buoyancy_Gradient.b_x_smooth # Rolling mean

weather_data = Weather_data.df_may_6_may_8

weather_data['ts'] = pd.to_datetime(weather_data['ts'])
df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])

df_combined.loc[df_combined['datetime'] == '2025-05-08 09:30:01', 'datetime'] = '2025-05-08 09:30:00' 

df_tau_matched = weather_data[weather_data['ts'].isin(df_combined['datetime'])].copy()
tau = df_tau_matched['tau']
tau_x = df_tau_matched['tau_x']
tau_y = df_tau_matched['tau_y']

lat = Buoyancy_Gradient.lat
lon = Buoyancy_Gradient.lon

f = gsw.f(lat)  # s^-1 

# Cite: https://journals.ametsoc.org/view/journals/phoc/53/12/JPO-D-23-0005.1.xml#bib51    --  for Ekman buoyancy flux



rho0 = df_combined['density'] # sea water density

cross_dot = tau_y * b_x - tau_x * b_x

EBF = cross_dot.values / (rho0 * f)

df_combined['EBF'] = EBF



Cp = 4000    # J/kg/K
alpha = 1e-4 # 1/K
g = 9.8      # m/s^2
f_fast = 1.22e-4

Q_ekman = (b_x * tau * Cp) / (f_fast * alpha * g)

# Vilken ska användas, EBF eller Q_ekman

#plt.plot(Q_ekman)
#plt.show()

# Plot:
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

#vmin, vmax = -1e-8, 1e-8 

sc = ax.scatter(lon, lat, c=Q_ekman, cmap='coolwarm', vmin = -500, vmax = 500, s=30, transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('W/m^2')

ax.set_title('Ekman Buoyancy Flux ')

plt.show()
'''