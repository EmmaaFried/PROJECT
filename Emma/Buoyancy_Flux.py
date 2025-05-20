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
import matplotlib.colors as mcolors


df_combined = Buoyancy_Gradient.df_combined
b_x = Buoyancy_Gradient.b_x
#b_x_smooth = Buoyancy_Gradient.b_x_smooth # Rolling mean

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


'''
rho0 = df_combined['density'] # sea water density
cross_dot = tau_y * b_x - tau_x * b_x
EBF = cross_dot.values / (rho0 * f)
df_combined['EBF'] = EBF
'''


Cp = 4000    # J/kg/K
alpha = 1e-4 # 1/K
g = 9.81      # m/s^2
f_fast = 1.22e-4

Q_ekman = (b_x.values * tau.values * Cp) / (f_fast * alpha * g)


'''
bx = 5e-7
t = 0.5

Q_ek_test = (bx*t*Cp)/(f_fast*alpha*g)

print(Q_ek_test) # = 1671.1 for tau = 0.1 and = 8355.5 for tau = 0.47 (tau.max())

'''


# Vilken ska användas, EBF eller Q_ekman?

dist = Buoyancy_Gradient.dist

##########################################
### Remove values outside of first CTD ###

lon_cutoff = 11.52267

mask = lon <= lon_cutoff

lat_filtered = lat[mask]
lon_filtered = lon[mask]
Q_ekman_filtered = Q_ekman[mask]
dist_filtered = dist[mask]

# Number of nan values

num_nans = np.isnan(Q_ekman_filtered).sum()
#print(len(Q_ekman_filtered))
#print(f"Antal NaN i Q_ekman_filtered: {num_nans}")

#########################################

'''
plt.figure(figsize=(10, 5))
plt.plot(dist_filtered/1000, Q_ekman_filtered)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Distance (km)')
plt.ylabel('W/m2')
plt.title('Q_ekman')
plt.legend()
plt.tight_layout()
plt.show()
'''

# Histogram
'''
plt.hist(Q_ekman_filtered, bins=1000)
plt.xlabel('Q_ekman (W/m²)')
plt.ylabel('Frekvens')
plt.title('Fördelning av Ekmanflöde')
plt.grid(True)
plt.xlim(-10000,10000)
plt.show()
'''
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

sc = ax.scatter(lon_filtered, lat_filtered, c=Q_ekman_filtered, vmin = -500, vmax = 500 ,cmap='coolwarm', s=30, transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('W/m^2')

ax.set_title('Ekman Buoyancy Flux ')

plt.show()
'''


# Two colorbars: 
# TODO: Put them closer and make it go higher for the reds, like 50 000 or something or maybe just the max and min value as vmax and v min?
'''
normal_mask = (Q_ekman_filtered >= -1600) & (Q_ekman_filtered <= 1600)
extreme_mask = (Q_ekman_filtered < -1600) | (Q_ekman_filtered > 1600)

lon_normal = lon_filtered[normal_mask]
lat_normal = lat_filtered[normal_mask]
Q_normal = Q_ekman_filtered[normal_mask]

lon_extreme = lon_filtered[extreme_mask]
lat_extreme = lat_filtered[extreme_mask]
Q_extreme = Q_ekman_filtered[extreme_mask]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.Mercator()})
ax.set_extent([lon_filtered.min()-0.05, lon_filtered.max()+0.05,
               lat_filtered.min()-0.05, lat_filtered.max()+0.05], crs=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')
gl.top_labels = gl.right_labels = False
gl.left_labels = True
gl.bottom_labels = True

sc1 = ax.scatter(lon_normal, lat_normal, c=Q_normal, cmap='Blues',
                 vmin=-1000, vmax=1000, s=30, transform=ccrs.PlateCarree())

sc2 = ax.scatter(lon_extreme, lat_extreme, c=Q_extreme, cmap='Reds',
                 vmin=-10000, vmax=10000, s=30, transform=ccrs.PlateCarree())

cbar1 = plt.colorbar(sc1, ax=ax, orientation='vertical', fraction=0.046, pad=0.21)
cbar1.set_label('Q_ekman (W/m²)')

cbar2 = plt.colorbar(sc2, ax=ax, orientation='vertical', fraction=0.046, pad=0.02)
cbar2.set_label('Q_ekman (W/m²)')

ax.set_title('Ekman Buoyancy Flux')
plt.show()
'''