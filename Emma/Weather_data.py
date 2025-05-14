import os
import pandas as pd
import numpy as np

df_w = pd.read_csv('Data/sk_weather.csv')
df_p = pd.read_csv('Data/sk_position.csv')

df_combined = pd.merge(df_p, df_w, on='ts', how='inner')

# Calcuate wind stress TODO KOLLA IGENOM DENNA BERÄKNINGEN!!!!!!!!

def air_density(temp_C, pressure_hPa, humidity_percent):

    T = temp_C + 273.15  # Celsius to Kelvin
    p = pressure_hPa * 100  # hPa to Pa
    RH = humidity_percent / 100.0

    Rd = 287.05     # Gas constant for dry air, J/(kg·K)
    Rv = 461.5      # Gas constant for water vapor, J/(kg·K)
    es = 6.112 * np.exp((17.67 * temp_C) / (temp_C + 243.5)) * 100  # saturation vapor pressure in Pa
    e = RH * es  # actual vapor pressure
    pd = p - e  # partial pressure of dry air

    rho = (pd / (Rd * T)) + (e / (Rv * T))

    return rho

def wind_stress_components(wind_speed, wind_dir_deg, air_temp, air_pressure, humidity):

    theta = np.radians(wind_dir_deg)
    rho_a = air_density(air_temp, air_pressure, humidity)

    # Drag coefficient
    Cd = 0.0015

    tau = rho_a * Cd * wind_speed**2

    # Wind direction is "from" direction, so reverse it
    tau_x = tau * np.sin(theta) * -1
    tau_y = tau * np.cos(theta) * -1

    return tau_x, tau_y, tau

df_combined['tau_x'], df_combined['tau_y'], df_combined['tau'] = zip(*df_combined.apply(
    lambda row: wind_stress_components(row['windspeed'],
                                       row['winddir'],
                                       row['airtemp'],
                                       row['airpressure'],
                                       row['humidity']), axis=1))


df_may_8 = df_combined[df_combined['ts'].astype(str).str.startswith('2025-05-08')]
df_may_7 = df_combined[df_combined['ts'].astype(str).str.startswith('2025-05-07')]
df_may_6 = df_combined[df_combined['ts'].astype(str).str.startswith('2025-05-06')]

df_may_6_may_8 = pd.concat([df_may_8, df_may_6], ignore_index=True)

df_may_6_may_8_tau = df_may_6_may_8['tau']
df_may_6_may_8_tau_x = df_may_6_may_8['tau_x']
df_may_6_may_8_tau_y = df_may_6_may_8['tau_y']
