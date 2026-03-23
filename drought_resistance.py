import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import pandas as pd
import rioxarray
import glob
from datetime import datetime



with xr.open_dataset(r'../result_data/spei_nt_annual_drought.nc') as data:
    spei_nt_annual_drought = data['spei']
with xr.open_dataset(r'../result_data/spei_nt_annual_wet.nc') as data:
    spei_nt_annual_wet = data['spei']
with xr.open_dataset(r'../result_data/spei_nt_annual_normal.nc') as data:
    spei_nt_annual_normal = data['spei']


spei_nt_annual_drought

# %%
spei_nt_annual_drought[0].plot()

# %%
with xr.open_dataset(r'../result_data/spei_sh_annual_drought.nc') as data:
    print(data)

# %%
with xr.open_dataset(r'../result_data/spei_sh_annual_drought.nc') as data:
    spei_sh_annual_drought = data['__xarray_dataarray_variable__']
    spei_sh_annual_drought.name = 'spei'
with xr.open_dataset(r'../result_data/spei_sh_annual_wet.nc') as data:
    spei_sh_annual_wet = data['__xarray_dataarray_variable__']
    spei_sh_annual_wet.name = 'spei'
with xr.open_dataset(r'../result_data/spei_sh_annual_normal.nc') as data:
    spei_sh_annual_normal = data['__xarray_dataarray_variable__']
    spei_sh_annual_normal.name = 'spei'

# %%
spei_sh_annual_drought[0].plot()



with xr.open_dataset(r'../result_data/plant_richness_log_05.nc') as data:
    plant_richness_log = data['richness']
plant_richness_log


plant_richness_log.plot()

# %%
with xr.open_dataset(r'../result_data/kndvi_tr_annual.nc') as data:
    kndvi_tr_annual = data['kndvi_annual_mean']
with xr.open_dataset(r'../result_data/kndvi_sh_annual.nc') as data:
    kndvi_sh_annual = data['kndvi_annual_mean']
with xr.open_dataset(r'../result_data/kndvi_nh_annual.nc') as data:
    kndvi_nh_annual = data['kndvi_annual_mean']




kndvi_sh_annual_trend = kndvi_sh_annual.polyfit(dim='year', deg=1)

# %%
(kndvi_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG', vmax = 0.002, vmin= -0.002)

# %%
kndvi_sh_annual_detrend = kndvi_sh_annual.copy()
for i in range(1, 39):
    kndvi_sh_annual_detrend[i] = kndvi_sh_annual_detrend[i] - kndvi_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
kndvi_sh_annual_detrend

# %%
kndvi_sh_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)


kndvi_nh_annual_trend = kndvi_nh_annual.polyfit(dim='year', deg=1)
(kndvi_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG', vmax = 0.002, vmin= -0.002)

# %%
kndvi_nh_annual_detrend = kndvi_nh_annual.copy()
for i in range(1, 40):
    kndvi_nh_annual_detrend[i] = kndvi_nh_annual_detrend[i] - kndvi_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
kndvi_nh_annual_detrend

# %%
kndvi_nh_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)


kndvi_tr_annual_trend = kndvi_tr_annual.polyfit(dim='year', deg=1)
(kndvi_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG', vmax = 0.002, vmin= -0.002)

# %%
kndvi_tr_annual_detrend = kndvi_tr_annual.copy()
for i in range(1, 40):
    kndvi_tr_annual_detrend[i] = kndvi_tr_annual_detrend[i] - kndvi_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
kndvi_tr_annual_detrend

# %%
kndvi_tr_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)

# %%
kndvi_nt_annual_detrend = xr.concat([kndvi_nh_annual_detrend, kndvi_tr_annual_detrend], dim='lat').sortby('lat')
kndvi_nt_annual_detrend

# %%
kndvi_nt_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)



# %%
spei_nt_normal_use_kndvi = spei_nt_annual_normal.sel(year = slice(1982, 2021))
spei_nt_drought_use_kndvi = spei_nt_annual_drought.sel(year = slice(1982, 2021))

spei_sh_normal_use_kndvi = spei_sh_annual_normal.sel(year = slice(1982, 2020))
spei_sh_drought_use_kndvi = spei_sh_annual_drought.sel(year = slice(1982, 2020))

spei_nt_normal_use_kndvi_after2000 = spei_nt_annual_normal.sel(year = slice(2000, 2021))
spei_nt_drought_use_kndvi_after2000 = spei_nt_annual_drought.sel(year = slice(2000, 2021))

spei_sh_normal_use_kndvi_after2000 = spei_sh_annual_normal.sel(year = slice(2000, 2020))
spei_sh_drought_use_kndvi_after2000 = spei_sh_annual_drought.sel(year = slice(2000, 2020))

# %%
spei_nt_normal_use_kndvi.plot.hist()

# %%
spei_nt_normal_use_kndvi.min()

# %%
spei_sh_normal_use_kndvi.plot.hist()

# %%
spei_sh_normal_use_kndvi.min()

# %%
kndvi_nt_annual_detrend = kndvi_nt_annual_detrend.where(kndvi_nt_annual_detrend >0)
kndvi_sh_annual_detrend = kndvi_sh_annual_detrend.where(kndvi_sh_annual_detrend >0)
kndvi_nt_normal_mean = kndvi_nt_annual_detrend.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
kndvi_nt_normal_mean

# %%
kndvi_nt_normal_mean.plot(center=False)

# %%
kndvi_sh_normal_mean = kndvi_sh_annual_detrend.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
kndvi_sh_normal_mean

# %%
kndvi_sh_normal_mean.plot(center=False)

# %%
kndvi_nt_normal_mean_after2000 = kndvi_nt_annual_detrend.where(spei_nt_normal_use_kndvi_after2000 > - 5).mean(dim='year')
kndvi_nt_normal_mean_after2000

# %%
kndvi_nt_normal_mean_after2000.plot(center=False)

# %%
kndvi_sh_normal_mean_after2000 = kndvi_sh_annual_detrend.where(spei_sh_normal_use_kndvi_after2000 > - 5).mean(dim='year')
kndvi_sh_normal_mean_after2000

# %%
kndvi_sh_normal_mean_after2000.plot(center=False)

# %%
spei_nt_drought_use_kndvi.min()

# %%
spei_sh_drought_use_kndvi.min()

# %%
kndvi_nt_resistance = kndvi_nt_normal_mean / (kndvi_nt_annual_detrend.where( spei_nt_drought_use_kndvi > -10) - kndvi_nt_normal_mean)
kndvi_nt_resistance = kndvi_nt_resistance.where(kndvi_nt_resistance < 0)
kndvi_nt_resistance.values = np.abs(kndvi_nt_resistance.values)
kndvi_nt_resistance

# %%
kndvi_nt_resistance.quantile(0.90)

# %%
kndvi_nt_resistance.where(kndvi_nt_resistance < 500).plot.hist()

# %%
np.log(kndvi_nt_resistance).plot.hist()

# %%
kndvi_nt_resistance.sel(year = slice(1982,1991)).plot(col = 'year', col_wrap = 5, vmax = 70)

# %%
kndvi_nt_resistance_mean = kndvi_nt_resistance.mean(dim='year').drop('quantile')
kndvi_nt_resistance_mean

# %%
kndvi_nt_resistance_mean.plot(vmax = 70)

# %%
kndvi_nt_resistance = kndvi_nt_resistance.transpose("year", "lat", "lon").drop('quantile')
kndvi_nt_resistance

# %%
kndvi_sh_resistance = kndvi_sh_normal_mean / (kndvi_sh_annual_detrend.where( spei_sh_drought_use_kndvi > -10) - kndvi_sh_normal_mean)
kndvi_sh_resistance = kndvi_sh_resistance.where(kndvi_sh_resistance < 0)
kndvi_sh_resistance.values = np.abs(kndvi_sh_resistance.values)
kndvi_sh_resistance

# %%
kndvi_sh_resistance.where(kndvi_sh_resistance < 500).plot.hist()

# %%
kndvi_sh_resistance.sel(year = slice(1987,1996)).plot(col = 'year', col_wrap = 5, vmax = 70)

# %%
kndvi_sh_resistance_mean = kndvi_sh_resistance.mean(dim='year').drop('quantile')
kndvi_sh_resistance_mean

# %%
kndvi_sh_resistance_mean.plot(vmax = 70)

# %%
kndvi_sh_resistance = kndvi_sh_resistance.transpose("year", "lat", "lon").drop('quantile')
kndvi_sh_resistance

# %%
kndvi_nt_resistance.name = 'kndvi_resistance'
kndvi_nt_resistance.to_netcdf(r'./python_output/fsc_drought/kndvi_nt_resistance.nc')

kndvi_sh_resistance.name = 'kndvi_resistance'
kndvi_sh_resistance.to_netcdf(r'./python_output/fsc_drought/kndvi_sh_resistance.nc')

# %%
kndvi_nt_resistance2 = kndvi_nt_annual_detrend.where( spei_nt_drought_use_kndvi > -10) / kndvi_nt_normal_mean
kndvi_nt_resistance2 = kndvi_nt_resistance2.where(kndvi_nt_resistance2 < 1)
kndvi_nt_resistance2

# %%
kndvi_nt_resistance2.quantile(0.90)

# %%
kndvi_nt_resistance2.plot.hist()

# %%
np.log(kndvi_nt_resistance2).plot.hist()

# %%
(kndvi_nt_resistance2**3).plot.hist()

# %%
kndvi_nt_resistance2.sel(year = slice(1982,1991)).plot(col = 'year', col_wrap = 5, vmax = 1)

# %%
kndvi_nt_resistance_mean2 = kndvi_nt_resistance2.mean(dim='year').drop('quantile')
kndvi_nt_resistance_mean2

# %%
kndvi_nt_resistance_mean2.plot(vmax = 0.95,vmin = 0.5)

# %%
kndvi_nt_resistance2 = kndvi_nt_resistance2.transpose("year", "lat", "lon").drop('quantile')
kndvi_nt_resistance2

# %%
kndvi_sh_resistance2 = kndvi_sh_annual_detrend.where( spei_sh_drought_use_kndvi > -10) / kndvi_sh_normal_mean
kndvi_sh_resistance2 = kndvi_sh_resistance2.where(kndvi_sh_resistance2 < 1)
kndvi_sh_resistance2

# %%
kndvi_sh_resistance2.plot.hist()

# %%
kndvi_sh_resistance2.sel(year = slice(1987,1996)).plot(col = 'year', col_wrap = 5, vmax = 1)

# %%
kndvi_sh_resistance2_mean = kndvi_sh_resistance2.mean(dim='year').drop('quantile')
kndvi_sh_resistance2_mean

# %%
kndvi_sh_resistance2_mean.plot()

# %%
kndvi_sh_resistance2 = kndvi_sh_resistance2.transpose("year", "lat", "lon").drop('quantile')
kndvi_sh_resistance2

# %%
kndvi_nt_resistance2.name = 'kndvi_resistance'
kndvi_nt_resistance2.to_netcdf(r'./python_output/fsc_drought/kndvi_nt_resistance2.nc')

kndvi_sh_resistance2.name = 'kndvi_resistance'
kndvi_sh_resistance2.to_netcdf(r'./python_output/fsc_drought/kndvi_sh_resistance2.nc')

# %% [markdown]
# ### 2.4 用2000之后的时间段

# %%
kndvi_nt_resistance_after2000 = kndvi_nt_normal_mean_after2000 / (kndvi_nt_annual_detrend.where( spei_nt_drought_use_kndvi_after2000 > -10) - kndvi_nt_normal_mean_after2000)
kndvi_nt_resistance_after2000 = kndvi_nt_resistance_after2000.where(kndvi_nt_resistance_after2000 < 0)
kndvi_nt_resistance_after2000.values = np.abs(kndvi_nt_resistance_after2000.values)
kndvi_nt_resistance_after2000

# %%
kndvi_nt_resistance_after2000.quantile(0.90)

# %%
kndvi_nt_resistance_after2000.where(kndvi_nt_resistance_after2000 < 500).plot.hist()

# %%
np.log(kndvi_nt_resistance_after2000).plot.hist()

# %%
kndvi_nt_resistance_after2000.sel(year = slice(2000,2009)).plot(col = 'year', col_wrap = 5, vmax = 70)

# %%
kndvi_nt_resistance_mean_after2000 = kndvi_nt_resistance_after2000.mean(dim='year').drop('quantile')
kndvi_nt_resistance_mean_after2000

# %%
kndvi_nt_resistance_mean_after2000.plot(vmax = 70)

# %%
kndvi_nt_resistance_after2000 = kndvi_nt_resistance_after2000.transpose("year", "lat", "lon").drop('quantile')
kndvi_nt_resistance_after2000

# %%
kndvi_sh_resistance_after2000 = kndvi_sh_normal_mean_after2000 / (kndvi_sh_annual_detrend.where( spei_sh_drought_use_kndvi_after2000 > -10) - kndvi_sh_normal_mean_after2000)
kndvi_sh_resistance_after2000 = kndvi_sh_resistance_after2000.where(kndvi_sh_resistance_after2000 < 0)
kndvi_sh_resistance_after2000.values = np.abs(kndvi_sh_resistance_after2000.values)
kndvi_sh_resistance_after2000

# %%
kndvi_sh_resistance_after2000.where(kndvi_sh_resistance_after2000 < 500).plot.hist()

# %%
kndvi_sh_resistance_after2000.sel(year = slice(2000,2009)).plot(col = 'year', col_wrap = 5, vmax = 70)

# %%
kndvi_sh_resistance_mean_after2000 = kndvi_sh_resistance_after2000.mean(dim='year').drop('quantile')
kndvi_sh_resistance_mean_after2000

# %%
kndvi_sh_resistance_mean_after2000.plot(vmax = 70)

# %%
kndvi_sh_resistance_after2000 = kndvi_sh_resistance_after2000.transpose("year", "lat", "lon").drop('quantile')
kndvi_sh_resistance_after2000

# %%
kndvi_nt_resistance_after2000.name = 'kndvi_resistance'
kndvi_nt_resistance_after2000.to_netcdf(r'./python_output/fsc_drought/kndvi_nt_resistance_after2000.nc')

kndvi_sh_resistance_after2000.name = 'kndvi_resistance'
kndvi_sh_resistance_after2000.to_netcdf(r'./python_output/fsc_drought/kndvi_sh_resistance_after2000.nc')

# %%
kndvi_nt_resistance2_after2000 = kndvi_nt_annual_detrend.where( spei_nt_drought_use_kndvi_after2000 > -10 ) / kndvi_nt_normal_mean_after2000
kndvi_nt_resistance2_after2000 = kndvi_nt_resistance2_after2000.where(kndvi_nt_resistance2_after2000 < 1)
kndvi_nt_resistance2_after2000

# %%
kndvi_nt_resistance2_after2000.quantile(0.90)

# %%
kndvi_nt_resistance2_after2000.plot.hist()

# %%
np.exp(kndvi_nt_resistance2_after2000).plot.hist()

# %%
kndvi_nt_resistance2_after2000.sel(year = slice(2000,2009)).plot(col = 'year', col_wrap = 5, vmax = 1)

# %%
kndvi_nt_resistance2_mean_after2000 = kndvi_nt_resistance2_after2000.mean(dim='year').drop('quantile')
kndvi_nt_resistance2_mean_after2000

# %%
kndvi_nt_resistance2_mean_after2000.plot(vmax = 1)

# %%
kndvi_nt_resistance2_after2000 = kndvi_nt_resistance2_after2000.transpose("year", "lat", "lon").drop('quantile')
kndvi_nt_resistance2_after2000

# %%
kndvi_sh_resistance2_after2000 = kndvi_sh_annual_detrend.where( spei_sh_drought_use_kndvi_after2000 > -10) / kndvi_sh_normal_mean_after2000
kndvi_sh_resistance2_after2000 = kndvi_sh_resistance2_after2000.where(kndvi_sh_resistance2_after2000 < 1)
kndvi_sh_resistance2_after2000

# %%
kndvi_sh_resistance2_after2000.plot.hist()

# %%
kndvi_sh_resistance2_after2000.sel(year = slice(2000,2009)).plot(col = 'year', col_wrap = 5, vmax = 1)

# %%
kndvi_sh_resistance2_mean_after2000 = kndvi_sh_resistance2_after2000.mean(dim='year').drop('quantile')
kndvi_sh_resistance2_mean_after2000

# %%
kndvi_sh_resistance2_mean_after2000.plot(vmax = 1)

# %%
kndvi_sh_resistance2_after2000 = kndvi_sh_resistance2_after2000.transpose("year", "lat", "lon").drop('quantile')
kndvi_sh_resistance2_after2000

# %%
kndvi_nt_resistance2_after2000.name = 'kndvi_resistance'
kndvi_nt_resistance2_after2000.to_netcdf(r'./python_output/fsc_drought/kndvi_nt_resistance2_after2000.nc')

kndvi_sh_resistance2_after2000.name = 'kndvi_resistance'
kndvi_sh_resistance2_after2000.to_netcdf(r'./python_output/fsc_drought/kndvi_sh_resistance2_after2000.nc')

# %%
kndvi_sh_resistance2_after2000.to_dataframe()

# %%
kndvi_sh_resistance_after2000.to_dataframe()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(kndvi_nt_resistance2_after2000.values.flatten(), kndvi_nt_resistance_after2000.values.flatten())
axes[0].set_ylim(0,150)
axes[0].set_xlabel('Y_drought / Y_normal')
axes[0].set_ylabel('Y_normal / abs(Y_drought - Y_normal)')

axes[1].scatter(kndvi_nt_resistance2_after2000.values.flatten(), np.log(kndvi_nt_resistance_after2000.values).flatten())
axes[1].set_ylim(0,8)
axes[1].set_xlabel('Y_drought / Y_normal')
axes[1].set_ylabel('ln(Y_normal / abs(Y_drought - Y_normal))')

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].hist(kndvi_nt_resistance2_after2000.values.flatten())
axes[0].set_xlabel('Y_drought / Y_normal')

axes[1].hist( kndvi_nt_resistance_after2000.where(kndvi_nt_resistance_after2000 < 300).values.flatten())
axes[1].set_xlabel('Y_normal / abs(Y_drought - Y_normal)')

axes[2].hist( np.log(kndvi_nt_resistance_after2000.where(kndvi_nt_resistance_after2000 < 300).values).flatten())
axes[2].set_xlabel('ln(Y_normal / abs(Y_drought - Y_normal))')



# %%
with xr.open_dataset('../result_data/sif_tr_annual.nc') as data:
    sif_tr_annual = data['sif_annual_mean']
with xr.open_dataset('../result_data/sif_sh_annual.nc') as data:
    sif_sh_annual = data['sif_annual_mean']
with xr.open_dataset('../result_data/sif_nh_annual.nc') as data:
    sif_nh_annual = data['sif_annual_mean']


sif_sh_annual_trend = sif_sh_annual.polyfit(dim='year', deg=1)
(sif_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG', vmax = 0.002, vmin= -0.002)

# %%
sif_sh_annual_detrend = sif_sh_annual.copy()
for i in range(1, 22):
    sif_sh_annual_detrend[i] = sif_sh_annual_detrend[i] - sif_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sif_sh_annual_detrend

# %%
sif_sh_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)


sif_nh_annual_trend = sif_nh_annual.polyfit(dim='year', deg=1)
(sif_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG', vmax = 0.002, vmin= -0.002)

# %%
sif_nh_annual_detrend = sif_nh_annual.copy()
for i in range(1, 23):
    sif_nh_annual_detrend[i] = sif_nh_annual_detrend[i] - sif_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sif_nh_annual_detrend

# %%
sif_nh_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)


sif_tr_annual_trend = sif_tr_annual.polyfit(dim='year', deg=1)
(sif_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG', vmax = 0.002, vmin= -0.002)

# %%
sif_tr_annual_detrend = sif_tr_annual.copy()
for i in range(1, 23):
    sif_tr_annual_detrend[i] = sif_tr_annual_detrend[i] - sif_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sif_tr_annual_detrend

# %%
sif_tr_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)

# %%
sif_nt_annual_detrend = xr.concat([sif_nh_annual_detrend, sif_tr_annual_detrend], dim='lat').sortby('lat')
sif_nt_annual_detrend

# %%
sif_nt_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)



# %%
spei_nt_normal_use_sif = spei_nt_annual_normal.sel(year = slice(2000, 2022))
spei_nt_drought_use_sif = spei_nt_annual_drought.sel(year = slice(2000, 2022))

spei_sh_normal_use_sif = spei_sh_annual_normal.sel(year = slice(2000, 2021))
spei_sh_drought_use_sif = spei_sh_annual_drought.sel(year = slice(2000, 2021))

# %%
spei_nt_normal_use_sif

# %%
sif_nt_annual_detrend = sif_nt_annual_detrend.where(sif_nt_annual_detrend >0)
sif_sh_annual_detrend = sif_sh_annual_detrend.where(sif_sh_annual_detrend >0)

# %%
sif_nt_normal_mean = sif_nt_annual_detrend.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
sif_nt_normal_mean

# %%
sif_nt_normal_mean.plot(center=False)

# %%
sif_sh_normal_mean = sif_sh_annual_detrend.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
sif_sh_normal_mean

# %%
sif_sh_normal_mean.plot(center=False)

# %%
spei_nt_drought_use_sif.min()

# %%
spei_sh_drought_use_sif.min()

# %%
sif_nt_resistance = sif_nt_normal_mean / (sif_nt_annual_detrend.where( spei_nt_drought_use_sif> -10) - sif_nt_normal_mean)
sif_nt_resistance = sif_nt_resistance.where(sif_nt_resistance < 0)
sif_nt_resistance.values = np.abs(sif_nt_resistance.values)
sif_nt_resistance

# %%
sif_nt_resistance.quantile(0.90)
sif_nt_resistance.where(sif_nt_resistance < 300).plot.hist()

# %%
sif_nt_resistance.sel(year = slice(2001,2010)).plot(col = 'year', col_wrap = 5, vmax = 70)

# %%
sif_nt_resistance_mean = sif_nt_resistance.mean(dim='year').drop('quantile')
sif_nt_resistance_mean

# %%
sif_nt_resistance_mean.plot(vmax = 70)

# %%
sif_nt_resistance = sif_nt_resistance.transpose("year", "lat", "lon").drop('quantile')
sif_nt_resistance

# %%
sif_sh_resistance = sif_sh_normal_mean / (sif_sh_annual_detrend.where( spei_sh_drought_use_sif > -10) - sif_sh_normal_mean)
sif_sh_resistance = sif_sh_resistance.where(sif_sh_resistance < 0)
sif_sh_resistance.values = np.abs(sif_sh_resistance.values)
sif_sh_resistance

# %%
sif_sh_resistance.where(sif_sh_resistance < 300).plot.hist()

# %%
sif_sh_resistance.sel(year = slice(2001,2010)).plot(col = 'year', col_wrap = 5, vmax = 70)

# %%
sif_sh_resistance_mean = sif_sh_resistance.mean(dim='year').drop('quantile')
sif_sh_resistance_mean

# %%
sif_sh_resistance_mean.plot(vmax = 70)

# %%
sif_sh_resistance = sif_sh_resistance.transpose("year", "lat", "lon").drop('quantile')
sif_sh_resistance

# %%
sif_nt_resistance.name = 'sif_resistance'
sif_nt_resistance.to_netcdf(r'./python_output/fsc_drought/sif_nt_resistance.nc')

sif_sh_resistance.name = 'sif_resistance'
sif_sh_resistance.to_netcdf(r'./python_output/fsc_drought/sif_sh_resistance.nc')

# %%
sif_nt_resistance2 = sif_nt_annual_detrend.where( spei_nt_drought_use_sif> -10) / sif_nt_normal_mean
sif_nt_resistance2 = sif_nt_resistance2.where(sif_nt_resistance2 < 1)
sif_nt_resistance2

# %%
sif_nt_resistance2.quantile(0.90)

# %%
sif_nt_resistance2.plot.hist()

# %%
sif_nt_resistance2.sel(year = slice(2001,2010)).plot(col = 'year', col_wrap = 5, vmax = 1)

# %%
sif_nt_resistance2_mean = sif_nt_resistance2.mean(dim='year').drop('quantile')
sif_nt_resistance2_mean

# %%
sif_nt_resistance2_mean.plot(vmax = 1)

# %%
sif_nt_resistance2 = sif_nt_resistance2.transpose("year", "lat", "lon").drop('quantile')
sif_nt_resistance2

# %%
sif_sh_resistance2 = sif_sh_annual_detrend.where( spei_sh_drought_use_sif > -10) / sif_sh_normal_mean
sif_sh_resistance2 = sif_sh_resistance2.where(sif_sh_resistance2 < 1)
sif_sh_resistance2

# %%
sif_sh_resistance2.plot.hist()

# %%
sif_sh_resistance2.sel(year = slice(2001,2010)).plot(col = 'year', col_wrap = 5, vmax = 1)

# %%
sif_sh_resistance2_mean = sif_sh_resistance2.mean(dim='year').drop('quantile')
sif_sh_resistance2_mean

# %%
sif_sh_resistance2_mean.plot(vmax = 1)

# %%
sif_sh_resistance2 = sif_sh_resistance2.transpose("year", "lat", "lon").drop('quantile')
sif_sh_resistance2

# %%
sif_nt_resistance2.name = 'sif_resistance'
sif_nt_resistance2.to_netcdf(r'./python_output/fsc_drought/sif_nt_resistance2.nc')

sif_sh_resistance2.name = 'sif_resistance'
sif_sh_resistance2.to_netcdf(r'./python_output/fsc_drought/sif_sh_resistance2.nc')

# %%
sif_sh_resistance2.where(sif_sh_resistance2 > -1).sum()


