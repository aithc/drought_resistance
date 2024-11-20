#!/usr/bin/env python
# coding: utf-8


import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import glob
import warnings

warnings.filterwarnings('ignore')

for year_n in range(1979,2023):
    print(year_n)
    sm_year_path = glob.glob(r'soil_moisture_esa_cci/{}/*nc'.format(year_n))
    sm_year_path = sorted(sm_year_path)
    print(len(sm_year_path))

    with xr.open_mfdataset(sm_year_path)['sm'] as data:
        sm_year = data
    
    sm_year_month = sm_year.resample(time='M').median()
    sm_year_month_co = sm_year_month.coarsen(lat=2, lon =2).mean()

    sm_year_month_co.to_netcdf(r'data/esa_soil_moisture/ESACCI_SOILMOISTURE_L3S_SSMV_COMBINED_{}.nc'.format(year_n))


# ### 3 detrend
sm_use_path = glob.glob(r'data/esa_soil_moisture/*nc')
sm_use_path

sm_month_all = []
for sm_path_n in sm_use_path[3:]:
    print(sm_path_n)
    with xr.open_dataset(sm_path_n)['sm'] as data:
        sm_month_all.append(data)

sm_month_all = xr.concat(sm_month_all, dim='time')
sm_month_all


sm_annual = sm_month_all.groupby('time.year').mean()

## detrend
sm_annual_trend = sm_annual.polyfit(dim='year', deg=1)

sm_annual_detrend = sm_annual.copy()
for i in range(1, 41):
    sm_annual_detrend[i] = sm_annual_detrend[i] - sm_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sm_annual_detrend

sm_annual_detrend = sm_annual_detrend.where(sm_annual_detrend > 0)
sm_annual_detrend = sm_annual_detrend.where(sm_annual_detrend < 1)


# ### 4 
with xr.open_dataset(r'result_data/spei_annual_drought.nc')  as  data:
    spei_annual_drought = data['spei'].sel(year = slice(1982,2022))

with xr.open_dataset(r'result_data/spei_annual_normal.nc')  as  data:
    spei_annual_normal = data['spei'].sel(year = slice(1982,2022))


sm_normal_mean = sm_annual_detrend.where(spei_annual_normal > - 5).mean(dim='year')
sm_normal_mean

sm_change = sm_normal_mean / (sm_annual_detrend.where( spei_annual_drought > -10) - sm_normal_mean)
sm_change


sm_change = sm_change.transpose("year", "lat", "lon")
sm_change


sm_change = sm_change.where(sm_change < 0)
sm_change.values = np.abs(sm_change.values)

sm_change

sm_change = sm_change.where(sm_change < 195.06403809)

sm_change_mean = sm_change.mean(dim='year')

sm_change.name = 'sm_change'
sm_change.to_netcdf(r'result_data/sm_change.nc')


sm_change_mean.name = 'sm_change'
sm_change_mean.to_netcdf(r'result_data/sm_change_mean.nc')

