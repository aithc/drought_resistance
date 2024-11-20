#!/usr/bin/env python
# coding: utf-8

import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import rioxarray
import glob


# ### 1 spei data

with xr.open_dataset(r'data/spei/spei06.nc')  as data:
    spei = data['spei']
spei

# #### 1.2  thresholds

spei_annual = spei.groupby(spei.time.dt.year).mean()
spei_annual

## detrend
spei_annual_trend = spei_annual.polyfit(dim='year', deg=1)

spei_annual_detrend = spei_annual.copy()
for i in range(1, 122):
    spei_annual_detrend[i] = spei_annual_detrend[i] - spei_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
spei_annual_detrend

spei_annual_detrend_low = spei_annual_detrend.quantile(0.1, dim='year')
spei_annual_detrend_high = spei_annual_detrend.quantile(0.9, dim='year')

spei_annual_drought = spei_annual_detrend.where(spei_annual_detrend < spei_annual_detrend_low)
spei_annual_wet = spei_annual_detrend.where(spei_annual_detrend > spei_annual_detrend_high)

spei_annual_normal = spei_annual_detrend.where(spei_annual_detrend > spei_annual_detrend_low)
spei_annual_normal = spei_annual_normal.where(spei_annual_normal < spei_annual_detrend_high)

spei_annual_drought


spei_annual_drought.to_netcdf(r'result_data/spei_annual_drought.nc')
spei_annual_wet.to_netcdf(r'result_data/spei_annual_wet.nc')
spei_annual_normal.to_netcdf(r'result_data/spei_annual_normal.nc')

spei_annual_detrend_low.to_netcdf(r'result_data/spei_annual_threshold.nc')


# #### 1.3 richness

with rioxarray.open_rasterio(r'D:/data/plant_richness/S_mean_raster.tif') as data:
    plant_richness = data.where(data > 0)
plant_richness

plant_richness = plant_richness.coarsen(x =20, y =20).mean()
plant_richness

lat_list = np.arange(-89.75, 90, 0.5)
lon_list = np.arange(-179.75, 180, 0.5)

plant_richness = xr.DataArray(plant_richness[0].values[::-1,:], coords=[lat_list, lon_list], dims = ['lat','lon'])
plant_richness

plant_richness_log = xr.DataArray(np.log(plant_richness.values), coords=[plant_richness.lat, plant_richness.lon], dims = ['lat','lon'])
plant_richness_log

plant_richness_log.name = 'richness'
plant_richness_log.to_netcdf(r'result_data/plant_richness_log_05.nc')


# #### 1.4  ndvi data

gimms_path = glob.glob(r'gimms_3g_2022/*.nc4')
len(gimms_path)

## landcover
with xr.open_dataset(r'modis_landcover/landcover_gimms_mask.nc').modis_landcover as ld:
    landCover = ld
landCover

gimms_ndvi = []
for gimms_path_n in gimms_path:
    print(gimms_path_n)
    with  xr.open_dataset(gimms_path_n)  as data:
        gimms_ndvi_n = data['ndvi'].where(data['percentile'] <= 2000)
        gimms_ndvi_n = gimms_ndvi_n * 0.0001
        gimms_ndvi_n = gimms_ndvi_n.where( gimms_ndvi_n > 0)
        gimms_ndvi_n = gimms_ndvi_n.where( landCover > 0)
        gimms_ndvi_n = gimms_ndvi_n.coarsen(lat=6,lon=6).mean()
        gimms_ndvi.append( gimms_ndvi_n )
gimms_ndvi = xr.concat(gimms_ndvi, dim='time')

gimms_ndvi = gimms_ndvi.where(plant_richness>0)
gimms_ndvi

gimms_ndvi.sel(lat=60,lon = 50, method='nearest').plot()

### na
def na_count(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return np.isnan(data).sum()

gimms_ndvi_na_count = xr.apply_ufunc(
    na_count,
    gimms_ndvi,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)

gimms_ndvi_month = gimms_ndvi.resample(time="1MS").max()
gimms_ndvi_month

gimms_ndvi_na_month = xr.apply_ufunc(
    na_count,
    gimms_ndvi_month,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)

gimms_ndvi_na_month.plot()

gimms_ndvi_na_month.where(gimms_ndvi_na_month == 0).plot()

gimms_ndvi_month_north = gimms_ndvi_month.sel(lat = slice(45,90))
gimms_ndvi_month_south = gimms_ndvi_month.sel(lat = slice(-90,45))

gimms_ndvi_month_north = gimms_ndvi_month_north.sel(time = gimms_ndvi_month_north.time.dt.month.isin([5,6,7,8,9]))

gimms_ndvi_month_north



gimms_ndvi_na_north = xr.apply_ufunc(
    na_count,
    gimms_ndvi_month_north,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)


gimms_ndvi_month_north = gimms_ndvi_month_north.where(gimms_ndvi_na_north == 0)

gimms_ndvi_annual_north = gimms_ndvi_month_north.groupby('time.year').mean()

gimms_ndvi_annual_north


gimms_ndvi_na_south = xr.apply_ufunc(
    na_count,
    gimms_ndvi_month_south,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)


gimms_ndvi_month_south = gimms_ndvi_month_south.where(gimms_ndvi_na_south == 0)

gimms_ndvi_annual_south = gimms_ndvi_month_south.groupby('time.year').mean()
gimms_ndvi_annual_south

gimms_ndvi_annual = xr.concat([gimms_ndvi_annual_south, gimms_ndvi_annual_north], dim='lat')
gimms_ndvi_annual

gimms_ndvi_annual.name = 'ndvi_annual_mean'
gimms_ndvi_annual.to_netcdf(r'result_data/gimms_ndvi_annual.nc')


# #### 1.5 detrend

gimms_ndvi_annual_trend = gimms_ndvi_annual.polyfit(dim='year', deg=1)

gimms_ndvi_annual_detrend = gimms_ndvi_annual.copy()
for i in range(1, 41):
    gimms_ndvi_annual_detrend[i] = gimms_ndvi_annual_detrend[i] - gimms_ndvi_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
gimms_ndvi_annual_detrend

gimms_ndvi_annual_detrend[0:10].plot(col = 'year', col_wrap = 5, center = False)


# ### 2 resistance

spei_normal_use = spei_annual_normal.sel(year = slice(1982, 2022))
spei_drought_use = spei_annual_drought.sel(year = slice(1982, 2022))

gimms_ndvi_annual_detrend = gimms_ndvi_annual_detrend.where(gimms_ndvi_annual_detrend >0)
gimms_ndvi_annual_detrend = gimms_ndvi_annual_detrend.where(gimms_ndvi_annual_detrend <1)

ndvi_normal_mean = gimms_ndvi_annual_detrend.where(spei_normal_use > - 5).mean(dim='year')
ndvi_normal_mean

gimms_resistance = ndvi_normal_mean / (gimms_ndvi_annual_detrend.where( spei_drought_use> -10) - ndvi_normal_mean)
gimms_resistance.values = np.abs(gimms_resistance.values)
gimms_resistance


gimms_resistance_mean = gimms_resistance.mean(dim='year')
gimms_resistance_mean


gimms_resistance = gimms_resistance.transpose("year", "lat", "lon")
gimms_resistance



gimms_resistance = xr.DataArray(gimms_resistance.values, coords=[gimms_resistance.year,gimms_resistance.lat,gimms_resistance.lon], dims=['year','lat','lon'])

gimms_resistance.name = 'gimms_resistance'
gimms_resistance.to_netcdf(r'result_data/gimms_resistance.nc')


# #### 2.2 recover

gimms_drought_year = (gimms_ndvi_annual_detrend.where( spei_drought_use> -10) - ndvi_normal_mean)
gimms_drought_year = gimms_drought_year.sel(year = slice(1982, 2020))
gimms_drought_year


gimms_drought_year = xr.DataArray(gimms_drought_year.values,coords=[np.arange(1984,2023),gimms_drought_year.lat,gimms_drought_year.lon], dims=['year','lat','lon'])
gimms_drought_year

spei_drought_use_2year = spei_drought_use.sel(year = slice(1982,2020))
spei_drought_use_2year = xr.DataArray(spei_drought_use_2year.values, coords=[np.arange(1984,2023), spei_drought_use_2year.lat, spei_drought_use_2year.lon], dims=['year','lat','lon'])
spei_drought_use_2year

gimms_drought_2year = (gimms_ndvi_annual_detrend.sel(year=slice(1984,2022)).where( spei_drought_use_2year > -10) - ndvi_normal_mean)
gimms_drought_2year

gimms_resilience = gimms_drought_year / gimms_drought_2year
gimms_resilience.values = np.abs(gimms_resilience.values)
gimms_resilience


gimms_resilience_mean = gimms_resilience.mean(dim='year')
gimms_resilience_mean

gimms_resilience.name = 'gimms_resilience'
gimms_resilience.to_netcdf(r'result_data/gimms_resilience.nc')

