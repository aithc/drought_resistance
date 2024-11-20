#!/usr/bin/env python
# coding: utf-8

# ##  drought chars

import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs


# ### read data


with xr.open_dataset(r'data/spei/spei06.nc')  as data:
    spei = data['spei']
spei

spei_time = spei.time

spei_time 


## detrend
spei = xr.DataArray(spei.values, coords=[np.arange(1464), spei.lat, spei.lon], dims=['month','lat','lon'])
spei_trend = spei.polyfit(dim='month', deg=1)


(spei_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, cmap = 'PiYG')


spei_detrend = spei.copy()
for i in range(1, 1464):
    spei_detrend[i] = spei_detrend[i] - spei_trend['polyfit_coefficients'].sel(degree = 1) * i 
spei_detrend


### threshold
spei_low = spei_detrend.quantile(0.1, dim='month')

spei_low 


spei_drought = spei_detrend.where(spei_detrend < spei_low)
spei_drought



spei_drought = xr.DataArray(spei_drought.values, coords=[spei_time, spei_drought.lat, spei_drought.lon], dims=['time','lat','lon'])
spei_drought


def get_drought(data):
    data_values = np.asarray(data.copy())
    data_values[~np.isnan(data_values)] = 0
    data_values[np.isnan(data_values)] = 1

    data_use = np.concatenate(([1], data_values, [1]))

    count = 0
    dura = 1
    durations = []
    severity = []
    for i in range(len(data_use)-1):
        if (data_use[i] == 1) & (data_use[i+1] == 0):
            count += 1
            sever_sum = data[i]
        if (data_use[i] == 0) & (data_use[i+1] == 0):
            dura += 1
            sever_sum = sever_sum + data[i]
        if (data_use[i] == 0) & (data_use[i+1] == 1):
            durations.append(dura)
            dura = 1
            severity.append(sever_sum)

    if len(durations) > 0:
        dura_mean = np.mean(durations)
    else:
        dura_mean = 0
    
    severity_use = np.asarray(severity) / np.asarray(durations)
    
    if len(severity_use) > 0:
        sever_mean = np.mean(severity_use)
    else:
        sever_mean = 0


    return count, dura_mean, sever_mean



spei_drought_use = spei_drought.sel(time = (spei_drought.time.dt.year > 1981))

spei_drought_use


drought_char = xr.apply_ufunc(
    get_drought,
    spei_drought_use,
    input_core_dims=[['time']],
    output_core_dims=[[],[],[]],
    vectorize= True,
    output_dtypes = ['int','float64','float64']
)


drought_char


drought_chars_new = xr.Dataset({'severity':drought_char[2], 
                           'count':drought_char[0],
                           'duration':drought_char[1]})


drought_chars_new.to_netcdf(r'result_data/drought_chars_1982_2022_new.nc')



# ### 15 year moving

spei_drought_use

drought_counts = []
drought_duras = []
drought_severs = []

for year_n in range(1989,2016):
    print(year_n)
    spei_drought_n = spei_drought_use.sel(time = (spei_drought_use.time.dt.year > (year_n - 8) ))
    spei_drought_n = spei_drought_n.sel(time = (spei_drought_n.time.dt.year < (year_n + 8) ))

    drought_char = xr.apply_ufunc(
        get_drought,
        spei_drought_n,
        input_core_dims=[['time']],
        output_core_dims=[[],[],[]],
        vectorize= True,
        output_dtypes = ['int','float64','float64']
    )

    drought_counts.append(drought_char[0])
    drought_duras.append(drought_char[1])
    drought_severs.append(drought_char[2])


drought_counts = xr.concat(drought_counts, dim='year')
drought_duras = xr.concat(drought_duras, dim='year')
drought_severs = xr.concat(drought_severs, dim='year')


drought_counts = xr.DataArray(drought_counts.values, coords= [np.arange(1989,2016),drought_counts.lat, drought_counts.lon], dims=['year','lat','lon'])
drought_duras = xr.DataArray(drought_duras.values, coords= [np.arange(1989,2016),drought_duras.lat, drought_duras.lon], dims=['year','lat','lon'])
drought_severs = xr.DataArray(drought_severs.values, coords= [np.arange(1989,2016),drought_severs.lat, drought_severs.lon], dims=['year','lat','lon'])


drought_counts.name = 'drought_count'
drought_duras.name = 'drought_duration'
drought_severs.name = 'drought_severity'

drought_counts.to_netcdf('result_data/drought_counts_change.nc')
drought_duras.to_netcdf('result_data/drought_duration_change.nc')
drought_severs.to_netcdf('result_data/drought_severity_change.nc')


# #### trend

import pymannkendall as mk
def mk_trend_ve(x):
    if np.isnan(x).sum() > 15:
        return (np.nan ,np.nan)
    else :
        mk_result = mk.original_test(x)
        slope = mk_result.slope
        p = mk_result.p
        return (slope ,p)


with xr.open_dataset('result_data/drought_counts_change.nc')['drought_count'] as data:
    drought_counts = data.where(data > 0)
with xr.open_dataset('result_data/drought_duration_change.nc')['drought_duration'] as data:
    drought_duras = data.where(data > 0)
with xr.open_dataset('result_data/drought_severity_change.nc')['drought_severity'] as data:
    drought_severs = data.where(data < 0)


drought_count_trend = xr.apply_ufunc(
    mk_trend_ve,
    drought_counts,
    input_core_dims = [['year']],
    output_core_dims = [[],[]],
    vectorize=True
)


drought_dura_trend = xr.apply_ufunc(
    mk_trend_ve,
    drought_duras,
    input_core_dims = [['year']],
    output_core_dims = [[],[]],
    vectorize=True
)



drought_sev_trend = xr.apply_ufunc(
    mk_trend_ve,
    drought_severs,
    input_core_dims = [['year']],
    output_core_dims = [[],[]],
    vectorize=True
)


drought_count_trend_ds = xr.Dataset({'trend':drought_count_trend[0], 'p_val':drought_count_trend[1]})
drought_dura_trend_ds = xr.Dataset({'trend':drought_dura_trend[0], 'p_val':drought_dura_trend[1]})
drought_sev_trend_ds = xr.Dataset({'trend':drought_sev_trend[0], 'p_val':drought_sev_trend[1]})


drought_count_trend_ds.to_netcdf(r'result_data/drought_count_trend.nc')
drought_dura_trend_ds.to_netcdf(r'result_data/drought_dura_trend.nc')
drought_sev_trend_ds.to_netcdf(r'result_data/drought_sev_trend.nc')



