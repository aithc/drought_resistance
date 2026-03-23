# %% [markdown]
# # 读取预处理一下 lst 数据
# 先 算到日均值 保存就行
# 计算变化的时候  可以算成类似于 zscore的东西

# %%
import numpy as np
import xarray as xr
import dask
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import glob
import warnings
from scipy import stats

# %% [markdown]
# ## 预处理 era lst数据

# %% [markdown]
# ### 1.1 小时数据计算成日均  并行读取并保存

# %%
##with xr.open_dataset('E:/lst_download/era5_land_lst.nc' ) as data:
##    print(data)

# %%
##with xr.open_dataset('E:/lst_download/era5_land_lst.nc', chunks = {'latitude':100, 'longitude':200} ) as data:
##    lst_era = data['skt']
##lst_era

# %%
##lst_era = lst_era.rename({'latitude':'lat', 'longitude':'lon', 'valid_time':'time'})
##lst_era

# %%
##lst_era.resample(time='ME').mean().to_netcdf('E:/lst_download/era5_land_lst_daytime.nc', encoding={'skt': {'dtype': 'float32', 'scale_factor': 0.01}} )

# %% [markdown]
# ### 1.2 读取数据

# %%
with xr.open_dataset('E:/lst_download/era5_land_lst_daytime.nc') as data:
    lst_data = data['skt'].drop('number')
lst_data

# %%
lst_data[0].plot()

# %%
lst_data.lon.values[:6]

# %%
lst_data.lon.values[-6:]

# %%
lst_data.lon.values[-1800:0:-1]* -1

# %%
lst_data = lst_data.assign_coords({'lon': np.concatenate([lst_data.lon.values[:1800], lst_data.lon.values[-1800:0:-1]* -1]) })
lst_data

# %%
lst_data = lst_data.sortby('lon')
lst_data

# %%
lst_data[0].plot()

# %% [markdown]
# ### 1.3 处理landcover

# %%
# 自定义众数函数（在groupby_reduce里会用）
def mode_func(x, axis):
    mode_result = stats.mode(x, axis=axis, nan_policy='omit', keepdims=False)
    return mode_result.mode

# %%
## 读取landcover数据
with xr.open_dataset(r'D:/data/modis_landcover/modis_IGBP_2001_2022.nc').modis_landcover as ld:
    landCover = ld
landCover


# %%
def ld_unchanged(ld_st):
    if np.isnan(ld_st).all():
        change = 2
    else:
        ld_len = len(np.unique(ld_st))
        if ld_len > 1:
            change = 1
        else:
            change = 0
    return change  

# %%
modis_ld_01 = landCover.coarsen(lat=2, lon=2).reduce(mode_func)
modis_ld_01

# %%
ld_change_01 =  xr.apply_ufunc(
    ld_unchanged,
    modis_ld_01,
    input_core_dims=[['year']],
    output_core_dims=[[]],
    vectorize= True
)

# %%
modis_ld_01_use = modis_ld_01[-1].where(ld_change_01 == 0)
modis_ld_01_use = modis_ld_01_use.where(modis_ld_01_use >0)
modis_ld_01_use = modis_ld_01_use.where(modis_ld_01_use <11)
modis_ld_01_use = modis_ld_01_use.drop('year')
modis_ld_01_use = modis_ld_01_use.sortby('lat', ascending=False)
modis_ld_01_use

# %%
modis_ld_01_use.plot()

# %%
lst_data = lst_data.interp_like(modis_ld_01_use, method = 'nearest')
lst_data

# %%
lst_data = lst_data.where(modis_ld_01_use >0)
lst_data

# %%
lst_data[0].plot()

# %%
lst_data = lst_data.coarsen(lat=5, lon=5).mean()

# %% [markdown]
# ### 1.4  分南北半球

# %%
lst_sh = lst_data.sel(lat = slice(-23.5,-90))
lst_nh = lst_data.sel(lat = slice(90, 23.5))
lst_tr = lst_data.sel(lat = slice(23.5,-23.5))

# %%
lst_sh

# %%
lst_tr

# %%
lst_sh = xr.DataArray(lst_sh.values, coords=[pd.Series(lst_sh.time.values) - pd.DateOffset(months=6), lst_sh.lat, lst_sh.lon], dims=['time','lat','lon'])
lst_sh

# %%
lst_sh[0:12].plot(x = 'lon',y='lat',col = 'time', col_wrap = 3)

# %%
lst_sh = lst_sh.sel(time = slice('19820101','20211231'))
lst_nh = lst_nh.sel(time = slice('19820101','20221231'))
lst_tr = lst_tr.sel(time = slice('19820101','20221231'))

# %%
lst_sh = lst_sh.sel(time = lst_sh.time.dt.month.isin([5,6,7,8,9]))
lst_nh = lst_nh.sel(time = lst_nh.time.dt.month.isin([5,6,7,8,9]))

# %%
### 统计一下缺失值数量
def na_count(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return np.isnan(data).sum()

# %%
lst_na_sh = xr.apply_ufunc(
    na_count,
    lst_sh,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
lst_na_sh.plot()  ## era的数据 基本没有缺失值

# %%
lst_sh_annual = lst_sh.groupby('time.year').mean()
lst_nh_annual = lst_nh.groupby('time.year').mean()
lst_tr_annual = lst_tr.groupby('time.year').mean()

# %%
lst_sh_annual = lst_sh_annual.transpose('year','lat','lon')
lst_nh_annual = lst_nh_annual.transpose('year','lat','lon')
lst_tr_annual = lst_tr_annual.transpose('year','lat','lon')

# %%
lst_sh_annual.name = 'lst'
lst_nh_annual.name = 'lst'
lst_tr_annual.name = 'lst'

# %%
lst_sh_annual.to_netcdf('E:/python_output/fsc_drought/lst_sh_annual.nc')
lst_nh_annual.to_netcdf('E:/python_output/fsc_drought/lst_nh_annual.nc')
lst_tr_annual.to_netcdf('E:/python_output/fsc_drought/lst_tr_annual.nc')

# %% [markdown]
# ### 1.5 去趋势

# %%
lst_sh_annual_trend = lst_sh_annual.polyfit(dim='year', deg=1)

# %%
(lst_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
lst_sh_annual_detrend = lst_sh_annual.copy()
for i in range(1, 40):
    lst_sh_annual_detrend[i] = lst_sh_annual_detrend[i] - lst_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
lst_sh_annual_detrend

# %%
lst_sh_annual_detrend[0:4].plot(col = 'year')

# %%
lst_nh_annual_trend = lst_nh_annual.polyfit(dim='year', deg=1)
(lst_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
lst_nh_annual_detrend = lst_nh_annual.copy()
for i in range(1, 41):
    lst_nh_annual_detrend[i] = lst_nh_annual_detrend[i] - lst_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
lst_nh_annual_detrend

# %%
lst_nh_annual_detrend[0:4].plot(col = 'year')

# %%
lst_tr_annual_trend = lst_tr_annual.polyfit(dim='year', deg=1)
(lst_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
lst_tr_annual_detrend = lst_tr_annual.copy()
for i in range(1, 41):
    lst_tr_annual_detrend[i] = lst_tr_annual_detrend[i] - lst_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
lst_tr_annual_detrend

# %%
lst_tr_annual_detrend[0:4].plot(col = 'year')

# %%
lst_nt_annual_detrend = xr.concat([lst_nh_annual_detrend, lst_tr_annual_detrend], dim='lat').sortby('lat')
lst_nt_annual_detrend

# %%
lst_nt_annual_detrend[0:4].plot(col='year')

# %% [markdown]
# ## 2 干旱年份的LST变化

# %%
with xr.open_dataset(r'../result_data/spei_nt_annual_drought.nc') as data:
    spei_nt_annual_drought = data['spei']
with xr.open_dataset(r'../result_data/spei_nt_annual_wet.nc') as data:
    spei_nt_annual_wet = data['spei']
with xr.open_dataset(r'../result_data/spei_nt_annual_normal.nc') as data:
    spei_nt_annual_normal = data['spei']
with xr.open_dataset(r'../result_data/spei_sh_annual_drought.nc') as data:
    spei_sh_annual_drought = data['__xarray_dataarray_variable__']
    spei_sh_annual_drought.name = 'spei'
with xr.open_dataset(r'../result_data/spei_sh_annual_wet.nc') as data:
    spei_sh_annual_wet = data['__xarray_dataarray_variable__']
    spei_sh_annual_wet.name = 'spei'
with xr.open_dataset(r'../result_data/spei_sh_annual_normal.nc') as data:
    spei_sh_annual_normal = data['__xarray_dataarray_variable__']
    spei_sh_annual_normal.name = 'spei'

# %% [markdown]
# ### 2.1 kndvi对应的

# %%
spei_nt_normal_use_kndvi = spei_nt_annual_normal.sel(year = slice(1982, 2021))
spei_nt_drought_use_kndvi = spei_nt_annual_drought.sel(year = slice(1982, 2021))

spei_sh_normal_use_kndvi = spei_sh_annual_normal.sel(year = slice(1982, 2020))
spei_sh_drought_use_kndvi = spei_sh_annual_drought.sel(year = slice(1982, 2020))

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance.nc') as data:
    kndvi_nt_resistance = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance.nc') as data:
    kndvi_sh_resistance = data['kndvi_resistance']

# %%
lst_nt_annual_detrend_kndvi = lst_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi, method='nearest')
lst_sh_annual_detrend_kndvi = lst_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi, method='nearest')

# %%
lst_nt_annual_detrend_kndvi

# %%
lst_sh_annual_detrend_kndvi

# %%
lst_nt_normal_mean_kndvi = lst_nt_annual_detrend_kndvi.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
lst_nt_normal_mean_kndvi

# %%
lst_nt_normal_std_kndvi = lst_nt_annual_detrend_kndvi.where(spei_nt_normal_use_kndvi > - 5).std(dim='year')
lst_nt_normal_std_kndvi

# %%
lst_nt_change_kndvi = lst_nt_normal_mean_kndvi / (lst_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - lst_nt_normal_mean_kndvi)
lst_nt_change_kndvi

# %%
lst_nt_change_kndvi = lst_nt_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
lst_nt_change_kndvi

# %%
lst_nt_change_kndvi[0].plot(vmin = -100, vmax = 100)

# %%
lst_nt_change_kndvi.quantile([0.1,0.5,0.9])

# %%
lst_nt_change_kndvi.plot.hist(bins = np.arange(-1200,1200, 200))

# %%
lst_nt_change_kndvi.where(kndvi_nt_resistance > 0).plot.hist(bins = np.arange(-1200,1200, 200))

# %%
lst_nt_change_kndvi = lst_nt_change_kndvi.where(kndvi_nt_resistance > 0)

# %%
np.isfinite(lst_nt_change_kndvi)

# %%
lst_nt_change_kndvi = lst_nt_change_kndvi.where(np.isfinite(lst_nt_change_kndvi)==1)
lst_nt_change_kndvi

# %%
lst_nt_change_kndvi.plot.hist()

# %%
lst_nt_zs_kndvi =  (lst_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - lst_nt_normal_mean_kndvi) / lst_nt_normal_std_kndvi
lst_nt_zs_kndvi

# %%
lst_nt_zs_kndvi[0].plot(vmin = -1, vmax = 1)

# %%
lst_nt_zs_kndvi.quantile([0.1,0.5,0.9])

# %%
lst_nt_zs_kndvi.plot.hist(bins = np.arange(-1,3, 0.2))

# %%
lst_nt_zs_kndvi = lst_nt_zs_kndvi.where(kndvi_nt_resistance > 0)
lst_nt_zs_kndvi.plot.hist()

# %%
lst_sh_normal_mean_kndvi = lst_sh_annual_detrend_kndvi.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
lst_sh_normal_mean_kndvi

# %%
lst_sh_normal_std_kndvi = lst_sh_annual_detrend_kndvi.where(spei_sh_normal_use_kndvi > - 5).std(dim='year')
lst_sh_normal_std_kndvi

# %%
lst_sh_change_kndvi = lst_sh_normal_mean_kndvi / (lst_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - lst_sh_normal_mean_kndvi)
lst_sh_change_kndvi = lst_sh_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
lst_sh_change_kndvi

# %%
lst_sh_change_kndvi = lst_sh_change_kndvi.where(kndvi_sh_resistance > 0)
lst_sh_change_kndvi = lst_sh_change_kndvi.where(np.isfinite(lst_sh_change_kndvi)==1)
lst_sh_change_kndvi

# %%
lst_sh_change_kndvi.plot.hist()

# %%
lst_nt_change_kndvi.quantile(0.95)

# %%
lst_sh_change_kndvi.quantile(0.95)

# %%
lst_nt_change_kndvi[:8].plot(col = 'year', col_wrap = 4, vmax = 1000)

# %%
lst_sh_change_kndvi[:8].plot(col = 'year', col_wrap = 4, vmax = 1000)

# %%
lst_sh_zs_kndvi =  (lst_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - lst_sh_normal_mean_kndvi) / lst_sh_normal_std_kndvi
lst_sh_zs_kndvi

# %%
lst_sh_zs_kndvi[0].plot(vmin = -1, vmax = 1)

# %%
lst_sh_zs_kndvi.quantile([0.1,0.5,0.9])

# %%
lst_sh_zs_kndvi.plot.hist(bins = np.arange(-1,3, 0.2))

# %%
lst_sh_zs_kndvi = lst_sh_zs_kndvi.where(kndvi_sh_resistance > 0)
lst_sh_zs_kndvi.plot.hist()

# %%
lst_nt_change_kndvi.name = 'lst_change'
lst_nt_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/lst_nt_change_kndvi.nc')

lst_sh_change_kndvi.name = 'lst_change'
lst_sh_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/lst_sh_change_kndvi.nc')

lst_nt_zs_kndvi.name = 'lst_zs'
lst_nt_zs_kndvi.to_netcdf(r'E:/python_output/fsc_drought/lst_nt_zs_kndvi.nc')

lst_sh_zs_kndvi.name = 'lst_zs'
lst_sh_zs_kndvi.to_netcdf(r'E:/python_output/fsc_drought/lst_sh_zs_kndvi.nc')

# %% [markdown]
# ### 2.1  kndvi 2000年之后

# %%
spei_nt_normal_use_kndvi_after2000 = spei_nt_annual_normal.sel(year = slice(2000, 2021))
spei_nt_drought_use_kndvi_after2000 = spei_nt_annual_drought.sel(year = slice(2000, 2021))

spei_sh_normal_use_kndvi_after2000 = spei_sh_annual_normal.sel(year = slice(2000, 2020))
spei_sh_drought_use_kndvi_after2000 = spei_sh_annual_drought.sel(year = slice(2000, 2020))

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance_after2000.nc') as data:
    kndvi_nt_resistance_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance_after2000.nc') as data:
    kndvi_sh_resistance_after2000 = data['kndvi_resistance']

# %%
lst_nt_annual_detrend_kndvi_after2000 = lst_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi_after2000, method='nearest')
lst_sh_annual_detrend_kndvi_after2000 = lst_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi_after2000, method='nearest')

# %%
lst_nt_annual_detrend_kndvi_after2000

# %%
lst_sh_annual_detrend_kndvi_after2000

# %%
lst_nt_normal_mean_kndvi_after2000 = lst_nt_annual_detrend_kndvi_after2000.where(spei_nt_normal_use_kndvi_after2000 > - 5).mean(dim='year')
lst_nt_normal_mean_kndvi_after2000

# %%
lst_nt_normal_std_kndvi_after2000 = lst_nt_annual_detrend_kndvi_after2000.where(spei_nt_normal_use_kndvi_after2000 > - 5).std(dim='year')
lst_nt_normal_std_kndvi_after2000

# %%
lst_nt_change_kndvi_after2000 = lst_nt_normal_mean_kndvi_after2000 / (lst_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - lst_nt_normal_mean_kndvi_after2000)
lst_nt_change_kndvi_after2000

# %%
lst_nt_change_kndvi_after2000 = lst_nt_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
lst_nt_change_kndvi_after2000

# %%
lst_nt_change_kndvi_after2000[0].plot(vmin = -100, vmax = 100)

# %%
lst_nt_change_kndvi_after2000.quantile([0.1,0.5,0.9])

# %%
lst_nt_change_kndvi_after2000.plot.hist(bins = np.arange(-1200,1200, 200))

# %%
lst_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0).plot.hist(bins = np.arange(-1200,1200, 200))

# %%
lst_nt_change_kndvi_after2000 = lst_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)

# %%
np.isfinite(lst_nt_change_kndvi_after2000)

# %%
lst_nt_change_kndvi_after2000 = lst_nt_change_kndvi_after2000.where(np.isfinite(lst_nt_change_kndvi_after2000)==1)
lst_nt_change_kndvi_after2000

# %%
lst_nt_change_kndvi_after2000.plot.hist()

# %%
lst_nt_zs_kndvi_after2000 =  (lst_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - lst_nt_normal_mean_kndvi_after2000) / lst_nt_normal_std_kndvi_after2000
lst_nt_zs_kndvi_after2000

# %%
lst_nt_zs_kndvi_after2000[0].plot(vmin = -1, vmax = 1)

# %%
lst_nt_zs_kndvi_after2000.quantile([0.1,0.5,0.9])

# %%
lst_nt_zs_kndvi_after2000.plot.hist(bins = np.arange(-1,3, 0.2))

# %%
lst_nt_zs_kndvi_after2000 = lst_nt_zs_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)
lst_nt_zs_kndvi_after2000.plot.hist()

# %%
lst_sh_normal_mean_kndvi_after2000 = lst_sh_annual_detrend_kndvi_after2000.where(spei_sh_normal_use_kndvi_after2000 > - 5).mean(dim='year')
lst_sh_normal_mean_kndvi_after2000

# %%
lst_sh_normal_std_kndvi_after2000 = lst_sh_annual_detrend_kndvi_after2000.where(spei_sh_normal_use_kndvi_after2000 > - 5).std(dim='year')
lst_sh_normal_std_kndvi_after2000

# %%
lst_sh_change_kndvi_after2000 = lst_sh_normal_mean_kndvi_after2000 / (lst_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - lst_sh_normal_mean_kndvi_after2000)
lst_sh_change_kndvi_after2000 = lst_sh_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
lst_sh_change_kndvi_after2000

# %%
lst_sh_change_kndvi_after2000 = lst_sh_change_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
lst_sh_change_kndvi_after2000 = lst_sh_change_kndvi_after2000.where(np.isfinite(lst_sh_change_kndvi_after2000)==1)
lst_sh_change_kndvi_after2000

# %%
lst_sh_change_kndvi_after2000.plot.hist()

# %%
lst_nt_change_kndvi_after2000.quantile(0.95)

# %%
lst_sh_change_kndvi_after2000.quantile(0.95)

# %%
lst_nt_change_kndvi_after2000[:8].plot(col = 'year', col_wrap = 4, vmax = 1000)

# %%
lst_sh_change_kndvi_after2000[:8].plot(col = 'year', col_wrap = 4, vmax = 1000)

# %%
lst_sh_zs_kndvi_after2000 =  (lst_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - lst_sh_normal_mean_kndvi_after2000) / lst_sh_normal_std_kndvi_after2000
lst_sh_zs_kndvi_after2000

# %%
lst_sh_zs_kndvi_after2000[0].plot(vmin = -1, vmax = 1)

# %%
lst_sh_zs_kndvi_after2000.quantile([0.1,0.5,0.9])

# %%
lst_sh_zs_kndvi_after2000.plot.hist(bins = np.arange(-1,3, 0.2))

# %%
lst_sh_zs_kndvi_after2000 = lst_sh_zs_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
lst_sh_zs_kndvi_after2000.plot.hist()

# %%
lst_nt_change_kndvi_after2000.name = 'lst_change'
lst_nt_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/lst_nt_change_kndvi_after2000.nc')

lst_sh_change_kndvi_after2000.name = 'lst_change'
lst_sh_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/lst_sh_change_kndvi_after2000.nc')

lst_nt_zs_kndvi_after2000.name = 'lst_zs'
lst_nt_zs_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/lst_nt_zs_kndvi_after2000.nc')

lst_sh_zs_kndvi_after2000.name = 'lst_zs'
lst_sh_zs_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/lst_sh_zs_kndvi_after2000.nc')

# %% [markdown]
# ### 2.2 sif 对应的

# %%
spei_nt_normal_use_sif = spei_nt_annual_normal.sel(year = slice(2000, 2022))
spei_nt_drought_use_sif = spei_nt_annual_drought.sel(year = slice(2000, 2022))

spei_sh_normal_use_sif = spei_sh_annual_normal.sel(year = slice(2000, 2022))
spei_sh_drought_use_sif = spei_sh_annual_drought.sel(year = slice(2000, 2022))

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance.nc') as data:
    sif_nt_resistance = data['sif_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance.nc') as data:
    sif_sh_resistance = data['sif_resistance']
sif_nt_resistance

# %%
sif_sh_resistance

# %%
lst_nt_annual_detrend_sif = lst_nt_annual_detrend.interp_like(spei_nt_normal_use_sif, method='nearest')
lst_sh_annual_detrend_sif = lst_sh_annual_detrend.interp_like(spei_sh_normal_use_sif, method='nearest')

# %%
lst_nt_annual_detrend_sif

# %%
spei_sh_normal_use_sif

# %%
lst_sh_annual_detrend_sif

# %%
lst_nt_normal_mean_sif = lst_nt_annual_detrend_sif.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
lst_nt_normal_mean_sif

# %%
lst_nt_normal_std_sif = lst_nt_annual_detrend_sif.where(spei_nt_normal_use_sif > - 5).std(dim='year')
lst_nt_normal_std_sif

# %%
lst_nt_change_sif = lst_nt_normal_mean_sif / (lst_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - lst_nt_normal_mean_sif)
lst_nt_change_sif

# %%
lst_nt_change_sif = lst_nt_change_sif.transpose("year", "lat", "lon").drop('quantile')
lst_nt_change_sif

# %%
lst_nt_change_sif[0].plot(vmin = -100, vmax = 100)

# %%
lst_nt_change_sif.quantile([0.1,0.5,0.9])

# %%
lst_nt_change_sif.plot.hist(bins = np.arange(-1200,1200, 200))

# %%
lst_nt_change_sif.where(sif_nt_resistance > 0).plot.hist(bins = np.arange(-1200,1200, 200))

# %%
lst_nt_change_sif = lst_nt_change_sif.where(sif_nt_resistance > 0)
lst_nt_change_sif = lst_nt_change_sif.where(np.isfinite(lst_nt_change_sif)==1)
lst_nt_change_sif

# %%
lst_nt_change_sif.plot.hist()

# %%
lst_nt_zs_sif =  (lst_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - lst_nt_normal_mean_sif) / lst_nt_normal_std_sif
lst_nt_zs_sif

# %%
lst_nt_zs_sif[0].plot(vmin = -1, vmax = 1)

# %%
lst_nt_zs_sif.quantile([0.1,0.5,0.9])

# %%
lst_nt_zs_sif.plot.hist(bins = np.arange(-1,3, 0.2))

# %%
lst_nt_zs_sif = lst_nt_zs_sif.where(sif_nt_resistance > 0)
lst_nt_zs_sif.plot.hist()

# %%
lst_sh_normal_mean_sif = lst_sh_annual_detrend_sif.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
lst_sh_normal_mean_sif

# %%
lst_sh_normal_std_sif = lst_sh_annual_detrend_sif.where(spei_sh_normal_use_sif > - 5).std(dim='year')
lst_sh_normal_std_sif

# %%
lst_sh_change_sif = lst_sh_normal_mean_sif / (lst_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - lst_sh_normal_mean_sif)
lst_sh_change_sif = lst_sh_change_sif.transpose("year", "lat", "lon").drop('quantile')
lst_sh_change_sif

# %%
lst_sh_change_sif = lst_sh_change_sif.where(sif_sh_resistance > 0)
lst_sh_change_sif = lst_sh_change_sif.where(np.isfinite(lst_sh_change_sif)==1)
lst_sh_change_sif

# %%
lst_sh_change_sif.plot.hist()

# %%
lst_nt_change_sif.quantile(0.95)

# %%
lst_sh_change_sif.quantile(0.95)

# %%
lst_nt_change_sif[:8].plot(col = 'year', col_wrap = 4, vmax = 1000)

# %%
lst_sh_change_sif[:8].plot(col = 'year', col_wrap = 4, vmax = 1000)

# %%
lst_sh_zs_sif =  (lst_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - lst_sh_normal_mean_sif) / lst_sh_normal_std_sif
lst_sh_zs_sif

# %%
lst_sh_zs_sif[0].plot(vmin = -1, vmax = 1)

# %%
lst_sh_zs_sif.quantile([0.1,0.5,0.9])

# %%
lst_sh_zs_sif.plot.hist(bins = np.arange(-1,3, 0.2))

# %%
lst_sh_zs_sif = lst_sh_zs_sif.where(sif_sh_resistance > 0)
lst_sh_zs_sif.plot.hist()

# %%
lst_nt_change_sif.name = 'lst_change'
lst_nt_change_sif.to_netcdf(r'E:/python_output/fsc_drought/lst_nt_change_sif.nc')

lst_sh_change_sif.name = 'lst_change'
lst_sh_change_sif.to_netcdf(r'E:/python_output/fsc_drought/lst_sh_change_sif.nc')

lst_nt_zs_sif.name = 'lst_zs'
lst_nt_zs_sif.to_netcdf(r'E:/python_output/fsc_drought/lst_nt_zs_sif.nc')

lst_sh_zs_sif.name = 'lst_zs'
lst_sh_zs_sif.to_netcdf(r'E:/python_output/fsc_drought/lst_sh_zs_sif.nc')

# %% [markdown]
# ## change log
# 1. 2025.11.11 计算了干旱时 era5-land  LST的变化  分为两种  一种是和之前一样算的变化率   另一种是算成 类似 zscore
# 2. 2025.12.11 计算了2000年以后 对应kndvi的结果

# %%



