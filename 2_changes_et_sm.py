# %% [markdown]
# # 再计算 与 干旱抵抗力对应的 蒸散发和 土壤水分的变化
# 
# 和之前的区别在于  只看对应的干旱抵抗力的 格点的 蒸散发和土壤水分的变化  土壤水分可以考虑加上gleam的数据
# 
# 在数据的原始分辨率上就要先排除发生了 landcover变化 的点

# %%
import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import glob
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1 土壤水分数据

# %% [markdown]
# ### 1.0 处理一下 landcover的变化
# 
# 先用众数把landcover重采样到土壤水分数据的原始分辨率，然后再在这个分辨率上看landcover的有没有发生变化

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
## eas 是 0.25度分辨率    gleam 是0.1度分辨率
modis_ld_025 = landCover.coarsen(lat=5, lon=5).reduce(mode_func)
modis_ld_025

# %%
modis_ld_01 = landCover.coarsen(lat=2, lon=2).reduce(mode_func)
modis_ld_01

# %%
ld_change_025 =  xr.apply_ufunc(
    ld_unchanged,
    modis_ld_025,
    input_core_dims=[['year']],
    output_core_dims=[[]],
    vectorize= True
)

# %%
ld_change_01 =  xr.apply_ufunc(
    ld_unchanged,
    modis_ld_01,
    input_core_dims=[['year']],
    output_core_dims=[[]],
    vectorize= True
)

# %%
modis_ld_025_use = modis_ld_025[-1].where(ld_change_025 == 0)
modis_ld_025_use = modis_ld_025_use.where(modis_ld_025_use >0)
modis_ld_025_use = modis_ld_025_use.where(modis_ld_025_use <11)
modis_ld_025_use = modis_ld_025_use.drop('year')
modis_ld_025_use = modis_ld_025_use.sortby('lat', ascending=False)
modis_ld_025_use

# %%
modis_ld_025_use.plot()

# %%
modis_ld_025_use.name = 'landcover'

# %%
modis_ld_025_use.to_netcdf(r'E:/python_output/fsc_drought/landcover_025_mask.nc')

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
modis_ld_01_use.name = 'landcover'

# %%
modis_ld_01_use.to_netcdf(r'E:/python_output/fsc_drought/landcover_01_mask.nc')

# %% [markdown]
# ### 1.1 esa 土壤水分

# %%
with xr.open_dataset(r'D:/data/esa_sm_091/v09_1/1979/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-19790101000000-fv09.1.nc')['sm'] as data:
    sm_esa_exam = data
sm_esa_exam
sm_esa_exam.plot()

# %%
modis_ld_025_use = modis_ld_025_use.interp_like(sm_esa_exam, method= 'nearest')

# %%
sm_esa_exam.where(modis_ld_025_use > 0)

# %%
for year_n in range(1979,2023):
    print(year_n)
    sm_year_path = glob.glob(r'D:/data/esa_sm_091/v09_1/{}/*nc'.format(year_n))
    sm_year_path = sorted(sm_year_path)
    print(len(sm_year_path))

    with xr.open_mfdataset(sm_year_path)['sm'] as data:
        sm_year = data
    
    sm_year_month = sm_year.resample(time='M').median()
    sm_year_month = sm_year_month.where(modis_ld_025_use > 0)
    sm_year_month_co = sm_year_month.coarsen(lat=2, lon =2).mean()

    sm_year_month_co.to_netcdf(r'D:/data/esa_sm_091/esa_monthly_ld/ESACCI_SOILMOISTURE_L3S_SSMV_COMBINED_{}.nc'.format(year_n))


# %%
sm_use_path = glob.glob(r'D:/data/esa_sm_091/esa_monthly_ld/*nc')
sm_use_path

# %%
sm_month_all = []
for sm_path_n in sm_use_path[3:]:
    print(sm_path_n)
    with xr.open_dataset(sm_path_n)['sm'] as data:
        sm_month_all.append(data)

sm_month_all = xr.concat(sm_month_all, dim='time')
sm_month_all

# %%
sm_month_all[:12].plot(col = 'time', col_wrap=3)

# %% [markdown]
# #### 1.1.2 分南北半球

# %%
sm_month_all

# %%
sm_sh = sm_month_all.sel(lat = slice(-23.5,-90))
sm_nh = sm_month_all.sel(lat = slice(90, 23.5))
sm_tr = sm_month_all.sel(lat = slice(23.5,-23.5))

# %%
sm_nh

# %%
sm_tr

# %%
sm_sh

# %%
import pandas as pd
sm_sh = xr.DataArray(sm_sh.values, coords=[pd.Series(sm_sh.time.values) - pd.DateOffset(months=6), sm_sh.lat, sm_sh.lon], dims=['time','lat','lon'])
sm_sh

# %%
sm_sh[0:12].plot(x = 'lon',y='lat',col = 'time', col_wrap = 3)

# %%
pd.Series(sm_sh.time.values).dt.year.value_counts()

# %%
sm_sh = sm_sh.sel(time = slice('19820101','20211231'))
sm_nh = sm_nh.sel(time = slice('19820101','20221231'))
sm_tr = sm_tr.sel(time = slice('19820101','20221231'))

# %%
sm_sh

# %%
sm_sh = sm_sh.sel(time = sm_sh.time.dt.month.isin([5,6,7,8,9]))
sm_nh = sm_nh.sel(time = sm_nh.time.dt.month.isin([5,6,7,8,9]))

# %%
sm_sh

# %%
### 统计一下缺失值数量
def na_count(data):
    if np.isnan(data).all():
        return np.nan
    else:
        return np.isnan(data).sum()

# %%
sm_na_sh = xr.apply_ufunc(
    na_count,
    sm_sh,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
sm_na_sh.plot()

# %%
sm_na_nh = xr.apply_ufunc(
    na_count,
    sm_nh,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
sm_na_nh.plot()

# %%
sm_na_tr = xr.apply_ufunc(
    na_count,
    sm_tr,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
sm_na_tr.plot()

# %% [markdown]
# #### 1.1.3 简单填补缺失值

# %%
print(sm_nh.shape, sm_sh.shape, sm_tr.shape)

# %%
def na_replace(data, value, isna_data):
    if  isna_data < 67:    ## 数字根据序列长度修改
        data = np.nan_to_num(data, nan = value)
    
    return data
def na_replace_tr(data, value, isna_data):
    if  isna_data < 164:    ## 数字根据序列长度修改
        data = np.nan_to_num(data, nan = value)
    
    return data
def month_na_replace_run(sm_data_month, isna_data, is_tr):

    sm_month_per50 = sm_data_month.quantile(0.5, dim = 'time', method = 'lower')
    
    if is_tr:
        sm_month_na_replaced = xr.apply_ufunc(
            na_replace_tr,
            sm_data_month,
            sm_month_per50,
            isna_data,
            input_core_dims=[['time'],[],[]],
            output_core_dims=[['time']],
            dask = 'parallelized',
            vectorize= True,
            output_dtypes=[float]
        )
    else:
        sm_month_na_replaced = xr.apply_ufunc(
            na_replace,
            sm_data_month,
            sm_month_per50,
            isna_data,
            input_core_dims=[['time'],[],[]],
            output_core_dims=[['time']],
            dask = 'parallelized',
            vectorize= True,
            output_dtypes=[float]
        )
    
    
    return sm_month_na_replaced

# %%
import dask
from dask.diagnostics import ProgressBar
months = [5,6,7,8,9]
month_na_replaced_all = []
for k in range(len(months)):
    print(months[k])
    sm_data_month = sm_sh.groupby('time.month')[months[k]]
        
    month_na_replaced = month_na_replace_run(sm_data_month, sm_na_sh, is_tr=False)
    month_na_replaced_all.append(month_na_replaced)
        
month_na_replaced_final = xr.concat(month_na_replaced_all, dim = 'time')
month_na_replaced_final.name = 'sm'
    
with ProgressBar():
    sm_sh_nareplaced = month_na_replaced_final.compute()
sm_sh_nareplaced = sm_sh_nareplaced.drop('quantile')
sm_sh_nareplaced

# %%
sm_na_sh_after = xr.apply_ufunc(
    na_count,
    sm_sh_nareplaced,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
sm_na_sh_after.where(sm_na_sh_after > 0).plot()

# %%
months = [5,6,7,8,9]
month_na_replaced_all = []
for k in range(len(months)):
    print(months[k])
    sm_data_month = sm_nh.groupby('time.month')[months[k]]
        
    month_na_replaced = month_na_replace_run(sm_data_month, sm_na_nh, is_tr=False)
    month_na_replaced_all.append(month_na_replaced)
        
month_na_replaced_final = xr.concat(month_na_replaced_all, dim = 'time')
month_na_replaced_final.name = 'sm'
    
with ProgressBar():
    sm_nh_nareplaced = month_na_replaced_final.compute()
sm_nh_nareplaced = sm_nh_nareplaced.drop('quantile')
sm_nh_nareplaced

# %%
sm_na_nh_after = xr.apply_ufunc(
    na_count,
    sm_nh_nareplaced,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
sm_na_nh_after.where(sm_na_nh_after>0).plot()

# %%
months = [1,2,3,4,5,6,7,8,9,10,11,12]
month_na_replaced_all = []
for k in range(len(months)):
    print(months[k])
    sm_data_month = sm_tr.groupby('time.month')[months[k]]
        
    month_na_replaced = month_na_replace_run(sm_data_month, sm_na_tr, is_tr=True)
    month_na_replaced_all.append(month_na_replaced)
        
month_na_replaced_final = xr.concat(month_na_replaced_all, dim = 'time')
month_na_replaced_final.name = 'sm'
    
with ProgressBar():
    sm_tr_nareplaced = month_na_replaced_final.compute()
sm_tr_nareplaced = sm_tr_nareplaced.drop('quantile')
sm_tr_nareplaced

# %%
sm_na_tr_after = xr.apply_ufunc(
    na_count,
    sm_tr_nareplaced,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
sm_na_tr_after.where(sm_na_tr_after > 0).plot()

# %%
sm_sh_use = sm_sh_nareplaced.where(sm_na_sh_after ==0)
sm_nh_use = sm_nh_nareplaced.where(sm_na_nh_after ==0)
sm_tr_use = sm_tr_nareplaced.where(sm_na_tr_after ==0)

# %%
sm_sh_annual = sm_sh_use.groupby('time.year').mean()
sm_nh_annual = sm_nh_use.groupby('time.year').mean()
sm_tr_annual = sm_tr_use.groupby('time.year').mean()

# %%
sm_sh_annual = sm_sh_annual.transpose('year','lat','lon')
sm_nh_annual = sm_nh_annual.transpose('year','lat','lon')
sm_tr_annual = sm_tr_annual.transpose('year','lat','lon')

# %%
sm_sh_annual

# %%
sm_nh_annual

# %%
sm_tr_annual

# %%
sm_sh_annual.to_netcdf('E:/python_output/fsc_drought/sm_sh_annual.nc')
sm_nh_annual.to_netcdf('E:/python_output/fsc_drought/sm_nh_annual.nc')
sm_tr_annual.to_netcdf('E:/python_output/fsc_drought/sm_tr_annual.nc')

# %% [markdown]
# #### 1.1.4 去趋势

# %%
print(sm_sh_annual.shape, sm_nh_annual.shape, sm_tr_annual.shape)

# %%
## 去趋势
sm_sh_annual_trend = sm_sh_annual.polyfit(dim='year', deg=1)
(sm_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.001, vmin= -0.001, cmap = 'PiYG')

# %%
sm_sh_annual_detrend = sm_sh_annual.copy()
for i in range(1, 40):
    sm_sh_annual_detrend[i] = sm_sh_annual_detrend[i] - sm_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sm_sh_annual_detrend

# %%
sm_sh_annual_detrend.plot.hist(bins = [-0.5,0,0.5,1])

# %%
sm_sh_annual_detrend = sm_sh_annual_detrend.where(sm_sh_annual_detrend > 0)
sm_sh_annual_detrend = sm_sh_annual_detrend.where(sm_sh_annual_detrend < 1)
sm_sh_annual_detrend[0:4].plot(col = 'year')

# %%
## 去趋势
sm_nh_annual_trend = sm_nh_annual.polyfit(dim='year', deg=1)
(sm_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.001, vmin= -0.001, cmap = 'PiYG')

# %%
sm_nh_annual_detrend = sm_nh_annual.copy()
for i in range(1, 41):
    sm_nh_annual_detrend[i] = sm_nh_annual_detrend[i] - sm_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sm_nh_annual_detrend

# %%
sm_nh_annual_detrend = sm_nh_annual_detrend.where(sm_nh_annual_detrend > 0)
sm_nh_annual_detrend = sm_nh_annual_detrend.where(sm_nh_annual_detrend < 1)
sm_nh_annual_detrend[0:4].plot(col = 'year')

# %%
## 去趋势
sm_tr_annual_trend = sm_tr_annual.polyfit(dim='year', deg=1)
(sm_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.001, vmin= -0.001, cmap = 'PiYG')

# %%
sm_tr_annual_detrend = sm_tr_annual.copy()
for i in range(1, 41):
    sm_tr_annual_detrend[i] = sm_tr_annual_detrend[i] - sm_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
sm_tr_annual_detrend

# %%
sm_tr_annual_detrend = sm_tr_annual_detrend.where(sm_tr_annual_detrend > 0)
sm_tr_annual_detrend = sm_tr_annual_detrend.where(sm_tr_annual_detrend < 1)
sm_tr_annual_detrend[0:4].plot(col = 'year')

# %%
sm_nt_annual_detrend = xr.concat([sm_nh_annual_detrend, sm_tr_annual_detrend], dim='lat').sortby('lat')
sm_nt_annual_detrend

# %%
sm_nt_annual_detrend[0:4].plot(col = 'year')

# %% [markdown]
# ### 1.2 gleam

# %%
ea_paths = glob.glob(r'D:/data/gleam/monthly/Actual_evaporation/*.nc')
et_paths = glob.glob(r'D:/data/gleam/monthly/Transpiration/*.nc')
smrz_paths = glob.glob(r'D:/data/gleam/monthly/SMrz/*.nc')

# %%
ea_paths

# %%
et_paths

# %%
smrz_paths

# %% [markdown]
# #### 1.2.1 土壤水分

# %%
with xr.open_dataset(smrz_paths[0]) as data:
    print(data)

# %%
with xr.open_dataset(smrz_paths[0]) as data:
    smrz_exam = data['SMrz']

# %%
modis_ld_01_use = modis_ld_01_use.interp_like(smrz_exam, method='nearest')
modis_ld_01_use

# %%
## 原始数据是0.1度  转到0.5度
smrz_month_all = []
for smrz_n in smrz_paths:
    print(smrz_n)
    with xr.open_dataset(smrz_n)['SMrz'] as data:
        data = data.where(modis_ld_01_use > 0)
        smrz_month_all.append(data.coarsen(lat = 5, lon=5).mean())

smrz_month_all = xr.concat(smrz_month_all, dim='time')

# %%
smrz_month_all

# %%
smrz_month_all[0].plot()

# %%
smrz_sh = smrz_month_all.sel(lat = slice(-23.5,-90))
smrz_nh = smrz_month_all.sel(lat = slice(90, 23.5))
smrz_tr = smrz_month_all.sel(lat = slice(23.5,-23.5))

# %%
smrz_nh

# %%
smrz_tr

# %%
smrz_sh


# %%
smrz_sh = xr.DataArray(smrz_sh.values, coords=[pd.Series(smrz_sh.time.values) - pd.DateOffset(months=6), smrz_sh.lat, smrz_sh.lon], dims=['time','lat','lon'])
smrz_sh

# %%
smrz_sh[0:12].plot(x = 'lon',y='lat',col = 'time', col_wrap = 3)

# %%
pd.Series(smrz_sh.time.values).dt.year.value_counts()

# %%
smrz_sh = smrz_sh.sel(time = slice('19820101','20211231'))
smrz_nh = smrz_nh.sel(time = slice('19820101','20221231'))
smrz_tr = smrz_tr.sel(time = slice('19820101','20221231'))

# %%
smrz_sh

# %%
smrz_sh = smrz_sh.sel(time = smrz_sh.time.dt.month.isin([5,6,7,8,9]))
smrz_nh = smrz_nh.sel(time = smrz_nh.time.dt.month.isin([5,6,7,8,9]))

# %%
smrz_na_sh = xr.apply_ufunc(
    na_count,
    smrz_sh,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
smrz_na_sh.plot()

# %%
smrz_na_nh = xr.apply_ufunc(
    na_count,
    smrz_nh,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
smrz_na_nh.plot()

# %%
smrz_na_tr = xr.apply_ufunc(
    na_count,
    smrz_tr,
    input_core_dims=[['time']],
    output_core_dims=[[]],
    vectorize= True
)
smrz_na_tr.plot()

# %%
## 缺失值基本没有 
smrz_sh_use = smrz_sh.where(smrz_na_sh ==0)
smrz_nh_use = smrz_nh.where(smrz_na_nh ==0)
smrz_tr_use = smrz_tr.where(smrz_na_tr ==0)

# %%
smrz_sh_annual = smrz_sh_use.groupby('time.year').mean()
smrz_nh_annual = smrz_nh_use.groupby('time.year').mean()
smrz_tr_annual = smrz_tr_use.groupby('time.year').mean()

# %%
smrz_sh_annual = smrz_sh_annual.transpose('year','lat','lon')
smrz_nh_annual = smrz_nh_annual.transpose('year','lat','lon')
smrz_tr_annual = smrz_tr_annual.transpose('year','lat','lon')

# %%
smrz_sh_annual.name = 'SMrz'
smrz_nh_annual.name = 'SMrz'
smrz_tr_annual.name = 'SMrz'

# %%
smrz_sh_annual.to_netcdf('E:/python_output/fsc_drought/smrz_sh_annual.nc')
smrz_nh_annual.to_netcdf('E:/python_output/fsc_drought/smrz_nh_annual.nc')
smrz_tr_annual.to_netcdf('E:/python_output/fsc_drought/smrz_tr_annual.nc')

# %%
with xr.open_dataset('E:/python_output/fsc_drought/smrz_sh_annual.nc') as data:
    smrz_sh_annual = data['SMrz']
smrz_sh_annual
with xr.open_dataset('E:/python_output/fsc_drought/smrz_nh_annual.nc') as data:
    smrz_nh_annual = data['SMrz']
smrz_nh_annual
with xr.open_dataset('E:/python_output/fsc_drought/smrz_tr_annual.nc') as data:
    smrz_tr_annual = data['SMrz']
smrz_tr_annual

# %% [markdown]
# #### 1.2.2 去趋势

# %%
## 去趋势
smrz_sh_annual_trend = smrz_sh_annual.polyfit(dim='year', deg=1)
(smrz_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.001, vmin= -0.001, cmap = 'PiYG')

# %%
smrz_sh_annual_detrend = smrz_sh_annual.copy()
for i in range(1, 40):
    smrz_sh_annual_detrend[i] = smrz_sh_annual_detrend[i] - smrz_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
smrz_sh_annual_detrend

# %%
smrz_sh_annual_detrend = smrz_sh_annual_detrend.where(smrz_sh_annual_detrend > 0)
smrz_sh_annual_detrend = smrz_sh_annual_detrend.where(smrz_sh_annual_detrend < 1)
smrz_sh_annual_detrend[0:4].plot(col = 'year')

# %%
## 去趋势
smrz_nh_annual_trend = smrz_nh_annual.polyfit(dim='year', deg=1)
(smrz_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.001, vmin= -0.001, cmap = 'PiYG')

# %%
smrz_nh_annual_detrend = smrz_nh_annual.copy()
for i in range(1, 41):
    smrz_nh_annual_detrend[i] = smrz_nh_annual_detrend[i] - smrz_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
smrz_nh_annual_detrend

# %%
smrz_nh_annual_detrend = smrz_nh_annual_detrend.where(smrz_nh_annual_detrend > 0)
smrz_nh_annual_detrend = smrz_nh_annual_detrend.where(smrz_nh_annual_detrend < 1)
smrz_nh_annual_detrend[0:4].plot(col = 'year')

# %%
## 去趋势
smrz_tr_annual_trend = smrz_tr_annual.polyfit(dim='year', deg=1)
(smrz_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.001, vmin= -0.001, cmap = 'PiYG')

# %%
smrz_tr_annual_detrend = smrz_tr_annual.copy()
for i in range(1, 41):
    smrz_tr_annual_detrend[i] = smrz_tr_annual_detrend[i] - smrz_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
smrz_tr_annual_detrend

# %%
smrz_tr_annual_detrend = smrz_tr_annual_detrend.where(smrz_tr_annual_detrend > 0)
smrz_tr_annual_detrend = smrz_tr_annual_detrend.where(smrz_tr_annual_detrend < 1)
smrz_tr_annual_detrend[0:4].plot(col = 'year')

# %%
smrz_nt_annual_detrend = xr.concat([smrz_nh_annual_detrend, smrz_tr_annual_detrend], dim='lat').sortby('lat')
smrz_nt_annual_detrend
smrz_nt_annual_detrend[0:4].plot(col = 'year')

# %% [markdown]
# ## 2 gleam的 蒸散发

# %% [markdown]
# ### 2.1 读取数据

# %%
ea_month_all = []
for ea_n in ea_paths:
    print(ea_n)
    with xr.open_dataset(ea_n)['E'] as data:
        data = data.where(modis_ld_01_use > 0)
        ea_month_all.append(data.coarsen(lat = 5, lon=5).mean())

ea_month_all = xr.concat(ea_month_all, dim='time')
ea_month_all

# %%
ea_month_all[0].plot()

# %%
et_month_all = []
for et_n in et_paths:
    print(et_n)
    with xr.open_dataset(et_n)['Et'] as data:
        data = data.where(modis_ld_01_use > 0)
        et_month_all.append(data.coarsen(lat = 5, lon=5).mean())

et_month_all = xr.concat(et_month_all, dim='time')
et_month_all

# %%
et_month_all[0].plot()

# %%
### 1.2 分南北半球
et_sh = et_month_all.sel(lat = slice(-23.5,-90))
et_nh = et_month_all.sel(lat = slice(90, 23.5))
et_tr = et_month_all.sel(lat = slice(23.5,-23.5))

# %%
et_sh

# %%
et_nh

# %%
et_tr

# %%
ea_sh = ea_month_all.sel(lat = slice(-23.5,-90))
ea_nh = ea_month_all.sel(lat = slice(90, 23.5))
ea_tr = ea_month_all.sel(lat = slice(23.5,-23.5))

# %%
ea_sh

# %%
ea_nh

# %%
ea_tr

# %%
ea_sh = xr.DataArray(ea_sh.values, coords=[pd.Series(ea_sh.time.values) - pd.DateOffset(months=6), ea_sh.lat, ea_sh.lon], dims=['time','lat','lon'])
ea_sh

# %%
et_sh = xr.DataArray(et_sh.values, coords=[pd.Series(et_sh.time.values) - pd.DateOffset(months=6), et_sh.lat, et_sh.lon], dims=['time','lat','lon'])
et_sh

# %%
ea_sh = ea_sh.sel(time = slice('19820101','20221231'))
ea_nh = ea_nh.sel(time = slice('19820101','20221231'))
ea_tr = ea_tr.sel(time = slice('19820101','20221231'))

# %%
et_sh = et_sh.sel(time = slice('19820101','20221231'))
et_nh = et_nh.sel(time = slice('19820101','20221231'))
et_tr = et_tr.sel(time = slice('19820101','20221231'))

# %%
ea_sh = ea_sh.sel(time = ea_sh.time.dt.month.isin([5,6,7,8,9]))
ea_nh = ea_nh.sel(time = ea_nh.time.dt.month.isin([5,6,7,8,9]))
et_sh = et_sh.sel(time = et_sh.time.dt.month.isin([5,6,7,8,9]))
et_nh = et_nh.sel(time = et_nh.time.dt.month.isin([5,6,7,8,9]))

# %%
## 基本没缺失值
ea_sh_annual = ea_sh.groupby('time.year').mean()
ea_nh_annual = ea_nh.groupby('time.year').mean()
ea_tr_annual = ea_tr.groupby('time.year').mean()
et_sh_annual = et_sh.groupby('time.year').mean()
et_nh_annual = et_nh.groupby('time.year').mean()
et_tr_annual = et_tr.groupby('time.year').mean()

# %%
ea_sh_annual.to_netcdf('E:/python_output/fsc_drought/ea_sh_annual.nc')
ea_nh_annual.to_netcdf('E:/python_output/fsc_drought/ea_nh_annual.nc')
ea_tr_annual.to_netcdf('E:/python_output/fsc_drought/ea_tr_annual.nc')
et_sh_annual.to_netcdf('E:/python_output/fsc_drought/et_sh_annual.nc')
et_nh_annual.to_netcdf('E:/python_output/fsc_drought/et_nh_annual.nc')
et_tr_annual.to_netcdf('E:/python_output/fsc_drought/et_tr_annual.nc')

# %% [markdown]
# ### 2.2 去趋势

# %%
## 去趋势
ea_sh_annual_trend = ea_sh_annual.polyfit(dim='year', deg=1)
(ea_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
ea_sh_annual_detrend = ea_sh_annual.copy()
for i in range(1, 41):
    ea_sh_annual_detrend[i] = ea_sh_annual_detrend[i] - ea_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
ea_sh_annual_detrend

# %%
ea_sh_annual_detrend = ea_sh_annual_detrend.where(ea_sh_annual_detrend>0)

# %%

ea_sh_annual_detrend[0:4].plot(col = 'year')

# %%
ea_nh_annual_trend = ea_nh_annual.polyfit(dim='year', deg=1)
(ea_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
ea_nh_annual_detrend = ea_nh_annual.copy()
for i in range(1, 41):
    ea_nh_annual_detrend[i] = ea_nh_annual_detrend[i] - ea_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
ea_nh_annual_detrend

# %%
ea_nh_annual_detrend = ea_nh_annual_detrend.where(ea_nh_annual_detrend > 0)
ea_nh_annual_detrend[0:4].plot(col = 'year')

# %%
ea_tr_annual_trend = ea_tr_annual.polyfit(dim='year', deg=1)
(ea_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
ea_tr_annual_detrend = ea_tr_annual.copy()
for i in range(1, 41):
    ea_tr_annual_detrend[i] = ea_tr_annual_detrend[i] - ea_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
ea_tr_annual_detrend

# %%
ea_tr_annual_detrend = ea_tr_annual_detrend.where(ea_tr_annual_detrend > 0)
ea_tr_annual_detrend[0:4].plot(col = 'year')

# %%
et_sh_annual_trend = et_sh_annual.polyfit(dim='year', deg=1)
(et_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
et_sh_annual_detrend = et_sh_annual.copy()
for i in range(1, 41):
    et_sh_annual_detrend[i] = et_sh_annual_detrend[i] - et_sh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
et_sh_annual_detrend

# %%
et_sh_annual_detrend = et_sh_annual_detrend.where(et_sh_annual_detrend>0)
et_sh_annual_detrend[0:4].plot(col = 'year')

# %%
et_nh_annual_trend = et_nh_annual.polyfit(dim='year', deg=1)
(et_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
et_nh_annual_detrend = et_nh_annual.copy()
for i in range(1, 41):
    et_nh_annual_detrend[i] = et_nh_annual_detrend[i] - et_nh_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
et_nh_annual_detrend

# %%
et_nh_annual_detrend = et_nh_annual_detrend.where(et_nh_annual_detrend > 0)
et_nh_annual_detrend[0:4].plot(col = 'year')

# %%
et_tr_annual_trend = et_tr_annual.polyfit(dim='year', deg=1)
(et_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) ).plot(center = False, vmax= 0.1, vmin= -0.1, cmap = 'PiYG')

# %%
et_tr_annual_detrend = et_tr_annual.copy()
for i in range(1, 41):
    et_tr_annual_detrend[i] = et_tr_annual_detrend[i] - et_tr_annual_trend['polyfit_coefficients'].sel(degree = 1) * i
et_tr_annual_detrend

# %%
et_tr_annual_detrend = et_tr_annual_detrend.where(et_tr_annual_detrend > 0)
et_tr_annual_detrend[0:4].plot(col ='year')

# %%
ea_nt_annual_detrend = xr.concat([ea_nh_annual_detrend, ea_tr_annual_detrend], dim='lat').sortby('lat')
ea_nt_annual_detrend

# %%
ea_nt_annual_detrend[0:4].plot(col='year')

# %%
et_nt_annual_detrend = xr.concat([et_nh_annual_detrend, et_tr_annual_detrend], dim='lat').sortby('lat')
et_nt_annual_detrend

# %%
et_nt_annual_detrend[0:4].plot(col='year')

# %% [markdown]
# ## 3 之前计算的干旱

# %%
with xr.open_dataset(r'../result_data/spei_nt_annual_drought.nc') as data:
    spei_nt_annual_drought = data['spei']
with xr.open_dataset(r'../result_data/spei_nt_annual_wet.nc') as data:
    spei_nt_annual_wet = data['spei']
with xr.open_dataset(r'../result_data/spei_nt_annual_normal.nc') as data:
    spei_nt_annual_normal = data['spei']

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

# %% [markdown]
# ## 3 kndvi 对应的

# %% [markdown]
# ### 3.1 读取 干旱抵抗力的数据

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
kndvi_nt_resistance

# %%
kndvi_sh_resistance

# %% [markdown]
# ### 3.2 eas 土壤水分

# %%
spei_nt_normal_use_kndvi

# %%
sm_nt_annual_detrend_kndvi = sm_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi, method='nearest')
sm_sh_annual_detrend_kndvi = sm_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi, method='nearest')

# %%
sm_nt_annual_detrend_kndvi

# %%
sm_sh_annual_detrend_kndvi

# %%
sm_nt_normal_mean_kndvi = sm_nt_annual_detrend_kndvi.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
sm_nt_normal_mean_kndvi

# %%
sm_nt_change_kndvi = sm_nt_normal_mean_kndvi / (sm_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - sm_nt_normal_mean_kndvi)
sm_nt_change_kndvi

# %%
sm_nt_change_kndvi = sm_nt_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
sm_nt_change_kndvi

# %%
sm_nt_change_kndvi.where(sm_nt_change_kndvi >0).plot.hist()

# %%
sm_nt_change_kndvi.where(kndvi_nt_resistance > 0).plot.hist()

# %%
sm_nt_change_kndvi = sm_nt_change_kndvi.where(kndvi_nt_resistance > 0)
sm_nt_change_kndvi = sm_nt_change_kndvi.where(sm_nt_change_kndvi < 0)
sm_nt_change_kndvi.values = np.abs(sm_nt_change_kndvi.values)

# %%
sm_nt_change_kndvi.where(sm_nt_change_kndvi < 100).plot.hist()

# %%
sm_sh_normal_mean_kndvi = sm_sh_annual_detrend_kndvi.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
sm_sh_normal_mean_kndvi

# %%
sm_sh_change_kndvi = sm_sh_normal_mean_kndvi / (sm_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - sm_sh_normal_mean_kndvi)
sm_sh_change_kndvi = sm_sh_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
sm_sh_change_kndvi

# %%
sm_sh_change_kndvi = sm_sh_change_kndvi.where(kndvi_sh_resistance > 0)
sm_sh_change_kndvi = sm_sh_change_kndvi.where(sm_sh_change_kndvi < 0)
sm_sh_change_kndvi.values = np.abs(sm_sh_change_kndvi.values)
sm_sh_change_kndvi

# %%
sm_nt_change_kndvi.quantile(0.95)

# %%
sm_sh_change_kndvi.quantile(0.95)

# %%
sm_nt_change_kndvi[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
sm_sh_change_kndvi[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
sm_nt_change_kndvi.name = 'sm_change'
sm_nt_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/sm_nt_change_kndvi.nc')

sm_sh_change_kndvi.name = 'sm_change'
sm_sh_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/sm_sh_change_kndvi.nc')

# %% [markdown]
# ### 3.3 gleam  土壤水分

# %%
smrz_nt_annual_detrend_kndvi = smrz_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi, method='nearest')
smrz_sh_annual_detrend_kndvi = smrz_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi, method='nearest')

# %%
smrz_nt_annual_detrend_kndvi

# %%
smrz_sh_annual_detrend_kndvi

# %%
smrz_nt_normal_mean_kndvi = smrz_nt_annual_detrend_kndvi.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
smrz_nt_normal_mean_kndvi

# %%
smrz_nt_change_kndvi = smrz_nt_normal_mean_kndvi / (smrz_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - smrz_nt_normal_mean_kndvi)
smrz_nt_change_kndvi

# %%
smrz_nt_change_kndvi = smrz_nt_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
smrz_nt_change_kndvi

# %%
smrz_nt_change_kndvi.where(kndvi_nt_resistance > 0).plot.hist()

# %%
smrz_nt_change_kndvi = smrz_nt_change_kndvi.where(kndvi_nt_resistance > 0)
smrz_nt_change_kndvi = smrz_nt_change_kndvi.where(smrz_nt_change_kndvi < 0)
smrz_nt_change_kndvi.values = np.abs(smrz_nt_change_kndvi.values)
smrz_nt_change_kndvi.where(smrz_nt_change_kndvi < 100).plot.hist()

# %%
smrz_sh_normal_mean_kndvi = smrz_sh_annual_detrend_kndvi.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
smrz_sh_normal_mean_kndvi

# %%
smrz_sh_change_kndvi = smrz_sh_normal_mean_kndvi / (smrz_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - smrz_sh_normal_mean_kndvi)
smrz_sh_change_kndvi = smrz_sh_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
smrz_sh_change_kndvi

# %%
smrz_sh_change_kndvi = smrz_sh_change_kndvi.where(kndvi_sh_resistance > 0)
smrz_sh_change_kndvi = smrz_sh_change_kndvi.where(smrz_sh_change_kndvi < 0)
smrz_sh_change_kndvi.values = np.abs(smrz_sh_change_kndvi.values)
smrz_sh_change_kndvi

# %%
smrz_nt_change_kndvi[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
smrz_sh_change_kndvi[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
smrz_nt_change_kndvi.name = 'sm_change'
smrz_nt_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi.nc')

smrz_sh_change_kndvi.name = 'sm_change'
smrz_sh_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi.nc')

# %% [markdown]
# ### 3.4  蒸腾  ET

# %%
et_nt_annual_detrend_kndvi = et_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi, method='nearest')
et_nt_annual_detrend_kndvi

# %%
et_nt_normal_mean_kndvi = et_nt_annual_detrend_kndvi.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
et_nt_normal_mean_kndvi

# %%
et_nt_normal_mean_kndvi.plot()

# %%
et_nt_change_kndvi = et_nt_normal_mean_kndvi / (et_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - et_nt_normal_mean_kndvi)
et_nt_change_kndvi = et_nt_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
et_nt_change_kndvi

# %%
et_nt_change_kndvi[:4].plot(col = 'year', vmax = 100)

# %%
et_nt_change_kndvi = et_nt_change_kndvi.where(kndvi_nt_resistance > 0)
et_nt_change_kndvi

# %%
et_nt_change_kndvi[:4].plot(col = 'year', vmax = 100)

# %%
et_sh_annual_detrend_kndvi = et_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi, method='nearest')
et_sh_annual_detrend_kndvi

# %%
et_sh_normal_mean_kndvi = et_sh_annual_detrend_kndvi.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
et_sh_normal_mean_kndvi

# %%
et_sh_change_kndvi = et_sh_normal_mean_kndvi / (et_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - et_sh_normal_mean_kndvi)
et_sh_change_kndvi = et_sh_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
et_sh_change_kndvi

# %%
et_sh_change_kndvi[:4].plot(col ='year', vmax = 100)

# %%
print(et_nt_change_kndvi.quantile(0.05).values,et_nt_change_kndvi.quantile(0.95).values)

# %%
et_nt_change_kndvi.plot.hist(bins = np.arange(-60,61,10))

# %%
print(et_sh_change_kndvi.quantile(0.05).values,et_sh_change_kndvi.quantile(0.95).values)

# %%
et_sh_change_kndvi.plot.hist(bins = np.arange(-30,31,5))

# %%
et_sh_change_kndvi = et_sh_change_kndvi.where(kndvi_sh_resistance > 0)
et_sh_change_kndvi

# %%
et_sh_change_kndvi[:4].plot(col ='year', vmax = 100)

# %%
et_sh_change_kndvi.plot.hist(bins = np.arange(-30,31,5))

# %%
et_nt_change_kndvi.name = 'et_change'
et_nt_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/et_nt_change_kndvi.nc')

et_sh_change_kndvi.name = 'et_change'
et_sh_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/et_sh_change_kndvi.nc')

# %% [markdown]
# #### 蒸腾 反过来计算变化

# %%
et_nt_change2_kndvi =  (et_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - et_nt_normal_mean_kndvi) / et_nt_normal_mean_kndvi 
et_nt_change2_kndvi = et_nt_change2_kndvi.transpose("year", "lat", "lon").drop('quantile')
et_nt_change2_kndvi

# %%
et_nt_change2_kndvi[:4].plot(col = 'year' )

# %%
et_nt_change2_kndvi = et_nt_change2_kndvi.where(kndvi_nt_resistance > 0)
et_nt_change2_kndvi

# %%
et_nt_change2_kndvi[:4].plot(col = 'year', vmax = 1)

# %%
et_sh_change2_kndvi =  (et_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - et_sh_normal_mean_kndvi) /et_sh_normal_mean_kndvi
et_sh_change2_kndvi = et_sh_change2_kndvi.transpose("year", "lat", "lon").drop('quantile')
et_sh_change2_kndvi

# %%
et_sh_change2_kndvi[:4].plot(col ='year', vmax = 0.5)

# %%
print(et_nt_change2_kndvi.quantile(0.05).values,et_nt_change2_kndvi.quantile(0.95).values)

# %%
et_nt_change2_kndvi.plot.hist(bins = np.arange(-60,61,10))

# %%
print(et_sh_change2_kndvi.quantile(0.05).values,et_sh_change2_kndvi.quantile(0.95).values)

# %%
et_sh_change2_kndvi.plot.hist(bins = np.arange(-30,31,5))

# %%
et_sh_change2_kndvi = et_sh_change2_kndvi.where(kndvi_sh_resistance > 0)
et_sh_change2_kndvi

# %%
et_sh_change2_kndvi[:4].plot(col ='year', vmax = 0.5)

# %%
et_sh_change2_kndvi.plot.hist(bins = np.arange(-30,31,5))

# %%
et_nt_change2_kndvi.name = 'et_change2'
et_nt_change2_kndvi.to_netcdf(r'E:/python_output/fsc_drought/et_nt_change2_kndvi.nc')

et_sh_change2_kndvi.name = 'et_change2'
et_sh_change2_kndvi.to_netcdf(r'E:/python_output/fsc_drought/et_sh_change2_kndvi.nc')

# %% [markdown]
# ### 3.5 蒸散发 EA

# %%
ea_nt_annual_detrend_kndvi = ea_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi, method='nearest')
ea_nt_annual_detrend_kndvi

# %%
ea_nt_normal_mean_kndvi = ea_nt_annual_detrend_kndvi.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
ea_nt_normal_mean_kndvi

# %%
ea_nt_normal_mean_kndvi.plot()

# %%
ea_nt_change_kndvi = ea_nt_normal_mean_kndvi / (ea_nt_annual_detrend_kndvi.where( spei_nt_drought_use_kndvi > -10) - ea_nt_normal_mean_kndvi)
ea_nt_change_kndvi = ea_nt_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
ea_nt_change_kndvi

# %%
ea_nt_change_kndvi[:4].plot(col = 'year', vmax = 100)

# %%
ea_nt_change_kndvi = ea_nt_change_kndvi.where(kndvi_nt_resistance > 0)
ea_nt_change_kndvi

# %%
ea_nt_change_kndvi[:4].plot(col = 'year', vmax = 100)

# %%
ea_sh_annual_detrend_kndvi = ea_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi, method='nearest')
ea_sh_annual_detrend_kndvi

# %%
ea_sh_normal_mean_kndvi = ea_sh_annual_detrend_kndvi.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
ea_sh_normal_mean_kndvi

# %%
ea_sh_change_kndvi = ea_sh_normal_mean_kndvi / (ea_sh_annual_detrend_kndvi.where( spei_sh_drought_use_kndvi > -10) - ea_sh_normal_mean_kndvi)
ea_sh_change_kndvi = ea_sh_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
ea_sh_change_kndvi

# %%
ea_sh_change_kndvi[:4].plot(col ='year', vmax = 100)

# %%
print(ea_nt_change_kndvi.quantile(0.05).values,ea_nt_change_kndvi.quantile(0.95).values)

# %%
ea_nt_change_kndvi.plot.hist(bins = np.arange(-60,61,10))

# %%
print(ea_sh_change_kndvi.quantile(0.05).values,ea_sh_change_kndvi.quantile(0.95).values)

# %%
ea_sh_change_kndvi.plot.hist(bins = np.arange(-30,31,5))

# %%
ea_sh_change_kndvi = ea_sh_change_kndvi.where(kndvi_sh_resistance > 0)
ea_sh_change_kndvi

# %%
ea_sh_change_kndvi[:4].plot(col ='year', vmax = 100)

# %%
ea_sh_change_kndvi.plot.hist(bins = np.arange(-30,31,5))

# %%
ea_nt_change_kndvi.name = 'ea_change'
ea_nt_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/ea_nt_change_kndvi.nc')

ea_sh_change_kndvi.name = 'ea_change'
ea_sh_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/ea_sh_change_kndvi.nc')

# %% [markdown]
# ### 3.6  蒸腾/蒸发  ET/EA

# %%
ret_ea_nt_kndvi = et_nt_annual_detrend_kndvi /  ea_nt_annual_detrend_kndvi
ret_ea_sh_kndvi = et_sh_annual_detrend_kndvi /  ea_sh_annual_detrend_kndvi

# %%
ret_ea_nt_kndvi = ret_ea_nt_kndvi.where(ret_ea_nt_kndvi<1)
ret_ea_sh_kndvi = ret_ea_sh_kndvi.where(ret_ea_sh_kndvi<1)

# %%
ret_ea_nt_normal_mean_kndvi = ret_ea_nt_kndvi.where(spei_nt_normal_use_kndvi > - 5).mean(dim='year')
ret_ea_nt_normal_mean_kndvi

# %%
ret_ea_nt_normal_mean_kndvi.plot()

# %%
ret_ea_nt_change_kndvi = ret_ea_nt_normal_mean_kndvi / (ret_ea_nt_kndvi.where( spei_nt_drought_use_kndvi > -10) - ret_ea_nt_normal_mean_kndvi)
ret_ea_nt_change_kndvi

# %%
ret_ea_nt_change_kndvi = ret_ea_nt_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
ret_ea_nt_change_kndvi

# %%
ret_ea_nt_change_kndvi[:4].plot(col = 'year', vmax = 100)

# %%
ret_ea_nt_change_kndvi = ret_ea_nt_change_kndvi.where(kndvi_nt_resistance > 0)

# %%
ret_ea_nt_change_kndvi[:4].plot(col = 'year', vmax = 100)

# %%
ret_ea_sh_normal_mean_kndvi = ret_ea_sh_kndvi.where(spei_sh_normal_use_kndvi > - 5).mean(dim='year')
ret_ea_sh_normal_mean_kndvi

# %%
ret_ea_sh_normal_mean_kndvi.plot()

# %%
ret_ea_sh_change_kndvi = ret_ea_sh_normal_mean_kndvi / (ret_ea_sh_kndvi.where( spei_sh_drought_use_kndvi > -10) - ret_ea_sh_normal_mean_kndvi)
ret_ea_sh_change_kndvi = ret_ea_sh_change_kndvi.transpose("year", "lat", "lon").drop('quantile')
ret_ea_sh_change_kndvi

# %%
ret_ea_sh_change_kndvi[:4].plot(col = 'year', vmax = 50)

# %%
ret_ea_sh_change_kndvi = ret_ea_sh_change_kndvi.where(kndvi_sh_resistance > 0)

# %%
ret_ea_sh_change_kndvi[:4].plot(col = 'year', vmax = 50)

# %%
print(ret_ea_nt_change_kndvi.quantile(0.05).values,ret_ea_nt_change_kndvi.quantile(0.95).values)

# %%
ret_ea_nt_change_kndvi.plot.hist(bins = np.arange(-100,110,10))

# %%
print(ret_ea_sh_change_kndvi.quantile(0.05).values,ret_ea_sh_change_kndvi.quantile(0.95).values)

# %%
ret_ea_sh_change_kndvi.plot.hist(bins = np.arange(-80,90,10))

# %%
ret_ea_nt_change_kndvi.name = 'ret_ea_change'
ret_ea_nt_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/ret_ea_nt_change_kndvi.nc')

ret_ea_sh_change_kndvi.name = 'ret_ea_change'
ret_ea_sh_change_kndvi.to_netcdf(r'E:/python_output/fsc_drought/ret_ea_sh_change_kndvi.nc')

# %% [markdown]
# ## 3 kndvi 对应的 2000年以后

# %% [markdown]
# ### 3.1 读取 干旱抵抗力的数据

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
kndvi_nt_resistance_after2000

# %%
kndvi_sh_resistance_after2000

# %% [markdown]
# ### 3.2 eas 土壤水分

# %%
spei_nt_normal_use_kndvi_after2000

# %%
sm_nt_annual_detrend_kndvi_after2000 = sm_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi_after2000, method='nearest')
sm_sh_annual_detrend_kndvi_after2000 = sm_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi_after2000, method='nearest')
sm_nt_annual_detrend_kndvi_after2000

# %%
sm_sh_annual_detrend_kndvi_after2000

# %%
sm_nt_normal_mean_kndvi_after2000 = sm_nt_annual_detrend_kndvi_after2000.where(spei_nt_normal_use_kndvi_after2000 > - 5).mean(dim='year')
sm_nt_normal_mean_kndvi_after2000

# %%
sm_nt_change_kndvi_after2000 = sm_nt_normal_mean_kndvi_after2000 / (sm_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - sm_nt_normal_mean_kndvi_after2000)
sm_nt_change_kndvi_after2000

# %%
sm_nt_change_kndvi_after2000 = sm_nt_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
sm_nt_change_kndvi_after2000

# %%
sm_nt_change_kndvi_after2000.where(sm_nt_change_kndvi_after2000 >0).plot.hist()

# %%
sm_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0).plot.hist()

# %%
sm_nt_change_kndvi_after2000 = sm_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)
sm_nt_change_kndvi_after2000 = sm_nt_change_kndvi_after2000.where(sm_nt_change_kndvi_after2000 < 0)
sm_nt_change_kndvi_after2000.values = np.abs(sm_nt_change_kndvi_after2000.values)
sm_nt_change_kndvi_after2000.where(sm_nt_change_kndvi_after2000 < 100).plot.hist()

# %%
sm_sh_normal_mean_kndvi_after2000 = sm_sh_annual_detrend_kndvi_after2000.where(spei_sh_normal_use_kndvi_after2000 > - 5).mean(dim='year')
sm_sh_normal_mean_kndvi_after2000

# %%
sm_sh_change_kndvi_after2000 = sm_sh_normal_mean_kndvi_after2000 / (sm_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - sm_sh_normal_mean_kndvi_after2000)
sm_sh_change_kndvi_after2000 = sm_sh_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
sm_sh_change_kndvi_after2000

# %%
sm_sh_change_kndvi_after2000 = sm_sh_change_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
sm_sh_change_kndvi_after2000 = sm_sh_change_kndvi_after2000.where(sm_sh_change_kndvi_after2000 < 0)
sm_sh_change_kndvi_after2000.values = np.abs(sm_sh_change_kndvi_after2000.values)
sm_sh_change_kndvi_after2000

# %%
sm_nt_change_kndvi_after2000.quantile(0.95)

# %%
sm_sh_change_kndvi_after2000.quantile(0.95)

# %%
sm_nt_change_kndvi_after2000[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
sm_sh_change_kndvi_after2000[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
sm_nt_change_kndvi_after2000.name = 'sm_change'
sm_nt_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/sm_nt_change_kndvi_after2000.nc')

sm_sh_change_kndvi_after2000.name = 'sm_change'
sm_sh_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/sm_sh_change_kndvi_after2000.nc')

# %% [markdown]
# ### 3.3 gleam  土壤水分

# %%
smrz_nt_annual_detrend_kndvi_after2000 = smrz_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi_after2000, method='nearest')
smrz_sh_annual_detrend_kndvi_after2000 = smrz_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi_after2000, method='nearest')
smrz_nt_annual_detrend_kndvi_after2000

# %%
smrz_sh_annual_detrend_kndvi_after2000

# %%
smrz_nt_normal_mean_kndvi_after2000 = smrz_nt_annual_detrend_kndvi_after2000.where(spei_nt_normal_use_kndvi_after2000 > - 5).mean(dim='year')
smrz_nt_normal_mean_kndvi_after2000

# %%
smrz_nt_change_kndvi_after2000 = smrz_nt_normal_mean_kndvi_after2000 / (smrz_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - smrz_nt_normal_mean_kndvi_after2000)
smrz_nt_change_kndvi_after2000

# %%
smrz_nt_change_kndvi_after2000 = smrz_nt_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
smrz_nt_change_kndvi_after2000

# %%
smrz_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0).plot.hist()

# %%
smrz_nt_change_kndvi_after2000 = smrz_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)
smrz_nt_change_kndvi_after2000 = smrz_nt_change_kndvi_after2000.where(smrz_nt_change_kndvi_after2000 < 0)
smrz_nt_change_kndvi_after2000.values = np.abs(smrz_nt_change_kndvi_after2000.values)
smrz_nt_change_kndvi_after2000.where(smrz_nt_change_kndvi_after2000 < 100).plot.hist()

# %%
smrz_sh_normal_mean_kndvi_after2000 = smrz_sh_annual_detrend_kndvi_after2000.where(spei_sh_normal_use_kndvi_after2000 > - 5).mean(dim='year')
smrz_sh_normal_mean_kndvi_after2000

# %%
smrz_sh_change_kndvi_after2000 = smrz_sh_normal_mean_kndvi_after2000 / (smrz_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - smrz_sh_normal_mean_kndvi_after2000)
smrz_sh_change_kndvi_after2000 = smrz_sh_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
smrz_sh_change_kndvi_after2000

# %%
smrz_sh_change_kndvi_after2000 = smrz_sh_change_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
smrz_sh_change_kndvi_after2000 = smrz_sh_change_kndvi_after2000.where(smrz_sh_change_kndvi_after2000 < 0)
smrz_sh_change_kndvi_after2000.values = np.abs(smrz_sh_change_kndvi_after2000.values)
smrz_sh_change_kndvi_after2000

# %%
smrz_nt_change_kndvi_after2000[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
smrz_sh_change_kndvi_after2000[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
smrz_nt_change_kndvi_after2000.name = 'sm_change'
smrz_nt_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi_after2000.nc')

smrz_sh_change_kndvi_after2000.name = 'sm_change'
smrz_sh_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi_after2000.nc')

# %% [markdown]
# ### 3.4  蒸腾  ET

# %%
et_nt_annual_detrend_kndvi_after2000 = et_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi_after2000, method='nearest')
et_nt_annual_detrend_kndvi_after2000

# %%
et_nt_normal_mean_kndvi_after2000 = et_nt_annual_detrend_kndvi_after2000.where(spei_nt_normal_use_kndvi_after2000 > - 5).mean(dim='year')
et_nt_normal_mean_kndvi_after2000

# %%
et_nt_normal_mean_kndvi_after2000.plot()

# %%
et_nt_change_kndvi_after2000 = et_nt_normal_mean_kndvi_after2000 / (et_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - et_nt_normal_mean_kndvi_after2000)
et_nt_change_kndvi_after2000 = et_nt_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
et_nt_change_kndvi_after2000

# %%
et_nt_change_kndvi_after2000[:4].plot(col = 'year', vmax = 100)

# %%
et_nt_change_kndvi_after2000 = et_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)
et_nt_change_kndvi_after2000

# %%
et_nt_change_kndvi_after2000[:4].plot(col = 'year', vmax = 100)

# %%
et_sh_annual_detrend_kndvi_after2000 = et_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi_after2000, method='nearest')
et_sh_annual_detrend_kndvi_after2000

# %%
et_sh_normal_mean_kndvi_after2000 = et_sh_annual_detrend_kndvi_after2000.where(spei_sh_normal_use_kndvi_after2000 > - 5).mean(dim='year')
et_sh_normal_mean_kndvi_after2000

# %%
et_sh_change_kndvi_after2000 = et_sh_normal_mean_kndvi_after2000 / (et_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - et_sh_normal_mean_kndvi_after2000)
et_sh_change_kndvi_after2000 = et_sh_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
et_sh_change_kndvi_after2000

# %%
et_sh_change_kndvi_after2000[:4].plot(col ='year', vmax = 100)

# %%
print(et_nt_change_kndvi_after2000.quantile(0.05).values,et_nt_change_kndvi_after2000.quantile(0.95).values)

# %%
et_nt_change_kndvi_after2000.plot.hist(bins = np.arange(-60,61,10))

# %%
print(et_sh_change_kndvi_after2000.quantile(0.05).values,et_sh_change_kndvi_after2000.quantile(0.95).values)

# %%
et_sh_change_kndvi_after2000.plot.hist(bins = np.arange(-30,31,5))

# %%
et_sh_change_kndvi_after2000 = et_sh_change_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
et_sh_change_kndvi_after2000

# %%
et_sh_change_kndvi_after2000[:4].plot(col ='year', vmax = 100)

# %%
et_sh_change_kndvi_after2000.plot.hist(bins = np.arange(-30,31,5))

# %%
et_nt_change_kndvi_after2000.name = 'et_change'
et_nt_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/et_nt_change_kndvi_after2000.nc')

et_sh_change_kndvi_after2000.name = 'et_change'
et_sh_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/et_sh_change_kndvi_after2000.nc')

# %% [markdown]
# #### 蒸腾 反过来计算变化

# %%
et_nt_change2_kndvi_after2000 =  (et_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - et_nt_normal_mean_kndvi_after2000) / et_nt_normal_mean_kndvi_after2000 
et_nt_change2_kndvi_after2000 = et_nt_change2_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
et_nt_change2_kndvi_after2000

# %%
et_nt_change2_kndvi_after2000[:4].plot(col = 'year' )

# %%
et_nt_change2_kndvi_after2000 = et_nt_change2_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)
et_nt_change2_kndvi_after2000

# %%
et_nt_change2_kndvi_after2000[:4].plot(col = 'year', vmax = 1)

# %%
et_sh_change2_kndvi_after2000 =  (et_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - et_sh_normal_mean_kndvi_after2000) /et_sh_normal_mean_kndvi_after2000
et_sh_change2_kndvi_after2000 = et_sh_change2_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
et_sh_change2_kndvi_after2000

# %%
et_sh_change2_kndvi_after2000[:4].plot(col ='year', vmax = 0.5)

# %%
print(et_nt_change2_kndvi_after2000.quantile(0.05).values,et_nt_change2_kndvi_after2000.quantile(0.95).values)

# %%
et_nt_change2_kndvi_after2000.plot.hist(bins = np.arange(-60,61,10))

# %%
print(et_sh_change2_kndvi_after2000.quantile(0.05).values,et_sh_change2_kndvi_after2000.quantile(0.95).values)

# %%
et_sh_change2_kndvi_after2000.plot.hist(bins = np.arange(-30,31,5))

# %%
et_sh_change2_kndvi_after2000 = et_sh_change2_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
et_sh_change2_kndvi_after2000

# %%
et_sh_change2_kndvi_after2000[:4].plot(col ='year', vmax = 0.5)

# %%
et_sh_change2_kndvi_after2000.plot.hist(bins = np.arange(-30,31,5))

# %%
et_nt_change2_kndvi_after2000.name = 'et_change2'
et_nt_change2_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/et_nt_change2_kndvi_after2000.nc')

et_sh_change2_kndvi_after2000.name = 'et_change2'
et_sh_change2_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/et_sh_change2_kndvi_after2000.nc')

# %% [markdown]
# ### 3.5 蒸散发 EA

# %%
ea_nt_annual_detrend_kndvi_after2000 = ea_nt_annual_detrend.interp_like(spei_nt_normal_use_kndvi_after2000, method='nearest')
ea_nt_annual_detrend_kndvi_after2000

# %%
ea_nt_normal_mean_kndvi_after2000 = ea_nt_annual_detrend_kndvi_after2000.where(spei_nt_normal_use_kndvi_after2000 > - 5).mean(dim='year')
ea_nt_normal_mean_kndvi_after2000

# %%
ea_nt_normal_mean_kndvi_after2000.plot()

# %%
ea_nt_change_kndvi_after2000 = ea_nt_normal_mean_kndvi_after2000 / (ea_nt_annual_detrend_kndvi_after2000.where( spei_nt_drought_use_kndvi_after2000 > -10) - ea_nt_normal_mean_kndvi_after2000)
ea_nt_change_kndvi_after2000 = ea_nt_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
ea_nt_change_kndvi_after2000

# %%
ea_nt_change_kndvi_after2000[:4].plot(col = 'year', vmax = 100)

# %%
ea_nt_change_kndvi_after2000 = ea_nt_change_kndvi_after2000.where(kndvi_nt_resistance_after2000 > 0)
ea_nt_change_kndvi_after2000

# %%
ea_nt_change_kndvi_after2000[:4].plot(col = 'year', vmax = 100)

# %%
ea_sh_annual_detrend_kndvi_after2000 = ea_sh_annual_detrend.interp_like(spei_sh_normal_use_kndvi_after2000, method='nearest')
ea_sh_annual_detrend_kndvi_after2000

# %%
ea_sh_normal_mean_kndvi_after2000 = ea_sh_annual_detrend_kndvi_after2000.where(spei_sh_normal_use_kndvi_after2000 > - 5).mean(dim='year')
ea_sh_normal_mean_kndvi_after2000

# %%
ea_sh_change_kndvi_after2000 = ea_sh_normal_mean_kndvi_after2000 / (ea_sh_annual_detrend_kndvi_after2000.where( spei_sh_drought_use_kndvi_after2000 > -10) - ea_sh_normal_mean_kndvi_after2000)
ea_sh_change_kndvi_after2000 = ea_sh_change_kndvi_after2000.transpose("year", "lat", "lon").drop('quantile')
ea_sh_change_kndvi_after2000

# %%
ea_sh_change_kndvi_after2000[:4].plot(col ='year', vmax = 100)

# %%
print(ea_nt_change_kndvi_after2000.quantile(0.05).values,ea_nt_change_kndvi_after2000.quantile(0.95).values)

# %%
ea_nt_change_kndvi_after2000.plot.hist(bins = np.arange(-60,61,10))

# %%
print(ea_sh_change_kndvi_after2000.quantile(0.05).values,ea_sh_change_kndvi_after2000.quantile(0.95).values)

# %%
ea_sh_change_kndvi_after2000.plot.hist(bins = np.arange(-30,31,5))

# %%
ea_sh_change_kndvi_after2000 = ea_sh_change_kndvi_after2000.where(kndvi_sh_resistance_after2000 > 0)
ea_sh_change_kndvi_after2000

# %%
ea_sh_change_kndvi_after2000[:4].plot(col ='year', vmax = 100)

# %%
ea_sh_change_kndvi_after2000.plot.hist(bins = np.arange(-30,31,5))

# %%
ea_nt_change_kndvi_after2000.name = 'ea_change'
ea_nt_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/ea_nt_change_kndvi_after2000.nc')

ea_sh_change_kndvi_after2000.name = 'ea_change'
ea_sh_change_kndvi_after2000.to_netcdf(r'E:/python_output/fsc_drought/ea_sh_change_kndvi_after2000.nc')

# %% [markdown]
# ## 4 sif 对应的

# %% [markdown]
# ### 4.1 干旱抵抗力数据

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

# %% [markdown]
# ### 4.2 eas 土壤水分

# %%
spei_nt_normal_use_sif

# %%
sm_nt_annual_detrend_sif = sm_nt_annual_detrend.interp_like(spei_nt_normal_use_sif, method='nearest')
sm_sh_annual_detrend_sif = sm_sh_annual_detrend.interp_like(spei_sh_normal_use_sif, method='nearest')
sm_nt_annual_detrend_sif

# %%
sm_sh_annual_detrend_sif

# %%
sm_nt_normal_mean_sif = sm_nt_annual_detrend_sif.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
sm_nt_normal_mean_sif

# %%
sm_nt_change_sif = sm_nt_normal_mean_sif / (sm_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - sm_nt_normal_mean_sif)
sm_nt_change_sif

# %%
sm_nt_change_sif = sm_nt_change_sif.transpose("year", "lat", "lon").drop('quantile')
sm_nt_change_sif

# %%
sm_nt_change_sif.where(sm_nt_change_sif >0).plot.hist()

# %%
sm_nt_change_sif.where(sif_nt_resistance > 0).plot.hist()

# %%
sm_nt_change_sif = sm_nt_change_sif.where(sif_nt_resistance > 0)
sm_nt_change_sif = sm_nt_change_sif.where(sm_nt_change_sif < 0)
sm_nt_change_sif.values = np.abs(sm_nt_change_sif.values)
sm_nt_change_sif.where(sm_nt_change_sif < 100).plot.hist()

# %%
sm_sh_normal_mean_sif = sm_sh_annual_detrend_sif.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
sm_sh_normal_mean_sif

# %%
sm_sh_change_sif = sm_sh_normal_mean_sif / (sm_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - sm_sh_normal_mean_sif)
sm_sh_change_sif = sm_sh_change_sif.transpose("year", "lat", "lon").drop('quantile')
sm_sh_change_sif

# %%
sm_sh_change_sif = sm_sh_change_sif.where(sif_sh_resistance > 0)
sm_sh_change_sif = sm_sh_change_sif.where(sm_sh_change_sif < 0)
sm_sh_change_sif.values = np.abs(sm_sh_change_sif.values)
sm_sh_change_sif

# %%
sm_nt_change_sif.quantile(0.95)

# %%
sm_sh_change_sif.quantile(0.95)

# %%
sm_nt_change_sif[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
sm_sh_change_sif[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
sm_nt_change_sif.name = 'sm_change'
sm_nt_change_sif.to_netcdf(r'E:/python_output/fsc_drought/sm_nt_change_sif.nc')

sm_sh_change_sif.name = 'sm_change'
sm_sh_change_sif.to_netcdf(r'E:/python_output/fsc_drought/sm_sh_change_sif.nc')

# %% [markdown]
# ### 4.3 gleam  土壤水分

# %%
smrz_nt_annual_detrend_sif = smrz_nt_annual_detrend.interp_like(spei_nt_normal_use_sif, method='nearest')
smrz_sh_annual_detrend_sif = smrz_sh_annual_detrend.interp_like(spei_sh_normal_use_sif, method='nearest')
smrz_nt_annual_detrend_sif

# %%
smrz_sh_annual_detrend_sif

# %%
smrz_nt_normal_mean_sif = smrz_nt_annual_detrend_sif.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
smrz_nt_normal_mean_sif

# %%
smrz_nt_change_sif = smrz_nt_normal_mean_sif / (smrz_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - smrz_nt_normal_mean_sif)
smrz_nt_change_sif

# %%
smrz_nt_change_sif = smrz_nt_change_sif.transpose("year", "lat", "lon").drop('quantile')
smrz_nt_change_sif

# %%
smrz_nt_change_sif.where(sif_nt_resistance > 0).plot.hist()

# %%
smrz_nt_change_sif = smrz_nt_change_sif.where(sif_nt_resistance > 0)
smrz_nt_change_sif = smrz_nt_change_sif.where(smrz_nt_change_sif < 0)
smrz_nt_change_sif.values = np.abs(smrz_nt_change_sif.values)
smrz_nt_change_sif.where(smrz_nt_change_sif < 100).plot.hist()

# %%
smrz_sh_normal_mean_sif = smrz_sh_annual_detrend_sif.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
smrz_sh_normal_mean_sif

# %%
smrz_sh_change_sif = smrz_sh_normal_mean_sif / (smrz_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - smrz_sh_normal_mean_sif)
smrz_sh_change_sif = smrz_sh_change_sif.transpose("year", "lat", "lon").drop('quantile')
smrz_sh_change_sif

# %%
smrz_sh_change_sif = smrz_sh_change_sif.where(sif_sh_resistance > 0)
smrz_sh_change_sif = smrz_sh_change_sif.where(smrz_sh_change_sif < 0)
smrz_sh_change_sif.values = np.abs(smrz_sh_change_sif.values)
smrz_sh_change_sif

# %%
smrz_nt_change_sif[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
smrz_sh_change_sif[:8].plot(col = 'year', col_wrap = 4, vmax = 150)

# %%
smrz_nt_change_sif.name = 'sm_change'
smrz_nt_change_sif.to_netcdf(r'E:/python_output/fsc_drought/smrz_nt_change_sif.nc')

smrz_sh_change_sif.name = 'sm_change'
smrz_sh_change_sif.to_netcdf(r'E:/python_output/fsc_drought/smrz_sh_change_sif.nc')

# %% [markdown]
# ### 4.4  蒸腾  ET

# %%
et_nt_annual_detrend_sif = et_nt_annual_detrend.interp_like(spei_nt_normal_use_sif, method='nearest')
et_nt_annual_detrend_sif

# %%
et_nt_normal_mean_sif = et_nt_annual_detrend_sif.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
et_nt_normal_mean_sif

# %%
et_nt_normal_mean_sif.plot()

# %%
et_nt_change_sif = et_nt_normal_mean_sif / (et_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - et_nt_normal_mean_sif)
et_nt_change_sif = et_nt_change_sif.transpose("year", "lat", "lon").drop('quantile')
et_nt_change_sif

# %%
et_nt_change_sif[:4].plot(col = 'year', vmax = 100)

# %%
et_nt_change_sif = et_nt_change_sif.where(sif_nt_resistance > 0)
et_nt_change_sif

# %%
et_nt_change_sif[:4].plot(col = 'year', vmax = 100)

# %%
et_sh_annual_detrend_sif = et_sh_annual_detrend.interp_like(spei_sh_normal_use_sif, method='nearest')
et_sh_annual_detrend_sif

# %%
et_sh_normal_mean_sif = et_sh_annual_detrend_sif.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
et_sh_normal_mean_sif

# %%
et_sh_change_sif = et_sh_normal_mean_sif / (et_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - et_sh_normal_mean_sif)
et_sh_change_sif = et_sh_change_sif.transpose("year", "lat", "lon").drop('quantile')
et_sh_change_sif

# %%
et_sh_change_sif[:4].plot(col ='year', vmax = 100)

# %%
print(et_nt_change_sif.quantile(0.05).values,et_nt_change_sif.quantile(0.95).values)

# %%
et_nt_change_sif.plot.hist(bins = np.arange(-60,61,10))

# %%
print(et_sh_change_sif.quantile(0.05).values,et_sh_change_sif.quantile(0.95).values)

# %%
et_sh_change_sif.plot.hist(bins = np.arange(-30,31,5))

# %%
et_sh_change_sif = et_sh_change_sif.where(sif_sh_resistance > 0)
et_sh_change_sif

# %%
et_sh_change_sif[:4].plot(col ='year', vmax = 100)

# %%
et_sh_change_sif.plot.hist(bins = np.arange(-30,31,5))

# %%
et_nt_change_sif.name = 'et_change'
et_nt_change_sif.to_netcdf(r'E:/python_output/fsc_drought/et_nt_change_sif.nc')

et_sh_change_sif.name = 'et_change'
et_sh_change_sif.to_netcdf(r'E:/python_output/fsc_drought/et_sh_change_sif.nc')

# %% [markdown]
# #### 另一种算法

# %%
et_nt_change2_sif = (et_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - et_nt_normal_mean_sif) / et_nt_normal_mean_sif 
et_nt_change2_sif = et_nt_change2_sif.transpose("year", "lat", "lon").drop('quantile')
et_nt_change2_sif

# %%
et_nt_change2_sif[:4].plot(col = 'year', vmax = 0.5)

# %%
et_nt_change2_sif = et_nt_change2_sif.where(sif_nt_resistance > 0)
et_nt_change2_sif

# %%
et_nt_change2_sif[:4].plot(col = 'year', vmax =0.5)

# %%
et_sh_change2_sif = (et_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - et_sh_normal_mean_sif) / et_sh_normal_mean_sif
et_sh_change2_sif = et_sh_change2_sif.transpose("year", "lat", "lon").drop('quantile')
et_sh_change2_sif

# %%
et_sh_change2_sif[:4].plot(col ='year', vmax = 0.5)

# %%
print(et_nt_change2_sif.quantile(0.05).values,et_nt_change2_sif.quantile(0.95).values)

# %%
et_nt_change2_sif.plot.hist(bins = np.arange(-60,61,10))

# %%
print(et_sh_change2_sif.quantile(0.05).values,et_sh_change2_sif.quantile(0.95).values)

# %%
et_sh_change2_sif.plot.hist(bins = np.arange(-0.7,0.7,0.1))

# %%
et_sh_change2_sif = et_sh_change2_sif.where(sif_sh_resistance > 0)
et_sh_change2_sif

# %%
et_sh_change2_sif[:4].plot(col ='year', vmax = 0.5)

# %%
et_sh_change2_sif.plot.hist(bins = np.arange(-0.7,0.7,0.1))

# %%
et_nt_change2_sif.name = 'et_change2'
et_nt_change2_sif.to_netcdf(r'E:/python_output/fsc_drought/et_nt_change2_sif.nc')

et_sh_change2_sif.name = 'et_change2'
et_sh_change2_sif.to_netcdf(r'E:/python_output/fsc_drought/et_sh_change2_sif.nc')

# %% [markdown]
# ### 4.5 蒸散发 EA

# %%
ea_nt_annual_detrend_sif = ea_nt_annual_detrend.interp_like(spei_nt_normal_use_sif, method='nearest')
ea_nt_annual_detrend_sif

# %%
ea_nt_normal_mean_sif = ea_nt_annual_detrend_sif.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
ea_nt_normal_mean_sif

# %%
ea_nt_normal_mean_sif.plot()

# %%
ea_nt_change_sif = ea_nt_normal_mean_sif / (ea_nt_annual_detrend_sif.where( spei_nt_drought_use_sif > -10) - ea_nt_normal_mean_sif)
ea_nt_change_sif = ea_nt_change_sif.transpose("year", "lat", "lon").drop('quantile')
ea_nt_change_sif

# %%
ea_nt_change_sif[:4].plot(col = 'year', vmax = 100)

# %%
ea_nt_change_sif = ea_nt_change_sif.where(sif_nt_resistance > 0)
ea_nt_change_sif

# %%
ea_nt_change_sif[:4].plot(col = 'year', vmax = 100)

# %%
ea_sh_annual_detrend_sif = ea_sh_annual_detrend.interp_like(spei_sh_normal_use_sif, method='nearest')
ea_sh_annual_detrend_sif

# %%
ea_sh_normal_mean_sif = ea_sh_annual_detrend_sif.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
ea_sh_normal_mean_sif

# %%
ea_sh_change_sif = ea_sh_normal_mean_sif / (ea_sh_annual_detrend_sif.where( spei_sh_drought_use_sif > -10) - ea_sh_normal_mean_sif)
ea_sh_change_sif = ea_sh_change_sif.transpose("year", "lat", "lon").drop('quantile')
ea_sh_change_sif

# %%
ea_sh_change_sif[:4].plot(col ='year', vmax = 100)

# %%
print(ea_nt_change_sif.quantile(0.05).values,ea_nt_change_sif.quantile(0.95).values)

# %%
ea_nt_change_sif.plot.hist(bins = np.arange(-60,61,10))

# %%
print(ea_sh_change_sif.quantile(0.05).values,ea_sh_change_sif.quantile(0.95).values)

# %%
ea_sh_change_sif.plot.hist(bins = np.arange(-30,31,5))

# %%
ea_sh_change_sif = ea_sh_change_sif.where(sif_sh_resistance > 0)
ea_sh_change_sif

# %%
ea_sh_change_sif[:4].plot(col ='year', vmax = 100)

# %%
ea_sh_change_sif.plot.hist(bins = np.arange(-30,31,5))

# %%
ea_nt_change_sif.name = 'ea_change'
ea_nt_change_sif.to_netcdf(r'E:/python_output/fsc_drought/ea_nt_change_sif.nc')

ea_sh_change_sif.name = 'ea_change'
ea_sh_change_sif.to_netcdf(r'E:/python_output/fsc_drought/ea_sh_change_sif.nc')

# %% [markdown]
# ### 4.6  蒸腾/蒸发  ET/EA

# %%
ret_ea_nt_sif = et_nt_annual_detrend_sif /  ea_nt_annual_detrend_sif
ret_ea_sh_sif = et_sh_annual_detrend_sif /  ea_sh_annual_detrend_sif
ret_ea_nt_sif = ret_ea_nt_sif.where(ret_ea_nt_sif<1)
ret_ea_sh_sif = ret_ea_sh_sif.where(ret_ea_sh_sif<1)

# %%
ret_ea_nt_normal_mean_sif = ret_ea_nt_sif.where(spei_nt_normal_use_sif > - 5).mean(dim='year')
ret_ea_nt_normal_mean_sif

# %%
ret_ea_nt_normal_mean_sif.plot()

# %%
ret_ea_nt_change_sif = ret_ea_nt_normal_mean_sif / (ret_ea_nt_sif.where( spei_nt_drought_use_sif > -10) - ret_ea_nt_normal_mean_sif)
ret_ea_nt_change_sif

# %%
ret_ea_nt_change_sif = ret_ea_nt_change_sif.transpose("year", "lat", "lon").drop('quantile')
ret_ea_nt_change_sif

# %%
ret_ea_nt_change_sif[:4].plot(col = 'year', vmax = 100)

# %%
ret_ea_nt_change_sif = ret_ea_nt_change_sif.where(sif_nt_resistance > 0)

# %%
ret_ea_nt_change_sif[:4].plot(col = 'year', vmax = 100)

# %%
ret_ea_sh_normal_mean_sif = ret_ea_sh_sif.where(spei_sh_normal_use_sif > - 5).mean(dim='year')
ret_ea_sh_normal_mean_sif

# %%
ret_ea_sh_normal_mean_sif.plot()

# %%
ret_ea_sh_change_sif = ret_ea_sh_normal_mean_sif / (ret_ea_sh_sif.where( spei_sh_drought_use_sif > -10) - ret_ea_sh_normal_mean_sif)
ret_ea_sh_change_sif = ret_ea_sh_change_sif.transpose("year", "lat", "lon").drop('quantile')
ret_ea_sh_change_sif

# %%
ret_ea_sh_change_sif[:4].plot(col = 'year', vmax = 50)

# %%
ret_ea_sh_change_sif = ret_ea_sh_change_sif.where(sif_sh_resistance > 0)
ret_ea_sh_change_sif[:4].plot(col = 'year', vmax = 50)

# %%
print(ret_ea_nt_change_sif.quantile(0.05).values,ret_ea_nt_change_sif.quantile(0.95).values)

# %%
ret_ea_nt_change_sif.plot.hist(bins = np.arange(-100,110,10))

# %%
print(ret_ea_sh_change_sif.quantile(0.05).values,ret_ea_sh_change_sif.quantile(0.95).values)

# %%
ret_ea_sh_change_sif.plot.hist(bins = np.arange(-80,90,10))

# %%
ret_ea_nt_change_sif.name = 'ret_ea_change'
ret_ea_nt_change_sif.to_netcdf(r'E:/python_output/fsc_drought/ret_ea_nt_change_sif.nc')

ret_ea_sh_change_sif.name = 'ret_ea_change'
ret_ea_sh_change_sif.to_netcdf(r'E:/python_output/fsc_drought/ret_ea_sh_change_sif.nc')

# %% [markdown]
# ## change log
# 1. 2025.10.27  计算了对应干旱植被响应的  土壤水分 蒸散发 还有蒸腾蒸散比的变化
# 2. 2025.11.17  交换 除数和被除数之后 计算蒸腾变化

# %%


# %% [markdown]
# 


