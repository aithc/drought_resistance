# %% [markdown]
# # 用来做一些数据预处理

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
# ## 1 处理一下 gimms lai的数据  然后算一个生长季的均值 或者中位数

# %% [markdown]
# ### 1.1 先读取一个数据看看

# %%
with xr.open_dataset(r'D:/data/gimms_lai/GIMMS_LAI4g_V1.2_19820101.tif') as data:
    lai_exam = data['band_data']
lai_exam

# %%
lai_exam[0].where(lai_exam[0]<60000).plot()

# %%
lai_exam[1].where(lai_exam[1]<60000).plot()

# %%
lai_exam[1].where(lai_exam[1]<60000).plot.hist()

# %% [markdown]
# ### 1.2 处理一下 landcover的变化
# 先用众数把landcover重采样到土壤水分数据的原始分辨率，然后再在这个分辨率上看landcover的有没有发生变化
# 
# 因为这个数据是0.083度  所有还是简单做一下插值吧

# %%
## 读取landcover数据
with xr.open_dataset(r'D:/data/modis_landcover/modis_IGBP_2001_2022.nc').modis_landcover as ld:
    landCover = ld
landCover

# %%
lai_exam = lai_exam.sel(band = 1).drop(['band','spatial_ref']).rename({'x':'lon','y':'lat'})
lai_exam

# %%
landCover = landCover.interp_like(lai_exam)
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
ld_change =  xr.apply_ufunc(
    ld_unchanged,
    landCover,
    input_core_dims=[['year']],
    output_core_dims=[[]],
    vectorize= True
)

# %%
modis_ld_use = landCover[-1].where(ld_change == 0)
modis_ld_use = modis_ld_use.where(modis_ld_use >0)
modis_ld_use = modis_ld_use.where(modis_ld_use <11)
modis_ld_use = modis_ld_use.drop('year')
modis_ld_use = modis_ld_use.sortby('lat', ascending=False)
modis_ld_use

# %%
modis_ld_use.plot()

# %% [markdown]
# ### 1.3 读取所有的 lai数据
# 
# 分南北半球和热带来做  不然可能数据太大

# %%
lai_path = glob.glob(r'D:/data/gimms_lai/GIMMS_LAI4g_V1.2_*.tif')
lai_path

# %%
lai_path[0].split('_')[-1].split('.')[0][:6]

# %%
lai_nh = []
for lai_path_n in lai_path:
    print(lai_path_n)
    with xr.open_dataset(lai_path_n) as data:
        lai_n = data['band_data'].sel(y = slice(90,23.5), band = 1)
    lai_nh.append(lai_n)

lai_nh = xr.concat(lai_nh, dim = 'band')
lai_nh

# %%
lai_nh = lai_nh.where(lai_nh <60000)
lai_nh = lai_nh.drop('spatial_ref').rename({'band':'time','x':'lon','y':'lat'})
lai_nh

# %%
import pandas as pd

# %%
lai_nh.name = 'lai'
lai_nh = lai_nh.assign_coords(time = pd.date_range('1982-01-01', '2020-12-31', freq='SM'))
lai_nh

# %%
lai_nh[0].plot()

# %%
lai_nh = lai_nh.sel(time = lai_nh.time.dt.month.isin([5,6,7,8,9])).groupby('time.year').mean(dim='time')
lai_nh

# %%
lai_nh[0].plot()

# %%
lai_tr = []
for lai_path_n in lai_path:
    print(lai_path_n)
    with xr.open_dataset(lai_path_n) as data:
        lai_n = data['band_data'].sel(y = slice(23.5,-23.5), band = 1)
    lai_tr.append(lai_n)

lai_tr = xr.concat(lai_tr, dim = 'band')
lai_tr

# %%
lai_tr = lai_tr.where(lai_tr <60000)
lai_tr = lai_tr.drop('spatial_ref').rename({'band':'time','x':'lon','y':'lat'})
lai_tr

# %%
lai_tr.name = 'lai'
lai_tr = lai_tr.assign_coords(time = pd.date_range('1982-01-01', '2020-12-31', freq='SM'))
lai_tr

# %%
lai_tr[0].plot()

# %%
lai_tr = lai_tr.groupby('time.year').mean(dim='time')
lai_tr

# %%
lai_tr[0].plot()

# %%
lai_sh = []
for lai_path_n in lai_path:
    print(lai_path_n)
    with xr.open_dataset(lai_path_n) as data:
        lai_n = data['band_data'].sel(y = slice(-23.5,-90), band = 1)
    lai_sh.append(lai_n)

lai_sh = xr.concat(lai_sh, dim = 'band')
lai_sh

# %%
lai_sh = lai_sh.where(lai_sh <60000)
lai_sh = lai_sh.drop('spatial_ref').rename({'band':'time','x':'lon','y':'lat'})
lai_sh

# %%
lai_sh.name = 'lai'
lai_sh = lai_sh.assign_coords(time = pd.date_range('1981-07-01', '2020-06-30', freq='SM'))  ## 南半球时间调整半年
lai_sh

# %%
lai_sh[0].plot()

# %%
lai_sh = lai_sh.sel(time = lai_sh.time.dt.month.isin([5,6,7,8,9])).groupby('time.year').mean(dim='time')
lai_sh

# %%
lai_sh[0].plot()

# %%
lai_sh = lai_sh.sel(year = slice(1982,2019)).mean(dim='year')
lai_nh = lai_nh.sel(year = slice(1982,2019)).mean(dim='year')
lai_tr = lai_tr.sel(year = slice(1982,2019)).mean(dim='year')

# %%
lai_gs = xr.concat([lai_sh,lai_nh,lai_tr], dim = 'lat').sortby('lat')
lai_gs

# %%
lai_gs.plot()

# %%
lai_gs = lai_gs.where(modis_ld_use > 0)
lai_gs.plot()

# %%
lai_gs_05 = lai_gs.coarsen(lat = 6,lon = 6).mean()
lai_gs_05.plot()

# %%
lai_gs_05

# %%
lai_gs_05.to_netcdf(r'E:/python_output/fsc_drought/lai_gs_05.nc')

# %%



