# %% [markdown]
# # 用机器学习看看干旱抵抗力的影响因素
# 
# 试一下 新算的那些结果  vod,sif,kndvi + csc

# %% [markdown]
# ## 1 先读取数据   用 geo虚拟环境

# %%
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import glob

# %% [markdown]
# ### 1.1 干旱抵抗力

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance.nc') as data:
    kndvi_nt_resistance = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance.nc') as data:
    kndvi_sh_resistance = data['kndvi_resistance']

# %%
kndvi_nt_resistance

# %%
kndvi_sh_resistance

# %%
kndvi_nt_resistance[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance_after2000.nc') as data:
    kndvi_nt_resistance_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance_after2000.nc') as data:
    kndvi_sh_resistance_after2000 = data['kndvi_resistance']

# %%
kndvi_nt_resistance_after2000

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2.nc') as data:
    kndvi_nt_resistance2 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2.nc') as data:
    kndvi_sh_resistance2 = data['kndvi_resistance']

# %%
kndvi_nt_resistance2

# %%
kndvi_nt_resistance2[:6].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 3, cmap = 'RdYlGn')

# %%
kndvi_sh_resistance2

# %%
kndvi_sh_resistance2[:6].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 3, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2_after2000.nc') as data:
    kndvi_nt_resistance2_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2_after2000.nc') as data:
    kndvi_sh_resistance2_after2000 = data['kndvi_resistance']

# %%
kndvi_nt_resistance2_after2000[:6].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 3, cmap = 'RdYlGn')

# %%
kndvi_sh_resistance2_after2000[:6].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 3, cmap = 'RdYlGn')

# %% [markdown]
# ### 1.2 干旱指数

# %%
with xr.open_dataset(r'../result_data/spei_nt_annual_drought.nc') as data:
    spei_nt_annual_drought = data['spei']

with xr.open_dataset(r'../result_data/spei_sh_annual_drought.nc') as data:
    spei_sh_annual_drought = data['__xarray_dataarray_variable__']
    spei_sh_annual_drought.name = 'spei'

# %%
spei_nt_drought_use_kndvi = spei_nt_annual_drought.sel(year = slice(1982, 2021))
spei_sh_drought_use_kndvi = spei_sh_annual_drought.sel(year = slice(1982, 2020))

# %%
spei_nt_drought_use_kndvi_after2000 = spei_nt_annual_drought.sel(year = slice(2000, 2021))
spei_sh_drought_use_kndvi_after2000 = spei_sh_annual_drought.sel(year = slice(2000, 2020))

# %%
spei_nt_drought_use_kndvi[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
spei_sh_drought_use_kndvi[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %% [markdown]
# ### 1.3 气候背景

# %%
era_temp_path = glob.glob(r'D:/data/era5_land_annual/era5_land_mean*.nc')
era_prec_path = glob.glob(r'D:/data/era5_land_annual/era5_land_pre*.nc')
## 年均温
annual_temp = []
for path_n in era_temp_path:
    print(path_n)
    with xr.open_dataset(path_n)['t2m']  as data:
        data = xr.DataArray(data.values, coords=[data.year, data.lat, data.lon], dims=['year','lat','lon'])
        annual_temp.append(data)

annual_temp = xr.concat(annual_temp, dim='year')
annual_temp = annual_temp.mean(dim='year')
annual_temp = annual_temp - 273.15
annual_temp.plot()

# %%
## 年降水
annual_prec = []
for path_n in era_prec_path:
    print(path_n)
    with xr.open_dataset(path_n)['tp']  as data:
        data = xr.DataArray(data.values, coords=[data.year, data.lat, data.lon], dims=['year','lat','lon'])
        annual_prec.append(data)

annual_prec = xr.concat(annual_prec, dim='year')
annual_prec = annual_prec.mean(dim='year')
annual_prec = annual_prec*1000*30
annual_prec = annual_prec.where(annual_prec < 7000)
annual_prec.plot()

# %% [markdown]
# ### 1.4 多样性和结构复杂度

# %%
with xr.open_dataset(r'../result_data/plant_richness_log_05.nc') as data:
    plant_richness = data['richness']
plant_richness

# %%
plant_richness.plot()

# %%
with xr.open_dataset(r'D:/data/fsc_from_su/data/global_forest_csc/global_forest_csc.tif')  as  data:
    fsc = data['band_data'][0].drop(['spatial_ref','band'])
fsc = fsc.rename({'x':'lon','y':'lat'})
fsc = fsc.coarsen(lat = 20, lon=20).mean()
fsc

# %%
fsc = fsc.interp_like(plant_richness, method='nearest')
fsc.plot()

# %% [markdown]
# ### 1.5 干旱指标

# %%
with xr.open_dataset(r'../result_data/drought_chars_1982_2022_new.nc') as data:
    drought_severity = data['severity']
    drought_count =  data['count']     
    drought_duration = data['duration']
drought_severity

# %%
drought_count

# %%
drought_duration

# %% [markdown]
# ### 1.6 土壤性质

# %%
with xr.open_dataset(r'D:/data/soil/soilgrids2.0_cec_060.nc')  as data:
    soil_cec = data['cec_060']
    soil_cec = soil_cec.where(soil_cec>0)
soil_cec = soil_cec.coarsen(lat =12, lon = 12).mean() * 0.001
soil_cec = soil_cec.interp_like(plant_richness, method='nearest')
soil_cec

# %%
with xr.open_dataset(r'D:/data/soil/soilgrids2.0_clay_060.nc')  as data:
    soil_clay = data['clay_060']
    soil_clay = soil_clay.where(soil_clay>0)
soil_clay = soil_clay.coarsen(lat =12, lon = 12).mean() * 0.001
soil_clay = soil_clay.interp_like(plant_richness, method='nearest')
soil_clay
soil_clay.plot()

# %% [markdown]
# ### 1.7 CTI

# %%
with xr.open_dataset(r'D:/data/Compound_topographic_index/cti_24.nc') as data:
    cti = data['cti']
cti

# %%
cti = cti.coarsen(lat = 10, lon =10, boundary='pad').mean()
cti = cti.interp_like(plant_richness, method='nearest')
cti

# %%
cti.plot(center = False)

# %% [markdown]
# ### 1.8 性状

# %%
with rioxarray.open_rasterio(r'D:/data/trait_Boonman_2020_geb/SLA.tif') as data:
    sla = data.where(data>0)
sla

# %%
sla = xr.DataArray(sla[0], coords=[sla.y, sla.x], dims=['lat','lon'])
sla = sla.interp_like(plant_richness, method='nearest')
sla

# %%
with rioxarray.open_rasterio(r'D:/data/trait_Boonman_2020_geb/Wood.density.tif') as data:
    wood_den = data.where(data>0)
wood_den

# %%
wood_den = xr.DataArray(wood_den[0], coords=[wood_den.y, wood_den.x], dims=['lat','lon'])
wood_den = wood_den.interp_like(plant_richness, method='nearest')
wood_den

# %% [markdown]
# ### 1.9 干旱指数 ai

# %%
## 干旱指数
with rioxarray.open_rasterio(r'D:/data/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif')  as data:
        ai_index= xr.DataArray(data.values[0], coords=[data.y, data.x], dims=['lat','lon'])
ai_index = ai_index.coarsen(lat=10,lon=10).mean()
ai_index

# %%
ai_index = ai_index * 0.0001
ai_index.where(ai_index<1.5).plot()

# %%
annual_temp = annual_temp.coarsen(lat = 5, lon=5, boundary='trim').mean()
annual_prec = annual_prec.coarsen(lat =5, lon=5, boundary='trim').mean()
ai_index = ai_index.coarsen(lat = 6, lon = 6).mean()
annual_temp = annual_temp.interp_like(plant_richness, method='nearest')
annual_prec = annual_prec.interp_like(plant_richness, method='nearest')
ai_index = ai_index.interp_like(plant_richness, method='nearest')

# %% [markdown]
# ### 1.10 土壤水分变化

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi.nc') as data:
    smrz_nt_change_kndvi = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi.nc') as data:
    smrz_sh_change_kndvi = data['sm_change']
smrz_nt_change_kndvi

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi_after2000.nc') as data:
    smrz_nt_change_kndvi_after2000 = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi_after2000.nc') as data:
    smrz_sh_change_kndvi_after2000 = data['sm_change']
smrz_nt_change_kndvi_after2000

# %%
smrz_nt_change_kndvi[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
smrz_sh_change_kndvi

# %%
smrz_sh_change_kndvi[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %% [markdown]
# ### 1.11 合并数据

# %%
dataset_nt_kndvi = xr.Dataset({
        'kndvi': kndvi_nt_resistance,
        'spei': spei_nt_drought_use_kndvi,
        'sm_change': smrz_nt_change_kndvi}).drop('quantile')
dataset_nt_kndvi

# %%
dataset_sh_kndvi = xr.Dataset({
        'kndvi': kndvi_sh_resistance,
        'spei': spei_sh_drought_use_kndvi,
        'sm_change': smrz_sh_change_kndvi}).drop('quantile')
dataset_sh_kndvi

# %%
df_nt_kndvi = dataset_nt_kndvi.to_dataframe().reset_index()
df_sh_kndvi = dataset_sh_kndvi.to_dataframe().reset_index()
df_nt_kndvi = df_nt_kndvi.dropna()
df_sh_kndvi = df_sh_kndvi.dropna()

# %%
df_nt_kndvi.head()

# %%
df_sh_kndvi.head()

# %%
df_all_kndvi = pd.concat([df_nt_kndvi, df_sh_kndvi])
df_all_kndvi.index = np.arange(df_all_kndvi.shape[0])
df_all_kndvi.head()

# %%
dataset_nt_kndvi_after2000 = xr.Dataset({
        'kndvi': kndvi_nt_resistance_after2000,
        'spei': spei_nt_drought_use_kndvi_after2000,
        'sm_change': smrz_nt_change_kndvi_after2000}).drop('quantile')
dataset_nt_kndvi_after2000

# %%
dataset_sh_kndvi_after2000 = xr.Dataset({
        'kndvi': kndvi_sh_resistance_after2000,
        'spei': spei_sh_drought_use_kndvi_after2000,
        'sm_change': smrz_sh_change_kndvi_after2000}).drop('quantile')
dataset_sh_kndvi_after2000

# %%
df_nt_kndvi_after2000 = dataset_nt_kndvi_after2000.to_dataframe().reset_index()
df_sh_kndvi_after2000 = dataset_sh_kndvi_after2000.to_dataframe().reset_index()
df_nt_kndvi_after2000 = df_nt_kndvi_after2000.dropna()
df_sh_kndvi_after2000 = df_sh_kndvi_after2000.dropna()

# %%
df_nt_kndvi_after2000.head()

# %%
df_sh_kndvi_after2000.head()

# %%
df_all_kndvi_after2000 = pd.concat([df_nt_kndvi_after2000, df_sh_kndvi_after2000])
df_all_kndvi_after2000.index = np.arange(df_all_kndvi_after2000.shape[0])
df_all_kndvi_after2000.head()

# %%
other_factor = xr.Dataset({
        'fsc': fsc,
        'annual_temp': annual_temp,
        'annual_prec': annual_prec,
        'plant_richness': plant_richness,
        'drought_severity': drought_severity,
        'drought_count': drought_count,
        'drought_duration': drought_duration,
        'soil_cec': soil_cec,
        'soil_clay': soil_clay,
        'cti': cti,
        'sla': sla,
        'wood_den': wood_den,
        'ai_index': ai_index
    }).to_dataframe().reset_index()
other_factor.head()

# %%
df_all_kndvi = pd.merge(df_all_kndvi, other_factor, on = ['lat','lon'])
df_all_kndvi.head()

# %%
df_all_kndvi = df_all_kndvi.dropna()
df_all_kndvi.index = np.arange(df_all_kndvi.shape[0])
df_all_kndvi.head()

# %%
df_all_kndvi.describe()

# %%
df_all_kndvi.to_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi_20251216.csv', index = False)

# %%
df_all_kndvi_after2000 = pd.merge(df_all_kndvi_after2000, other_factor, on = ['lat','lon'])
df_all_kndvi_after2000.head()

# %%
df_all_kndvi_after2000 = df_all_kndvi_after2000.dropna()
df_all_kndvi_after2000.index = np.arange(df_all_kndvi_after2000.shape[0])
df_all_kndvi_after2000.head()

# %%
df_all_kndvi_after2000.describe()

# %%
df_all_kndvi_after2000.to_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi_after2000_20251216.csv', index = False)

# %%
dataset_nt_kndvi2 = xr.Dataset({
        'kndvi': kndvi_nt_resistance2,
        'spei': spei_nt_drought_use_kndvi,
        'sm_change': smrz_nt_change_kndvi}).drop('quantile')
dataset_nt_kndvi2

# %%
dataset_sh_kndvi2 = xr.Dataset({
        'kndvi': kndvi_sh_resistance2,
        'spei': spei_sh_drought_use_kndvi,
        'sm_change': smrz_sh_change_kndvi}).drop('quantile')
dataset_sh_kndvi2

# %%
df_nt_kndvi2 = dataset_nt_kndvi2.to_dataframe().reset_index()
df_sh_kndvi2 = dataset_sh_kndvi2.to_dataframe().reset_index()
df_nt_kndvi2 = df_nt_kndvi2.dropna()
df_sh_kndvi2 = df_sh_kndvi2.dropna()

# %%
df_nt_kndvi2.head()

# %%
df_sh_kndvi2.head()

# %%
df_all_kndvi2 = pd.concat([df_nt_kndvi2, df_sh_kndvi2])
df_all_kndvi2.index = np.arange(df_all_kndvi2.shape[0])
df_all_kndvi2.head()

# %%
dataset_nt_kndvi2_after2000 = xr.Dataset({
        'kndvi': kndvi_nt_resistance2_after2000,
        'spei': spei_nt_drought_use_kndvi_after2000,
        'sm_change': smrz_nt_change_kndvi_after2000}).drop('quantile')
dataset_nt_kndvi2_after2000

# %%
dataset_sh_kndvi2_after2000 = xr.Dataset({
        'kndvi': kndvi_sh_resistance2_after2000,
        'spei': spei_sh_drought_use_kndvi_after2000,
        'sm_change': smrz_sh_change_kndvi_after2000}).drop('quantile')
dataset_sh_kndvi2_after2000

# %%
df_nt_kndvi2_after2000 = dataset_nt_kndvi2_after2000.to_dataframe().reset_index()
df_sh_kndvi2_after2000 = dataset_sh_kndvi2_after2000.to_dataframe().reset_index()
df_nt_kndvi2_after2000 = df_nt_kndvi2_after2000.dropna()
df_sh_kndvi2_after2000 = df_sh_kndvi2_after2000.dropna()

# %%
df_nt_kndvi2_after2000.head()

# %%
df_sh_kndvi2_after2000.head()

# %%
df_all_kndvi2_after2000 = pd.concat([df_nt_kndvi2_after2000, df_sh_kndvi2_after2000])
df_all_kndvi2_after2000.index = np.arange(df_all_kndvi2_after2000.shape[0])
df_all_kndvi2_after2000.head()

# %%
df_all_kndvi2 = pd.merge(df_all_kndvi2, other_factor, on = ['lat','lon'])
df_all_kndvi2.head()

# %%
df_all_kndvi2 = df_all_kndvi2.dropna()
df_all_kndvi2.index = np.arange(df_all_kndvi2.shape[0])
df_all_kndvi2.head()

# %%
df_all_kndvi2.describe()

# %%
df_all_kndvi2.to_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi2_20251216.csv', index = False)

# %%
df_all_kndvi2_after2000 = pd.merge(df_all_kndvi2_after2000, other_factor, on = ['lat','lon'])
df_all_kndvi2_after2000.head()

# %%
df_all_kndvi2_after2000 = df_all_kndvi2_after2000.dropna()
df_all_kndvi2_after2000.index = np.arange(df_all_kndvi2_after2000.shape[0])
df_all_kndvi2_after2000.head()

# %%
df_all_kndvi2_after2000.describe()

# %%
df_all_kndvi2_after2000.to_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi2_after2000_20251216.csv', index = False)

# %% [markdown]
# ## 2 sif的数据

# %% [markdown]
# ### 2.1 干旱抵抗力

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance.nc') as data:
    sif_nt_resistance = data['sif_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance.nc') as data:
    sif_sh_resistance = data['sif_resistance']
sif_nt_resistance

# %%
sif_sh_resistance

# %%
sif_nt_resistance[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance2.nc') as data:
    sif_nt_resistance2 = data['sif_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance2.nc') as data:
    sif_sh_resistance2 = data['sif_resistance']

# %%
sif_nt_resistance2

# %%
sif_sh_resistance2[:6].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 3, cmap = 'RdYlGn')

# %% [markdown]
# ### 2.2 干旱指数

# %%
spei_nt_drought_use_sif = spei_nt_annual_drought.sel(year = slice(2000, 2022))
spei_sh_drought_use_sif = spei_sh_annual_drought.sel(year = slice(2000, 2021))

# %%
spei_nt_drought_use_sif[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
spei_sh_drought_use_sif[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %% [markdown]
# ### 2.3 土壤水分变化

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_sif.nc') as data:
    smrz_nt_change_sif = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_sif.nc') as data:
    smrz_sh_change_sif = data['sm_change']
smrz_nt_change_sif

# %%
smrz_nt_change_sif[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
smrz_sh_change_sif

# %%
smrz_sh_change_sif[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %% [markdown]
# ### 2.4 合并数据

# %%
dataset_nt_sif = xr.Dataset({
        'sif': sif_nt_resistance,
        'spei': spei_nt_drought_use_sif,
        'sm_change': smrz_nt_change_sif}).drop('quantile')
dataset_nt_sif

# %%
dataset_sh_sif = xr.Dataset({
        'sif':sif_sh_resistance,
        'spei': spei_sh_drought_use_sif,
        'sm_change': smrz_sh_change_sif}).drop('quantile')
dataset_sh_sif

# %%
df_nt_sif = dataset_nt_sif.to_dataframe().reset_index()
df_sh_sif = dataset_sh_sif.to_dataframe().reset_index()
df_nt_sif = df_nt_sif.dropna()
df_sh_sif = df_sh_sif.dropna()
df_nt_sif.head()

# %%
df_sh_sif.head()

# %%
df_all_sif = pd.concat([df_nt_sif, df_sh_sif])
df_all_sif.index = np.arange(df_all_sif.shape[0])
df_all_sif.head()

# %%
df_all_sif = pd.merge(df_all_sif, other_factor, on = ['lat','lon'])
df_all_sif.head()

# %%
df_all_sif = df_all_sif.dropna()

# %%
df_all_sif.index = np.arange(df_all_sif.shape[0])
df_all_sif.head()

# %%
df_all_sif.describe()

# %%
df_all_sif.to_csv(r'E:/python_output/fsc_drought/df_ml_all_sif_20251213.csv', index = False)

# %%
dataset_nt_sif2 = xr.Dataset({
        'sif': sif_nt_resistance2,
        'spei': spei_nt_drought_use_sif,
        'sm_change': smrz_nt_change_sif}).drop('quantile')
dataset_nt_sif2

# %%
dataset_sh_sif2 = xr.Dataset({
        'sif':sif_sh_resistance2,
        'spei': spei_sh_drought_use_sif,
        'sm_change': smrz_sh_change_sif}).drop('quantile')
dataset_sh_sif2

# %%
df_nt_sif2 = dataset_nt_sif2.to_dataframe().reset_index()
df_sh_sif2 = dataset_sh_sif2.to_dataframe().reset_index()
df_nt_sif2 = df_nt_sif2.dropna()
df_sh_sif2 = df_sh_sif2.dropna()

# %%
df_nt_sif2.head()

# %%
df_sh_sif2.head()

# %%
df_all_sif2 = pd.concat([df_nt_sif2, df_sh_sif2])
df_all_sif2.index = np.arange(df_all_sif2.shape[0])
df_all_sif2.head()

# %%
df_all_sif2 = pd.merge(df_all_sif2, other_factor, on = ['lat','lon'])
df_all_sif2.head()

# %%
df_all_sif2 = df_all_sif2.dropna()
df_all_sif2.index = np.arange(df_all_sif2.shape[0])
df_all_sif2.head()

# %%
df_all_sif2.describe()

# %%
df_all_sif2.to_csv(r'E:/python_output/fsc_drought/df_ml_all_sif2_20251213.csv', index = False)

# %% [markdown]
# ## 3 做机器学习

# %%
import pandas as pd
import numpy as np
import shap 

import time
from datetime import timedelta
import matplotlib.pyplot as plt

import sklearn
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, KFold, cross_val_score
from sklearn.inspection import permutation_importance

## 试试 optuna
import optuna

# %%
from sklearn.inspection import partial_dependence
def plot_pdp(model, X, feature, target=False, label=False,return_pd=False, y_pct=True, figsize=(10,9), norm_hist=True, dec=.5):
    # Get partial dependence
    pardep = partial_dependence(model, X, [feature],kind='average' )
    #percentiles=(0.02, 0.98)
    # Get min & max values
    xmin = pardep['grid_values'][0].min()
    xmax = pardep['grid_values'][0].max()
    yrange = pardep['average'][0].max() - pardep['average'][0].min()
    ymin = pardep['average'][0].min()-0.05 *yrange
    ymax = pardep['average'][0].max()+0.05 *yrange
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.grid(alpha=.5, linewidth=1)
    
    # Plot partial dependence
    color = '#620284'
    ax1.plot(pardep['grid_values'][0], pardep['average'][0], color=color, linewidth=4)
    ax1.tick_params(axis='both', labelcolor='black', labelsize = 18)
    ax1.set_xlabel(label, fontsize=20)
    ax1.set_ylim([ymin, ymax])
    ax1.set_ylabel(r'Marginal impact on {}'.format(target), color='black', fontsize=20)
    #ax1.axvline(x=0, color='black', linestyle='dashed')
    
    ax2 = ax1.twinx()
    color = '#0A66C2'
    ax2.hist(X[feature], bins=200, range=(xmin, xmax), alpha=.25, color=color, density=False)
    ax2.tick_params(axis='both', labelcolor='black', labelsize = 18)
    ax2.set_ylabel('Frequency Distribution', color='black', fontsize=20)
    

    #plt.savefig('{}_{}_pdp.jpg'.format(target,label), dpi=500)
    return plt

# %%
col_dict = {'Climate':'dodgerblue','Landcover':'orange','Forest structure':'green','Drought':'orangered','Soil':'peru','Trait':'olive','Species richness':'green','CTI':'peru'}

# %%
y_labs = ['Wood density','Clay content','Specific leaf area','Compound topographic index','Cation exchange capacity','Drought counts',
          'Mean drought severity','Species richness','Mean drought duration','Mean annual precipitation',
         'Forest structural complexity', 'Landcover','Aridity index','Mean annual temperature']

# %%
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

# %% [markdown]
# ## 3.1 kndvi

# %% [markdown]
# ### 2.1 读取数据

# %%
drought_resistance_kndvi = pd.read_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi_20251216.csv')
drought_resistance_kndvi

# %%
drought_resistance_kndvi = drought_resistance_kndvi.drop(columns=['lon','lat','year'])
drought_resistance_kndvi.describe()

# %%
drought_resistance_kndvi_high = drought_resistance_kndvi.kndvi.quantile(0.9)
drought_resistance_kndvi = drought_resistance_kndvi[drought_resistance_kndvi.kndvi < drought_resistance_kndvi_high]

drought_resistance_kndvi.describe()

# %%
x_resistance_kndvi = drought_resistance_kndvi.drop(['kndvi'], axis=1)
y_resistance_kndvi = drought_resistance_kndvi['kndvi']

# %%
y_resistance_kndvi = np.log(y_resistance_kndvi)

# %%
y_resistance_kndvi.plot.hist()

# %% [markdown]
# ### 2.2 建模

# %%
X_train_kndvi, X_test_kndvi, y_train_kndvi, y_test_kndvi = sklearn.model_selection.train_test_split( x_resistance_kndvi, y_resistance_kndvi)
reg = xgb.XGBRegressor()
reg.fit(X_train_kndvi, y_train_kndvi)

# %%
reg.score(X_train_kndvi, y_train_kndvi)

# %%
reg.score(X_test_kndvi, y_test_kndvi)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 1, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 5, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 5, 10),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_kndvi, y_train_kndvi, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_kndvi = study.best_params

# %%
## 20251216 的调参结果
best_params_kndvi = {'n_estimators': 550,
 'learning_rate': 0.06581885381938615,
 'max_depth': 6,
 'subsample': 0.6666613632210401,
 'colsample_bytree': 0.6486228680741734,
 'min_child_weight': 4.491032007444079,
 'gamma': 1.0086192687999287,
 'reg_alpha': 9.973932227506042,
 'reg_lambda': 8.333055146335706}

# %%
## 这是上一次的调参结果 可能是20251119
best_params_kndvi = {'n_estimators': 950,   
 'learning_rate': 0.08216808463006084,
 'max_depth': 5,
 'subsample': 0.7212970720632401,
 'colsample_bytree': 0.7578553006479924,
 'min_child_weight': 3.87128353789333,
 'gamma': 1.3349285524430239,
 'reg_alpha': 5.82953593836527,
 'reg_lambda': 7.080782981849838}

# %%

finalmodel_resistance_kndvi = xgb.XGBRegressor(**best_params_kndvi,early_stopping_rounds=30)

## fit the model
finalmodel_resistance_kndvi.fit(X_train_kndvi, y_train_kndvi, eval_set=[(X_test_kndvi, y_test_kndvi)])

print('train: {}   test: {}'.format(finalmodel_resistance_kndvi.score(X_train_kndvi, y_train_kndvi),
                                    finalmodel_resistance_kndvi.score(X_test_kndvi, y_test_kndvi)) )

# %%
shap_explainer_kndvi = shap.TreeExplainer(finalmodel_resistance_kndvi)
shap_explainer_resistance_kndvi = shap_explainer_kndvi(x_resistance_kndvi)

# %%
shapdf_kndvi = pd.DataFrame(shap_explainer_resistance_kndvi.values,columns=x_resistance_kndvi.columns)
shapdf_kndvi.head()

# %%
shap_df_kndvi = shapdf_kndvi.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_kndvi = shap_df_kndvi.sort_values('sv')
shap_df_kndvi['relative'] = shap_df_kndvi['sv']/shap_df_kndvi['sv'].max()
shap_df_kndvi['method'] = "Shapley Value"

shap_df_kndvi

# %%
pi = permutation_importance(finalmodel_resistance_kndvi, x_resistance_kndvi, y_resistance_kndvi, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(list(x_resistance_kndvi.columns))[sorted_idx]

permimport_kndvi = pd.DataFrame({'mean' : pi.importances_mean[sorted_idx].T,\
                          'stdev' : pi.importances_std[sorted_idx].T,\
                           'variable': sorted_names})
permimport_kndvi

# %%
permimport_kndvi['relative'] = permimport_kndvi['mean']/permimport_kndvi['mean'].max()
permimport_kndvi['method'] = "Permutation Importance"

# %%
permimport_kndvi

# %%
var_dic = {'drought_count':'Drought counts','drought_severity':'Mean drought severity','drought_duration':'Mean drought duration',
           'fsc':'Forest structural complexity','plant_richness':'Tree species richness','soil_cec':'Cation exchange capacity',
           'sla':'Specific leaf area','cti':'Compound topographic index','soil_clay':'Clay content','ai_index':'Aridity index',
           'wood_den':'Wood density','annual_temp':'Mean annual temperature','annual_prec':'Mean annual precipitation',
           'sm_change':'Soil moisture change', 'spei':'SPEI'
}

# %%
y_labs_kndvi = [var_dic[i] for i in shap_df_kndvi['variable']]
y_labs_kndvi

# %%
permimport_kndvi["variable"] = pd.Categorical(permimport_kndvi["variable"], categories= shap_df_kndvi['variable'], ordered=True)
permimport_kndvi = permimport_kndvi.sort_values('variable')

permimport_kndvi

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_kndvi['variable'], shap_df_kndvi['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_kndvi['variable'], permimport_kndvi['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_kndvi)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
plot_pdp(finalmodel_resistance_kndvi,x_resistance_kndvi,'fsc',target='kndvi',label='fsc')

# %%
plot_pdp(finalmodel_resistance_kndvi,x_resistance_kndvi,'plant_richness',target='kndvi',label='plant_richness')

# %%
shap.plots.beeswarm(shap_explainer_resistance_kndvi)

# %%
shap.plots.scatter(shap_explainer_resistance_kndvi[:, 'fsc'])

# %%
shap.plots.scatter(shap_explainer_resistance_kndvi[:, 'plant_richness'])

# %%
pardep_kndvi = partial_dependence(finalmodel_resistance_kndvi, x_resistance_kndvi,'fsc',kind='average' )

# %%
pardep_kndvi

# %%
import statsmodels.api as sm

# %%
loess_result = sm.nonparametric.lowess(shap_explainer_resistance_kndvi[:, 'fsc'].values, shap_explainer_resistance_kndvi[:, 'fsc'].data, frac=0.3)  # frac=0.3 代表使用 30% 数据进行局部回归
x_loess, y_loess = loess_result[:, 0], loess_result[:, 1]

# %%
xmin = pardep_kndvi['grid_values'][0].min()
xmax = pardep_kndvi['grid_values'][0].max()
ymin = pardep_kndvi['average'][0].min()
ymax = pardep_kndvi['average'][0].max()
y_ran = ymax -ymin

fig, ax1 = plt.subplots()
ax1.plot(pardep_kndvi['grid_values'][0], pardep_kndvi['average'][0], color='#20B2AA', linewidth=3, label = 'pdp')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim([ymin-0.25*y_ran, ymax+0.25*y_ran])

ax2 = ax1.twinx()

ax2.hist(x_resistance_kndvi['fsc'], bins=200, alpha=.25, color = 'grey', density= True)
ax2.set_ylim([0, 5])
ax2.yaxis.set_visible(False)
#ax2.tick_params(axis='y', labelcolor='black')
#ax2.set_ylabel('Frequency Distribution', color='black', fontsize=14)

ax3 = ax1.twinx()
len_loess = len(x_loess)
ax3.scatter(shap_explainer_resistance_kndvi[:, 'fsc'].data, shap_explainer_resistance_kndvi[:, 'fsc'].values, s =0.4, color = '#8A2BE2', alpha=0.2)
ax3.plot(x_loess[int(len_loess*0.02):-int(len_loess*0.02)], y_loess[int(len_loess*0.02):-int(len_loess*0.02)], color = '#8A2BE2', linewidth = 3, label = 'shap')

ymin = shap_explainer_resistance_kndvi[:, 'fsc'].values.min()
ymax = shap_explainer_resistance_kndvi[:, 'fsc'].values.max()
y_ran = ymax -ymin

ax3.set_ylim([ymin-0.2*y_ran, ymax+0.2*y_ran])
ax3.yaxis.set_visible(True)

ax1.legend(loc= [0.1,0.8])
ax3.legend(loc= [0.1,0.9])

# %% [markdown]
# ## 3 kndvi after 2000

# %% [markdown]
# ### 3.1 读取数据

# %%
drought_resistance_kndvi_after2000 = pd.read_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi_after2000_20251216.csv')
drought_resistance_kndvi_after2000

# %%
drought_resistance_kndvi_after2000 = drought_resistance_kndvi_after2000.drop(columns=['lon','lat','year'])
drought_resistance_kndvi_after2000.describe()

# %%
drought_resistance_kndvi_high_after2000 = drought_resistance_kndvi_after2000.kndvi.quantile(0.9)
drought_resistance_kndvi_after2000 = drought_resistance_kndvi_after2000[drought_resistance_kndvi_after2000.kndvi < drought_resistance_kndvi_high_after2000]

drought_resistance_kndvi_after2000.describe()

# %%
x_resistance_kndvi_after2000 = drought_resistance_kndvi_after2000.drop(['kndvi'], axis=1)
y_resistance_kndvi_after2000 = drought_resistance_kndvi_after2000['kndvi']
y_resistance_kndvi_after2000 = np.log(y_resistance_kndvi_after2000)
y_resistance_kndvi_after2000.plot.hist()

# %% [markdown]
# ### 3.2 建模

# %%
X_train_kndvi_after2000, X_test_kndvi_after2000, y_train_kndvi_after2000, y_test_kndvi_after2000 = sklearn.model_selection.train_test_split( x_resistance_kndvi_after2000, y_resistance_kndvi_after2000)
reg = xgb.XGBRegressor()
reg.fit(X_train_kndvi_after2000, y_train_kndvi_after2000)
reg.score(X_train_kndvi_after2000, y_train_kndvi_after2000)

# %%
reg.score(X_test_kndvi_after2000, y_test_kndvi_after2000)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 2, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 5, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 5, 10),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_kndvi_after2000, y_train_kndvi_after2000, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_kndvi_after2000 = study.best_params

# %%
## 20251216 的调参结果
best_params_kndvi_after2000 

# %%
best_params_kndvi_after2000 = {'n_estimators': 900,
 'learning_rate': 0.08005629765138192,
 'max_depth': 5,
 'subsample': 0.557636439177856,
 'colsample_bytree': 0.7489858831826437,
 'min_child_weight': 4.32465897881975,
 'gamma': 2.1364833432474533,
 'reg_alpha': 5.021957336159852,
 'reg_lambda': 6.947100967189815}

# %%
finalmodel_resistance_kndvi_after2000 = xgb.XGBRegressor(**best_params_kndvi_after2000,early_stopping_rounds=30)

## fit the model
finalmodel_resistance_kndvi_after2000.fit(X_train_kndvi_after2000, y_train_kndvi_after2000, eval_set=[(X_test_kndvi_after2000, y_test_kndvi_after2000)])

print('train: {}   test: {}'.format(finalmodel_resistance_kndvi_after2000.score(X_train_kndvi_after2000, y_train_kndvi_after2000),
                                    finalmodel_resistance_kndvi_after2000.score(X_test_kndvi_after2000, y_test_kndvi_after2000)) )

# %%
shap_explainer_kndvi_after2000 = shap.TreeExplainer(finalmodel_resistance_kndvi_after2000)
shap_explainer_resistance_kndvi_after2000 = shap_explainer_kndvi_after2000(x_resistance_kndvi_after2000)
shapdf_kndvi_after2000 = pd.DataFrame(shap_explainer_resistance_kndvi_after2000.values,columns=x_resistance_kndvi_after2000.columns)
shapdf_kndvi_after2000.head()

# %%
shap_df_kndvi_after2000 = shapdf_kndvi_after2000.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_kndvi_after2000 = shap_df_kndvi_after2000.sort_values('sv')
shap_df_kndvi_after2000['relative'] = shap_df_kndvi_after2000['sv']/shap_df_kndvi_after2000['sv'].max()
shap_df_kndvi_after2000['method'] = "Shapley Value"

shap_df_kndvi_after2000

# %%
pi_after2000 = permutation_importance(finalmodel_resistance_kndvi_after2000, x_resistance_kndvi_after2000, y_resistance_kndvi_after2000, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx_after2000 = pi_after2000.importances_mean.argsort()
sorted_names_after2000 = np.array(list(x_resistance_kndvi_after2000.columns))[sorted_idx_after2000]

permimport_kndvi_after2000 = pd.DataFrame({'mean' : pi_after2000.importances_mean[sorted_idx_after2000].T,\
                          'stdev' : pi_after2000.importances_std[sorted_idx_after2000].T,\
                           'variable': sorted_names_after2000})
permimport_kndvi_after2000

# %%
permimport_kndvi_after2000['relative'] = permimport_kndvi_after2000['mean']/permimport_kndvi_after2000['mean'].max()
permimport_kndvi_after2000['method'] = "Permutation Importance"
permimport_kndvi_after2000

# %%
y_labs_kndvi_after2000 = [var_dic[i] for i in shap_df_kndvi_after2000['variable']]
y_labs_kndvi_after2000

# %%
permimport_kndvi_after2000["variable"] = pd.Categorical(permimport_kndvi_after2000["variable"], categories= shap_df_kndvi_after2000['variable'], ordered=True)
permimport_kndvi_after2000 = permimport_kndvi_after2000.sort_values('variable')

permimport_kndvi_after2000

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_kndvi_after2000['variable'], shap_df_kndvi_after2000['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_kndvi_after2000['variable'], permimport_kndvi_after2000['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_kndvi_after2000)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
pardep_kndvi_after2000 = partial_dependence(finalmodel_resistance_kndvi_after2000, x_resistance_kndvi_after2000,'fsc',kind='average' )
pardep_kndvi_after2000

# %%
loess_result_after2000 = sm.nonparametric.lowess(shap_explainer_resistance_kndvi_after2000[:, 'fsc'].values, shap_explainer_resistance_kndvi_after2000[:, 'fsc'].data, frac=0.3)  # frac=0.3 代表使用 30% 数据进行局部回归
x_loess_after2000, y_loess_after2000 = loess_result_after2000[:, 0], loess_result_after2000[:, 1]

# %%
xmin = pardep_kndvi_after2000['grid_values'][0].min()
xmax = pardep_kndvi_after2000['grid_values'][0].max()
ymin = pardep_kndvi_after2000['average'][0].min()
ymax = pardep_kndvi_after2000['average'][0].max()
y_ran = ymax -ymin

# %%
fig, ax1 = plt.subplots()
ax1.plot(pardep_kndvi_after2000['grid_values'][0], pardep_kndvi_after2000['average'][0], color='#20B2AA', linewidth=3, label = 'pdp')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim([ymin-0.25*y_ran, ymax+0.25*y_ran])

ax2 = ax1.twinx()

ax2.hist(x_resistance_kndvi_after2000['fsc'], bins=200, alpha=.25, color = 'grey', density= True)
ax2.set_ylim([0, 5])
ax2.yaxis.set_visible(False)
#ax2.tick_params(axis='y', labelcolor='black')
#ax2.set_ylabel('Frequency Distribution', color='black', fontsize=14)

ax3 = ax1.twinx()
len_loess = len(x_loess_after2000)
ax3.scatter(shap_explainer_resistance_kndvi_after2000[:, 'fsc'].data, shap_explainer_resistance_kndvi_after2000[:, 'fsc'].values, s =0.4, color = '#8A2BE2', alpha=0.2)
ax3.plot(x_loess_after2000[int(len_loess*0.02):-int(len_loess*0.02)], y_loess_after2000[int(len_loess*0.02):-int(len_loess*0.02)], color = '#8A2BE2', linewidth = 3, label = 'shap')

ymin = shap_explainer_resistance_kndvi_after2000[:, 'fsc'].values.min()
ymax = shap_explainer_resistance_kndvi_after2000[:, 'fsc'].values.max()
y_ran = ymax -ymin

ax3.set_ylim([ymin-0.2*y_ran, ymax+0.2*y_ran])
ax3.yaxis.set_visible(True)

ax1.legend(loc= [0.1,0.8])
ax3.legend(loc= [0.1,0.9])

# %% [markdown]
# ## 4 sif + csc mid

# %% [markdown]
# ### 4.1 读取数据

# %%
drought_resistance_sif = pd.read_csv(r'E:/python_output/fsc_drought/df_ml_all_sif_20251213.csv')
drought_resistance_sif

# %%
drought_resistance_sif = drought_resistance_sif.drop(columns=['lon','lat','year'])
drought_resistance_sif.describe()

# %%
drought_resistance_sif_high = drought_resistance_sif.sif.quantile(0.9)
drought_resistance_sif = drought_resistance_sif[drought_resistance_sif.sif < drought_resistance_sif_high]

# %%
drought_resistance_sif.describe()

# %%
x_resistance_sif = drought_resistance_sif.drop(['sif'], axis=1)
y_resistance_sif = drought_resistance_sif['sif']
y_resistance_sif = np.log(y_resistance_sif)
y_resistance_sif.plot.hist()

# %% [markdown]
# ### 4.2 建模

# %%
X_train_sif, X_test_sif, y_train_sif, y_test_sif = sklearn.model_selection.train_test_split( x_resistance_sif, y_resistance_sif)
reg = xgb.XGBRegressor()
reg.fit(X_train_sif, y_train_sif)

# %%
reg.score(X_train_sif, y_train_sif)

# %%
reg.score(X_test_sif, y_test_sif)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 5,20),
        "gamma": trial.suggest_float("gamma", 2, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 5, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 5, 20),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_sif, y_train_sif, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_sif = study.best_params

# %%
## 之前2025.11.13确定的参数
best_params_sif = {'n_estimators': 750,
 'learning_rate': 0.0826838204086382,
 'max_depth': 6,
 'subsample': 0.5345882028868246,
 'colsample_bytree': 0.7264498299131363,
 'min_child_weight': 18.789291902610714,
 'gamma': 2.027403008253123,
 'reg_alpha': 5.730605221018445,
 'reg_lambda': 10.940782669851025}

# %%

finalmodel_resistance_sif = xgb.XGBRegressor(**best_params_sif,early_stopping_rounds=30)   ## 还是用2025.11.13确定的参数

## fit the model
finalmodel_resistance_sif.fit(X_train_sif, y_train_sif, eval_set=[(X_test_sif, y_test_sif)])

print('train: {}   test: {}'.format(finalmodel_resistance_sif.score(X_train_sif, y_train_sif),
                                    finalmodel_resistance_sif.score(X_test_sif, y_test_sif)) )

# %%
shap_explainer_sif = shap.TreeExplainer(finalmodel_resistance_sif)
shap_explainer_resistance_sif = shap_explainer_sif(x_resistance_sif)

# %%
shapdf_sif = pd.DataFrame(shap_explainer_resistance_sif.values,columns=x_resistance_sif.columns)
shapdf_sif.head()

# %%
shap_df_sif = shapdf_sif.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_sif = shap_df_sif.sort_values('sv')
shap_df_sif['relative'] = shap_df_sif['sv']/shap_df_sif['sv'].max()
shap_df_sif['method'] = "Shapley Value"

shap_df_sif

# %%
pi = permutation_importance(finalmodel_resistance_sif, x_resistance_sif, y_resistance_sif, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(list(x_resistance_sif.columns))[sorted_idx]

permimport_sif = pd.DataFrame({'mean' : pi.importances_mean[sorted_idx].T,\
                          'stdev' : pi.importances_std[sorted_idx].T,\
                           'variable': sorted_names})
permimport_sif

# %%
permimport_sif['relative'] = permimport_sif['mean']/permimport_sif['mean'].max()
permimport_sif['method'] = "Permutation Importance"

# %%
y_labs_sif = [var_dic[i] for i in shap_df_sif['variable']]
y_labs_sif

# %%
permimport_sif["variable"] = pd.Categorical(permimport_sif["variable"], categories= shap_df_sif['variable'], ordered=True)
permimport_sif = permimport_sif.sort_values('variable')

permimport_sif

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_sif['variable'], shap_df_sif['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_sif['variable'], permimport_sif['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_sif)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
plot_pdp(finalmodel_resistance_sif,x_resistance_sif,'fsc',target='resisitance_mid',label='fsc')

# %%
plot_pdp(finalmodel_resistance_sif,x_resistance_sif,'plant_richness',target='sif',label='plant_richness')

# %%
shap.plots.beeswarm(shap_explainer_resistance_sif)

# %%
shap.plots.scatter(shap_explainer_resistance_sif[:, 'fsc'])

# %%
shap.plots.scatter(shap_explainer_resistance_sif[:, 'richness'])

# %% [markdown]
# ## 3 kndvi  Ydrou/Ymean

# %% [markdown]
# ### 3.1 读取数据

# %%
drought_resistance_kndvi2 = pd.read_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi2_20251216.csv')
drought_resistance_kndvi2

# %%
drought_resistance_kndvi2 = drought_resistance_kndvi2.drop(columns=['lon','lat','year'])
drought_resistance_kndvi2.describe()

# %%
drought_resistance_kndvi2_high = drought_resistance_kndvi2.kndvi.quantile(0.9)
drought_resistance_kndvi2 = drought_resistance_kndvi2[drought_resistance_kndvi2.kndvi < drought_resistance_kndvi2_high]

drought_resistance_kndvi2.describe()

# %%
x_resistance_kndvi2 = drought_resistance_kndvi2.drop(['kndvi'], axis=1)
y_resistance_kndvi2 = drought_resistance_kndvi2['kndvi']

# %%
p = np.arange(0.1,0.99,0.01)
logit = np.log(p / (1 - p))
plt.scatter(p,logit)

# %%
np.log(y_resistance_kndvi2/(1-y_resistance_kndvi2)).plot.hist()

# %%
#y_resistance_kndvi2 = np.log(y_resistance_kndvi2)
#y_resistance_kndvi2.plot.hist()
##  先不做转换看看
y_resistance_kndvi2.plot.hist()

# %%
y_resistance_kndvi2_log = np.log(y_resistance_kndvi2/(1-y_resistance_kndvi2))

# %%
y_resistance_kndvi2_log.plot.hist()

# %% [markdown]
# ### 3.2 建模

# %%
X_train_kndvi2, X_test_kndvi2, y_train_kndvi2, y_test_kndvi2 = sklearn.model_selection.train_test_split( x_resistance_kndvi2, y_resistance_kndvi2)
reg = xgb.XGBRegressor()
reg.fit(X_train_kndvi2, y_train_kndvi2)
reg.score(X_train_kndvi2, y_train_kndvi2)

# %%
reg.score(X_test_kndvi2, y_test_kndvi2)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 0.1, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 0.5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 0.5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_kndvi2, y_train_kndvi2, scoring='r2', cv=cv).mean()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_kndvi2 = study.best_params

# %%
best_params_kndvi2

# %%
## 20260105 的调参结果
best_params_kndvi2 = {'n_estimators': 550,
 'learning_rate': 0.06581885381938615,
 'max_depth': 6,
 'subsample': 0.6666613632210401,
 'colsample_bytree': 0.6486228680741734,
 'min_child_weight': 4.491032007444079,
 'gamma': 1.0086192687999287,
 'reg_alpha': 9.973932227506042,
 'reg_lambda': 8.333055146335706}

# %%
finalmodel_resistance_kndvi2 = xgb.XGBRegressor(**best_params_kndvi2,early_stopping_rounds=30)

## fit the model
finalmodel_resistance_kndvi2.fit(X_train_kndvi2, y_train_kndvi2, eval_set=[(X_test_kndvi2, y_test_kndvi2)])

print('train: {}   test: {}'.format(finalmodel_resistance_kndvi2.score(X_train_kndvi2, y_train_kndvi2),
                                    finalmodel_resistance_kndvi2.score(X_test_kndvi2, y_test_kndvi2)) )

# %%
shap_explainer_kndvi2 = shap.TreeExplainer(finalmodel_resistance_kndvi2)
shap_explainer_resistance_kndvi2 = shap_explainer_kndvi2(x_resistance_kndvi2)

# %%
shapdf_kndvi2 = pd.DataFrame(shap_explainer_resistance_kndvi2.values,columns=x_resistance_kndvi2.columns)
shapdf_kndvi2.head()

# %%
shap_df_kndvi2 = shapdf_kndvi2.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_kndvi2 = shap_df_kndvi2.sort_values('sv')
shap_df_kndvi2['relative'] = shap_df_kndvi2['sv']/shap_df_kndvi2['sv'].max()
shap_df_kndvi2['method'] = "Shapley Value"

shap_df_kndvi2

# %%
pi = permutation_importance(finalmodel_resistance_kndvi2, x_resistance_kndvi2, y_resistance_kndvi2, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(list(x_resistance_kndvi2.columns))[sorted_idx]

permimport_kndvi2 = pd.DataFrame({'mean' : pi.importances_mean[sorted_idx].T,\
                          'stdev' : pi.importances_std[sorted_idx].T,\
                           'variable': sorted_names})
permimport_kndvi2

# %%
permimport_kndvi2['relative'] = permimport_kndvi2['mean']/permimport_kndvi2['mean'].max()
permimport_kndvi2['method'] = "Permutation Importance"
permimport_kndvi2

# %%
var_dic = {'drought_count':'Drought counts','drought_severity':'Mean drought severity','drought_duration':'Mean drought duration',
           'fsc':'Forest structural complexity','plant_richness':'Tree species richness','soil_cec':'Cation exchange capacity',
           'sla':'Specific leaf area','cti':'Compound topographic index','soil_clay':'Clay content','ai_index':'Aridity index',
           'wood_den':'Wood density','annual_temp':'Mean annual temperature','annual_prec':'Mean annual precipitation',
           'sm_change':'Soil moisture change', 'spei':'SPEI'
}

# %%
y_labs_kndvi2 = [var_dic[i] for i in shap_df_kndvi2['variable']]
y_labs_kndvi2

# %%
permimport_kndvi2["variable"] = pd.Categorical(permimport_kndvi2["variable"], categories= shap_df_kndvi2['variable'], ordered=True)
permimport_kndvi2 = permimport_kndvi2.sort_values('variable')

permimport_kndvi2

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_kndvi2['variable'], shap_df_kndvi2['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_kndvi2['variable'], permimport_kndvi2['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_kndvi2)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
plot_pdp(finalmodel_resistance_kndvi2,x_resistance_kndvi2,'fsc',target='kndvi',label='fsc')

# %%
plot_pdp(finalmodel_resistance_kndvi2,x_resistance_kndvi2,'plant_richness',target='kndvi',label='plant_richness')

# %%
shap.plots.beeswarm(shap_explainer_resistance_kndvi2)

# %%
shap.plots.scatter(shap_explainer_resistance_kndvi2[:, 'fsc'])

# %%
shap.plots.scatter(shap_explainer_resistance_kndvi2[:, 'plant_richness'])

# %%
pardep_kndvi2 = partial_dependence(finalmodel_resistance_kndvi2, x_resistance_kndvi2,'fsc',kind='average' )
pardep_kndvi2

# %%
import statsmodels.api as sm
loess_result = sm.nonparametric.lowess(shap_explainer_resistance_kndvi2[:, 'fsc'].values, shap_explainer_resistance_kndvi2[:, 'fsc'].data, frac=0.3)  # frac=0.3 代表使用 30% 数据进行局部回归

# %%
x_loess, y_loess = loess_result[:, 0], loess_result[:, 1]
xmin = pardep_kndvi2['grid_values'][0].min()
xmax = pardep_kndvi2['grid_values'][0].max()
ymin = pardep_kndvi2['average'][0].min()
ymax = pardep_kndvi2['average'][0].max()
y_ran = ymax -ymin

fig, ax1 = plt.subplots()
ax1.plot(pardep_kndvi2['grid_values'][0], pardep_kndvi2['average'][0], color='#20B2AA', linewidth=3, label = 'pdp')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim([ymin-0.25*y_ran, ymax+0.25*y_ran])

ax2 = ax1.twinx()

ax2.hist(x_resistance_kndvi2['fsc'], bins=200, alpha=.25, color = 'grey', density= True)
ax2.set_ylim([0, 5])
ax2.yaxis.set_visible(False)
#ax2.tick_params(axis='y', labelcolor='black')
#ax2.set_ylabel('Frequency Distribution', color='black', fontsize=14)

ax3 = ax1.twinx()
len_loess = len(x_loess)
ax3.scatter(shap_explainer_resistance_kndvi2[:, 'fsc'].data, shap_explainer_resistance_kndvi2[:, 'fsc'].values, s =0.4, color = '#8A2BE2', alpha=0.2)
ax3.plot(x_loess[int(len_loess*0.02):-int(len_loess*0.02)], y_loess[int(len_loess*0.02):-int(len_loess*0.02)], color = '#8A2BE2', linewidth = 3, label = 'shap')

ymin = shap_explainer_resistance_kndvi2[:, 'fsc'].values.min()
ymax = shap_explainer_resistance_kndvi2[:, 'fsc'].values.max()
y_ran = ymax -ymin

ax3.set_ylim([ymin-0.2*y_ran, ymax+0.2*y_ran])
ax3.yaxis.set_visible(True)

ax1.legend(loc= [0.1,0.8])
ax3.legend(loc= [0.1,0.9])

# %% [markdown]
# ### 3.3 建模 log

# %%
X_train_kndvi2_log, X_test_kndvi2_log, y_train_kndvi2_log, y_test_kndvi2_log = sklearn.model_selection.train_test_split( x_resistance_kndvi2, y_resistance_kndvi2_log)
reg = xgb.XGBRegressor()
reg.fit(X_train_kndvi2_log, y_train_kndvi2_log)

# %%
reg.score(X_train_kndvi2_log, y_train_kndvi2_log)

# %%
reg.score(X_test_kndvi2_log, y_test_kndvi2_log)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 2, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 2, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 2, 10),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_kndvi2_log, y_train_kndvi2_log, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_kndvi2_log = study.best_params

# %%
best_params_kndvi2_log
## 20260105 的调参结果

# %%
finalmodel_resistance_kndvi2_log = xgb.XGBRegressor(**best_params_kndvi2_log,early_stopping_rounds=30)

## fit the model
finalmodel_resistance_kndvi2_log.fit(X_train_kndvi2_log, y_train_kndvi2_log, eval_set=[(X_test_kndvi2_log, y_test_kndvi2_log)])

print('train: {}   test: {}'.format(finalmodel_resistance_kndvi2_log.score(X_train_kndvi2_log, y_train_kndvi2_log),
                                    finalmodel_resistance_kndvi2_log.score(X_test_kndvi2_log, y_test_kndvi2_log)) )

# %%
shap_explainer_kndvi2_log = shap.TreeExplainer(finalmodel_resistance_kndvi2_log)
shap_explainer_resistance_kndvi2_log = shap_explainer_kndvi2_log(x_resistance_kndvi2)
shapdf_kndvi2_log = pd.DataFrame(shap_explainer_resistance_kndvi2_log.values,columns=x_resistance_kndvi2.columns)
shapdf_kndvi2_log.head()

# %%
shap_df_kndvi2_log = shapdf_kndvi2_log.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_kndvi2_log = shap_df_kndvi2_log.sort_values('sv')
shap_df_kndvi2_log['relative'] = shap_df_kndvi2_log['sv']/shap_df_kndvi2_log['sv'].max()
shap_df_kndvi2_log['method'] = "Shapley Value"

shap_df_kndvi2_log

# %%
pi = permutation_importance(finalmodel_resistance_kndvi2_log, x_resistance_kndvi2, y_resistance_kndvi2_log, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(list(x_resistance_kndvi2.columns))[sorted_idx]

permimport_kndvi2_log = pd.DataFrame({'mean' : pi.importances_mean[sorted_idx].T,\
                          'stdev' : pi.importances_std[sorted_idx].T,\
                           'variable': sorted_names})
permimport_kndvi2_log

# %%
permimport_kndvi2_log['relative'] = permimport_kndvi2_log['mean']/permimport_kndvi2_log['mean'].max()
permimport_kndvi2_log['method'] = "Permutation Importance"
permimport_kndvi2_log

# %%
y_labs_kndvi2_log = [var_dic[i] for i in shap_df_kndvi2_log['variable']]
y_labs_kndvi2_log

# %%
permimport_kndvi2_log["variable"] = pd.Categorical(permimport_kndvi2_log["variable"], categories= shap_df_kndvi2_log['variable'], ordered=True)
permimport_kndvi2_log = permimport_kndvi2_log.sort_values('variable')

permimport_kndvi2_log

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_kndvi2_log['variable'], shap_df_kndvi2_log['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_kndvi2_log['variable'], permimport_kndvi2_log['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_kndvi2_log)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
plot_pdp(finalmodel_resistance_kndvi2_log,x_resistance_kndvi2,'fsc',target='kndvi',label='fsc')

# %%
plot_pdp(finalmodel_resistance_kndvi2_log,x_resistance_kndvi2,'plant_richness',target='kndvi',label='plant_richness')

# %%
shap.plots.beeswarm(shap_explainer_resistance_kndvi2_log)

# %%
shap.plots.scatter(shap_explainer_resistance_kndvi2_log[:, 'fsc'])

# %% [markdown]
# ## 3 kndvi after2000 Ydrou/Ymean

# %% [markdown]
# ### 3.1 读取数据

# %%
drought_resistance_kndvi2_after2000 = pd.read_csv(r'E:/python_output/fsc_drought/df_ml_all_kndvi2_after2000_20251216.csv')
drought_resistance_kndvi2_after2000

# %%
drought_resistance_kndvi2_after2000 = drought_resistance_kndvi2_after2000.drop(columns=['lon','lat','year'])
drought_resistance_kndvi2_after2000.describe()

# %%
drought_resistance_kndvi2_high_after2000 = drought_resistance_kndvi2_after2000.kndvi.quantile(0.9)
drought_resistance_kndvi2_after2000 = drought_resistance_kndvi2_after2000[drought_resistance_kndvi2_after2000.kndvi < drought_resistance_kndvi2_high_after2000]

drought_resistance_kndvi2_after2000.describe()

# %%
x_resistance_kndvi2_after2000 = drought_resistance_kndvi2_after2000.drop(['kndvi'], axis=1)
y_resistance_kndvi2_after2000 = drought_resistance_kndvi2_after2000['kndvi']

# %%
#y_resistance_kndvi2_after2000 = np.log(y_resistance_kndvi2_after2000)
y_resistance_kndvi2_after2000.plot.hist()

# %%
y_resistance_kndvi2_after2000_log = np.log(y_resistance_kndvi2_after2000/(1-y_resistance_kndvi2_after2000))
y_resistance_kndvi2_after2000_log.plot.hist()

# %% [markdown]
# ### 3.2 建模

# %%
X_train_kndvi2_after2000, X_test_kndvi2_after2000, y_train_kndvi2_after2000, y_test_kndvi2_after2000 = sklearn.model_selection.train_test_split( x_resistance_kndvi2_after2000, y_resistance_kndvi2_after2000)
reg = xgb.XGBRegressor()
reg.fit(X_train_kndvi2_after2000, y_train_kndvi2_after2000)
reg.score(X_train_kndvi2_after2000, y_train_kndvi2_after2000)

# %%
reg.score(X_test_kndvi2_after2000, y_test_kndvi2_after2000)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_kndvi2_after2000, y_train_kndvi2_after2000, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_kndvi2_after2000 = study.best_params


# %%
## 20251216 的调参结果
best_params_kndvi2_after2000 

# %%
finalmodel_resistance_kndvi2_after2000 = xgb.XGBRegressor(**best_params_kndvi2_after2000,early_stopping_rounds=30)

## fit the model
finalmodel_resistance_kndvi2_after2000.fit(X_train_kndvi2_after2000, y_train_kndvi2_after2000, eval_set=[(X_test_kndvi2_after2000, y_test_kndvi2_after2000)])

print('train: {}   test: {}'.format(finalmodel_resistance_kndvi2_after2000.score(X_train_kndvi2_after2000, y_train_kndvi2_after2000),
                                    finalmodel_resistance_kndvi2_after2000.score(X_test_kndvi2_after2000, y_test_kndvi2_after2000)) )

# %%
shap_explainer_kndvi2_after2000 = shap.TreeExplainer(finalmodel_resistance_kndvi2_after2000)
shap_explainer_resistance_kndvi2_after2000 = shap_explainer_kndvi2_after2000(x_resistance_kndvi2_after2000)
shapdf_kndvi2_after2000 = pd.DataFrame(shap_explainer_resistance_kndvi2_after2000.values,columns=x_resistance_kndvi2_after2000.columns)
shapdf_kndvi2_after2000.head()

# %%
shap_df_kndvi2_after2000 = shapdf_kndvi2_after2000.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_kndvi2_after2000 = shap_df_kndvi2_after2000.sort_values('sv')
shap_df_kndvi2_after2000['relative'] = shap_df_kndvi2_after2000['sv']/shap_df_kndvi2_after2000['sv'].max()
shap_df_kndvi2_after2000['method'] = "Shapley Value"

shap_df_kndvi2_after2000

# %%
pi_after2000 = permutation_importance(finalmodel_resistance_kndvi2_after2000, x_resistance_kndvi2_after2000, y_resistance_kndvi2_after2000, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx_after2000 = pi_after2000.importances_mean.argsort()
sorted_names_after2000 = np.array(list(x_resistance_kndvi2_after2000.columns))[sorted_idx_after2000]

permimport_kndvi2_after2000 = pd.DataFrame({'mean' : pi_after2000.importances_mean[sorted_idx_after2000].T,\
                          'stdev' : pi_after2000.importances_std[sorted_idx_after2000].T,\
                           'variable': sorted_names_after2000})
permimport_kndvi2_after2000

# %%
permimport_kndvi2_after2000['relative'] = permimport_kndvi2_after2000['mean']/permimport_kndvi2_after2000['mean'].max()
permimport_kndvi2_after2000['method'] = "Permutation Importance"
permimport_kndvi2_after2000

# %%
y_labs_kndvi2_after2000 = [var_dic[i] for i in shap_df_kndvi2_after2000['variable']]
y_labs_kndvi2_after2000

# %%
permimport_kndvi2_after2000["variable"] = pd.Categorical(permimport_kndvi2_after2000["variable"], categories= shap_df_kndvi2_after2000['variable'], ordered=True)
permimport_kndvi2_after2000 = permimport_kndvi2_after2000.sort_values('variable')

permimport_kndvi2_after2000

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_kndvi2_after2000['variable'], shap_df_kndvi2_after2000['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_kndvi2_after2000['variable'], permimport_kndvi2_after2000['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_kndvi2_after2000)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
pardep_kndvi2_after2000 = partial_dependence(finalmodel_resistance_kndvi2_after2000, x_resistance_kndvi2_after2000,'fsc',kind='average' )
pardep_kndvi2_after2000

# %%
loess_result_after2000 = sm.nonparametric.lowess(shap_explainer_resistance_kndvi2_after2000[:, 'fsc'].values, shap_explainer_resistance_kndvi2_after2000[:, 'fsc'].data, frac=0.3)  # frac=0.3 代表使用 30% 数据进行局部回归

# %%
x_loess_after2000, y_loess_after2000 = loess_result_after2000[:, 0], loess_result_after2000[:, 1]
xmin = pardep_kndvi2_after2000['grid_values'][0].min()
xmax = pardep_kndvi2_after2000['grid_values'][0].max()
ymin = pardep_kndvi2_after2000['average'][0].min()
ymax = pardep_kndvi2_after2000['average'][0].max()
y_ran = ymax -ymin
fig, ax1 = plt.subplots()
ax1.plot(pardep_kndvi2_after2000['grid_values'][0], pardep_kndvi2_after2000['average'][0], color='#20B2AA', linewidth=3, label = 'pdp')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim([ymin-0.25*y_ran, ymax+0.25*y_ran])

ax2 = ax1.twinx()

ax2.hist(x_resistance_kndvi2_after2000['fsc'], bins=200, alpha=.25, color = 'grey', density= True)
ax2.set_ylim([0, 5])
ax2.yaxis.set_visible(False)
#ax2.tick_params(axis='y', labelcolor='black')
#ax2.set_ylabel('Frequency Distribution', color='black', fontsize=14)

ax3 = ax1.twinx()
len_loess = len(x_loess_after2000)
ax3.scatter(shap_explainer_resistance_kndvi2_after2000[:, 'fsc'].data, shap_explainer_resistance_kndvi2_after2000[:, 'fsc'].values, s =0.4, color = '#8A2BE2', alpha=0.2)
ax3.plot(x_loess_after2000[int(len_loess*0.02):-int(len_loess*0.02)], y_loess_after2000[int(len_loess*0.02):-int(len_loess*0.02)], color = '#8A2BE2', linewidth = 3, label = 'shap')

ymin = shap_explainer_resistance_kndvi2_after2000[:, 'fsc'].values.min()
ymax = shap_explainer_resistance_kndvi2_after2000[:, 'fsc'].values.max()
y_ran = ymax -ymin

ax3.set_ylim([ymin-0.2*y_ran, ymax+0.2*y_ran])
ax3.yaxis.set_visible(True)

ax1.legend(loc= [0.1,0.8])
ax3.legend(loc= [0.1,0.9])

# %% [markdown]
# ### 3.3 建模 log

# %%
X_train_kndvi2_after2000_log, X_test_kndvi2_after2000_log, y_train_kndvi2_after2000_log, y_test_kndvi2_after2000_log = sklearn.model_selection.train_test_split( x_resistance_kndvi2_after2000, y_resistance_kndvi2_after2000_log)
reg = xgb.XGBRegressor()
reg.fit(X_train_kndvi2_after2000_log, y_train_kndvi2_after2000_log)

# %%
reg.score(X_train_kndvi2_after2000_log, y_train_kndvi2_after2000_log)

# %%
reg.score(X_test_kndvi2_after2000_log, y_test_kndvi2_after2000_log)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 2, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 2, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 2, 5),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_kndvi2_after2000_log, y_train_kndvi2_after2000_log, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_kndvi2_after2000_log = study.best_params

# %%
## 20251216 的调参结果
best_params_kndvi2_after2000_log 

# %%
finalmodel_resistance_kndvi2_after2000_log = xgb.XGBRegressor(**best_params_kndvi2_after2000_log,early_stopping_rounds=30)

## fit the model
finalmodel_resistance_kndvi2_after2000_log.fit(X_train_kndvi2_after2000_log, y_train_kndvi2_after2000_log, eval_set=[(X_test_kndvi2_after2000_log, y_test_kndvi2_after2000_log)])

print('train: {}   test: {}'.format(finalmodel_resistance_kndvi2_after2000_log.score(X_train_kndvi2_after2000_log, y_train_kndvi2_after2000_log),
                                    finalmodel_resistance_kndvi2_after2000_log.score(X_test_kndvi2_after2000_log, y_test_kndvi2_after2000_log)) )

# %%
shap_explainer_kndvi2_after2000_log = shap.TreeExplainer(finalmodel_resistance_kndvi2_after2000_log)
shap_explainer_resistance_kndvi2_after2000_log = shap_explainer_kndvi2_after2000_log(x_resistance_kndvi2_after2000)
shapdf_kndvi2_after2000_log = pd.DataFrame(shap_explainer_resistance_kndvi2_after2000_log.values,columns=x_resistance_kndvi2_after2000.columns)
shapdf_kndvi2_after2000_log.head()

# %%
shap_df_kndvi2_after2000_log = shapdf_kndvi2_after2000_log.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_kndvi2_after2000_log = shap_df_kndvi2_after2000_log.sort_values('sv')
shap_df_kndvi2_after2000_log['relative'] = shap_df_kndvi2_after2000_log['sv']/shap_df_kndvi2_after2000_log['sv'].max()
shap_df_kndvi2_after2000_log['method'] = "Shapley Value"

shap_df_kndvi2_after2000_log

# %%
pi_after2000_log = permutation_importance(finalmodel_resistance_kndvi2_after2000_log, x_resistance_kndvi2_after2000, y_resistance_kndvi2_after2000_log, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx_after2000_log = pi_after2000_log.importances_mean.argsort()
sorted_names_after2000_log = np.array(list(x_resistance_kndvi2_after2000.columns))[sorted_idx_after2000_log]

permimport_kndvi2_after2000_log = pd.DataFrame({'mean' : pi_after2000_log.importances_mean[sorted_idx_after2000_log].T,\
                          'stdev' : pi_after2000_log.importances_std[sorted_idx_after2000_log].T,\
                           'variable': sorted_names_after2000_log})
permimport_kndvi2_after2000_log

# %%
permimport_kndvi2_after2000_log['relative'] = permimport_kndvi2_after2000_log['mean']/permimport_kndvi2_after2000_log['mean'].max()
permimport_kndvi2_after2000_log['method'] = "Permutation Importance"
permimport_kndvi2_after2000_log

# %%
y_labs_kndvi2_after2000_log = [var_dic[i] for i in shap_df_kndvi2_after2000_log['variable']]
y_labs_kndvi2_after2000_log

# %%
permimport_kndvi2_after2000_log["variable"] = pd.Categorical(permimport_kndvi2_after2000_log["variable"], categories= shap_df_kndvi2_after2000_log['variable'], ordered=True)
permimport_kndvi2_after2000_log = permimport_kndvi2_after2000_log.sort_values('variable')

permimport_kndvi2_after2000_log

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_kndvi2_after2000_log['variable'], shap_df_kndvi2_after2000_log['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_kndvi2_after2000_log['variable'], permimport_kndvi2_after2000_log['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_kndvi2_after2000_log)
ax.set_xlabel('Relative Importance')
ax.legend()

# %% [markdown]
# ## 3 sif  Ydrou/Ymean

# %% [markdown]
# ### 3.1 读取数据

# %%
drought_resistance_sif2 = pd.read_csv(r'E:/python_output/fsc_drought/df_ml_all_sif2_20251213.csv')
drought_resistance_sif2

# %%
drought_resistance_sif2 = drought_resistance_sif2.drop(columns=['lon','lat','year'])
drought_resistance_sif2.describe()

# %%
drought_resistance_sif2_high = drought_resistance_sif2.sif.quantile(0.9)
drought_resistance_sif2 = drought_resistance_sif2[drought_resistance_sif2.sif < drought_resistance_sif2_high]
drought_resistance_sif2.describe()

# %%
x_resistance_sif2 = drought_resistance_sif2.drop(['sif'], axis=1)
y_resistance_sif2 = drought_resistance_sif2['sif']
#y_resistance_sif2 = np.log(y_resistance_sif2)
y_resistance_sif2.plot.hist()

# %%
y_resistance_sif2_log = np.log(y_resistance_sif2/(1-y_resistance_sif2))
y_resistance_sif2_log.plot.hist()

# %% [markdown]
# ### 3.2 建模

# %%
X_train_sif2, X_test_sif2, y_train_sif2, y_test_sif2 = sklearn.model_selection.train_test_split( x_resistance_sif2, y_resistance_sif2)
reg = xgb.XGBRegressor()
reg.fit(X_train_sif2, y_train_sif2)

# %%
reg.score(X_train_sif2, y_train_sif2)

# %%
reg.score(X_test_sif2, y_test_sif2)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_sif2, y_train_sif2, scoring='r2', cv=cv).mean()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_sif2 = study.best_params

# %%
## 之前2026.01.06确定的参数
best_params_sif2

# %%

finalmodel_resistance_sif2 = xgb.XGBRegressor(**best_params_sif2,early_stopping_rounds=30)   ## 还是用2025.11.13确定的参数

## fit the model
finalmodel_resistance_sif2.fit(X_train_sif2, y_train_sif2, eval_set=[(X_test_sif2, y_test_sif2)])

print('train: {}   test: {}'.format(finalmodel_resistance_sif2.score(X_train_sif2, y_train_sif2),
                                    finalmodel_resistance_sif2.score(X_test_sif2, y_test_sif2)) )

# %%
shap_explainer_sif2 = shap.TreeExplainer(finalmodel_resistance_sif2)
shap_explainer_resistance_sif2 = shap_explainer_sif2(x_resistance_sif2)
shapdf_sif2 = pd.DataFrame(shap_explainer_resistance_sif2.values,columns=x_resistance_sif2.columns)
shapdf_sif2.head()

# %%
shap_df_sif2 = shapdf_sif2.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_sif2 = shap_df_sif2.sort_values('sv')
shap_df_sif2['relative'] = shap_df_sif2['sv']/shap_df_sif2['sv'].max()
shap_df_sif2['method'] = "Shapley Value"

shap_df_sif2

# %%
pi = permutation_importance(finalmodel_resistance_sif2, x_resistance_sif2, y_resistance_sif2, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(list(x_resistance_sif2.columns))[sorted_idx]

permimport_sif2 = pd.DataFrame({'mean' : pi.importances_mean[sorted_idx].T,\
                          'stdev' : pi.importances_std[sorted_idx].T,\
                           'variable': sorted_names})
permimport_sif2

# %%
permimport_sif2['relative'] = permimport_sif2['mean']/permimport_sif2['mean'].max()
permimport_sif2['method'] = "Permutation Importance"
y_labs_sif2 = [var_dic[i] for i in shap_df_sif2['variable']]
y_labs_sif2

# %%
permimport_sif2["variable"] = pd.Categorical(permimport_sif2["variable"], categories= shap_df_sif2['variable'], ordered=True)
permimport_sif2 = permimport_sif2.sort_values('variable')

permimport_sif2

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_sif2['variable'], shap_df_sif2['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_sif2['variable'], permimport_sif2['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_sif2)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
plot_pdp(finalmodel_resistance_sif2,x_resistance_sif2,'fsc',target='sif',label='fsc')

# %%
plot_pdp(finalmodel_resistance_sif2,x_resistance_sif2,'plant_richness',target='sif',label='plant_richness')

# %%
shap.plots.beeswarm(shap_explainer_resistance_sif2)

# %%
shap.plots.scatter(shap_explainer_resistance_sif2[:, 'fsc'])

# %% [markdown]
# ### 3.3 建模 log

# %%
X_train_sif2_log, X_test_sif2_log, y_train_sif2_log, y_test_sif2_log = sklearn.model_selection.train_test_split( x_resistance_sif2, y_resistance_sif2_log)
reg = xgb.XGBRegressor()
reg.fit(X_train_sif2_log, y_train_sif2_log)

# %%
reg.score(X_train_sif2_log, y_train_sif2_log)

# %%
reg.score(X_test_sif2_log, y_test_sif2_log)

# %%
## 用r2 调参
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1,10),
        "gamma": trial.suggest_float("gamma", 2, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 3, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 3, 10),
        "random_state": 1412,
        "objective": "reg:squarederror",
    }

    model = xgb.XGBRegressor(**params)

    # 用交叉验证评价
    cv = KFold(n_splits=5, shuffle=True, random_state=1412)
    score = cross_val_score(model, X_train_sif2_log, y_train_sif2_log, scoring='r2', cv=cv).min()

    return score  # 最大化 R²

# 开始调参
study = optuna.create_study(direction="maximize", study_name="xgb_opt")
study.optimize(objective, n_trials=50, timeout=1800)  # 50轮或30分钟

# 输出最优结果
print("Best trial:")
print(study.best_trial)

# 最优参数
best_params_sif2_log = study.best_params

# %%
## 之前2026.01.06确定的参数
best_params_sif2_log

# %%
finalmodel_resistance_sif2_log = xgb.XGBRegressor(**best_params_sif2_log,early_stopping_rounds=30)   ## 还是用2025.11.13确定的参数

## fit the model
finalmodel_resistance_sif2_log.fit(X_train_sif2_log, y_train_sif2_log, eval_set=[(X_test_sif2_log, y_test_sif2_log)])

print('train: {}   test: {}'.format(finalmodel_resistance_sif2_log.score(X_train_sif2_log, y_train_sif2_log),
                                    finalmodel_resistance_sif2_log.score(X_test_sif2_log, y_test_sif2_log)) )

# %%
shap_explainer_sif2_log = shap.TreeExplainer(finalmodel_resistance_sif2_log)
shap_explainer_resistance_sif2_log = shap_explainer_sif2_log(x_resistance_sif2)
shapdf_sif2_log = pd.DataFrame(shap_explainer_resistance_sif2_log.values,columns=x_resistance_sif2.columns)
shapdf_sif2_log.head()

# %%
shap_df_sif2_log = shapdf_sif2_log.abs().sum().to_frame(name='sv').reset_index().rename(columns = {'index':'variable'})
shap_df_sif2_log = shap_df_sif2_log.sort_values('sv')
shap_df_sif2_log['relative'] = shap_df_sif2_log['sv']/shap_df_sif2_log['sv'].max()
shap_df_sif2_log['method'] = "Shapley Value"

shap_df_sif2_log

# %%
pi = permutation_importance(finalmodel_resistance_sif2_log, x_resistance_sif2, y_resistance_sif2_log, n_repeats=100,
                            random_state=1412, n_jobs=-1)

sorted_idx = pi.importances_mean.argsort()
sorted_names = np.array(list(x_resistance_sif2.columns))[sorted_idx]

permimport_sif2_log = pd.DataFrame({'mean' : pi.importances_mean[sorted_idx].T,\
                          'stdev' : pi.importances_std[sorted_idx].T,\
                           'variable': sorted_names})
permimport_sif2_log

# %%
permimport_sif2_log['relative'] = permimport_sif2_log['mean']/permimport_sif2_log['mean'].max()
permimport_sif2_log['method'] = "Permutation Importance"
y_labs_sif2_log = [var_dic[i] for i in shap_df_sif2_log['variable']]
y_labs_sif2_log

# %%
permimport_sif2_log["variable"] = pd.Categorical(permimport_sif2_log["variable"], categories= shap_df_sif2_log['variable'], ordered=True)
permimport_sif2_log = permimport_sif2_log.sort_values('variable')

permimport_sif2_log

# %%
fig, ax = plt.subplots()
ax.barh(shap_df_sif2_log['variable'], shap_df_sif2_log['relative'], color = '#8A2BE2', alpha = 0.6, label = 'Shapley Value')
ax.barh(permimport_sif2_log['variable'], permimport_sif2_log['relative'], color = '#20B2AA', alpha = 0.6, label = 'Permutation Importance')
ax.set_yticklabels(y_labs_sif2_log)
ax.set_xlabel('Relative Importance')
ax.legend()

# %%
plot_pdp(finalmodel_resistance_sif2_log,x_resistance_sif2,'fsc',target='sif',label='fsc')

# %%
shap.plots.beeswarm(shap_explainer_resistance_sif2_log)

# %%
shap.plots.scatter(shap_explainer_resistance_sif2_log[:, 'fsc'])

# %% [markdown]
# ## 6 画图

# %%
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'bold'
from matplotlib.ticker import MultipleLocator

# %%
## 函数
def importance_bar(ax, shap_data, pi_data, shap_col, pi_col, y_labs):
    ax.barh(shap_data['variable'], shap_data['relative'], color = shap_col, alpha = 0.6, label = 'Shapley Value')
    ax.barh(pi_data['variable'], pi_data['relative'], color = pi_col, alpha = 0.6, label = 'Permutation Importance')
    ax.set_yticklabels(y_labs)
    ax.set_xlabel('Relative Importance')
    ax.legend(frameon=False, loc = [0.13,0.05], fontsize = 9)


def pdp_shap(ax, shap_data,model,X,shap_col, pi_col, yrang_f = [0.25,0.25], MLocator = 0.01):
    pardep = partial_dependence(model, X,'fsc',kind='average' )
    xmin = pardep['grid_values'][0].min()
    xmax = pardep['grid_values'][0].max()
    ymin = shap_data[:, 'fsc'].values.min()
    ymax = shap_data[:, 'fsc'].values.max()
    y_ran = ymax -ymin

    
    loess_result = sm.nonparametric.lowess(shap_data[:, 'fsc'].values, shap_data[:, 'fsc'].data, frac=0.3)  # frac=0.3 代表使用 30% 数据进行局部回归
    x_loess, y_loess = loess_result[:, 0], loess_result[:, 1]

    len_loess = len(x_loess)
    ax.scatter(shap_data[:, 'fsc'].data, shap_data[:, 'fsc'].values, s =0.4, color = "#A253EC", alpha=0.1)
    ax.plot(x_loess[int(len_loess*0.02):-int(len_loess*0.02)], y_loess[int(len_loess*0.02):-int(len_loess*0.02)], color = shap_col, linewidth = 3, label = 'Shapley Value')

    ax.set_ylim([ymin-yrang_f[0]*y_ran, ymax+yrang_f[0]*y_ran])
    ax.yaxis.set_visible(True)
    ax.set_xlabel('Forest structural complexity')


    ymin = pardep['average'][0].min()
    ymax = pardep['average'][0].max()
    y_ran = ymax -ymin
    ax3 = ax.twinx()
    ax3.plot(pardep['grid_values'][0], pardep['average'][0], color=pi_col, linewidth=3, label = 'Partial Dependence')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.set_ylim([ymin-yrang_f[1]*y_ran, ymax+yrang_f[1]*y_ran])
    ax3.yaxis.set_major_locator(MultipleLocator(MLocator))
    ax3.set_xlabel('Forest structural complexity')
    ax3.set_ylabel('Impact on Resistance')

    ax2 = ax.twinx()

    ax2.hist(shap_data[:, 'fsc'].data, bins=200, alpha=.25, color = 'grey', density= True)
    ax2.set_ylim([0, 5])
    ax2.yaxis.set_visible(False)
    #ax2.tick_params(axis='y', labelcolor='black')
    #ax2.set_ylabel('Frequency Distribution', color='black', fontsize=14)


    ax3.legend(loc= [0.06,0.84],frameon=False)
    ax.legend(loc= [0.06,0.9],frameon=False)

# %%
from scipy.stats import gaussian_kde
import warnings


# TODO: remove unused title argument / use title argument
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def violin_ax(
    ax,
    shap_values,
    features=None,
    feature_names=None,
    max_display=None,
    color='coolwarm',
    axis_color="#333333",
    axvl_color = '#999999',
    which_xlim = 'abs_minmax',
    x_lab = '',
    alpha=1,
    show=True,
    sort=True,
    color_bar=True,
    layered_violin_max_num_bins=20,
    class_names=None,
    class_inds=None,
    color_bar_label="Feature Value",
    use_log_scale=False,
):
    """Create a SHAP violin plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation, or numpy.array
        For single output explanations, this is a matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a ``feature_names`` list as
        shorthand.

    feature_names : list
        Names of the features (length: # features).

    max_display : int
        How many top features to include in the plot (default is 20).

    plot_type : "violin", or "layered_violin".
        What type of summary plot to produce. A "layered_violin" plot shows the
        distribution of the SHAP values of each variable. A "violin" plot is the same,
        except with outliers drawn as scatter points.

    color_bar : bool
        Whether to draw the color bar (legend).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If ``None`` is passed, then the size of the current figure will be left
        unchanged.

    Examples
    --------
    See `violin plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/violin.html>`_.

    """
    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names

    num_features = shap_values.shape[1]

    if use_log_scale:
        plt.xscale("symlog")

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    ax.axvline(x=0, color=axvl_color, linewidth = 1, zorder=-1)

    num_x_points = 200
    bins = (
        np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype("int")
        )  # the indices of the feature data corresponding to each bin
    shap_min, shap_max = np.min(shap_values), np.max(shap_values)
    shap_05,shap_95 = np.quantile(shap_values,0.05), np.quantile(shap_values,0.95)
    x_points = np.linspace(shap_min, shap_max, num_x_points)

    # loop through each feature and plot:
    for pos, ind in enumerate(feature_order):
        # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
        # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
        feature = features[:, ind]
        unique, counts = np.unique(feature, return_counts=True)
        if unique.shape[0] <= layered_violin_max_num_bins:
            order = np.argsort(unique)
            thesebins = np.cumsum(counts[order])
            thesebins = np.insert(thesebins, 0, 0)
        else:
            thesebins = bins
        nbins = thesebins.shape[0] - 1
        # order the feature data so we can apply percentiling
        order = np.argsort(feature)
        # x axis is located at y0 = pos, with pos being there for offset
        # y0 = np.ones(num_x_points) * pos
        # calculate kdes:
        ys = np.zeros((nbins, num_x_points))
        for i in range(nbins):
            # get shap values in this bin:
            shaps = shap_values[order[thesebins[i] : thesebins[i + 1]], ind]
                # if there's only one element, then we can't
            if shaps.shape[0] == 1:
                warnings.warn(
                    "not enough data in bin #%d for feature %s, so it'll be ignored. Try increasing the number of records to plot."
                    % (i, feature_names[ind])
                )
                # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                # nothing to do if i == 0
                if i > 0:
                    ys[i, :] = ys[i - 1, :]
                continue
            # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
            ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
            # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
            # do nothing, but when we've gone with the unique option, this will matter - e.g. if 99% are male and 1%
            # female, we want the 1% to appear a lot smaller.
            size = thesebins[i + 1] - thesebins[i]
            bin_size_if_even = features.shape[0] / nbins
            relative_bin_size = size / bin_size_if_even
            ys[i, :] *= relative_bin_size
        # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
        # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
        # whitespace
        ys = np.cumsum(ys, axis=0)
        width = 0.8
        scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
        for i in range(nbins - 1, -1, -1):
            y = ys[i, :] / scale
            c = (
                plt.get_cmap(color)(i / (nbins - 1)) if color in plt.colormaps else color
                )  # if color is a cmap, use it, otherwise use a color
            ax.fill_between(x_points, pos - y, pos + y, facecolor=c, edgecolor="face")
    if which_xlim == 'abs_minmax':
        ax.set_xlim(shap_min, shap_max)
    elif which_xlim == 'qua':
        ax.set_xlim(shap_05, shap_95)
    else:
        ax.set_xlim(which_xlim[0], which_xlim[1])

    import matplotlib.cm as cm

    m = cm.ScalarMappable(cmap=plt.get_cmap(color))
    m.set_array([0, 1])
    cb = plt.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
    cb.set_ticklabels(['Low','High'])
    cb.set_label(color_bar_label,  labelpad=0)
    cb.ax.tick_params( length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False) 

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    #ax.spines["right"].set_visible(False)
    #ax.spines["top"].set_visible(False)
    #ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_order)), [])
    #ax.set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x")
    ax.set_ylim(-1, len(feature_order))
    ax.set_xlabel('SHAP value (impact on {})'.format(x_lab))

# %% [markdown]
# ### 6.1 kndvi 和  sif

# %%
shap_explainer_kndvi

# %%
shap_explainer_kndvi_after2000

# %%
#fig, axes = plt.subplots(2,3,figsize = (12,8)) 
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1])

axes = []
for i in range(6):
    axes.append(fig.add_subplot(gs[i//3,i%3]))

importance_bar(axes[0],shap_df_kndvi,permimport_kndvi,'#8A2BE2', '#20B2AA', y_labs_kndvi)
axes[0].set_xlabel('')
axes[0].set_title('(a)',loc='left', fontweight='bold')

violin_ax( axes[1],shap_explainer_resistance_kndvi.values, features=shap_explainer_resistance_kndvi.data, 
            feature_names=x_resistance_kndvi.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[1].set_xlabel('')
axes[1].set_xlim([-0.8,0.6])
axes[1].set_title('(b)',loc='left', fontweight='bold')

pdp_shap(axes[2],shap_explainer_resistance_kndvi,finalmodel_resistance_kndvi, x_resistance_kndvi,'#8A2BE2', '#20B2AA')
axes[2].set_xlabel('')
axes[2].set_xlim([8.5,11.7])
axes[2].set_title('(c)',loc='left', fontweight='bold')

importance_bar(axes[3],shap_df_sif,permimport_sif,'#8A2BE2', '#20B2AA', y_labs_sif)
axes[3].set_title('(d)',loc='left', fontweight='bold')

violin_ax( axes[4],shap_explainer_resistance_sif.values, features=shap_explainer_resistance_sif.data, 
            feature_names=x_resistance_sif.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[4].set_xlim([-0.8,0.5])
axes[4].set_title('(e)',loc='left', fontweight='bold')

pdp_shap(axes[5],shap_explainer_resistance_sif,finalmodel_resistance_sif, x_resistance_sif,'#8A2BE2', '#20B2AA')
axes[5].set_xlim([8.5,11.7])
axes[5].set_title('(f)',loc='left', fontweight='bold')

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.2, right=0.95)

plt.savefig(r'result_figure/figure_use_20251212/ml_results_kndvi&sif.png', dpi = 600)

# %%
#fig, axes = plt.subplots(2,3,figsize = (12,8)) 
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1])

axes = []
for i in range(6):
    axes.append(fig.add_subplot(gs[i//3,i%3]))

importance_bar(axes[0],shap_df_kndvi_after2000,permimport_kndvi_after2000,'#8A2BE2', '#20B2AA', y_labs_kndvi_after2000)
axes[0].set_xlabel('')
axes[0].set_title('(a)',loc='left', fontweight='bold')

violin_ax( axes[1],shap_explainer_resistance_kndvi_after2000.values, features=shap_explainer_resistance_kndvi_after2000.data, 
            feature_names=x_resistance_kndvi_after2000.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[1].set_xlabel('')
axes[1].set_xlim([-0.8,0.6])
axes[1].set_title('(b)',loc='left', fontweight='bold')

pdp_shap(axes[2],shap_explainer_resistance_kndvi_after2000,finalmodel_resistance_kndvi_after2000, x_resistance_kndvi_after2000,'#8A2BE2', '#20B2AA')
axes[2].set_xlabel('')
axes[2].set_xlim([8.5,11.7])
axes[2].set_title('(c)',loc='left', fontweight='bold')

importance_bar(axes[3],shap_df_sif,permimport_sif,'#8A2BE2', '#20B2AA', y_labs_sif)
axes[3].set_title('(d)',loc='left', fontweight='bold')

violin_ax( axes[4],shap_explainer_resistance_sif.values, features=shap_explainer_resistance_sif.data, 
            feature_names=x_resistance_sif.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[4].set_xlim([-0.8,0.5])
axes[4].set_title('(e)',loc='left', fontweight='bold')

pdp_shap(axes[5],shap_explainer_resistance_sif,finalmodel_resistance_sif, x_resistance_sif,'#8A2BE2', '#20B2AA')
axes[5].set_xlim([8.5,11.7])
axes[5].set_title('(f)',loc='left', fontweight='bold')

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.2, right=0.95)

plt.savefig(r'result_figure/figure_use_20251212/ml_results_kndvi_after2000&sif.png', dpi = 600)

# %% [markdown]
# ### 6.2 另一个抵抗力指标  Ydrou/Ymean

# %%
shap_explainer_kndvi2

# %%
shap_explainer_kndvi2_after2000

# %%
from matplotlib.ticker import MultipleLocator

# %%
#fig, axes = plt.subplots(2,3,figsize = (12,8)) 
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1])

axes = []
for i in range(6):
    axes.append(fig.add_subplot(gs[i//3,i%3]))

importance_bar(axes[0],shap_df_kndvi2,permimport_kndvi2,'#8A2BE2', '#20B2AA', y_labs_kndvi2)
axes[0].set_xlabel('')
axes[0].set_title('(a)',loc='left', fontweight='bold')

violin_ax( axes[1],shap_explainer_resistance_kndvi2.values, features=shap_explainer_resistance_kndvi2.data, 
            feature_names=x_resistance_kndvi2.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[1].set_xlabel('')
axes[1].set_xlim([-0.06,0.03])
axes[1].set_title('(b)',loc='left', fontweight='bold')

pdp_shap(axes[2],shap_explainer_resistance_kndvi2,finalmodel_resistance_kndvi2, x_resistance_kndvi2,'#8A2BE2', '#20B2AA',[0.04,0.25])
axes[2].set_xlabel('')
axes[2].set_xlim([8.5,11.7])
axes[2].set_title('(c)',loc='left', fontweight='bold')

importance_bar(axes[3],shap_df_sif2,permimport_sif2,'#8A2BE2', '#20B2AA', y_labs_sif2)
axes[3].set_title('(d)',loc='left', fontweight='bold')

violin_ax( axes[4],shap_explainer_resistance_sif2.values, features=shap_explainer_resistance_sif2.data, 
            feature_names=x_resistance_sif2.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[4].set_xlim([-0.065,0.04])
axes[4].set_title('(e)',loc='left', fontweight='bold')

pdp_shap(axes[5],shap_explainer_resistance_sif2,finalmodel_resistance_sif2, x_resistance_sif2,'#8A2BE2', '#20B2AA',[0.04, 0.25])
axes[5].set_xlim([8.5,11.7])
axes[5].set_title('(f)',loc='left', fontweight='bold')

#fig.align_ylabels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.19, right=0.94, wspace=0.2)

plt.savefig(r'result_figure/figure_use_20260105/ml_results_kndvi&sif2.png', dpi = 600)

# %%
#fig, axes = plt.subplots(2,3,figsize = (12,8)) 
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1])

axes = []
for i in range(6):
    axes.append(fig.add_subplot(gs[i//3,i%3]))

importance_bar(axes[0],shap_df_kndvi2_after2000,permimport_kndvi2_after2000,'#8A2BE2', '#20B2AA', y_labs_kndvi2_after2000)
axes[0].set_xlabel('')
axes[0].set_title('(a)',loc='left', fontweight='bold')

violin_ax( axes[1],shap_explainer_resistance_kndvi2_after2000.values, features=shap_explainer_resistance_kndvi2_after2000.data, 
            feature_names=x_resistance_kndvi2_after2000.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[1].set_xlabel('')
axes[1].set_xlim([-0.06,0.03])
axes[1].set_title('(b)',loc='left', fontweight='bold')

pdp_shap(axes[2],shap_explainer_resistance_kndvi2_after2000,finalmodel_resistance_kndvi2_after2000, x_resistance_kndvi2_after2000,'#8A2BE2', '#20B2AA',[0.04,0.25])
axes[2].set_xlabel('')
axes[2].set_xlim([8.5,11.7])
axes[2].set_title('(c)',loc='left', fontweight='bold')

importance_bar(axes[3],shap_df_sif2,permimport_sif2,'#8A2BE2', '#20B2AA', y_labs_sif2)
axes[3].set_title('(d)',loc='left', fontweight='bold')

violin_ax( axes[4],shap_explainer_resistance_sif2.values, features=shap_explainer_resistance_sif2.data, 
            feature_names=x_resistance_sif2.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[4].set_xlim([-0.05,0.035])
axes[4].set_title('(e)',loc='left', fontweight='bold')

pdp_shap(axes[5],shap_explainer_resistance_sif2,finalmodel_resistance_sif2, x_resistance_sif2,'#8A2BE2', '#20B2AA',[0.04,0.25])
axes[5].set_xlim([8.5,11.7])
axes[5].set_title('(f)',loc='left', fontweight='bold')

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.19, right=0.94)

plt.savefig(r'result_figure/figure_use_20260105/ml_results_kndvi_after2000&sif2.png', dpi = 600)

# %% [markdown]
# ### 6.3 Ydrou/Ymean  log

# %%
shap_explainer_kndvi2_log

# %%
shap_explainer_kndvi2_after2000_log

# %%
#fig, axes = plt.subplots(2,3,figsize = (12,8)) 
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1])

axes = []
for i in range(6):
    axes.append(fig.add_subplot(gs[i//3,i%3]))

importance_bar(axes[0],shap_df_kndvi2_log,permimport_kndvi2_log,'#8A2BE2', '#20B2AA', y_labs_kndvi2_log)
axes[0].set_xlabel('')
axes[0].set_title('(a)',loc='left', fontweight='bold')

violin_ax( axes[1],shap_explainer_resistance_kndvi2_log.values, features=shap_explainer_resistance_kndvi2_log.data, 
            feature_names=x_resistance_kndvi2.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[1].set_xlabel('')
axes[1].set_xlim([-0.8,0.6])
axes[1].set_title('(b)',loc='left', fontweight='bold')

pdp_shap(axes[2],shap_explainer_resistance_kndvi2_log,finalmodel_resistance_kndvi2_log, x_resistance_kndvi2,'#8A2BE2', '#20B2AA',[0.04,0.25],0.2)
axes[2].set_xlabel('')
axes[2].set_xlim([8.5,11.7])
axes[2].set_title('(c)',loc='left', fontweight='bold')

importance_bar(axes[3],shap_df_sif2_log,permimport_sif2_log,'#8A2BE2', '#20B2AA', y_labs_sif2)
axes[3].set_title('(d)',loc='left', fontweight='bold')

violin_ax( axes[4],shap_explainer_resistance_sif2_log.values, features=shap_explainer_resistance_sif2_log.data, 
            feature_names=x_resistance_sif2.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[4].set_xlim([-0.8,0.6])
axes[4].set_title('(e)',loc='left', fontweight='bold')

pdp_shap(axes[5],shap_explainer_resistance_sif2_log,finalmodel_resistance_sif2_log, x_resistance_sif2,'#8A2BE2', '#20B2AA',[0.04, 0.25],0.2)
axes[5].set_xlim([8.5,11.7])
axes[5].set_title('(f)',loc='left', fontweight='bold')

#fig.align_ylabels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.19, right=0.94, wspace=0.2)

plt.savefig(r'result_figure/figure_use_20260105/ml_results_kndvi&sif2_log.png', dpi = 600)

# %%
#fig, axes = plt.subplots(2,3,figsize = (12,8)) 
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.25, 1])

axes = []
for i in range(6):
    axes.append(fig.add_subplot(gs[i//3,i%3]))

importance_bar(axes[0],shap_df_kndvi2_after2000_log,permimport_kndvi2_after2000_log,'#8A2BE2', '#20B2AA', y_labs_kndvi2_after2000_log)
axes[0].set_xlabel('')
axes[0].set_title('(a)',loc='left', fontweight='bold')

violin_ax( axes[1],shap_explainer_resistance_kndvi2_after2000_log.values, features=shap_explainer_resistance_kndvi2_after2000_log.data, 
            feature_names=x_resistance_kndvi2_after2000.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[1].set_xlabel('')
axes[1].set_xlim([-0.8,0.55])
axes[1].set_title('(b)',loc='left', fontweight='bold')

pdp_shap(axes[2],shap_explainer_resistance_kndvi2_after2000_log,finalmodel_resistance_kndvi2_after2000_log, x_resistance_kndvi2_after2000,'#8A2BE2', '#20B2AA',[0.04,0.25],0.2)
axes[2].set_xlabel('')
axes[2].set_xlim([8.5,11.7])
axes[2].set_title('(c)',loc='left', fontweight='bold')

importance_bar(axes[3],shap_df_sif2_log,permimport_sif2_log,'#8A2BE2', '#20B2AA', y_labs_sif2_log)
axes[3].set_title('(d)',loc='left', fontweight='bold')

violin_ax( axes[4],shap_explainer_resistance_sif2_log.values, features=shap_explainer_resistance_sif2_log.data, 
            feature_names=x_resistance_sif2.columns, color = 'RdBu_r',
            axvl_color='red' ,axis_color = '#0d0d0d', which_xlim='abs_minmax', x_lab='Resistance')
axes[4].set_xlim([-0.85,0.55])
axes[4].set_title('(e)',loc='left', fontweight='bold')

pdp_shap(axes[5],shap_explainer_resistance_sif2_log,finalmodel_resistance_sif2_log, x_resistance_sif2,'#8A2BE2', '#20B2AA',[0.04,0.25],0.2)
axes[5].set_xlim([8.5,11.7])
axes[5].set_title('(f)',loc='left', fontweight='bold')

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.08, left=0.19, right=0.94)

plt.savefig(r'result_figure/figure_use_20260105/ml_results_kndvi_after2000&sif2_log.png', dpi = 600)

# %% [markdown]
# ## change log
# 1. 2025.11.19  机器学习
# 2. 2025.12.16  重新跑了一下  加上了 kndvi 2000年之后的结果
# 3. 2026.01.07  加上了用 Ydrou/Ymean 作为干旱抵抗力指标的结果

# %%



