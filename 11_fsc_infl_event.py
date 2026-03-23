# %% [markdown]
# # 按照时间重新做前面的分析

# %%
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import glob

# %% [markdown]
# ## 1 先读取数据   
# 用之前做机器学习保存的数据
# log转换
# 分位数筛选

# %% [markdown]
# ### 1.1 干旱抵抗力

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance.nc') as data:
    kndvi_nt_resistance = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance.nc') as data:
    kndvi_sh_resistance = data['kndvi_resistance']
kndvi_nt_resistance

# %%
kndvi_sh_resistance

# %%
kndvi_nt_resistance[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2.nc') as data:
    kndvi_nt_resistance2 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2.nc') as data:
    kndvi_sh_resistance2 = data['kndvi_resistance']
kndvi_nt_resistance2

# %%
kndvi_sh_resistance2

# %%
kndvi_nt_resistance2[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %% [markdown]
# ### 1.2 干旱指数

# %%
with xr.open_dataset(r'../result_data/spei_nt_annual_drought.nc') as data:
    spei_nt_annual_drought = data['spei']

with xr.open_dataset(r'../result_data/spei_sh_annual_drought.nc') as data:
    spei_sh_annual_drought = data['__xarray_dataarray_variable__']
    spei_sh_annual_drought.name = 'spei'
spei_nt_drought_use_kndvi = spei_nt_annual_drought.sel(year = slice(1982, 2021))
spei_sh_drought_use_kndvi = spei_sh_annual_drought.sel(year = slice(1982, 2020))

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

# %%
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

# %%
## 干旱指数 ai
with rioxarray.open_rasterio(r'D:/data/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif')  as data:
        ai_index= xr.DataArray(data.values[0], coords=[data.y, data.x], dims=['lat','lon'])
ai_index = ai_index.coarsen(lat=10,lon=10).mean()
ai_index
ai_index = ai_index * 0.0001
ai_index.where(ai_index<1.5).plot()

# %% [markdown]
# ### 1.4 多样性和结构复杂度  LAI

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

# %%
with xr.open_dataset('E:/python_output/fsc_drought/lai_gs_05.nc') as data:
    lai_gs = data['lai'].interp_like(fsc, method='nearest')
lai_gs

# %%
lai_gs.plot()

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

# %%
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

# %%
annual_temp = annual_temp.coarsen(lat = 5, lon=5, boundary='trim').mean()
annual_prec = annual_prec.coarsen(lat =5, lon=5, boundary='trim').mean()
ai_index = ai_index.coarsen(lat = 6, lon = 6).mean()
annual_temp = annual_temp.interp_like(plant_richness, method='nearest')
annual_prec = annual_prec.interp_like(plant_richness, method='nearest')
ai_index = ai_index.interp_like(plant_richness, method='nearest')

# %% [markdown]
# ### 1.9 土壤水分变化 蒸腾变化 和  LST变化

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi.nc') as data:
    smrz_nt_change_kndvi = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi.nc') as data:
    smrz_sh_change_kndvi = data['sm_change']
smrz_nt_change_kndvi

# %%
smrz_nt_change_kndvi[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
smrz_sh_change_kndvi

# %%
smrz_sh_change_kndvi[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
## 蒸腾2
with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change2_kndvi.nc') as data:
    et_nt_change_kndvi = data['et_change2']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change2_kndvi.nc') as data:
    et_sh_change_kndvi = data['et_change2']
et_nt_change_kndvi

# %%
et_nt_change_kndvi[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)

# %%
et_sh_change_kndvi

# %%
et_sh_change_kndvi[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)

# %%
## LST
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_zs_kndvi.nc') as data:
    lst_nt_zs_kndvi = data['lst_zs']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_zs_kndvi.nc') as data:
    lst_sh_zs_kndvi = data['lst_zs']
lst_nt_zs_kndvi[:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)

# %% [markdown]
# ### 1.10 biome

# %%
with xr.open_dataset(r'D:/data/official_teow/biome.tif')  as data:
    biome = data['band_data'][0].drop('spatial_ref')
    biome.name = 'biome'
biome

# %%
biome = biome.where(biome>0)
biome = biome.where(biome<90)
biome.plot()

# %%
biome = biome.rename({'x':'lon','y':'lat'})
biome = biome.interp_like(plant_richness, method='nearest').drop('band')
biome

# %% [markdown]
# ### 1.11 合并数据

# %%
dataset_nt = xr.Dataset({
        'kndvi': kndvi_nt_resistance,
        'kndvi2': kndvi_nt_resistance2,
        'spei': spei_nt_drought_use_kndvi,
        'sm_change': smrz_nt_change_kndvi,
        'et_change': et_nt_change_kndvi,
        'lst_zs': lst_nt_zs_kndvi}).drop('quantile')
dataset_nt

# %%
dataset_sh = xr.Dataset({
        'kndvi': kndvi_sh_resistance,
        'kndvi2': kndvi_sh_resistance2,
        'spei': spei_sh_drought_use_kndvi,
        'sm_change': smrz_sh_change_kndvi,
        'et_change': et_sh_change_kndvi,
        'lst_zs': lst_sh_zs_kndvi}).drop('quantile')
dataset_sh

# %%
df_nt = dataset_nt.to_dataframe().reset_index()
df_sh = dataset_sh.to_dataframe().reset_index()
df_nt = df_nt.dropna()
df_sh = df_sh.dropna()
df_nt.head()

# %%
df_sh.head()

# %%
df_all_kndvi = pd.concat([df_nt, df_sh])
df_all_kndvi.index = np.arange(df_all_kndvi.shape[0])
df_all_kndvi.head()

# %%
df_all_kndvi.describe()

# %%
other_factor = xr.Dataset({
        'fsc': fsc,
        'annual_temp': annual_temp,
        'annual_prec': annual_prec,
        'ai_index': ai_index,
        'plant_richness': plant_richness,
        'lai_gs': lai_gs,
        'drought_severity': drought_severity,
        'drought_count': drought_count,
        'drought_duration': drought_duration,
        'soil_cec': soil_cec,
        'soil_clay': soil_clay,
        'cti': cti,
        'sla': sla,
        'wood_den': wood_den,
        'biome': biome
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
df_all_kndvi.to_csv(r'E:/python_output/fsc_drought/df_all_kndvi_events_with_lai.csv', index = False)

# %% [markdown]
# ### 1.12 2000以后的 kndvi

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance_after2000.nc') as data:
    kndvi_nt_resistance_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance_after2000.nc') as data:
    kndvi_sh_resistance_after2000 = data['kndvi_resistance']
kndvi_nt_resistance_after2000

# %%
kndvi_sh_resistance_after2000

# %%
kndvi_nt_resistance_after2000[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2_after2000.nc') as data:
    kndvi_nt_resistance2_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2_after2000.nc') as data:
    kndvi_sh_resistance2_after2000 = data['kndvi_resistance']
kndvi_nt_resistance2_after2000

# %%
kndvi_sh_resistance2_after2000

# %%
kndvi_nt_resistance2_after2000[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
spei_nt_drought_use_kndvi_after2000 = spei_nt_annual_drought.sel(year = slice(2000, 2021))
spei_sh_drought_use_kndvi_after2000 = spei_sh_annual_drought.sel(year = slice(2000, 2020))

# %% [markdown]
# ### 1.13 2000以后的 水分 蒸腾 和 lst

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi_after2000.nc') as data:
    smrz_nt_change_kndvi_after2000 = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi_after2000.nc') as data:
    smrz_sh_change_kndvi_after2000 = data['sm_change']
smrz_nt_change_kndvi_after2000

# %%
smrz_nt_change_kndvi_after2000[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
smrz_sh_change_kndvi_after2000

# %%
smrz_sh_change_kndvi_after2000[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
## 蒸腾2
with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change2_kndvi_after2000.nc') as data:
    et_nt_change_kndvi_after2000 = data['et_change2']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change2_kndvi_after2000.nc') as data:
    et_sh_change_kndvi_after2000 = data['et_change2']
et_nt_change_kndvi_after2000

# %%
et_nt_change_kndvi_after2000[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)

# %%
et_sh_change_kndvi_after2000

# %%
et_sh_change_kndvi_after2000[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)

# %%
## LST
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_zs_kndvi_after2000.nc') as data:
    lst_nt_zs_kndvi_after2000 = data['lst_zs']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_zs_kndvi_after2000.nc') as data:
    lst_sh_zs_kndvi_after2000 = data['lst_zs']
lst_nt_zs_kndvi_after2000[:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)

# %%
dataset_nt = xr.Dataset({
        'kndvi': kndvi_nt_resistance_after2000,
        'kndvi2': kndvi_nt_resistance2_after2000,
        'spei': spei_nt_drought_use_kndvi_after2000,
        'sm_change': smrz_nt_change_kndvi_after2000,
        'et_change': et_nt_change_kndvi_after2000,
        'lst_zs': lst_nt_zs_kndvi_after2000}).drop('quantile')
dataset_nt

# %%
dataset_sh = xr.Dataset({
        'kndvi': kndvi_sh_resistance_after2000,
        'kndvi2': kndvi_sh_resistance2_after2000,
        'spei': spei_sh_drought_use_kndvi_after2000,
        'sm_change': smrz_sh_change_kndvi_after2000,
        'et_change': et_sh_change_kndvi_after2000,
        'lst_zs': lst_sh_zs_kndvi_after2000}).drop('quantile')
dataset_sh

# %%
df_nt = dataset_nt.to_dataframe().reset_index()
df_sh = dataset_sh.to_dataframe().reset_index()
df_nt = df_nt.dropna()
df_sh = df_sh.dropna()
df_nt.head()

# %%
df_sh.head()

# %%
df_all_kndvi_after2000 = pd.concat([df_nt, df_sh])
df_all_kndvi_after2000.index = np.arange(df_all_kndvi_after2000.shape[0])
df_all_kndvi_after2000.head()

# %%
df_all_kndvi_after2000.describe()

# %%
df_all_kndvi_after2000 = pd.merge(df_all_kndvi_after2000, other_factor, on = ['lat','lon'])
df_all_kndvi_after2000.head()

# %%
df_all_kndvi_after2000 = df_all_kndvi_after2000.dropna()
df_all_kndvi_after2000.index = np.arange(df_all_kndvi_after2000.shape[0])
df_all_kndvi_after2000.head()

# %%
df_all_kndvi_after2000.describe()

# %% [markdown]
# ## 2 sif数据

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
sif_nt_resistance2

# %%
sif_sh_resistance2

# %%
sif_nt_resistance2[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

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
# ### 2.3 土壤水分 et lst

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

# %%
## 蒸腾
with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change2_sif.nc') as data:
    et_nt_change_sif = data['et_change2']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change2_sif.nc') as data:
    et_sh_change_sif = data['et_change2']
et_nt_change_sif

# %%
et_nt_change_sif[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -5, vmax = 5)

# %%
et_sh_change_sif

# %%
et_sh_change_sif[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -5, vmax = 5)

# %%
## LST
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_zs_sif.nc') as data:
    lst_nt_zs_sif = data['lst_zs']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_zs_sif.nc') as data:
    lst_sh_zs_sif = data['lst_zs']
lst_nt_zs_sif[:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)

# %% [markdown]
# ### 2.4 合并数据

# %%
dataset_nt_sif = xr.Dataset({
        'sif': sif_nt_resistance,
        'sif2': sif_nt_resistance2,
        'spei': spei_nt_drought_use_sif,
        'sm_change': smrz_nt_change_sif,
        'et_change': et_nt_change_sif,
        'lst_zs': lst_nt_zs_sif}).drop('quantile')
dataset_nt_sif

# %%
dataset_sh_sif = xr.Dataset({
        'sif': sif_sh_resistance,
        'sif2': sif_sh_resistance2,
        'spei': spei_sh_drought_use_sif,
        'sm_change': smrz_sh_change_sif,
        'et_change': et_sh_change_sif,
        'lst_zs': lst_sh_zs_sif}).drop('quantile')
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
df_all_sif.index = np.arange(df_all_sif.shape[0])
df_all_sif.head()

# %%
df_all_sif.describe()

# %%
df_all_sif.to_csv(r'E:/python_output/fsc_drought/df_all_sif_events_with_lai.csv', index = False)

# %% [markdown]
# ## 3 预处理数据

# %%
df_sm_kndvi = df_all_kndvi.copy()
df_sm_kndvi_after2000 = df_all_kndvi_after2000.copy()
df_sm_sif = df_all_sif.copy()

sm_kndvi_q95 = df_sm_kndvi['sm_change'].quantile(0.95)
sm_kndvi_q95_after2000 = df_sm_kndvi_after2000['sm_change'].quantile(0.95)
sm_sif_q95 = df_sm_sif['sm_change'].quantile(0.95)

df_sm_kndvi = df_sm_kndvi[df_sm_kndvi['sm_change']<sm_kndvi_q95]
df_sm_kndvi_after2000 = df_sm_kndvi_after2000[df_sm_kndvi_after2000['sm_change']<sm_kndvi_q95_after2000]
df_sm_sif = df_sm_sif[df_sm_sif['sm_change']<sm_sif_q95]

# %%
df_sm_kndvi['sm_log'] = np.log(df_sm_kndvi['sm_change'])
df_sm_kndvi_after2000['sm_log'] = np.log(df_sm_kndvi_after2000['sm_change'])
df_sm_sif['sm_log'] = np.log(df_sm_sif['sm_change'])
df_sm_kndvi.describe()

# %%
df_sm_kndvi_after2000.describe()

# %%
df_sm_sif.describe()

# %%
kndvi_q95 = df_all_kndvi['kndvi'].quantile(0.95)
kndvi_q95_after2000 = df_all_kndvi_after2000['kndvi'].quantile(0.95)
sif_q95 = df_all_sif['sif'].quantile(0.95)

df_all_kndvi = df_all_kndvi[df_all_kndvi['kndvi']<kndvi_q95]
df_all_kndvi_after2000 = df_all_kndvi_after2000[df_all_kndvi_after2000['kndvi']<kndvi_q95_after2000]
df_all_sif = df_all_sif[df_all_sif['sif']<sif_q95]

# %%
df_all_kndvi['resis_log'] = np.log(df_all_kndvi['kndvi'])
df_all_kndvi_after2000['resis_log'] = np.log(df_all_kndvi_after2000['kndvi'])
df_all_sif['resis_log'] = np.log(df_all_sif['sif'])

# %%
df_all_kndvi['resis_log2'] = np.log(df_all_kndvi['kndvi2']/(1-df_all_kndvi['kndvi2']))   
df_all_kndvi_after2000['resis_log2'] = np.log(df_all_kndvi_after2000['kndvi2']/(1-df_all_kndvi_after2000['kndvi2']))
df_all_sif['resis_log2'] = np.log(df_all_sif['sif2']/(1-df_all_sif['sif2']))

# %%
df_all_kndvi.describe()

# %%
df_all_kndvi_after2000.describe()

# %%
df_all_sif.describe()

# %%
df_all_kndvi.to_csv(r'E:/python_output/fsc_drought/df_all_kndvi_events_data.csv', index = False)
df_all_kndvi_after2000.to_csv(r'E:/python_output/fsc_drought/df_all_kndvi_events_data_after2000.csv', index = False)
df_all_sif.to_csv(r'E:/python_output/fsc_drought/df_all_sif_events_data.csv', index = False)

# %% [markdown]
# ## 4 kndvi

# %% [markdown]
# ### 4.1 先做相关

# %% [markdown]
# #### 4.1.1 因子间的简单相关
# - 排除那些 sm et 和 lst变化

# %%
df_all_kndvi.columns

# %%
drou_resis_cor_kndvi = df_all_kndvi.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log','resis_log2','kndvi2','spei','fsc','plant_richness','lai_gs','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi

# %%
var_name = ['Drought resistance','Drought resistance2_log','Drought resistance2','SPEI','Forest structural complexity','Tree species richness','LAI','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']

# %%
drou_resis_np_kndvi = np.asarray(drou_resis_cor_kndvi)
drou_resis_np_kndvi

# %%
p_value_kndvi = np.full_like(drou_resis_np_kndvi, fill_value=np.nan)
p_value_kndvi

# %%
var_name_raw = ['resis_log','resis_log2','kndvi2','spei','fsc','plant_richness','lai_gs','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

# %%
from scipy.stats import spearmanr
for i in range(drou_resis_np_kndvi.shape[0]):
    for j in range(drou_resis_np_kndvi.shape[1]):
        
        p_value_kndvi[i,j] = spearmanr(df_all_kndvi[var_name_raw[i]],df_all_kndvi[var_name_raw[j]])[1]
p_value_kndvi

# %%
drou_resis_np_kndvi = np.where(p_value_kndvi<0.001,drou_resis_np_kndvi, np.nan)
for i in range(drou_resis_np_kndvi.shape[0]):
    for j in range(drou_resis_np_kndvi.shape[1]):
        if i <= j:
            drou_resis_np_kndvi[i,j] = np.nan
drou_resis_np_kndvi

# %%
drou_resis_np_kndvi.shape

# %%
plt.rc('font',family='Times New Roman', size = 15)

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(17))
ax.set_yticks(ticks=np.arange(17))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,18):
    for j in range(17):
        if ~np.isnan(drou_resis_np_kndvi[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/cor_resistance_kndvi_csc_kndvi_events.png', dpi = 600)

# %% [markdown]
# ##### 画之前用的 抵抗力指标的简单相关

# %%
df_all_kndvi.columns

# %%
drou_resis_cor_kndvi = df_all_kndvi.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']

# %%
drou_resis_np_kndvi = np.asarray(drou_resis_cor_kndvi)
drou_resis_np_kndvi

# %%
p_value_kndvi = np.full_like(drou_resis_np_kndvi, fill_value=np.nan)
p_value_kndvi

# %%
var_name_raw = ['resis_log','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

# %%
from scipy.stats import spearmanr
for i in range(drou_resis_np_kndvi.shape[0]):
    for j in range(drou_resis_np_kndvi.shape[1]):
        
        p_value_kndvi[i,j] = spearmanr(df_all_kndvi[var_name_raw[i]],df_all_kndvi[var_name_raw[j]])[1]
p_value_kndvi

# %%
drou_resis_np_kndvi = np.where(p_value_kndvi<0.001,drou_resis_np_kndvi, np.nan)

# %%
for i in range(drou_resis_np_kndvi.shape[0]):
    for j in range(drou_resis_np_kndvi.shape[1]):
        if i <= j:
            drou_resis_np_kndvi[i,j] = np.nan
drou_resis_np_kndvi

# %%
drou_resis_np_kndvi.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_kndvi[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/cor_resistance_kndvi_csc_kndvi_events.png', dpi = 600)

# %% [markdown]
# ##### 画之后用的 Ydrou/Ymean 抵抗力指标的简单相关

# %%
df_all_kndvi.columns

# %%
drou_resis_cor_kndvi2 = df_all_kndvi.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['kndvi2','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi2

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']

# %%
drou_resis_np_kndvi2 = np.asarray(drou_resis_cor_kndvi2)
drou_resis_np_kndvi2


# %%
p_value_kndvi2 = np.full_like(drou_resis_np_kndvi2, fill_value=np.nan)
p_value_kndvi2

# %%
var_name_raw = ['kndvi2','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_kndvi2.shape[0]):
    for j in range(drou_resis_np_kndvi2.shape[1]):
        
        p_value_kndvi2[i,j] = spearmanr(df_all_kndvi[var_name_raw[i]],df_all_kndvi[var_name_raw[j]])[1]
p_value_kndvi2

# %%
drou_resis_np_kndvi2 = np.where(p_value_kndvi2<0.001,drou_resis_np_kndvi2, np.nan)
for i in range(drou_resis_np_kndvi2.shape[0]):
    for j in range(drou_resis_np_kndvi2.shape[1]):
        if i <= j:
            drou_resis_np_kndvi2[i,j] = np.nan
drou_resis_np_kndvi2

# %%
drou_resis_np_kndvi2.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi2[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_kndvi2[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi2[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/cor_resistance_kndvi_csc_kndvi2_events.png', dpi = 600)

# %% [markdown]
# ##### 画之后用的 Ydrou/Ymean log 抵抗力指标的简单相关

# %%
df_all_kndvi.columns

# %%
drou_resis_cor_kndvi2_log = df_all_kndvi.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log2','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi2_log

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drou_resis_np_kndvi2_log = np.asarray(drou_resis_cor_kndvi2_log)
drou_resis_np_kndvi2_log

# %%
p_value_kndvi2_log = np.full_like(drou_resis_np_kndvi2_log, fill_value=np.nan)
p_value_kndvi2_log

# %%
var_name_raw = ['resis_log2','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_kndvi2_log.shape[0]):
    for j in range(drou_resis_np_kndvi2_log.shape[1]):
        
        p_value_kndvi2_log[i,j] = spearmanr(df_all_kndvi[var_name_raw[i]],df_all_kndvi[var_name_raw[j]])[1]
p_value_kndvi2_log

# %%
drou_resis_np_kndvi2_log = np.where(p_value_kndvi2_log<0.001,drou_resis_np_kndvi2_log, np.nan)
for i in range(drou_resis_np_kndvi2_log.shape[0]):
    for j in range(drou_resis_np_kndvi2_log.shape[1]):
        if i <= j:
            drou_resis_np_kndvi2_log[i,j] = np.nan
drou_resis_np_kndvi2_log

# %%
drou_resis_np_kndvi2_log.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi2_log[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_kndvi2_log[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi2_log[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/cor_resistance_kndvi_csc_kndvi2_events_log.png', dpi = 600)

# %% [markdown]
# #### 4.1.2 偏相关

# %%
import pingouin as pg
from scipy import stats

# %%
grouped_resistance_kndvi_by_bi = df_all_kndvi.groupby("biome")

# %%
biome_short_dic = {1: 'Trop.&Subtrop. Moist Broad. Forests',
 2: 'Trop.&Subtrop. Dry Broad. Forests',
 3: 'Trop.&Subtrop. Coni. Forests',
 4: 'Temp. Broad.&Mixed Forests',
 5: 'Temp. Coni. Forests',
 6: 'Boreal Forests',
 7: 'Trop.&Subtrop. Sav.&Shrub.',
 8: 'Temp. Sav.&Shrub.',
 9: 'Flooded Savannas',
 10: 'Montane Shrublands',
 11: 'Tundra',
 12: 'Mediter. Forests&Woodlands',
 13: 'Deserts&Xeric Shrublands',
 14: 'Mangroves',
 0:'All'}

# %%
df_all_kndvi.biome.value_counts().index[:9]

# %%
df_all_kndvi.columns

# %%
## 把整个 偏相关的部分写成一个函数
def biome_partial_cor(df, var1, var2, z, y): 

    grouped_resistance_by_bi = df.groupby("biome")

    ## 先算简单相关
    bi_cor_var1 = {}
    bi_cor_var2 = {}
    for bi_n in df.biome.value_counts().index[:9]:
        print(bi_n)
        bi_group_df = grouped_resistance_by_bi.get_group(bi_n)
        bi_cor_var1[bi_n] = {'r':stats.spearmanr(bi_group_df[y],bi_group_df[var1])[0],
                        'p-val':stats.spearmanr(bi_group_df[y],bi_group_df[var1])[1]}
        bi_cor_var2[bi_n] = {'r':stats.spearmanr(bi_group_df[y],bi_group_df[var2])[0],
                        'p-val':stats.spearmanr(bi_group_df[y],bi_group_df[var2])[1]}
        
    spear_var1 = stats.spearmanr(df[y],df[var1])
    spear_var2 = stats.spearmanr(df[y],df[var2])

    bi_pd_var1_cor_df = pd.DataFrame(bi_cor_var1).T
    bi_pd_var1_cor_df['var'] = var1
    bi_pd_var1_cor_df = bi_pd_var1_cor_df.sort_index()

    bi_pd_var2_cor_df = pd.DataFrame(bi_cor_var2).T
    bi_pd_var2_cor_df['var'] = var1
    bi_pd_var2_cor_df = bi_pd_var2_cor_df.sort_index()

    var1_cor_list = list(bi_pd_var1_cor_df.r)
    var1_cor_list.append(spear_var1[0])
    var2_cor_list = list(bi_pd_var2_cor_df.r)
    var2_cor_list.append(spear_var2[0])
    bi_cor_list = list(bi_pd_var2_cor_df.index)
    bi_cor_list.append(0)
    bi_cor_pd_r = pd.DataFrame({var1:var1_cor_list, var2:var2_cor_list, 'biome':bi_cor_list})
 
    var1_cor_list = list(bi_pd_var1_cor_df['p-val'])
    var1_cor_list.append(spear_var1[1])
    var2_cor_list = list(bi_pd_var2_cor_df['p-val'])
    var2_cor_list.append(spear_var2[1])
    bi_cor_list = list(bi_pd_var2_cor_df.index)
    bi_cor_list.append(0)
    bi_cor_pd_p = pd.DataFrame({var1:var1_cor_list, var2:var2_cor_list, 'biome':bi_cor_list})

    ## 偏相关
    bi_pd_var1_var2_pcor = {}
    bi_pd_var1_z_pcor = {}
    bi_pd_var2_var1_pcor = {}
    bi_pd_var2_z_pcor = {}
    for bi_n in df.biome.value_counts().index[:9]:
        print(bi_n)
        bi_group_df = grouped_resistance_by_bi.get_group(bi_n)
        var1_pcor = pg.partial_corr(data = bi_group_df,y=y,x=var1,covar=[var2],method='spearman').round(4)
        var1_z_pcor = pg.partial_corr(data = bi_group_df,y=y,x=var1,covar=[z],method='spearman').round(4)
        var2_pcor = pg.partial_corr(data = bi_group_df,y=y,x=var2,covar=[var1],method='spearman').round(4)
        var2_z_pcor = pg.partial_corr(data = bi_group_df,y=y,x=var2,covar=[z],method='spearman').round(4)
        bi_pd_var1_var2_pcor[bi_n] = {'r':var1_pcor['r'].values[0], 'p-val': var1_pcor['p-val'].values[0]}
        bi_pd_var2_var1_pcor[bi_n] = {'r':var2_pcor['r'].values[0], 'p-val': var2_pcor['p-val'].values[0]}
        bi_pd_var1_z_pcor[bi_n] = {'r':var1_z_pcor['r'].values[0], 'p-val': var1_z_pcor['p-val'].values[0]}
        bi_pd_var2_z_pcor[bi_n] = {'r':var2_z_pcor['r'].values[0], 'p-val': var2_z_pcor['p-val'].values[0]}

    pcor_var1_var2 = pg.partial_corr(data = df,y=y,x=var1,covar=[var2], method='spearman').round(3)
    pcor_var2_var1 = pg.partial_corr(data = df,y=y,x=var2,covar=[var1], method='spearman').round(3)
    pcor_var1_z = pg.partial_corr(data = df,y=y,x=var1,covar=[z], method='spearman').round(3)
    pcor_var2_z = pg.partial_corr(data = df,y=y,x=var2,covar=[z], method='spearman').round(3)

    bi_pd_var1_var2_pcor_df = pd.DataFrame(bi_pd_var1_var2_pcor).T
    bi_pd_var1_var2_pcor_df['var'] = var1+ '_' + var2
    bi_pd_var1_var2_pcor_df = bi_pd_var1_var2_pcor_df.sort_index()

    bi_pd_var2_var1_pcor_df = pd.DataFrame(bi_pd_var2_var1_pcor).T
    bi_pd_var2_var1_pcor_df['var'] = var2+ '_' + var1
    bi_pd_var2_var1_pcor_df = bi_pd_var2_var1_pcor_df.sort_index()

    bi_pd_var1_z_pcor_df = pd.DataFrame(bi_pd_var1_z_pcor).T
    bi_pd_var1_z_pcor_df['var'] = var1 + '_' + z
    bi_pd_var1_z_pcor_df = bi_pd_var1_z_pcor_df.sort_index()

    bi_pd_var2_z_pcor_df = pd.DataFrame(bi_pd_var2_z_pcor).T
    bi_pd_var2_z_pcor_df['var'] = var2 + '_' + z
    bi_pd_var2_z_pcor_df = bi_pd_var2_z_pcor_df.sort_index()

    var1_var2_list = list(bi_pd_var1_var2_pcor_df.r)
    var1_var2_list.append(pcor_var1_var2['r'].values[0])
    var1_z_list = list(bi_pd_var1_z_pcor_df.r)
    var1_z_list.append(pcor_var1_z['r'].values[0])

    var2_var1_list = list(bi_pd_var2_var1_pcor_df.r)
    var2_var1_list.append(pcor_var2_var1['r'].values[0])
    var2_z_list = list(bi_pd_var2_z_pcor_df.r)
    var2_z_list.append(pcor_var2_z['r'].values[0])
    
    bi_pcor_list = list(bi_pd_var2_z_pcor_df.index)
    bi_pcor_list.append(0)
    bi_pcor_df = pd.DataFrame({var1+'_'+z:var1_z_list, var1+'_'+var2:var1_var2_list,var2+'_'+z:var2_z_list, var2+'_'+var1:var2_var1_list, 'biome':bi_pcor_list})

    var1_var2_list = list(bi_pd_var1_var2_pcor_df['p-val'])
    var1_var2_list.append(pcor_var1_var2['p-val'].values[0])
    var1_z_list = list(bi_pd_var1_z_pcor_df['p-val'])
    var1_z_list.append(pcor_var1_z['p-val'].values[0])

    var2_var1_list = list(bi_pd_var2_var1_pcor_df['p-val'])
    var2_var1_list.append(pcor_var2_var1['p-val'].values[0])
    var2_z_list = list(bi_pd_var2_z_pcor_df['p-val'])
    var2_z_list.append(pcor_var2_z['p-val'].values[0])

    bi_pcor_list = list(bi_pd_var2_z_pcor_df.index)
    bi_pcor_list.append(0)
    bi_pval_df = pd.DataFrame({var1+'_'+z:var1_z_list, var1+'_'+var2:var1_var2_list,var2+'_'+z:var2_z_list, var2+'_'+var1:var2_var1_list, 'biome':bi_pcor_list})

    bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
    bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')

    return {'pcor':bi_pd_pcor_all, 'p-val':bi_pd_pval_all}

# %%
biome_partial_cor(df_all_kndvi, 'fsc', 'plant_richness', 'annual_prec', 'resis_log')

# %%
'''
bi_cor_fsc= {}
bi_cor_rich = {}
for bi_n in df_all_kndvi.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[1]}
    bi_cor_rich[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['plant_richness'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['plant_richness'])[1]}
    

bi_cor_fsc
'''

# %%
#bi_cor_rich

# %%
#stats.spearmanr(df_all_kndvi['resis_log'],df_all_kndvi['fsc'])

# %%
#stats.spearmanr(df_all_kndvi['resis_log'],df_all_kndvi['plant_richness'])

# %%
'''
bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df
'''

# %%
'''
bi_pd_rich_cor_df = pd.DataFrame(bi_cor_rich).T
bi_pd_rich_cor_df['var'] = 'rich'
bi_pd_rich_cor_df = bi_pd_rich_cor_df.sort_index()
bi_pd_rich_cor_df
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.447)
rich_cor_list = list(bi_pd_rich_cor_df.r)
rich_cor_list.append(0.408)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)
'''

# %%
'''
bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
rich_cor_list = list(bi_pd_rich_cor_df['p-val'])
rich_cor_list.append(0)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)
'''

# %%
'''
bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p
'''

# %%
'''
bi_pd_fsc_rich_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_rich_fsc_pcor = {}
bi_pd_rich_prec_pcor = {}
for bi_n in df_all_kndvi.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['plant_richness'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['annual_prec'],method='spearman').round(4)
    rich_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='plant_richness',covar=['fsc'],method='spearman').round(4)
    rich_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='plant_richness',covar=['annual_prec'],method='spearman').round(4)
    bi_pd_fsc_rich_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_rich_fsc_pcor[bi_n] = {'r':rich_pcor['r'].values[0], 'p-val': rich_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_rich_prec_pcor[bi_n] = {'r':rich_prec_pcor['r'].values[0], 'p-val': rich_prec_pcor['p-val'].values[0]}
'''

# %%
# bi_pd_fsc_rich_pcor

# %%
# bi_pd_fsc_prec_pcor

# %%
# bi_pd_rich_fsc_pcor

# %%
# bi_pd_rich_prec_pcor

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='plant_richness',covar=['annual_prec'], method='spearman').round(3)

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='plant_richness',covar=['fsc'], method='spearman').round(3)

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='fsc',covar=['annual_prec'], method='spearman').round(3)

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='fsc',covar=['plant_richness'], method='spearman').round(3)

# %%
'''
bi_pd_fsc_rich_pcor_df = pd.DataFrame(bi_pd_fsc_rich_pcor).T
bi_pd_fsc_rich_pcor_df['var'] = 'fsc_rich'
bi_pd_fsc_rich_pcor_df = bi_pd_fsc_rich_pcor_df.sort_index()
bi_pd_fsc_rich_pcor_df
'''

# %%
'''
bi_pd_rich_fsc_pcor_df = pd.DataFrame(bi_pd_rich_fsc_pcor).T
bi_pd_rich_fsc_pcor_df['var'] = 'rich_fsc'
bi_pd_rich_fsc_pcor_df = bi_pd_rich_fsc_pcor_df.sort_index()
bi_pd_rich_fsc_pcor_df
'''

# %%
'''
bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df
'''

# %%
'''
bi_pd_rich_prec_pcor_df = pd.DataFrame(bi_pd_rich_prec_pcor).T
bi_pd_rich_prec_pcor_df['var'] = 'rich_prec'
bi_pd_rich_prec_pcor_df = bi_pd_rich_prec_pcor_df.sort_index()
bi_pd_rich_prec_pcor_df
'''

# %%
'''
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df.r)
fsc_rich_list.append(0.455)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.267)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df.r)
rich_fsc_list.append(0.417)
rich_prec_list = list(bi_pd_rich_prec_pcor_df.r)
rich_prec_list.append(0.011)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)
'''

# %%
#bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
#bi_pcor_df

# %%
'''
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df['p-val'])
fsc_rich_list.append(0)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df['p-val'])
fsc_prec_list.append(0)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df['p-val'])
rich_fsc_list.append(0)
rich_prec_list = list(bi_pd_rich_prec_pcor_df['p-val'])
rich_prec_list.append(0.03)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)
'''

# %%
#bi_pval_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
#bi_pval_df

# %%
#bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
#bi_pd_pcor_all

# %%
#bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
#bi_pd_pval_all

# %%
#bi_pd_pcor_all.to_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_kndvi_mid_event.csv', index = False)
#bi_pd_pval_all.to_csv('E:/python_output/fsc_drought/bi_pd_pval_all_kndvi_mid_event.csv', index = False)

# %%
kndvi_pcor_result = biome_partial_cor(df_all_kndvi, 'fsc', 'plant_richness', 'annual_prec', 'resis_log')
kndvi_pcor_result

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events.png', dpi = 600)

# %%
kndvi2_pcor_result = biome_partial_cor(df_all_kndvi, 'fsc', 'plant_richness', 'annual_prec', 'kndvi2')
kndvi2_pcor_result

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events.png', dpi = 600)

# %%
kndvi2_log_pcor_result = biome_partial_cor(df_all_kndvi, 'fsc', 'plant_richness', 'annual_prec', 'resis_log2')
kndvi2_log_pcor_result

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_log_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events.png', dpi = 600)

# %% [markdown]
# #### 4.1.3 FSC 和 LAI的偏相关

# %%
df_all_kndvi.columns

# %%
kndvi_lai_pcor_result = biome_partial_cor(df_all_kndvi, 'fsc', 'lai_gs', 'annual_prec', 'resis_log')
kndvi_lai_pcor_result

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events.png', dpi = 600)

# %%
kndvi2_lai_pcor_result = biome_partial_cor(df_all_kndvi, 'fsc', 'lai_gs', 'annual_prec', 'kndvi2')
kndvi2_lai_pcor_result

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events.png', dpi = 600)

# %%
kndvi2_log_lai_pcor_result = biome_partial_cor(df_all_kndvi, 'fsc', 'lai_gs', 'annual_prec', 'resis_log2')
kndvi2_log_lai_pcor_result

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_log_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events.png', dpi = 600)

# %%
'''
bi_cor_fsc= {}
bi_cor_lai = {}
for bi_n in df_all_kndvi.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[1]}
    bi_cor_lai[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['lai_gs'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['lai_gs'])[1]}
bi_cor_fsc
'''

# %%
# bi_cor_lai

# %%
# stats.spearmanr(df_all_kndvi['resis_log'],df_all_kndvi['fsc'])

# %%
# stats.spearmanr(df_all_kndvi['resis_log'],df_all_kndvi['lai_gs'])

# %%
'''
bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df
'''

# %%
'''
bi_pd_lai_cor_df = pd.DataFrame(bi_cor_lai).T
bi_pd_lai_cor_df['var'] = 'lai'
bi_pd_lai_cor_df = bi_pd_lai_cor_df.sort_index()
bi_pd_lai_cor_df
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.448)
lai_cor_list = list(bi_pd_lai_cor_df.r)
lai_cor_list.append(0.634)
bi_cor_list = list(bi_pd_lai_cor_df.index)
bi_cor_list.append(0)
bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'lai':lai_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
lai_cor_list = list(bi_pd_lai_cor_df['p-val'])
lai_cor_list.append(0)
bi_cor_list = list(bi_pd_lai_cor_df.index)
bi_cor_list.append(0)
bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'lai':lai_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p
'''

# %%
'''
bi_pd_fsc_lai_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_lai_fsc_pcor = {}
bi_pd_lai_prec_pcor = {}
for bi_n in df_all_kndvi.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['lai_gs'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['annual_prec'],method='spearman').round(4)
    lai_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='lai_gs',covar=['fsc'],method='spearman').round(4)
    lai_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='lai_gs',covar=['annual_prec'],method='spearman').round(4)
    bi_pd_fsc_lai_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_lai_fsc_pcor[bi_n] = {'r':lai_pcor['r'].values[0], 'p-val': lai_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_lai_prec_pcor[bi_n] = {'r':lai_prec_pcor['r'].values[0], 'p-val': lai_prec_pcor['p-val'].values[0]}
bi_pd_fsc_lai_pcor
'''

# %%
# bi_pd_fsc_prec_pcor

# %%
# bi_pd_lai_fsc_pcor

# %%
# bi_pd_lai_prec_pcor

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='lai_gs',covar=['annual_prec'], method='spearman').round(3)

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='lai_gs',covar=['fsc'], method='spearman').round(3)

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='fsc',covar=['annual_prec'], method='spearman').round(3)

# %%
# pg.partial_corr(data = df_all_kndvi,y='resis_log',x='fsc',covar=['lai_gs'], method='spearman').round(3)

# %%
'''
bi_pd_fsc_lai_pcor_df = pd.DataFrame(bi_pd_fsc_lai_pcor).T
bi_pd_fsc_lai_pcor_df['var'] = 'fsc_lai'
bi_pd_fsc_lai_pcor_df = bi_pd_fsc_lai_pcor_df.sort_index()
bi_pd_fsc_lai_pcor_df
'''

# %%
'''
bi_pd_lai_fsc_pcor_df = pd.DataFrame(bi_pd_lai_fsc_pcor).T
bi_pd_lai_fsc_pcor_df['var'] = 'lai_fsc'
bi_pd_lai_fsc_pcor_df = bi_pd_lai_fsc_pcor_df.sort_index()
bi_pd_lai_fsc_pcor_df
'''

# %%
'''
bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df
'''

# %%
'''
bi_pd_lai_prec_pcor_df = pd.DataFrame(bi_pd_lai_prec_pcor).T
bi_pd_lai_prec_pcor_df['var'] = 'lai_prec'
bi_pd_lai_prec_pcor_df = bi_pd_lai_prec_pcor_df.sort_index()
bi_pd_lai_prec_pcor_df
'''

# %%
'''
fsc_lai_list = list(bi_pd_fsc_lai_pcor_df.r)
fsc_lai_list.append(0.076)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.276)
lai_fsc_list = list(bi_pd_lai_fsc_pcor_df.r)
lai_fsc_list.append(0.506)
lai_prec_list = list(bi_pd_lai_prec_pcor_df.r)
lai_prec_list.append(0.405)
bi_pcor_list = list(bi_pd_lai_prec_pcor_df.index)
bi_pcor_list.append(0)
bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_lai':fsc_lai_list,'lai_prec':lai_prec_list, 'lai_fsc':lai_fsc_list, 'biome':bi_pcor_list})
bi_pcor_df
'''

# %%
'''
fsc_lai_list = list(bi_pd_fsc_lai_pcor_df['p-val'])
fsc_lai_list.append(0)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df['p-val'])
fsc_prec_list.append(0)
lai_fsc_list = list(bi_pd_lai_fsc_pcor_df['p-val'])
lai_fsc_list.append(0)
lai_prec_list = list(bi_pd_lai_prec_pcor_df['p-val'])
lai_prec_list.append(0)
bi_pcor_list = list(bi_pd_lai_prec_pcor_df.index)
bi_pcor_list.append(0)
bi_pval_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_lai':fsc_lai_list,'lai_prec':lai_prec_list, 'lai_fsc':lai_fsc_list, 'biome':bi_pcor_list})
bi_pval_df
'''

# %%
'''
bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
bi_pd_pcor_all
'''

# %%
'''
bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
bi_pd_pval_all
'''

# %%
'''
draw_col = ['lai','lai_prec','lai_fsc','fsc','fsc_prec','fsc_lai']

fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(bi_pd_pcor_all[draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in bi_pd_pcor_all.biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if bi_pd_pval_all[draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif bi_pd_pval_all[draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif bi_pd_pval_all[draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()
'''

# %% [markdown]
# ### 4.2 fsc对 抵抗力  et sm lst的关系画图

# %%
print(df_all_kndvi.fsc.max(),df_all_kndvi.fsc.min())

# %%
df_all_kndvi['fsc_bins'] = pd.cut(df_all_kndvi.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
pd.unique(df_all_kndvi['fsc_bins'])

# %%
df_all_kndvi.plant_richness.max()

# %%
df_all_kndvi['rich_bins'] = pd.cut(df_all_kndvi.plant_richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %%
biome_dic2 = {1:'Trop.&Subtrop. Moist Broad. Forests',
          2:'Trop.&Subtrop. Dry Broad. Forests',
          3:'Trop.&Subtrop. Coni. Forests',
          4:'Temp. Broad.&Mix Forests',
          5:'Temp. Coni. Forests',
          6:'Boreal Forests',
          7:'Trop.&Subtrop. Sav.&Shrub.',
          8:'Temp. Sav.&Shrub.',
          9:'Flooded Savannas',
          10:'Montane Shrublands',
          11:'Tundra',
          12:'Mediter. Forests&Woodlands',
          13:'Deserts&Xeric Shrublands',
          14:'Mangroves',
          0:'All'}

# %% [markdown]
# #### 4.2.1 抵抗力 

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.resis_log[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.resis_log[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.resis_log[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 6.5, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,7)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_kndvi_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_kndvi.fsc_bins)
fsc_dfs = [df_all_kndvi.resis_log[df_all_kndvi.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_kndvi.resis_log[df_all_kndvi.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_kndvi.resis_log[df_all_kndvi.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,7)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_kndvi.rich_bins)
rich_dfs = [df_all_kndvi.resis_log[df_all_kndvi.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi.resis_log[df_all_kndvi.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi.resis_log[df_all_kndvi.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6.5, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,7)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_kndvi_csc_mid_log.png', dpi = 600)

# %%
df_all_kndvi.biome.value_counts()

# %%
grouped_resistance_kndvi_by_bi = df_all_kndvi.groupby("biome")

# %%
df_all_kndvi.biome.value_counts().index

# %%
df_all_kndvi.biome.value_counts().sort_values()

# %%
biome_short_dic = {1: 'Trop.&Subtrop. Moist Broad. Forests',
 2: 'Trop.&Subtrop. Dry Broad. Forests',
 3: 'Trop.&Subtrop. Coni. Forests',
 4: 'Temp. Broad.&Mixed Forests',
 5: 'Temp. Coni. Forests',
 6: 'Boreal Forests',
 7: 'Trop.&Subtrop. Sav.&Shrub.',
 8: 'Temp. Sav.&Shrub.',
 9: 'Flooded Savannas',
 10: 'Montane Shrublands',
 11: 'Tundra',
 12: 'Mediter. Forests&Woodlands',
 13: 'Deserts&Xeric Shrublands',
 14: 'Mangroves',
 0:'All'}
alpha_list = ['a','b','c','d','e','f','g','h','i']

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,0,0,0,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_kndvi_csc_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,0,0,0,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resis_log[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resis_log[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resis_log[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list[i]*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_kndvi_csc_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [12,11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5]
y_list_low = [9,8.5,9.5,9,8.5,8,8,8,8]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.fsc[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.fsc[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.fsc[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = (y_list[i] - y_list_low[i])*0.9 + y_list_low[i], s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Forest structural complexity')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_fsc.png', dpi = 600)

# %% [markdown]
# ##### kndvi2 画图

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.kndvi2[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.kndvi2[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.kndvi2[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 1.05, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance')
axes[0].set_xticklabels([])
axes[0].set_ylim(0.45,1.09)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_kndvi2.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_kndvi.fsc_bins)
fsc_dfs = [df_all_kndvi.kndvi2[df_all_kndvi.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_kndvi.kndvi2[df_all_kndvi.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_kndvi.kndvi2[df_all_kndvi.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 1.04, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0.4,1.09)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_kndvi.rich_bins)
rich_dfs = [df_all_kndvi.kndvi2[df_all_kndvi.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi.kndvi2[df_all_kndvi.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi.kndvi2[df_all_kndvi.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 1.04, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0.4,1.09)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_kndvi2.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [1.06,1.06,1.1,1.1,1.06,1.1,1.1,1.1,1.1]
y_list_low = [0.7,0.7,0.5,0.6,0.7,0.6,0.4,0.4,0.4]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.kndvi2[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.kndvi2[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.kndvi2[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_kndvi2.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [1.06,1.06,1.1,1.1,1.06,1.1,1.1,1.1,1.1]
y_list_low = [0.7,0.7,0.5,0.6,0.7,0.6,0.4,0.4,0.4]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.kndvi2[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.kndvi2[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.kndvi2[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_kndvi2.png', dpi = 600)

# %% [markdown]
# ##### kndvi2 log 画图

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.resis_log2[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.resis_log2[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.resis_log2[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 6.5, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(-1,7)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.plant_richness[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_kndvi.biome)
bi_dfs = [df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi.fsc[df_all_kndvi.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_kndvi2_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_kndvi.fsc_bins)
fsc_dfs = [df_all_kndvi.resis_log2[df_all_kndvi.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_kndvi.resis_log2[df_all_kndvi.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_kndvi.resis_log2[df_all_kndvi.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(-1,7)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_kndvi.rich_bins)
rich_dfs = [df_all_kndvi.resis_log2[df_all_kndvi.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi.resis_log2[df_all_kndvi.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi.resis_log2[df_all_kndvi.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6.5, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(-1,7)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi.fsc[df_all_kndvi.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_kndvi2_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,-1,0,0,-1,-1,-1,-1]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_kndvi2_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,-0.5,-1,0,-1,-1,-1,-1]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resis_log2[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resis_log2[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resis_log2[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_kndvi2_log.png', dpi = 600)

# %% [markdown]
# #### 4.2.2 sm change

# %%
print(df_sm_kndvi.fsc.min(), df_sm_kndvi.fsc.max() )

# %%
df_sm_kndvi['fsc_bins'] = pd.cut(df_sm_kndvi.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_sm_kndvi.fsc_bins)
fsc_dfs = [df_sm_kndvi.sm_log[df_sm_kndvi.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_sm_kndvi.sm_log[df_sm_kndvi.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_sm_kndvi.sm_log[df_sm_kndvi.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5.7, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,6.2)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/smrz_change_fsc_kndvi.png', dpi = 600)

# %%
df_sm_kndvi.biome.value_counts()

# %%
smrzchange_kndvi_fsc_bybi = df_sm_kndvi.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2]


for i in range(9):
    bi_n = df_sm_kndvi.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/smrz_change_csc_biome_kndvi_log.png', dpi = 600)

# %% [markdown]
# #### 4.2.3 et change

# %%
etchange_kndvi_use = df_all_kndvi.copy()
etchange_kndvi_use.head()

# %%
etchange_kndvi_use_po = etchange_kndvi_use[etchange_kndvi_use.et_change>0]
etchange_kndvi_use_ne = etchange_kndvi_use[etchange_kndvi_use.et_change<0]

etchange_kndvi_use_po.head()

# %%
etchange_kndvi_use_ne.head()

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use.fsc_bins)
fsc_dfs = [etchange_kndvi_use.et_change[etchange_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use.et_change[etchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use.et_change[etchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.32, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.8,0.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_kndvi.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_po.fsc_bins)
fsc_dfs = [etchange_kndvi_use_po.et_change[etchange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po.et_change[etchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po.et_change[etchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.35, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.02,0.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_kndvi_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_ne.fsc_bins)
fsc_dfs = [etchange_kndvi_use_ne.et_change[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne.et_change[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne.et_change[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.03, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.75,0.09)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_kndvi_ne.png', dpi = 600)

# %%
etchange_kndvi_use.biome.value_counts()

# %%
etchange_kndvi_fsc_bybi_po = etchange_kndvi_use_po.groupby("biome")
etchange_kndvi_fsc_bybi_ne = etchange_kndvi_use_ne.groupby("biome")
etchange_kndvi_fsc_bybi = etchange_kndvi_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.3,0.2,0.4,0.3,0.4,0.3,0.3,0.3,0.3]

for i in range(9):
    bi_n = etchange_kndvi_use_po.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.02,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.35,0.35,0.5,0.4,0.5,0.4,0.5,0.35,0.35]
y_list_low = [-0.65,-0.5,-0.6,-0.75,-0.5,-0.6,-0.9,-0.9,-0.75]

for i in range(9):
    bi_n = etchange_kndvi_use.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - y_list_low[i])*0.92 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_kndvi_use.png', dpi = 600)


# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.1,0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
y_list_low = [-0.75, -0.4 ,-0.8, -0.6, -0.75, -0.9, -0.6, -0.9, -0.8]

for i in range(9):
    bi_n = etchange_kndvi_use_ne.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i]-y_list_low[i]) * 0.92 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i], y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_kndvi_ne.png', dpi = 600)

# %% [markdown]
# #### 4.2.4 lst change

# %%
lstzs_kndvi_use = df_all_kndvi.copy()
lstzs_kndvi_use.head()

# %%
lstzs_kndvi_use['fsc_bins'] = pd.cut(lstzs_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
lstzs_kndvi_use_po = lstzs_kndvi_use[lstzs_kndvi_use.lst_zs>0]
lstzs_kndvi_use_ne = lstzs_kndvi_use[lstzs_kndvi_use.lst_zs<0]
lstzs_kndvi_use_po

# %%
lstzs_kndvi_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use_po.fsc_bins)
fsc_dfs = [lstzs_kndvi_use_po.lst_zs[lstzs_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_po.lst_zs[lstzs_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_po.lst_zs[lstzs_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.1,4.9)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_kndvi_po.png', dpi = 600)

# %%

fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use_ne.fsc_bins)
fsc_dfs = [lstzs_kndvi_use_ne.lst_zs[lstzs_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_ne.lst_zs[lstzs_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_ne.lst_zs[lstzs_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.02, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-1.75,0.1)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_kndvi_ne.png', dpi = 600)

# %%

fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use.fsc_bins)
fsc_dfs = [lstzs_kndvi_use.lst_zs[lstzs_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use.lst_zs[lstzs_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use.lst_zs[lstzs_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-2,5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_kndvi_use.png', dpi = 600)

# %%
lstzs_kndvi_fsc_bybi_po = lstzs_kndvi_use_po.groupby("biome")
lstzs_kndvi_fsc_bybi_ne = lstzs_kndvi_use_ne.groupby("biome")
lstzs_kndvi_fsc_bybi = lstzs_kndvi_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5,5,5,4.5,5,5,6.5,4.5,5]

for i in range(9):
    bi_n = lstzs_kndvi_use_po.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.2,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.1,5.1,5,4.5,5,6,6.5,4.5,5]
y_list_low = [-2.5,-2.1,-2.1,-2.1,-2.5,-2,-2.5,-2,-2]


for i in range(9):
    bi_n = lstzs_kndvi_use.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - y_list_low[i]) * 0.9 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_kndvi_use.png', dpi = 600)

# %%
lstzs_kndvi_use_ne.biome.value_counts()

# %%
fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [2,2,2]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstzs_kndvi_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = 0.05, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 0.2)
    axes[i].set_xlim(8.7,11.3)
    #axes[i].set_xticks(np.arange(9,11.3,0.5),labels = [])
    axes[i].grid(c = 'lightgray', alpha = 0.3)

    if i >= 0 :
        axes[i].set_xlabel('Forest structural complexity')
        axes[i].set_xticks(np.arange(9,11.3,0.5),labels = np.arange(9,11.3,0.5))
    if (i%3) == 0 :
        axes[i].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_kndvi_ne.png', dpi = 600)

# %% [markdown]
# ## 4 kndvi 2000以后

# %% [markdown]
# ### 4.1 先做相关
# #### 4.1.1 因子间的简单相关
# - 排除那些 sm et 和 lst变化

# %%
df_all_kndvi_after2000.columns

# %%
drou_resis_cor_kndvi_after2000 = df_all_kndvi_after2000.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log','resis_log2','kndvi2','spei','fsc','plant_richness','lai_gs','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi_after2000

# %%
var_name = ['Drought resistance','Drought resistance2_log','Drought resistance2','SPEI','Forest structural complexity','Tree species richness','LAI','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']

# %%
drou_resis_np_kndvi_after2000 = np.asarray(drou_resis_cor_kndvi_after2000)
drou_resis_np_kndvi_after2000

# %%
p_value_kndvi_after2000 = np.full_like(drou_resis_np_kndvi_after2000, fill_value=np.nan)
p_value_kndvi_after2000

# %%
var_name_raw = ['resis_log','resis_log2','kndvi2','spei','fsc','plant_richness','lai_gs','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

# %%
for i in range(drou_resis_np_kndvi_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi_after2000.shape[1]):
        
        p_value_kndvi_after2000[i,j] = spearmanr(df_all_kndvi_after2000[var_name_raw[i]],df_all_kndvi_after2000[var_name_raw[j]])[1]
p_value_kndvi_after2000

# %%
drou_resis_np_kndvi_after2000 = np.where(p_value_kndvi_after2000<0.001,drou_resis_np_kndvi_after2000, np.nan)
for i in range(drou_resis_np_kndvi_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi_after2000.shape[1]):
        if i <= j:
            drou_resis_np_kndvi_after2000[i,j] = np.nan
drou_resis_np_kndvi_after2000

# %%
drou_resis_np_kndvi_after2000.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi_after2000[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(17))
ax.set_yticks(ticks=np.arange(17))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,18):
    for j in range(17):
        if ~np.isnan(drou_resis_np_kndvi_after2000[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi_after2000[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/cor_resistance_kndvi_csc_kndvi_events_after2000.png', dpi = 600)

# %% [markdown]
# ##### 画之前用的 抵抗力指标的简单相关

# %%
df_all_kndvi_after2000.columns

# %%
drou_resis_cor_kndvi_after2000 = df_all_kndvi_after2000.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi_after2000

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drou_resis_np_kndvi_after2000 = np.asarray(drou_resis_cor_kndvi_after2000)
drou_resis_np_kndvi_after2000

# %%
p_value_kndvi_after2000 = np.full_like(drou_resis_np_kndvi_after2000, fill_value=np.nan)
p_value_kndvi_after2000

# %%
var_name_raw = ['resis_log','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_kndvi_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi_after2000.shape[1]):
        
        p_value_kndvi_after2000[i,j] = spearmanr(df_all_kndvi_after2000[var_name_raw[i]],df_all_kndvi_after2000[var_name_raw[j]])[1]
p_value_kndvi_after2000

# %%
drou_resis_np_kndvi_after2000 = np.where(p_value_kndvi_after2000<0.001,drou_resis_np_kndvi_after2000, np.nan)
for i in range(drou_resis_np_kndvi_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi_after2000.shape[1]):
        if i <= j:
            drou_resis_np_kndvi_after2000[i,j] = np.nan
drou_resis_np_kndvi_after2000

# %%
drou_resis_np_kndvi_after2000.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi_after2000[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_kndvi_after2000[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi_after2000[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use/cor_resistance_kndvi_after2000_events.png', dpi = 600)

# %% [markdown]
# ##### 画之后用的 Ydrou/Ymean 抵抗力指标的简单相关

# %%
df_all_kndvi_after2000.columns
drou_resis_cor_kndvi2_after2000 = df_all_kndvi_after2000.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['kndvi2','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi2_after2000

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drou_resis_np_kndvi2_after2000 = np.asarray(drou_resis_cor_kndvi2_after2000)
drou_resis_np_kndvi2_after2000

# %%
p_value_kndvi2_after2000 = np.full_like(drou_resis_np_kndvi2_after2000, fill_value=np.nan)
p_value_kndvi2_after2000

# %%
var_name_raw = ['kndvi2','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_kndvi2_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi2_after2000.shape[1]):
        
        p_value_kndvi2_after2000[i,j] = spearmanr(df_all_kndvi_after2000[var_name_raw[i]],df_all_kndvi_after2000[var_name_raw[j]])[1]
p_value_kndvi2_after2000

# %%
drou_resis_np_kndvi2_after2000 = np.where(p_value_kndvi2_after2000<0.001,drou_resis_np_kndvi2_after2000, np.nan)
for i in range(drou_resis_np_kndvi2_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi2_after2000.shape[1]):
        if i <= j:
            drou_resis_np_kndvi2_after2000[i,j] = np.nan
drou_resis_np_kndvi2_after2000

# %%
drou_resis_np_kndvi2_after2000.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi2[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_kndvi2[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi2[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/cor_resistance_kndvi2_after2000_events.png', dpi = 600)

# %% [markdown]
# ##### 画之后用的 Ydrou/Ymean log 抵抗力指标的简单相关

# %%
df_all_kndvi.columns

# %%
drou_resis_cor_kndvi2_log_after2000 = df_all_kndvi.drop(['lon', 'lat', 'year', 'kndvi', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log2','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_kndvi2_log_after2000

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drou_resis_np_kndvi2_log_after2000 = np.asarray(drou_resis_cor_kndvi2_log_after2000)
drou_resis_np_kndvi2_log_after2000

# %%
p_value_kndvi2_log_after2000 = np.full_like(drou_resis_np_kndvi2_log_after2000, fill_value=np.nan)
p_value_kndvi2_log_after2000

# %%
var_name_raw = ['resis_log2','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_kndvi2_log_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi2_log_after2000.shape[1]):
        
        p_value_kndvi2_log_after2000[i,j] = spearmanr(df_all_kndvi_after2000[var_name_raw[i]],df_all_kndvi_after2000[var_name_raw[j]])[1]
p_value_kndvi2_log_after2000

# %%
drou_resis_np_kndvi2_log_after2000 = np.where(p_value_kndvi2_log_after2000<0.001,drou_resis_np_kndvi2_log_after2000, np.nan)
for i in range(drou_resis_np_kndvi2_log_after2000.shape[0]):
    for j in range(drou_resis_np_kndvi2_log_after2000.shape[1]):
        if i <= j:
            drou_resis_np_kndvi2_log_after2000[i,j] = np.nan
drou_resis_np_kndvi2_log_after2000

# %%
drou_resis_np_kndvi2_log_after2000.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_kndvi2_log[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_kndvi2_log[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_kndvi2_log[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/cor_resistance_kndvi2_after2000_events_log.png', dpi = 600)

# %% [markdown]
# #### 4.1.2 偏相关

# %%
grouped_resistance_kndvi_by_bi_after2000 = df_all_kndvi_after2000.groupby("biome")
df_all_kndvi_after2000.biome.value_counts().index[:9]

# %%
df_all_kndvi_after2000.columns

# %%
kndvi_pcor_result_after2000 = biome_partial_cor(df_all_kndvi_after2000, 'fsc', 'plant_richness', 'annual_prec', 'resis_log')
kndvi_pcor_result_after2000

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_events_after2000.png', dpi = 600)


# %%
kndvi2_pcor_result_after2000 = biome_partial_cor(df_all_kndvi_after2000, 'fsc', 'plant_richness', 'annual_prec', 'kndvi2')
kndvi2_pcor_result_after2000

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi2_csc_events_after2000.png', dpi = 600)

# %%
kndvi2_log_pcor_result_after2000 = biome_partial_cor(df_all_kndvi_after2000, 'fsc', 'plant_richness', 'annual_prec', 'resis_log2')
kndvi2_log_pcor_result_after2000

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_log_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi2_csc_events_after2000.png', dpi = 600)

# %% [markdown]
# #### 4.1.3 FSC 和 LAI的偏相关

# %%
df_all_kndvi_after2000.columns

# %%
kndvi_lai_pcor_result_after2000 = biome_partial_cor(df_all_kndvi_after2000, 'fsc', 'lai_gs', 'annual_prec', 'resis_log')
kndvi_lai_pcor_result_after2000

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi_lai_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_lai_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_lai_events_after2000.png', dpi = 600)

# %%
kndvi2_lai_pcor_result_after2000 = biome_partial_cor(df_all_kndvi_after2000, 'fsc', 'lai_gs', 'annual_prec', 'kndvi2')
kndvi2_lai_pcor_result_after2000

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_lai_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_lai_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_lai_events_after2000.png', dpi = 600)

# %%
kndvi2_log_lai_pcor_result_after2000 = biome_partial_cor(df_all_kndvi_after2000, 'fsc', 'lai_gs', 'annual_prec', 'resis_log2')
kndvi2_log_lai_pcor_result_after2000

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(kndvi2_log_lai_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_lai_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use/pcor_biome_resistance_cor_pcor_kndvi_csc_lai_events_after2000.png', dpi = 600)

# %% [markdown]
# ### 4.2 fsc对 抵抗力  et sm lst的关系画图

# %%
print(df_all_kndvi_after2000.fsc.max(),df_all_kndvi_after2000.fsc.min())

# %%
df_all_kndvi_after2000['fsc_bins'] = pd.cut(df_all_kndvi_after2000.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
pd.unique(df_all_kndvi_after2000['fsc_bins'])

# %%
df_all_kndvi_after2000.plant_richness.max()

# %%
df_all_kndvi_after2000['rich_bins'] = pd.cut(df_all_kndvi_after2000.plant_richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %% [markdown]
# #### 4.2.1 抵抗力 

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 6.5, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,7)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_kndvi_log_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_kndvi_after2000.fsc_bins)
fsc_dfs = [df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,7)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_kndvi_after2000.rich_bins)
rich_dfs = [df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.resis_log[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6.5, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,7)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_kndvi_log_after2000.png', dpi = 600)


# %%
df_all_kndvi_after2000.biome.value_counts()

# %%
grouped_resistance_kndvi_by_bi_after2000 = df_all_kndvi_after2000.groupby("biome")
df_all_kndvi_after2000.biome.value_counts().index

# %%
df_all_kndvi_after2000.biome.value_counts().sort_values()

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,0,0,0,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_kndvi_log_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,0,0,0,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resis_log[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resis_log[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resis_log[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list[i]*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_kndvi_log_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [12,11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5]
y_list_low = [9,8.5,9.5,9,8.5,8,8,8,8]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.fsc[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.fsc[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.fsc[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = (y_list[i] - y_list_low[i])*0.9 + y_list_low[i], s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Forest structural complexity')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_fsc_after2000.png', dpi = 600)

# %% [markdown]
# ##### kndvi2 画图

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 1.05, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance')
axes[0].set_xticklabels([])
axes[0].set_ylim(0.4,1.09)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_kndvi2_after2000.png', dpi = 600)


# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_kndvi_after2000.fsc_bins)
fsc_dfs = [df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 1.05, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0.4,1.1)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_kndvi_after2000.rich_bins)
rich_dfs = [df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.kndvi2[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 1.05, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0.4,1.1)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_kndvi2_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [1.05,1.05,1.1,1.1,1.05,1.1,1.1,1.1,1.1]
y_list_low = [0.7,0.7,0.5,0.6,0.7,0.5,0.4,0.5,0.5]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.kndvi2[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.kndvi2[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.kndvi2[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_kndvi2_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [1.05,1.05,1.1,1.1,1.05,1.1,1.1,1.1,1.1]
y_list_low = [0.7,0.7,0.5,0.6,0.7,0.5,0.4,0.5,0.5]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.kndvi2[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.kndvi2[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.kndvi2[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list_low[i] + (y_list[i] - y_list_low[i])*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_kndvi2_after2000.png', dpi = 600)

# %% [markdown]
# ##### kndvi2 log 画图

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 6.5, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(-1,7)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.plant_richness[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_kndvi_after2000.biome)
bi_dfs = [df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_kndvi2_log_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_kndvi_after2000.fsc_bins)
fsc_dfs = [df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(-1,7)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_kndvi_after2000.rich_bins)
rich_dfs = [df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.resis_log2[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6.5, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(-1,7)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_kndvi_after2000.fsc[df_all_kndvi_after2000.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_kndvi2_log_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,-1.5,0,0,-1,-1,-1,-1]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_kndvi2_log_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [7,7,7,7,7,7,7,7,7]
y_list_low = [0,0,-1,-1,0,-1,-1,-1,-1]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi_after2000.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resis_log2[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resis_log2[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resis_log2[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list_low[i] + (y_list[i] - y_list_low[i])*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_kndvi2_log_after2000.png', dpi = 600)

# %% [markdown]
# #### 4.2.2 sm change

# %%
print(df_sm_kndvi_after2000.fsc.min(), df_sm_kndvi_after2000.fsc.max() )

# %%
df_sm_kndvi_after2000['fsc_bins'] = pd.cut(df_sm_kndvi_after2000.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_sm_kndvi_after2000.fsc_bins)
fsc_dfs = [df_sm_kndvi_after2000.sm_log[df_sm_kndvi_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_sm_kndvi_after2000.sm_log[df_sm_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_sm_kndvi_after2000.sm_log[df_sm_kndvi_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5.7, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,6.2)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/smrz_change_fsc_kndvi_after2000.png', dpi = 600)

# %%
df_sm_kndvi_after2000.biome.value_counts()
smrzchange_kndvi_fsc_bybi_after2000 = df_sm_kndvi_after2000.groupby("biome")
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.2,6.2,6.2,6.2,6.2,6.2,5.2,6.2,5.2]


for i in range(9):
    bi_n = df_sm_kndvi_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_kndvi_fsc_bybi_after2000.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_kndvi_fsc_bybi_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/smrz_change_csc_biome_kndvi_log_after2000.png', dpi = 600)

# %% [markdown]
# #### 4.2.3 et change

# %%
etchange_kndvi_use_after2000 = df_all_kndvi_after2000.copy()
etchange_kndvi_use_after2000.head()

# %%
etchange_kndvi_use_po_after2000 = etchange_kndvi_use_after2000[etchange_kndvi_use_after2000.et_change>0]
etchange_kndvi_use_ne_after2000 = etchange_kndvi_use_after2000[etchange_kndvi_use_after2000.et_change<0]

# %%
etchange_kndvi_use_po_after2000.head()

# %%
etchange_kndvi_use_ne_after2000.head()

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_after2000.fsc_bins)
fsc_dfs = [etchange_kndvi_use_after2000.et_change[etchange_kndvi_use_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_after2000.et_change[etchange_kndvi_use_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_after2000.et_change[etchange_kndvi_use_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.32, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.8,0.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_kndvi_after2000.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_po_after2000.fsc_bins)
fsc_dfs = [etchange_kndvi_use_po_after2000.et_change[etchange_kndvi_use_po_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po_after2000.et_change[etchange_kndvi_use_po_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po_after2000.et_change[etchange_kndvi_use_po_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.35, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.02,0.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_kndvi_po_after2000.png', dpi = 600)

# %%

fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_ne_after2000.fsc_bins)
fsc_dfs = [etchange_kndvi_use_ne_after2000.et_change[etchange_kndvi_use_ne_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne_after2000.et_change[etchange_kndvi_use_ne_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne_after2000.et_change[etchange_kndvi_use_ne_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.03, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.75,0.09)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_kndvi_ne_after2000.png', dpi = 600)

# %%
etchange_kndvi_use_after2000.biome.value_counts()

# %%
etchange_kndvi_fsc_bybi_po_after2000 = etchange_kndvi_use_po_after2000.groupby("biome")
etchange_kndvi_fsc_bybi_ne_after2000 = etchange_kndvi_use_ne_after2000.groupby("biome")
etchange_kndvi_fsc_bybi_after2000 = etchange_kndvi_use_after2000.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.3,0.2,0.4,0.3,0.4,0.3,0.3,0.3,0.3]

for i in range(9):
    bi_n = etchange_kndvi_use_po_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_po_after2000.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_po_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.02,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_kndvi_po_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.35,0.35,0.55,0.4,0.5,0.4,0.5,0.4,0.35]
y_list_low = [-0.65,-0.5,-0.6,-0.75,-0.5,-0.6,-0.9,-0.9,-0.75]

for i in range(9):
    bi_n = etchange_kndvi_use_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_after2000.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - y_list_low[i])*0.92 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_kndvi_use_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.1,0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
y_list_low = [-0.75, -0.4 ,-0.8, -0.6, -0.75, -0.9, -0.5, -0.9, -0.75]

for i in range(9):
    bi_n = etchange_kndvi_use_ne_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_ne_after2000.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_ne_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i]-y_list_low[i]) * 0.92 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i], y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_kndvi_ne_after2000.png', dpi = 600)

# %% [markdown]
# #### 4.2.4 lst change

# %%
lstzs_kndvi_use_after2000 = df_all_kndvi_after2000.copy()
lstzs_kndvi_use_after2000.head()

# %%
lstzs_kndvi_use_after2000['fsc_bins'] = pd.cut(lstzs_kndvi_use_after2000.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
lstzs_kndvi_use_po_after2000 = lstzs_kndvi_use_after2000[lstzs_kndvi_use_after2000.lst_zs>0]
lstzs_kndvi_use_ne_after2000 = lstzs_kndvi_use_after2000[lstzs_kndvi_use_after2000.lst_zs<0]

# %%
lstzs_kndvi_use_po_after2000

# %%
lstzs_kndvi_use_ne_after2000

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use_po_after2000.fsc_bins)
fsc_dfs = [lstzs_kndvi_use_po_after2000.lst_zs[lstzs_kndvi_use_po_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_po_after2000.lst_zs[lstzs_kndvi_use_po_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_po_after2000.lst_zs[lstzs_kndvi_use_po_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.1,4.9)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_kndvi_po_after2000.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use_ne_after2000.fsc_bins)
fsc_dfs = [lstzs_kndvi_use_ne_after2000.lst_zs[lstzs_kndvi_use_ne_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_ne_after2000.lst_zs[lstzs_kndvi_use_ne_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_ne_after2000.lst_zs[lstzs_kndvi_use_ne_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.02, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-1.8,0.1)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_kndvi_ne_after2000.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use_after2000.fsc_bins)
fsc_dfs = [lstzs_kndvi_use_after2000.lst_zs[lstzs_kndvi_use_after2000.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_after2000.lst_zs[lstzs_kndvi_use_after2000.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_after2000.lst_zs[lstzs_kndvi_use_after2000.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-2.5,5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_kndvi_use_after2000.png', dpi = 600)

# %%
lstzs_kndvi_fsc_bybi_po_after2000 = lstzs_kndvi_use_po_after2000.groupby("biome")
lstzs_kndvi_fsc_bybi_ne_after2000 = lstzs_kndvi_use_ne_after2000.groupby("biome")
lstzs_kndvi_fsc_bybi_after2000 = lstzs_kndvi_use_after2000.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5,6,5,4.5,4,5.5,6.5,4.5,4]

for i in range(9):
    bi_n = lstzs_kndvi_use_po_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_po_after2000.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_po_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.2,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_kndvi_po_after2000.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,6,5,4.5,5,6,6.5,4.5,5]
y_list_low = [-2.5,-2.1,-2.1,-2.1,-2.5,-2,-2.5,-2,-2]


for i in range(9):
    bi_n = lstzs_kndvi_use_after2000.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_after2000.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - y_list_low[i]) * 0.9 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_kndvi_use_after2000.png', dpi = 600)

# %%
lstzs_kndvi_use_ne_after2000.biome.value_counts()

# %%
fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [2,2,2]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstzs_kndvi_use_ne_after2000.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_ne_after2000.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_ne_after2000.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = 0.05, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 0.2)
    axes[i].set_xlim(8.7,11.3)
    #axes[i].set_xticks(np.arange(9,11.3,0.5),labels = [])
    axes[i].grid(c = 'lightgray', alpha = 0.3)

    if i >= 0 :
        axes[i].set_xlabel('Forest structural complexity')
        axes[i].set_xticks(np.arange(9,11.3,0.5),labels = np.arange(9,11.3,0.5))
    if (i%3) == 0 :
        axes[i].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_kndvi_ne_after2000.png', dpi = 600)

# %% [markdown]
# ## 5 sif

# %% [markdown]
# ### 5.1 先做相关

# %% [markdown]
# #### 5.1.1 因子间的简单相关

# %%
drou_resis_cor_sif = df_all_sif.drop(['lon', 'lat', 'year', 'sif', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log','resis_log2','sif2','spei','fsc','plant_richness','lai_gs','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_sif

# %%
drou_resis_np_sif = np.asarray(drou_resis_cor_sif)
drou_resis_np_sif

# %%
p_value_sif = np.full_like(drou_resis_np_sif, fill_value=np.nan)
p_value_sif

# %%
var_name = ['Drought resistance','Drought resistance2_log','Drought resistance2','SPEI','Forest structural complexity','Tree species richness','LAI','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
var_name_raw = ['resis_log','resis_log2','sif2','spei','fsc','plant_richness','lai_gs','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

# %%
for i in range(drou_resis_np_sif.shape[0]):
    for j in range(drou_resis_np_sif.shape[1]):
        
        p_value_sif[i,j] = spearmanr(df_all_sif[var_name_raw[i]],df_all_sif[var_name_raw[j]])[1]
p_value_sif

# %%
drou_resis_np_sif = np.where(p_value_sif<0.001,drou_resis_np_sif, np.nan)
for i in range(drou_resis_np_sif.shape[0]):
    for j in range(drou_resis_np_sif.shape[1]):
        if i <= j:
            drou_resis_np_sif[i,j] = np.nan
drou_resis_np_sif

# %%
drou_resis_np_sif.shape

# %%
plt.rc('font',family='Times New Roman', size = 15)
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_sif[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(17))
ax.set_yticks(ticks=np.arange(17))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,18):
    for j in range(17):
        if ~np.isnan(drou_resis_np_sif[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_sif[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/cor_resistance_sif_csc_sif_events.png', dpi = 600)

# %% [markdown]
# ##### sif2 画图

# %%
df_all_sif.columns

# %%
drou_resis_cor_sif2 = df_all_sif.drop(['lon', 'lat', 'year', 'sif', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['sif2','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_sif2

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drou_resis_np_sif2 = np.asarray(drou_resis_cor_sif2)
drou_resis_np_sif2

# %%
p_value_sif2 = np.full_like(drou_resis_np_sif2, fill_value=np.nan)
p_value_sif2

# %%
var_name_raw = ['sif2','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_sif2.shape[0]):
    for j in range(drou_resis_np_sif2.shape[1]):
        
        p_value_sif2[i,j] = spearmanr(df_all_sif[var_name_raw[i]],df_all_sif[var_name_raw[j]])[1]
p_value_sif2

# %%
drou_resis_np_sif2 = np.where(p_value_sif2<0.001,drou_resis_np_sif2, np.nan)
for i in range(drou_resis_np_sif2.shape[0]):
    for j in range(drou_resis_np_sif2.shape[1]):
        if i <= j:
            drou_resis_np_sif2[i,j] = np.nan
drou_resis_np_sif2

# %%
drou_resis_np_sif2.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_sif2[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_sif2[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_sif2[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/cor_resistance_sif2_events.png', dpi = 600)

# %% [markdown]
# ##### sif2 log画图

# %%
df_all_sif.columns

# %%
drou_resis_cor_sif2_log = df_all_sif.drop(['lon', 'lat', 'year', 'sif', 'biome','sm_change', 'et_change','lst_zs',],
                                         axis=1)[['resis_log2','spei','fsc','plant_richness','annual_temp','annual_prec',
                                                  'ai_index','drought_count','drought_duration','drought_severity',
                                                  'soil_cec','soil_clay','cti','sla','wood_den']].corr(method='spearman')
drou_resis_cor_sif2_log

# %%
var_name = ['Drought resistance','SPEI','Forest structural complexity','Tree species richness','Mean annual temperature','Mean annual precipitation','Aridity index',
            'Drought counts','Mean drought duration','Mean drought severity','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drou_resis_np_sif2_log = np.asarray(drou_resis_cor_sif2_log)
drou_resis_np_sif2_log

# %%
p_value_sif2_log = np.full_like(drou_resis_np_sif2_log, fill_value=np.nan)
p_value_sif2_log

# %%
var_name_raw = ['resis_log2','spei','fsc','plant_richness','annual_temp','annual_prec',
                'ai_index','drought_count','drought_duration','drought_severity',
                'soil_cec','soil_clay','cti','sla','wood_den']

for i in range(drou_resis_np_sif2_log.shape[0]):
    for j in range(drou_resis_np_sif2_log.shape[1]):
        
        p_value_sif2_log[i,j] = spearmanr(df_all_sif[var_name_raw[i]],df_all_sif[var_name_raw[j]])[1]
p_value_sif2_log

# %%
drou_resis_np_sif2_log = np.where(p_value_sif2_log<0.001,drou_resis_np_sif2_log, np.nan)
for i in range(drou_resis_np_sif2_log.shape[0]):
    for j in range(drou_resis_np_sif2_log.shape[1]):
        if i <= j:
            drou_resis_np_sif2_log[i,j] = np.nan
drou_resis_np_sif2_log

# %%
drou_resis_np_sif2_log.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drou_resis_np_sif2_log[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(14))
ax.set_yticks(ticks=np.arange(14))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,15):
    for j in range(14):
        if ~np.isnan(drou_resis_np_sif2_log[i,j]):
            ax.text(j, i-1, str(round(drou_resis_np_sif2_log[i,j],3)), size = 11,ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/cor_resistance_sif2_log_events.png', dpi = 600)

# %% [markdown]
# #### 5.1.2 偏相关

# %%
grouped_resistance_sif_by_bi = df_all_sif.groupby("biome")

# %%
df_all_sif.biome.value_counts().index[:9]

# %%
df_all_sif.columns

# %%
'''
bi_cor_fsc= {}
bi_cor_rich = {}
for bi_n in df_all_sif.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[1]}
    bi_cor_rich[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['plant_richness'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['plant_richness'])[1]}
'''

# %%
#bi_cor_fsc

# %%
#bi_cor_rich

# %%
#stats.spearmanr(df_all_sif['resis_log'],df_all_sif['fsc'])

# %%
#stats.spearmanr(df_all_sif['resis_log'],df_all_sif['plant_richness'])

# %%
'''
bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df
'''

# %%
'''
bi_pd_rich_cor_df = pd.DataFrame(bi_cor_rich).T
bi_pd_rich_cor_df['var'] = 'rich'
bi_pd_rich_cor_df = bi_pd_rich_cor_df.sort_index()
bi_pd_rich_cor_df
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.453)
rich_cor_list = list(bi_pd_rich_cor_df.r)
rich_cor_list.append(0.350)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)
bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
rich_cor_list = list(bi_pd_rich_cor_df['p-val'])
rich_cor_list.append(0)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)
bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p
'''

# %%
'''
bi_pd_fsc_rich_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_rich_fsc_pcor = {}
bi_pd_rich_prec_pcor = {}
for bi_n in df_all_sif.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['plant_richness'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['annual_prec'],method='spearman').round(4)
    rich_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='plant_richness',covar=['fsc'],method='spearman').round(4)
    rich_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='plant_richness',covar=['annual_prec'],method='spearman').round(4)
    bi_pd_fsc_rich_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_rich_fsc_pcor[bi_n] = {'r':rich_pcor['r'].values[0], 'p-val': rich_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_rich_prec_pcor[bi_n] = {'r':rich_prec_pcor['r'].values[0], 'p-val': rich_prec_pcor['p-val'].values[0]}
'''

# %%
#bi_pd_fsc_rich_pcor

# %%
#bi_pd_fsc_prec_pcor

# %%
#bi_pd_rich_fsc_pcor

# %%
#bi_pd_rich_prec_pcor

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='plant_richness',covar=['annual_prec'], method='spearman').round(3)

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='plant_richness',covar=['fsc'], method='spearman').round(3)

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='fsc',covar=['annual_prec'], method='spearman').round(3)

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='fsc',covar=['plant_richness'], method='spearman').round(3)

# %%
'''
bi_pd_fsc_rich_pcor_df = pd.DataFrame(bi_pd_fsc_rich_pcor).T
bi_pd_fsc_rich_pcor_df['var'] = 'fsc_rich'
bi_pd_fsc_rich_pcor_df = bi_pd_fsc_rich_pcor_df.sort_index()
bi_pd_fsc_rich_pcor_df
'''

# %%
'''
bi_pd_rich_fsc_pcor_df = pd.DataFrame(bi_pd_rich_fsc_pcor).T
bi_pd_rich_fsc_pcor_df['var'] = 'rich_fsc'
bi_pd_rich_fsc_pcor_df = bi_pd_rich_fsc_pcor_df.sort_index()
bi_pd_rich_fsc_pcor_df
'''

# %%
'''
bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df
'''

# %%
'''
bi_pd_rich_prec_pcor_df = pd.DataFrame(bi_pd_rich_prec_pcor).T
bi_pd_rich_prec_pcor_df['var'] = 'rich_prec'
bi_pd_rich_prec_pcor_df = bi_pd_rich_prec_pcor_df.sort_index()
bi_pd_rich_prec_pcor_df
'''

# %%
'''
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df.r)
fsc_rich_list.append(0.444)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.292)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df.r)
rich_fsc_list.append(0.337)
rich_prec_list = list(bi_pd_rich_prec_pcor_df.r)
rich_prec_list.append(-0.11)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)
bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pcor_df
'''

# %%
'''
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df['p-val'])
fsc_rich_list.append(0)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df['p-val'])
fsc_prec_list.append(0)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df['p-val'])
rich_fsc_list.append(0)
rich_prec_list = list(bi_pd_rich_prec_pcor_df['p-val'])
rich_prec_list.append(0)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)
bi_pval_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pval_df
'''

# %%
'''
bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
bi_pd_pcor_all
'''

# %%
'''
bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
bi_pd_pval_all
'''

# %%
'''
bi_pd_pcor_all.to_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_sif_mid_event.csv', index = False)
bi_pd_pval_all.to_csv('E:/python_output/fsc_drought/bi_pd_pval_all_sif_mid_event.csv', index = False)
'''

# %%
sif_pcor_result = biome_partial_cor(df_all_sif, 'fsc', 'plant_richness', 'annual_prec', 'resis_log')
sif_pcor_result

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(sif_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in sif_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/pcor_biome_resistance_cor_pcor_sif_csc_events.png', dpi = 600)

# %%
sif2_pcor_result = biome_partial_cor(df_all_sif, 'fsc', 'plant_richness', 'annual_prec', 'sif2')
sif2_pcor_result

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(sif2_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in sif2_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/pcor_biome_resistance_cor_pcor_sif_csc_events.png', dpi = 600)

# %%
sif2_log_pcor_result = biome_partial_cor(df_all_sif, 'fsc', 'plant_richness', 'annual_prec', 'resis_log2')
sif2_log_pcor_result

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(sif2_log_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in sif2_log_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/pcor_biome_resistance_cor_pcor_sif_csc_events.png', dpi = 600)

# %% [markdown]
# #### 5.1.3 fsc 和 lai的偏相关

# %%
'''
bi_cor_fsc= {}
bi_cor_lai = {}
for bi_n in df_all_sif.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['fsc'])[1]}
    bi_cor_lai[bi_n] = {'r':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['lai_gs'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resis_log'],bi_group_df['lai_gs'])[1]}
bi_cor_fsc
'''

# %%
#bi_cor_lai

# %%
#stats.spearmanr(df_all_sif['resis_log'],df_all_sif['fsc'])

# %%
#stats.spearmanr(df_all_sif['resis_log'],df_all_sif['lai_gs'])

# %%
'''
bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df
'''

# %%
'''
bi_pd_lai_cor_df = pd.DataFrame(bi_cor_lai).T
bi_pd_lai_cor_df['var'] = 'lai'
bi_pd_lai_cor_df = bi_pd_lai_cor_df.sort_index()
bi_pd_lai_cor_df
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.452)
lai_cor_list = list(bi_pd_lai_cor_df.r)
lai_cor_list.append(0.584)
bi_cor_list = list(bi_pd_lai_cor_df.index)
bi_cor_list.append(0)
bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'lai':lai_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r
'''

# %%
'''
fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
lai_cor_list = list(bi_pd_lai_cor_df['p-val'])
lai_cor_list.append(0)
bi_cor_list = list(bi_pd_lai_cor_df.index)
bi_cor_list.append(0)
bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'lai':lai_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p
'''

# %%
'''
bi_pd_fsc_lai_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_lai_fsc_pcor = {}
bi_pd_lai_prec_pcor = {}
for bi_n in df_all_sif.biome.value_counts().index[:9]:
    print(bi_n)
    bi_group_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['lai_gs'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['annual_prec'],method='spearman').round(4)
    lai_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='lai_gs',covar=['fsc'],method='spearman').round(4)
    lai_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='lai_gs',covar=['annual_prec'],method='spearman').round(4)
    bi_pd_fsc_lai_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_lai_fsc_pcor[bi_n] = {'r':lai_pcor['r'].values[0], 'p-val': lai_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_lai_prec_pcor[bi_n] = {'r':lai_prec_pcor['r'].values[0], 'p-val': lai_prec_pcor['p-val'].values[0]}
bi_pd_fsc_lai_pcor
'''

# %%
#bi_pd_fsc_prec_pcor

# %%
#bi_pd_lai_fsc_pcor

# %%
#bi_pd_lai_prec_pcor

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='lai_gs',covar=['annual_prec'], method='spearman').round(3)

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='lai_gs',covar=['fsc'], method='spearman').round(3)

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='fsc',covar=['annual_prec'], method='spearman').round(3)

# %%
#pg.partial_corr(data = df_all_sif,y='resis_log',x='fsc',covar=['lai_gs'], method='spearman').round(3)

# %%
'''
bi_pd_fsc_lai_pcor_df = pd.DataFrame(bi_pd_fsc_lai_pcor).T
bi_pd_fsc_lai_pcor_df['var'] = 'fsc_lai'
bi_pd_fsc_lai_pcor_df = bi_pd_fsc_lai_pcor_df.sort_index()
bi_pd_fsc_lai_pcor_df
'''

# %%
'''
bi_pd_lai_fsc_pcor_df = pd.DataFrame(bi_pd_lai_fsc_pcor).T
bi_pd_lai_fsc_pcor_df['var'] = 'lai_fsc'
bi_pd_lai_fsc_pcor_df = bi_pd_lai_fsc_pcor_df.sort_index()
bi_pd_lai_fsc_pcor_df
'''

# %%
'''
bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df
'''

# %%
'''
bi_pd_lai_prec_pcor_df = pd.DataFrame(bi_pd_lai_prec_pcor).T
bi_pd_lai_prec_pcor_df['var'] = 'lai_prec'
bi_pd_lai_prec_pcor_df = bi_pd_lai_prec_pcor_df.sort_index()
bi_pd_lai_prec_pcor_df
'''

# %%
'''
fsc_lai_list = list(bi_pd_fsc_lai_pcor_df.r)
fsc_lai_list.append(0.152)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.29)
lai_fsc_list = list(bi_pd_lai_fsc_pcor_df.r)
lai_fsc_list.append(0.337)
lai_prec_list = list(bi_pd_lai_prec_pcor_df.r)
lai_prec_list.append(0.438)
bi_pcor_list = list(bi_pd_lai_prec_pcor_df.index)
bi_pcor_list.append(0)
bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_lai':fsc_lai_list,'lai_prec':lai_prec_list, 'lai_fsc':lai_fsc_list, 'biome':bi_pcor_list})
bi_pcor_df
'''

# %%
'''
fsc_lai_list = list(bi_pd_fsc_lai_pcor_df['p-val'])
fsc_lai_list.append(0)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df['p-val'])
fsc_prec_list.append(0)
lai_fsc_list = list(bi_pd_lai_fsc_pcor_df['p-val'])
lai_fsc_list.append(0)
lai_prec_list = list(bi_pd_lai_prec_pcor_df['p-val'])
lai_prec_list.append(0)
bi_pcor_list = list(bi_pd_lai_prec_pcor_df.index)
bi_pcor_list.append(0)
bi_pval_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_lai':fsc_lai_list,'lai_prec':lai_prec_list, 'lai_fsc':lai_fsc_list, 'biome':bi_pcor_list})
bi_pval_df
'''

# %%
#bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
#bi_pd_pcor_all

# %%
#bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
#bi_pd_pval_all

# %%
'''
draw_col = ['lai','lai_prec','lai_fsc','fsc','fsc_prec','fsc_lai']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(bi_pd_pcor_all[draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in bi_pd_pcor_all.biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if bi_pd_pval_all[draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif bi_pd_pval_all[draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif bi_pd_pval_all[draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Tree species lainess',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()
'''

# %%
sif_lai_pcor_result = biome_partial_cor(df_all_sif, 'fsc', 'lai_gs', 'annual_prec', 'resis_log')
sif_lai_pcor_result

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(sif_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in sif_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/pcor_biome_resistance_cor_pcor_sif_lai_csc_events.png', dpi = 600)

# %%
sif2_lai_pcor_result = biome_partial_cor(df_all_sif, 'fsc', 'lai_gs', 'annual_prec', 'sif2')
sif2_lai_pcor_result

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(sif2_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in sif2_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/pcor_biome_resistance_cor_pcor_sif_lai_csc_events.png', dpi = 600)

# %%
sif2_log_lai_pcor_result = biome_partial_cor(df_all_sif, 'fsc', 'lai_gs', 'annual_prec', 'resis_log2')
sif2_log_lai_pcor_result

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(sif2_log_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in sif2_log_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 10.1, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 10.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.05)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

#plt.savefig('result_figure/figure_use_20251212/pcor_biome_resistance_cor_pcor_sif_lai_csc_events.png', dpi = 600)

# %% [markdown]
# ### 5.2 fsc 对抵抗力 et sm lst

# %%
print(df_all_sif.fsc.max(),df_all_sif.fsc.min())

# %%
df_all_sif['fsc_bins'] = pd.cut(df_all_sif.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
pd.unique(df_all_sif['fsc_bins'])

# %%
df_all_sif.plant_richness.max()

# %%
df_all_sif['rich_bins'] = pd.cut(df_all_sif.plant_richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %% [markdown]
# #### 5.2.1 抵抗力

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.resis_log[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.resis_log[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.resis_log[df_all_sif.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,6.5)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.fsc[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_sif_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_sif.fsc_bins)
fsc_dfs = [df_all_sif.resis_log[df_all_sif.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_sif.resis_log[df_all_sif.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_sif.resis_log[df_all_sif.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(-0.1,7)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_sif.rich_bins)
rich_dfs = [df_all_sif.resis_log[df_all_sif.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_sif.resis_log[df_all_sif.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_sif.resis_log[df_all_sif.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6.5, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(-0.1,7)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_sif_log.png', dpi = 600)

# %%
df_all_sif.biome.value_counts()

# %%
grouped_resistance_sif_by_bi = df_all_sif.groupby("biome")

# %%
df_all_sif.biome.value_counts().index

# %%
df_all_sif.biome.value_counts().sort_values()

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,5.5]
y_list_low = [0,0,0,0,0,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resis_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_sif_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.5,6.5,6.5,6.5,6.5,6.5,6.5,6.5,5.5]
y_list_low = [0,0,0,0,0,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resis_log[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resis_log[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resis_log[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list[i]*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_sif_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [12,11.5,11.5,11.5,11.5,11.5,12,11.5,11.5]
y_list_low = [9,8.5,9.5,9,8.5,8,8.5,8,8]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.fsc[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.fsc[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.fsc[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = (y_list[i] - y_list_low[i])*0.9 + y_list_low[i], s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Forest structural complexity')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_fsc_sif_log.png', dpi = 600)

# %% [markdown]
# ##### sif2 画图

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.sif2[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.sif2[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.sif2[df_all_sif.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 1.05, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance')
axes[0].set_xticklabels([])
axes[0].set_ylim(0.35,1.1)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.fsc[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_sif2.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_sif.fsc_bins)
fsc_dfs = [df_all_sif.sif2[df_all_sif.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_sif.sif2[df_all_sif.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_sif.sif2[df_all_sif.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 1.05, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0.35,1.1)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_sif.rich_bins)
rich_dfs = [df_all_sif.sif2[df_all_sif.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_sif.sif2[df_all_sif.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_sif.sif2[df_all_sif.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 1.05, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0.35,1.1)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_sif2.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [1.08,1.08,1.1,1.1,1.08,1.1,1.1,1.1,1.1]
y_list_low = [0.6,0.6,0.5,0.4,0.6,0.4,0.3,0.4,0.3]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.sif2[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.sif2[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.sif2[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_sif2.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [1.06,1.07,1.1,1.1,1.08,1.1,1.1,1.1,1.1]
y_list_low = [0.7,0.6,0.5,0.4,0.6,0.4,0.1,0.4,0.3]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.sif2[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.sif2[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.sif2[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list_low[i]+ ( y_list[i] - y_list_low[i])*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_sif2.png', dpi = 600)

# %% [markdown]
# ##### sif2 log 画图

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.resis_log2[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.resis_log2[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.resis_log2[df_all_sif.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(-1.5,6.5)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.plant_richness[df_all_sif.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Tree species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(df_all_sif.biome)
bi_dfs = [df_all_sif.fsc[df_all_sif.biome == labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(df_all_sif.fsc[df_all_sif.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.7, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/figure_use_20260105/resistance_biome_sif2_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_all_sif.fsc_bins)
fsc_dfs = [df_all_sif.resis_log2[df_all_sif.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_all_sif.resis_log2[df_all_sif.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_all_sif.resis_log2[df_all_sif.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(-1,6.5)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(df_all_sif.rich_bins)
rich_dfs = [df_all_sif.resis_log2[df_all_sif.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_sif.resis_log2[df_all_sif.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_sif.resis_log2[df_all_sif.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Tree species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(-1,6.5)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(df_all_sif.fsc[df_all_sif.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Tree species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/figure_use_20260105/richness_fsc_resistance_sif2_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6]
y_list_low = [-0.5,-0.5,-1,-1,-0.5,-1,-1.5,-1,-1]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resis_log2[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list_low[i] + (y_list[i]-y_list_low[i])*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

axes[2, 2].set_xlabel('Forest structural complexity')
#axes[2,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/fsc_resistance_biome_sif2_log.png', dpi = 600)


# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6]
y_list_low = [0,-0.5,-1.2,-1,-0.5,-1,-1.6,-1,-1]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = df_all_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resis_log2[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resis_log2[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resis_log2[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list_low[i]+ ( y_list[i] - y_list_low[i])*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance (log)')

plt.tight_layout()

fig.savefig('result_figure/figure_use_20260105/richness_resistance_biome_sif2_log.png', dpi = 600)

# %% [markdown]
# #### 5.2.2 sm change

# %%
df_sm_sif['fsc_bins'] = pd.cut(df_sm_sif.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(df_sm_sif.fsc_bins)
fsc_dfs = [df_sm_sif.sm_log[df_sm_sif.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(df_sm_sif.sm_log[df_sm_sif.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(df_sm_sif.sm_log[df_sm_sif.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5.7, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,6.2)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/smrz_change_fsc_sif.png', dpi = 600)

# %%
df_sm_sif.biome.value_counts()

# %%
smrzchange_sif_fsc_bybi = df_sm_sif.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2,6.2]


for i in range(9):
    bi_n = df_sm_sif.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.sm_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/smrz_change_csc_biome_sif_log.png', dpi = 600)

# %% [markdown]
# #### 5.2.3 et change

# %%
etchange_sif_use = df_all_sif.copy()
etchange_sif_use.head()

# %%
etchange_sif_use_po = etchange_sif_use[etchange_sif_use.et_change>0]
etchange_sif_use_ne = etchange_sif_use[etchange_sif_use.et_change<0]

etchange_sif_use_po.head()

# %%
etchange_sif_use_ne.head()

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_sif_use.fsc_bins)
fsc_dfs = [etchange_sif_use.et_change[etchange_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_sif_use.et_change[etchange_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_sif_use.et_change[etchange_sif_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.32, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.8,0.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_sif.png', dpi = 600)


# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_sif_use_po.fsc_bins)
fsc_dfs = [etchange_sif_use_po.et_change[etchange_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_sif_use_po.et_change[etchange_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_sif_use_po.et_change[etchange_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.35, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.02,0.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_sif_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange_sif_use_ne.fsc_bins)
fsc_dfs = [etchange_sif_use_ne.et_change[etchange_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_sif_use_ne.et_change[etchange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_sif_use_ne.et_change[etchange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.03, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Tr change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.75,0.09)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/et_change_fsc_sif_ne.png', dpi = 600)

# %%
etchange_sif_use.biome.value_counts()

# %%
etchange_sif_fsc_bybi_po = etchange_sif_use_po.groupby("biome")
etchange_sif_fsc_bybi_ne = etchange_sif_use_ne.groupby("biome")
etchange_sif_fsc_bybi = etchange_sif_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.3,0.2,0.3,0.3,0.4,0.3,0.3,0.5,0.3]

for i in range(9):
    bi_n = etchange_sif_use_po.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(etchange_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.02,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_sif_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.35,0.3,0.52,0.4,0.45,0.45,0.35,0.4,0.35]
y_list_low = [-0.65,-0.5,-0.6,-0.75,-0.6,-0.75,-0.75,-0.9,-0.75]

for i in range(9):
    bi_n = etchange_sif_use.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(etchange_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - y_list_low[i])*0.92 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_sif_use.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [0.1,0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
y_list_low = [-0.7, -0.4 ,-0.8, -0.8, -0.75, -0.9, -0.5, -0.9, -0.75]

for i in range(9):
    bi_n = etchange_sif_use_ne.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(etchange_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i]-y_list_low[i]) * 0.92 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i], y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Tr change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/et_change_csc_biome_sif_ne.png', dpi = 600)

# %% [markdown]
# #### 5.2.4 lst change

# %%
lstzs_sif_use = df_all_sif.copy()
lstzs_sif_use.head()

# %%
lstzs_sif_use['fsc_bins'] = pd.cut(lstzs_sif_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
lstzs_sif_use_po = lstzs_sif_use[lstzs_sif_use.lst_zs>0]
lstzs_sif_use_ne = lstzs_sif_use[lstzs_sif_use.lst_zs<0]
lstzs_sif_use_po

# %%
lstzs_sif_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_sif_use_po.fsc_bins)
fsc_dfs = [lstzs_sif_use_po.lst_zs[lstzs_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_sif_use_po.lst_zs[lstzs_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_sif_use_po.lst_zs[lstzs_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.1,4.9)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_sif_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_sif_use_ne.fsc_bins)
fsc_dfs = [lstzs_sif_use_ne.lst_zs[lstzs_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_sif_use_ne.lst_zs[lstzs_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_sif_use_ne.lst_zs[lstzs_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 0.04, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-2,0.18)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_sif_ne.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_sif_use.fsc_bins)
fsc_dfs = [lstzs_sif_use.lst_zs[lstzs_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_sif_use.lst_zs[lstzs_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_sif_use.lst_zs[lstzs_sif_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-2.2,5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/figure_use_20251212/lst_zs_fsc_sif_use.png', dpi = 600)

# %%
lstzs_sif_fsc_bybi_po = lstzs_sif_use_po.groupby("biome")
lstzs_sif_fsc_bybi_ne = lstzs_sif_use_ne.groupby("biome")
lstzs_sif_fsc_bybi = lstzs_sif_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5,6,4.8,4.5,5.5,6.5,6,4.5,4.5]

for i in range(9):
    bi_n = lstzs_sif_use_po.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstzs_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.2,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_sif_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.8,5,4.5,5,5.8,7,4.5,5]
y_list_low = [-2.5,-2.1,-2.1,-2.1,-2.5,-2,-2,-2,-2]


for i in range(9):
    bi_n = lstzs_sif_use.biome.value_counts().index[:9].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(lstzs_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - y_list_low[i]) * 0.9 + y_list_low[i], s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list_low[i],y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_sif_use.png', dpi = 600)

# %%
lstzs_sif_use_ne.biome.value_counts()
fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [2.5,2,2]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstzs_sif_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstzs_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_zs[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = 0.05, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 0.2)
    axes[i].set_xlim(8.7,11.3)
    #axes[i].set_xticks(np.arange(9,11.3,0.5),labels = [])
    axes[i].grid(c = 'lightgray', alpha = 0.3)

    if i >= 0 :
        axes[i].set_xlabel('Forest structural complexity')
        axes[i].set_xticks(np.arange(9,11.3,0.5),labels = np.arange(9,11.3,0.5))
    if (i%3) == 0 :
        axes[i].set_ylabel('LST change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/figure_use_20251212/lst_zs_csc_biome_sif_ne.png', dpi = 600)

# %% [markdown]
# ## 6 其他一些图

# %% [markdown]
# ### 6.1 hexplot

# %%
##  kndvi 和 sif
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(df_all_kndvi.plant_richness, df_all_kndvi.fsc, 
                C= df_all_kndvi.resis_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(df_all_kndvi.annual_prec), df_all_kndvi.plant_richness, 
                C= df_all_kndvi.resis_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(df_all_kndvi.annual_prec), df_all_kndvi.fsc, 
                C= df_all_kndvi.resis_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='both', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[2,3,4,5])

axes[1,0].hexbin(df_all_sif.plant_richness, df_all_sif.fsc, 
                C= df_all_sif.resis_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Tree species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.plant_richness, 
                C= df_all_sif.resis_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.fsc, 
                C= df_all_sif.resis_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,2].set_xlabel('Annual precipitation (log) / mm')
axes[1,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,2].set_title('(f)', loc='left', size = 14)

position2=fig.add_axes([0.9,0.12,0.01,0.3])
plt.colorbar(im, position2, extend='both', label = 'Resistance (SIF)',orientation='vertical',ticks=[2,3,4])


plt.subplots_adjust(top=0.93, bottom=0.1, right=0.88, left=0.06, wspace=0.27)

plt.savefig('result_figure/figure_use_20260105/resistance_richness_fsc_prec_kndvi&sif_event.png', dpi = 600)

# %%
##  kndvi after2000 和 sif
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(df_all_kndvi_after2000.plant_richness, df_all_kndvi_after2000.fsc, 
                C= df_all_kndvi_after2000.resis_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(df_all_kndvi_after2000.annual_prec), df_all_kndvi_after2000.plant_richness, 
                C= df_all_kndvi_after2000.resis_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(df_all_kndvi_after2000.annual_prec), df_all_kndvi_after2000.fsc, 
                C= df_all_kndvi_after2000.resis_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='both', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[2,3,4,5])

axes[1,0].hexbin(df_all_sif.plant_richness, df_all_sif.fsc, 
                C= df_all_sif.resis_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Tree species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.plant_richness, 
                C= df_all_sif.resis_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.fsc, 
                C= df_all_sif.resis_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,2].set_xlabel('Annual precipitation (log) / mm')
axes[1,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,2].set_title('(f)', loc='left', size = 14)

position2=fig.add_axes([0.9,0.12,0.01,0.3])
plt.colorbar(im, position2, extend='both', label = 'Resistance (SIF)',orientation='vertical',ticks=[2,3,4])


plt.subplots_adjust(top=0.93, bottom=0.1, right=0.88, left=0.06, wspace=0.27)

plt.savefig('result_figure/figure_use_20260105/resistance_richness_fsc_prec_kndvi&sif_event_after2000.png', dpi = 600)


# %%
##  kndvi2 和 sif2
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(df_all_kndvi.plant_richness, df_all_kndvi.fsc, 
                C= df_all_kndvi.kndvi2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(df_all_kndvi.annual_prec), df_all_kndvi.plant_richness, 
                C= df_all_kndvi.kndvi2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(df_all_kndvi.annual_prec), df_all_kndvi.fsc, 
                C= df_all_kndvi.kndvi2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='min', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[0.75,0.8,0.85,0.9,0.95])


axes[1,0].hexbin(df_all_sif.plant_richness, df_all_sif.fsc, 
                C= df_all_sif.sif2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Tree species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.plant_richness, 
                C= df_all_sif.sif2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.fsc, 
                C= df_all_sif.sif2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[1,2].set_xlabel('Annual precipitation (log) / mm')
axes[1,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,2].set_title('(f)', loc='left', size = 14)

position2=fig.add_axes([0.9,0.12,0.01,0.3])
plt.colorbar(im, position2, extend='min', label = 'Resistance (SIF)',orientation='vertical',ticks=[0.75,0.8,0.85,0.9,0.95])


plt.subplots_adjust(top=0.93, bottom=0.1, right=0.88, left=0.06, wspace=0.27)

plt.savefig('result_figure/figure_use_20260105/resistance_richness_fsc_prec_kndvi2&sif2_event.png', dpi = 600)

# %%
##  kndvi2 after2000 和 sif2
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(df_all_kndvi_after2000.plant_richness, df_all_kndvi_after2000.fsc, 
                C= df_all_kndvi_after2000.kndvi2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(df_all_kndvi_after2000.annual_prec), df_all_kndvi_after2000.plant_richness, 
                C= df_all_kndvi_after2000.kndvi2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(df_all_kndvi_after2000.annual_prec), df_all_kndvi_after2000.fsc, 
                C= df_all_kndvi_after2000.kndvi2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='min', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[0.75,0.8,0.85,0.9,0.95])

axes[1,0].hexbin(df_all_sif.plant_richness, df_all_sif.fsc, 
                C= df_all_sif.sif2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Tree species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.plant_richness, 
                C= df_all_sif.sif2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.fsc, 
                C= df_all_sif.sif2, gridsize = 100, 
                vmax=0.98, vmin=0.75,
                reduce_C_function = np.median)
axes[1,2].set_xlabel('Annual precipitation (log) / mm')
axes[1,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,2].set_title('(f)', loc='left', size = 14)

position2=fig.add_axes([0.9,0.12,0.01,0.3])
plt.colorbar(im, position2, extend='min', label = 'Resistance (SIF)',orientation='vertical',ticks=[0.75,0.8,0.85,0.9,0.95])


plt.subplots_adjust(top=0.93, bottom=0.1, right=0.88, left=0.06, wspace=0.27)

plt.savefig('result_figure/figure_use_20260105/resistance_richness_fsc_prec_kndvi2&sif2_event_after2000.png', dpi = 600)

# %%
##  kndvi2  和 sif2  logit
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(df_all_kndvi.plant_richness, df_all_kndvi.fsc, 
                C= df_all_kndvi.resis_log2, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(df_all_kndvi.annual_prec), df_all_kndvi.plant_richness, 
                C= df_all_kndvi.resis_log2, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(df_all_kndvi.annual_prec), df_all_kndvi.fsc, 
                C= df_all_kndvi.resis_log2, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='both', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[1,2,3,4,5])

axes[1,0].hexbin(df_all_sif.plant_richness, df_all_sif.fsc, 
                C= df_all_sif.resis_log2, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Tree species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.plant_richness, 
                C= df_all_sif.resis_log2, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.fsc, 
                C= df_all_sif.resis_log2, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1,2].set_xlabel('Annual precipitation (log) / mm')
axes[1,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,2].set_title('(f)', loc='left', size = 14)

position2=fig.add_axes([0.9,0.12,0.01,0.3])
plt.colorbar(im, position2, extend='both', label = 'Resistance (SIF)',orientation='vertical',ticks=[1,2,3,4])


plt.subplots_adjust(top=0.93, bottom=0.1, right=0.88, left=0.06, wspace=0.27)

plt.savefig('result_figure/figure_use_20260105/resistance_richness_fsc_prec_kndvi2&sif2_log_event.png', dpi = 600)

# %%
##  kndvi2 after2000 和 sif2  logit
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(df_all_kndvi_after2000.plant_richness, df_all_kndvi_after2000.fsc, 
                C= df_all_kndvi_after2000.resis_log2, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(df_all_kndvi_after2000.annual_prec), df_all_kndvi_after2000.plant_richness, 
                C= df_all_kndvi_after2000.resis_log2, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(df_all_kndvi_after2000.annual_prec), df_all_kndvi_after2000.fsc, 
                C= df_all_kndvi_after2000.resis_log2, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='both', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[1,2,3,4,5])

axes[1,0].hexbin(df_all_sif.plant_richness, df_all_sif.fsc, 
                C= df_all_sif.resis_log2, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Tree species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.plant_richness, 
                C= df_all_sif.resis_log2, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Tree species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(df_all_sif.annual_prec), df_all_sif.fsc, 
                C= df_all_sif.resis_log2, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1,2].set_xlabel('Annual precipitation (log) / mm')
axes[1,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,2].set_title('(f)', loc='left', size = 14)

position2=fig.add_axes([0.9,0.12,0.01,0.3])
plt.colorbar(im, position2, extend='both', label = 'Resistance (SIF)',orientation='vertical',ticks=[1,2,3,4])


plt.subplots_adjust(top=0.93, bottom=0.1, right=0.88, left=0.06, wspace=0.27)

plt.savefig('result_figure/figure_use_20260105/resistance_richness_fsc_prec_kndvi2&sif2_log_event_after2000.png', dpi = 600)

# %% [markdown]
# ### 6.2 偏相关

# %%
'''
bi_pd_pcor_all_kndvi = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_kndvi_mid_event.csv')
bi_pd_pval_all_kndvi = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pval_all_kndvi_mid_event.csv')
bi_pd_pcor_all_sif = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_sif_mid_event.csv')
bi_pd_pval_all_sif = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pval_all_sif_mid_event.csv')
'''

# %% [markdown]
# #### 之前的指标

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi&sif_event.png', dpi = 600)

# %%
## kndvi after2000
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi&sif_event_after2000.png', dpi = 600)

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi&sif_lai_event.png', dpi = 600)

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi_lai_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_lai_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi&sif_lai_event_after2000.png', dpi = 600)

# %%
## 加一列 fsc-lai的偏相关
kndvi_pcor_result_addlai = {}
kndvi_pcor_result_addlai['pcor'] = pd.merge(kndvi_pcor_result['pcor'],kndvi_lai_pcor_result['pcor'][['biome','fsc_lai_gs']],on = 'biome')
kndvi_pcor_result_addlai['p-val'] = pd.merge(kndvi_pcor_result['p-val'],kndvi_lai_pcor_result['p-val'][['biome','fsc_lai_gs']],on = 'biome')
kndvi_pcor_result_addlai

# %%
sif_pcor_result_addlai = {}
sif_pcor_result_addlai['pcor'] = pd.merge(sif_pcor_result['pcor'],sif_lai_pcor_result['pcor'][['biome','fsc_lai_gs']],on = 'biome')
sif_pcor_result_addlai['p-val'] = pd.merge(sif_pcor_result['p-val'],sif_lai_pcor_result['p-val'][['biome','fsc_lai_gs']],on = 'biome')
sif_pcor_result_addlai

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[0].set_xticks(ticks=np.arange(7))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_pcor_result_addlai['pcor'].biome])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[1].set_xticks(ticks=np.arange(7))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[1].set_yticklabels([])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if sif_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi&sif_event_addlai.png', dpi = 600)

# %%
## 加一列 fsc-lai的偏相关
kndvi_pcor_result_after2000_addlai = {}
kndvi_pcor_result_after2000_addlai['pcor'] = pd.merge(kndvi_pcor_result_after2000['pcor'],kndvi_lai_pcor_result_after2000['pcor'][['biome','fsc_lai_gs']],on = 'biome')
kndvi_pcor_result_after2000_addlai['p-val'] = pd.merge(kndvi_pcor_result_after2000['p-val'],kndvi_lai_pcor_result_after2000['p-val'][['biome','fsc_lai_gs']],on = 'biome')
kndvi_pcor_result_after2000_addlai

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi_pcor_result_after2000_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[0].set_xticks(ticks=np.arange(7))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi_pcor_result_after2000_addlai['pcor'].biome])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if kndvi_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[1].set_xticks(ticks=np.arange(7))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[1].set_yticklabels([])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if sif_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi&sif_event_after2000_addlai.png', dpi = 600)

# %% [markdown]
# #### 用Ydrou/Ymean

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_event.png', dpi = 600)

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_event_after2000.png', dpi = 600)

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_lai_event.png', dpi = 600)

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_lai_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_lai_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_lai_event_after2000.png', dpi = 600)

# %%
## 加一列 fsc-lai的偏相关
kndvi2_pcor_result_addlai = {}
kndvi2_pcor_result_addlai['pcor'] = pd.merge(kndvi2_pcor_result['pcor'],kndvi2_lai_pcor_result['pcor'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_pcor_result_addlai['p-val'] = pd.merge(kndvi2_pcor_result['p-val'],kndvi2_lai_pcor_result['p-val'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_pcor_result_addlai

# %%
sif2_pcor_result_addlai = {}
sif2_pcor_result_addlai['pcor'] = pd.merge(sif2_pcor_result['pcor'],sif2_lai_pcor_result['pcor'][['biome','fsc_lai_gs']],on = 'biome')
sif2_pcor_result_addlai['p-val'] = pd.merge(sif2_pcor_result['p-val'],sif2_lai_pcor_result['p-val'][['biome','fsc_lai_gs']],on = 'biome')
sif2_pcor_result_addlai

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[0].set_xticks(ticks=np.arange(7))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_pcor_result_addlai['pcor'].biome])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[1].set_xticks(ticks=np.arange(7))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[1].set_yticklabels([])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if sif2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_event_addlai.png', dpi = 600)

# %%
## 加一列 fsc-lai的偏相关
kndvi2_pcor_result_after2000_addlai = {}
kndvi2_pcor_result_after2000_addlai['pcor'] = pd.merge(kndvi2_pcor_result_after2000['pcor'],kndvi2_lai_pcor_result_after2000['pcor'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_pcor_result_after2000_addlai['p-val'] = pd.merge(kndvi2_pcor_result_after2000['p-val'],kndvi2_lai_pcor_result_after2000['p-val'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_pcor_result_after2000_addlai

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_pcor_result_after2000_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[0].set_xticks(ticks=np.arange(7))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_pcor_result_after2000_addlai['pcor'].biome])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[1].set_xticks(ticks=np.arange(7))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[1].set_yticklabels([])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if sif2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_event_after2000_addlai.png', dpi = 600)

# %% [markdown]
# #### Ydrou/Ymean log

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_log_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_log_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_log_event.png', dpi = 600)

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_log_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_log_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_log_event_after2000.png', dpi = 600)

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_log_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_lai_pcor_result['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_log_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_log_lai_event.png', dpi = 600)

# %%
draw_col = ['lai_gs','lai_gs_annual_prec','lai_gs_fsc','fsc','fsc_annual_prec','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_log_lai_pcor_result_after2000['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_lai_pcor_result_after2000['pcor'].biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_lai_pcor_result_after2000['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_log_lai_pcor_result['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','LAI'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_lai_pcor_result['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Leaf area index',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_log_lai_event_after2000.png', dpi = 600)

# %%
## 加一列 fsc-lai的偏相关
kndvi2_log_pcor_result_addlai = {}
kndvi2_log_pcor_result_addlai['pcor'] = pd.merge(kndvi2_log_pcor_result['pcor'],kndvi2_log_lai_pcor_result['pcor'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_log_pcor_result_addlai['p-val'] = pd.merge(kndvi2_log_pcor_result['p-val'],kndvi2_log_lai_pcor_result['p-val'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_log_pcor_result_addlai

# %%
sif2_log_pcor_result_addlai = {}
sif2_log_pcor_result_addlai['pcor'] = pd.merge(sif2_log_pcor_result['pcor'],sif2_log_lai_pcor_result['pcor'][['biome','fsc_lai_gs']],on = 'biome')
sif2_log_pcor_result_addlai['p-val'] = pd.merge(sif2_log_pcor_result['p-val'],sif2_log_lai_pcor_result['p-val'][['biome','fsc_lai_gs']],on = 'biome')
sif2_log_pcor_result_addlai

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_log_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[0].set_xticks(ticks=np.arange(7))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_pcor_result_addlai['pcor'].biome])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_log_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[1].set_xticks(ticks=np.arange(7))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[1].set_yticklabels([])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_log_event_addlai.png', dpi = 600)

# %%
## 加一列 fsc-lai的偏相关
kndvi2_log_pcor_result_after2000_addlai = {}
kndvi2_log_pcor_result_after2000_addlai['pcor'] = pd.merge(kndvi2_log_pcor_result_after2000['pcor'],kndvi2_log_lai_pcor_result_after2000['pcor'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_log_pcor_result_after2000_addlai['p-val'] = pd.merge(kndvi2_log_pcor_result_after2000['p-val'],kndvi2_log_lai_pcor_result_after2000['p-val'][['biome','fsc_lai_gs']],on = 'biome')
kndvi2_log_pcor_result_after2000_addlai

# %%
draw_col = ['plant_richness','plant_richness_annual_prec','plant_richness_fsc','fsc','fsc_annual_prec','fsc_plant_richness','fsc_lai_gs']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(kndvi2_log_pcor_result_after2000_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[0].set_xticks(ticks=np.arange(7))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in kndvi2_log_pcor_result_after2000_addlai['pcor'].biome])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if kndvi2_log_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif kndvi2_log_pcor_result_after2000_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[0].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[0].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[0].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[0].spines['right'].set_visible(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['left'].set_visible(False)
axes[0].spines['bottom'].set_visible(False)

# sif
axes[1].imshow(sif2_log_pcor_result_addlai['pcor'][draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.75)
axes[1].set_xticks(ticks=np.arange(7))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR','LAI'])
axes[1].set_yticklabels([])

for i in [6,5,4,3,2,1,0]:
    for j in range(10):
        if sif2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif sif2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif sif2_log_pcor_result_addlai['p-val'][draw_col].iloc[j,i] < 0.05:
            axes[1].text(i,j, '*', ha='center', va = 'center')
#for k in range(8):
#    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')
    #ax.text(2.5,k, round(ai_ld_std[ld_pd_pcor.ld[k]],2), ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
#ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

axes[1].text(1, 10.3, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
axes[1].text(4.5, 10.3, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

axes[1].spines['right'].set_visible(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['left'].set_visible(False)
axes[1].spines['bottom'].set_visible(False)

position1=fig.add_axes([0.935,0.2,0.01,0.6])
cb = plt.colorbar(im, position1, extend = 'both',shrink=0.7, pad = 0.05, ticks = [-0.5,-0.25,0,0.25,0.5])
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

axes[0].set_title('(a)  kNDVI', loc = 'left')
axes[1].set_title('(b)  SIF', loc = 'left')

plt.subplots_adjust(top=0.93, bottom=0.12, right=0.93, left=0.2, wspace=0.02)

plt.savefig('result_figure/figure_use_20260105/pcor_richness_fsc_prec_kndvi2&sif2_log_event_after2000_addlai.png', dpi = 600)

# %% [markdown]
# ## change log
# 1. 2025.11.21  用每一次事件 画了 fsc对 抵抗力  sm et lst的影响
# 2. 2025.12.10  加上一个 fsc 和 lai的 偏相关
# 3. 2025.12.12  加上了lai  加上了只考虑2000年以后的kndvi的结果
# 4. 2025.01.08  用另一个指标 Ydrou/Ymean 做了一遍

# %%



