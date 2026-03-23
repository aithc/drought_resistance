# %% [markdown]
# # 结构复杂度和 干旱时 土壤水分 蒸散发变化的关系

# %%
import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import rioxarray
import pandas as pd
import glob

# %%
plt.rc('font',family='Times New Roman', size = 15)

# %% [markdown]
# ## 1 读取数据

# %% [markdown]
# ### 1.1 多样性和结构复杂度

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
# ### 1.2 背景气候

# %%
era_temp_path = glob.glob(r'D:/data/era5_land_annual/era5_land_mean*.nc')
era_prec_path = glob.glob(r'D:/data/era5_land_annual/era5_land_pre*.nc')

# %%
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

# %%
## 干旱指数
with rioxarray.open_rasterio(r'D:/data/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif')  as data:
        ai_index= xr.DataArray(data.values[0], coords=[data.y, data.x], dims=['lat','lon'])
ai_index = ai_index.coarsen(lat=10,lon=10).mean()
ai_index
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
# ### 1.3 biome

# %%
with rioxarray.open_rasterio(r'D:/data/official_teow/biome.tif')  as data:
    biome = data
biome

# %%
biome = biome.where(biome>0)
biome = biome.where(biome<90)
biome.plot()

# %%
biome = xr.DataArray(biome[0].values, coords=[biome.y, biome.x], dims=['lat','lon'])
biome = biome.interp_like(plant_richness, method='nearest')
biome

# %% [markdown]
# ### 1.4 土壤性质

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
# ### 1.5 CTI

# %%
with xr.open_dataset(r'D:/data/Compound_topographic_index/cti_24.nc') as data:
    cti = data['cti']
cti
cti = cti.coarsen(lat = 10, lon =10, boundary='pad').mean()
cti = cti.interp_like(plant_richness, method='nearest')
cti
cti.plot(center = False)

# %% [markdown]
# ### 1.6 土壤水分变化

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/sm_nt_change_kndvi.nc') as data:
    sm_nt_change_kndvi = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/sm_sh_change_kndvi.nc') as data:
    sm_sh_change_kndvi = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/sm_nt_change_sif.nc') as data:
    sm_nt_change_sif = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/sm_sh_change_sif.nc') as data:
    sm_sh_change_sif = data['sm_change']

# %%
sm_nt_change_kndvi

# %%
sm_sh_change_kndvi

# %%
sm_nt_change_kndvi_mid = sm_nt_change_kndvi.median(dim='year')
sm_sh_change_kndvi_mid = sm_sh_change_kndvi.median(dim='year')

sm_nt_change_sif_mid = sm_nt_change_sif.median(dim='year')
sm_sh_change_sif_mid = sm_sh_change_sif.median(dim='year')

# %%
sm_change_kndvi = xr.concat([sm_nt_change_kndvi_mid,sm_sh_change_kndvi_mid], dim='lat').sortby('lat')
sm_change_kndvi

# %%
sm_change_kndvi.plot(vmax = 100)

# %%
sm_change_sif = xr.concat([sm_nt_change_sif_mid,sm_sh_change_sif_mid], dim='lat').sortby('lat')
sm_change_sif

# %%
sm_change_sif.plot(vmax = 100)

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi.nc') as data:
    smrz_nt_change_kndvi = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi.nc') as data:
    smrz_sh_change_kndvi = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_sif.nc') as data:
    smrz_nt_change_sif = data['sm_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_sif.nc') as data:
    smrz_sh_change_sif = data['sm_change']

# %%
smrz_nt_change_kndvi

# %%
smrz_sh_change_kndvi

# %%
smrz_nt_change_kndvi_mid = smrz_nt_change_kndvi.median(dim='year')
smrz_sh_change_kndvi_mid = smrz_sh_change_kndvi.median(dim='year')

# %%
smrz_nt_change_sif_mid = smrz_nt_change_sif.median(dim='year')
smrz_sh_change_sif_mid = smrz_sh_change_sif.median(dim='year')

# %%
smrz_change_kndvi = xr.concat([smrz_nt_change_kndvi_mid,smrz_sh_change_kndvi_mid], dim='lat').sortby('lat')
smrz_change_kndvi

# %%
smrz_change_kndvi.plot(vmax = 100)

# %%
smrz_change_sif = xr.concat([smrz_nt_change_sif_mid,smrz_sh_change_sif_mid], dim='lat').sortby('lat')
smrz_change_sif

# %%
smrz_change_sif.plot(vmax = 100)

# %% [markdown]
# ### 1.8 ET EA ET/EA

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change_kndvi.nc') as data:
    et_nt_change_kndvi = data['et_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change_kndvi.nc') as data:
    et_sh_change_kndvi = data['et_change']

with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change_sif.nc') as data:
    et_nt_change_sif = data['et_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change_sif.nc') as data:
    et_sh_change_sif = data['et_change']

# %%
et_nt_change_kndvi_mid = et_nt_change_kndvi.median(dim='year')
et_sh_change_kndvi_mid = et_sh_change_kndvi.median(dim='year')
et_nt_change_sif_mid = et_nt_change_sif.median(dim='year')
et_sh_change_sif_mid = et_sh_change_sif.median(dim='year')

# %%
et_change_kndvi = xr.concat([et_nt_change_kndvi_mid,et_sh_change_kndvi_mid], dim='lat').sortby('lat')
et_change_kndvi

# %%
et_change_kndvi.plot(vmax = 100)

# %%
et_change_sif = xr.concat([et_nt_change_sif_mid,et_sh_change_sif_mid], dim='lat').sortby('lat')
et_change_sif

# %%
et_change_sif.plot(vmax = 100)

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change2_kndvi.nc') as data:
    et_nt_change2_kndvi = data['et_change2']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change2_kndvi.nc') as data:
    et_sh_change2_kndvi = data['et_change2']

with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change2_sif.nc') as data:
    et_nt_change2_sif = data['et_change2']
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change2_sif.nc') as data:
    et_sh_change2_sif = data['et_change2']

# %%
et_nt_change2_kndvi_mid = et_nt_change2_kndvi.median(dim='year')
et_sh_change2_kndvi_mid = et_sh_change2_kndvi.median(dim='year')
et_nt_change2_sif_mid = et_nt_change2_sif.median(dim='year')
et_sh_change2_sif_mid = et_sh_change2_sif.median(dim='year')

# %%
et_change2_kndvi = xr.concat([et_nt_change2_kndvi_mid,et_sh_change2_kndvi_mid], dim='lat').sortby('lat')
et_change2_kndvi

# %%
et_change2_kndvi.plot(vmax = 1)

# %%
et_change2_sif = xr.concat([et_nt_change2_sif_mid,et_sh_change2_sif_mid], dim='lat').sortby('lat')
et_change2_sif

# %%
et_change2_sif.plot(vmax = 0.5)

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/ea_nt_change_kndvi.nc') as data:
    ea_nt_change_kndvi = data['ea_change']   
with xr.open_dataset(r'E:/python_output/fsc_drought/ea_sh_change_kndvi.nc') as data:
    ea_sh_change_kndvi = data['ea_change']

with xr.open_dataset(r'E:/python_output/fsc_drought/ea_nt_change_sif.nc') as data:
    ea_nt_change_sif = data['ea_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/ea_sh_change_sif.nc') as data:
    ea_sh_change_sif = data['ea_change']

# %%
ea_nt_change_kndvi_mid = ea_nt_change_kndvi.median(dim='year')
ea_sh_change_kndvi_mid = ea_sh_change_kndvi.median(dim='year')
ea_nt_change_sif_mid = ea_nt_change_sif.median(dim='year')
ea_sh_change_sif_mid = ea_sh_change_sif.median(dim='year')

# %%
ea_change_kndvi = xr.concat([ea_nt_change_kndvi_mid,ea_sh_change_kndvi_mid], dim='lat').sortby('lat')
ea_change_kndvi

# %%
ea_change_kndvi.plot(vmax = 100)

# %%
ea_change_sif = xr.concat([ea_nt_change_sif_mid,ea_sh_change_sif_mid], dim='lat').sortby('lat')
ea_change_sif

# %%
ea_change_sif.plot(vmax = 100)

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/ret_ea_nt_change_kndvi.nc') as data:
    ret_ea_nt_change_kndvi = data['ret_ea_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/ret_ea_sh_change_kndvi.nc') as data:
    ret_ea_sh_change_kndvi = data['ret_ea_change']

with xr.open_dataset(r'E:/python_output/fsc_drought/ret_ea_nt_change_sif.nc') as data:
    ret_ea_nt_change_sif = data['ret_ea_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/ret_ea_sh_change_sif.nc') as data:
    ret_ea_sh_change_sif = data['ret_ea_change']

# %%
ret_ea_nt_change_kndvi_mid = ret_ea_nt_change_kndvi.median(dim='year')
ret_ea_sh_change_kndvi_mid = ret_ea_sh_change_kndvi.median(dim='year')
ret_ea_nt_change_sif_mid = ret_ea_nt_change_sif.median(dim='year')
ret_ea_sh_change_sif_mid = ret_ea_sh_change_sif.median(dim='year')

# %%
ret_ea_change_kndvi = xr.concat([ret_ea_nt_change_kndvi_mid,ret_ea_sh_change_kndvi_mid], dim='lat').sortby('lat')
ret_ea_change_kndvi

# %%
ret_ea_change_kndvi.plot(vmax = 100)

# %%
ret_ea_change_sif = xr.concat([ret_ea_nt_change_sif_mid,ret_ea_sh_change_sif_mid], dim='lat').sortby('lat')
ret_ea_change_sif

# %%
ret_ea_change_sif.plot(vmax = 100)

# %% [markdown]
# ### 1.9 合并数据

# %%
smchange_all = xr.Dataset({
    'smchange_kndvi':sm_change_kndvi,
    'smchange_sif':sm_change_sif,
    'richness':plant_richness,
    'fsc':fsc,
    'temp':annual_temp,
    'prec':annual_prec,
    'ai':ai_index,
    'biome':biome,
    'clay': soil_clay,
    'cec':soil_cec,
    'cti':cti
})
smchange_all

# %%
gleam_change_all = xr.Dataset({
    'smrz_change_kndvi':smrz_change_kndvi,
    'smrz_change_sif':smrz_change_sif,
    'ea_change_kndvi':ea_change_kndvi,
    'ea_change_sif':ea_change_sif,
    'et_change_kndvi':et_change_kndvi,  
    'et_change_sif':et_change_sif,
    'et_change2_kndvi':et_change2_kndvi,  
    'et_change2_sif':et_change2_sif,
    'ret_ea_change_kndvi':ret_ea_change_kndvi,
    'ret_ea_change_sif':ret_ea_change_sif,
    'richness':plant_richness,
    'fsc':fsc,
    'temp':annual_temp,
    'prec':annual_prec,
    'ai':ai_index,
    'biome':biome,
    'clay': soil_clay,
    'cec':soil_cec,
    'cti':cti
})
gleam_change_all

# %%
smchange_all_df = smchange_all.to_dataframe()
smchange_all_df.dropna(how='all')
smchange_all_df.describe()

# %%
smchange_kndvi_df = smchange_all_df.drop(['smchange_sif'], axis=1)
smchange_kndvi_df = smchange_kndvi_df.dropna(how='any')
smchange_kndvi_df.describe()

# %%
smchange_sif_df = smchange_all_df.drop(['smchange_kndvi'], axis=1)
smchange_sif_df = smchange_sif_df.dropna(how='any')
smchange_sif_df.describe()

# %%
gleam_change_all_df = gleam_change_all.to_dataframe()
gleam_change_all_df.dropna(how='all')
gleam_change_all_df.describe()

# %%
gleam_change_all_df.columns

# %%
smrzchange_kndvi_df = gleam_change_all_df.drop(['smrz_change_sif', 'ea_change_kndvi',
       'ea_change_sif', 'et_change_kndvi', 'et_change_sif', 'et_change2_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
smrzchange_kndvi_df = smrzchange_kndvi_df.dropna(how='any')
smrzchange_kndvi_df.describe()

# %%
smrzchange_sif_df = gleam_change_all_df.drop(['smrz_change_kndvi', 'ea_change_kndvi',
       'ea_change_sif', 'et_change_kndvi', 'et_change_sif','et_change2_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
smrzchange_sif_df = smrzchange_sif_df.dropna(how='any')
smrzchange_sif_df.describe()

# %%
eachange_kndvi_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_sif', 'et_change_kndvi', 'et_change_sif','et_change2_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
eachange_kndvi_df = eachange_kndvi_df.dropna(how='any')
eachange_kndvi_df.describe()

# %%
eachange_sif_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_kndvi', 'et_change_kndvi', 'et_change_sif','et_change2_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
eachange_sif_df = eachange_sif_df.dropna(how='any')
eachange_sif_df.describe()

# %%
etchange_kndvi_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_sif', 'ea_change_kndvi', 'et_change_sif', 'et_change2_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
etchange_kndvi_df = etchange_kndvi_df.dropna(how='any')
etchange_kndvi_df.describe()

# %%
etchange_sif_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_kndvi', 'et_change_kndvi', 'ea_change_sif', 'et_change2_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
etchange_sif_df = etchange_sif_df.dropna(how='any')
etchange_sif_df.describe()

# %%
etchange2_kndvi_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_sif', 'ea_change_kndvi', 'et_change_sif', 'et_change_kndvi', 'et_change2_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
etchange2_kndvi_df = etchange2_kndvi_df.dropna(how='any')
etchange2_kndvi_df.describe()

# %%
etchange2_sif_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_kndvi', 'et_change_kndvi', 'ea_change_sif', 'et_change2_kndvi', 'et_change_sif',
       'ret_ea_change_kndvi', 'ret_ea_change_sif'], axis=1)
etchange2_sif_df = etchange2_sif_df.dropna(how='any')
etchange2_sif_df.describe()

# %%
retea_change_kndvi_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_sif', 'ea_change_kndvi', 'et_change_sif',
       'et_change_kndvi', 'ret_ea_change_sif'], axis=1)
retea_change_kndvi_df = retea_change_kndvi_df.dropna(how='any')
retea_change_kndvi_df.describe()

# %%
retea_change_sif_df = gleam_change_all_df.drop(['smrz_change_sif', 'smrz_change_kndvi',
       'ea_change_sif', 'ea_change_kndvi', 'et_change_sif',
       'et_change_kndvi', 'ret_ea_change_kndvi'], axis=1)
retea_change_sif_df = retea_change_sif_df.dropna(how='any')
retea_change_sif_df.describe()

# %% [markdown]
# ## 2 kndvi 对应分析

# %% [markdown]
# ### 2.1 esa sm

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
smchange_kndvi_q90 = smchange_kndvi_df.smchange_kndvi.quantile(0.9)
smchange_kndvi_q90

# %%
smchange_kndvi_use = smchange_kndvi_df[smchange_kndvi_df.smchange_kndvi < smchange_kndvi_q90]
smchange_kndvi_use

# %%
plt.hexbin(x =smchange_kndvi_use.fsc, y= smchange_kndvi_use.smchange_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(smchange_kndvi_use.fsc.min(), smchange_kndvi_use.fsc.max() )

# %%
smchange_kndvi_use['fsc_bins'] = pd.cut(smchange_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
smchange_kndvi_use.boxplot('smchange_kndvi', by='fsc_bins')

# %%
smchange_kndvi_use['smchange_log'] = np.log(smchange_kndvi_use.smchange_kndvi)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smchange_kndvi_use.fsc_bins)
fsc_dfs = [smchange_kndvi_use.smchange_kndvi[smchange_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smchange_kndvi_use.smchange_kndvi[smchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smchange_kndvi_use.smchange_kndvi[smchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 100, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,110)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_kndvi.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smchange_kndvi_use.fsc_bins)
fsc_dfs = [smchange_kndvi_use.smchange_log[smchange_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smchange_kndvi_use.smchange_log[smchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smchange_kndvi_use.smchange_log[smchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0.5,5.5)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_kndvi_log.png', dpi = 600)

# %%
smchange_kndvi_use.biome.value_counts()

# %%
smchange_kndvi_fsc_bybi = smchange_kndvi_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [120,70,120,110,120,80,50,70,40]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = smchange_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = smchange_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smchange_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smchange_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smchange_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smchange_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.3)
    axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = np.arange(9,11.4,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/sm_change_csc_biome_kndvi.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]


for i in range(9):
    bi_n = smchange_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = smchange_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smchange_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smchange_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smchange_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smchange_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(1,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.3)
    axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = np.arange(9,11.4,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/sm_change_csc_biome_kndvi_log.png', dpi = 600)

# %%
plt.hexbin(np.log(smchange_kndvi_use.prec), smchange_kndvi_use.fsc, 
                C= smchange_kndvi_use.smchange_kndvi, gridsize = 100, 
                vmax=80, vmin=25,
                reduce_C_function = np.median)

# %%
plt.hexbin(smchange_kndvi_use.prec, smchange_kndvi_use.fsc, 
                C= smchange_kndvi_use.smchange_kndvi, gridsize = 100, 
                vmax=70, vmin=15,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smchange_kndvi_use.prec), smchange_kndvi_use.fsc, 
                C= smchange_kndvi_use.smchange_kndvi, gridsize = 100, 
                vmax=60, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_prec_kndvi.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smchange_kndvi_use.prec), smchange_kndvi_use.fsc, 
                C= smchange_kndvi_use.smchange_log, gridsize = 100, 
                vmax=4, vmin=2.5,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change (log)', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_prec_kndvi_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smchange_kndvi_use.ai), smchange_kndvi_use.fsc, 
                C= smchange_kndvi_use.smchange_kndvi, gridsize = 100, 
                vmax=70, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Aridity index')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([-3, 2])
axes.set_xticks(np.arange(-3,2))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

#plt.savefig('result_fig/sm_change_fsc_ai.png', dpi = 600)

# %% [markdown]
# ### 2.2 gleam smrz

# %%
smrzchange_kndvi_df.head()

# %%
smrzchange_kndvi_q90 = smrzchange_kndvi_df.smrz_change_kndvi.quantile(0.9)
smrzchange_kndvi_q90

# %%
smrzchange_kndvi_use = smrzchange_kndvi_df[smrzchange_kndvi_df.smrz_change_kndvi < smrzchange_kndvi_q90]
smrzchange_kndvi_use

# %%
plt.hexbin(x =smrzchange_kndvi_use.fsc, y= smrzchange_kndvi_use.smrz_change_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(smrzchange_kndvi_use.fsc.min(), smrzchange_kndvi_use.fsc.max() )

# %%
smrzchange_kndvi_use['fsc_bins'] = pd.cut(smrzchange_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
smrzchange_kndvi_use.boxplot('smrz_change_kndvi', by='fsc_bins')

# %%
smrzchange_kndvi_use['smrz_change_log'] = np.log(smrzchange_kndvi_use.smrz_change_kndvi)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smrzchange_kndvi_use.fsc_bins)
fsc_dfs = [smrzchange_kndvi_use.smrz_change_kndvi[smrzchange_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smrzchange_kndvi_use.smrz_change_kndvi[smrzchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smrzchange_kndvi_use.smrz_change_kndvi[smrzchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 100, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,115)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_kndvi.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smrzchange_kndvi_use.fsc_bins)
fsc_dfs = [smrzchange_kndvi_use.smrz_change_log[smrzchange_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smrzchange_kndvi_use.smrz_change_log[smrzchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smrzchange_kndvi_use.smrz_change_log[smrzchange_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0.5,5.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_kndvi_log.png', dpi = 600)

# %%
smrzchange_kndvi_use.biome.value_counts()
smrzchange_kndvi_fsc_bybi = smrzchange_kndvi_use.groupby("biome")
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [125,70,125,110,125,80,40,70,40]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = smrzchange_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smrz_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.9,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/smrz_change_csc_biome_kndvi.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]


for i in range(9):
    bi_n = smrzchange_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_kndvi_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_kndvi_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smrz_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
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

fig.savefig('result_figure/smrz_change_csc_biome_kndvi_log.png', dpi = 600)

# %%
plt.hexbin(np.log(smrzchange_kndvi_use.prec), smrzchange_kndvi_use.fsc, 
                C= smrzchange_kndvi_use.smrz_change_kndvi, gridsize = 100, 
                vmax=80, vmin=20,
                reduce_C_function = np.median)

# %%
plt.hexbin(smrzchange_kndvi_use.prec, smrzchange_kndvi_use.fsc, 
                C= smrzchange_kndvi_use.smrz_change_kndvi, gridsize = 100, 
                vmax=70, vmin=15,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smrzchange_kndvi_use.prec), smrzchange_kndvi_use.fsc, 
                C= smrzchange_kndvi_use.smrz_change_kndvi, gridsize = 100, 
                vmax=60, vmin=10,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_prec_kndvi.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smrzchange_kndvi_use.prec), smrzchange_kndvi_use.fsc, 
                C= smrzchange_kndvi_use.smrz_change_log, gridsize = 100, 
                vmax=4, vmin=2.5,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change (log)', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_prec_kndvi_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smrzchange_kndvi_use.ai), smrzchange_kndvi_use.fsc, 
                C= smrzchange_kndvi_use.smrz_change_kndvi, gridsize = 100, 
                vmax=70, vmin=10,
                reduce_C_function = np.median)
axes.set_xlabel('Aridity index')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([-3, 2])
axes.set_xticks(np.arange(-3,2))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

#plt.savefig('result_fig/sm_change_fsc_ai.png', dpi = 600)

# %% [markdown]
# ### 2.3 gleam ea

# %%
eachange_kndvi_q95 = eachange_kndvi_df.ea_change_kndvi.quantile(0.95)
eachange_kndvi_q95

# %%
eachange_kndvi_q5 = eachange_kndvi_df.ea_change_kndvi.quantile(0.05)
eachange_kndvi_q5

# %%
eachange_kndvi_use = eachange_kndvi_df[eachange_kndvi_df.ea_change_kndvi < eachange_kndvi_q95]
eachange_kndvi_use = eachange_kndvi_use[eachange_kndvi_use.ea_change_kndvi > eachange_kndvi_q5]
eachange_kndvi_use

# %%
plt.hexbin(x =eachange_kndvi_use.fsc, y= eachange_kndvi_use.ea_change_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(eachange_kndvi_use.fsc.min(), eachange_kndvi_use.fsc.max())

# %%
eachange_kndvi_use['fsc_bins'] = pd.cut(eachange_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
eachange_kndvi_use[eachange_kndvi_use.ea_change_kndvi>0].boxplot('ea_change_kndvi', by='fsc_bins')

# %%
eachange_kndvi_use[eachange_kndvi_use.ea_change_kndvi<0].boxplot('ea_change_kndvi', by='fsc_bins')

# %%
eachange_kndvi_use_po = eachange_kndvi_use[eachange_kndvi_use.ea_change_kndvi>0]
eachange_kndvi_use_ne = eachange_kndvi_use[eachange_kndvi_use.ea_change_kndvi<0]

# %%
eachange_kndvi_use_po['ea_change_log'] = np.log(eachange_kndvi_use_po.ea_change_kndvi)
eachange_kndvi_use_ne['ea_change_log'] = np.log(eachange_kndvi_use_ne.ea_change_kndvi * -1) * -1

# %%
eachange_kndvi_use_po

# %%
eachange_kndvi_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(eachange_kndvi_use_po.fsc_bins)
fsc_dfs = [eachange_kndvi_use_po.ea_change_kndvi[eachange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_po.ea_change_kndvi[eachange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_po.ea_change_kndvi[eachange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 85, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,90)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_kndvi_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(eachange_kndvi_use_po.fsc_bins)
fsc_dfs = [eachange_kndvi_use_po.ea_change_log[eachange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_po.ea_change_log[eachange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_po.ea_change_log[eachange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,5.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_kndvi_po_log.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(eachange_kndvi_use_ne.fsc_bins)
fsc_dfs = [eachange_kndvi_use_ne.ea_change_kndvi[eachange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_ne.ea_change_kndvi[eachange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_ne.ea_change_kndvi[eachange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -85, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-90,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_kndvi_ne.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(eachange_kndvi_use_ne.fsc_bins)
fsc_dfs = [eachange_kndvi_use_ne.ea_change_log[eachange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_ne.ea_change_log[eachange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_kndvi_use_ne.ea_change_log[eachange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-5.5,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_kndvi_ne_log.png', dpi = 600)

# %%
eachange_kndvi_use.biome.value_counts()

# %%
eachange_kndvi_fsc_bybi_po = eachange_kndvi_use_po.groupby("biome")
eachange_kndvi_fsc_bybi_ne = eachange_kndvi_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [90,90,90,90,80,90,90,80,90]

for i in range(9):
    bi_n = eachange_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(eachange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5,5.5,5.5,5,5.5]

for i in range(9):
    bi_n = eachange_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(eachange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_kndvi_po_log.png', dpi = 600)

# %%

fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [90,80,80,90,90,60,90,40,40]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = eachange_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(eachange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_kndvi_ne.png', dpi = 600)

# %%

fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5, 5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]


for i in range(9):
    bi_n = eachange_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(eachange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_kndvi_ne_log.png', dpi = 600)

# %%
plt.hexbin(np.log(eachange_kndvi_use_po.prec), eachange_kndvi_use_po.fsc, 
                C= eachange_kndvi_use_po.ea_change_kndvi, gridsize = 100, 
                vmax=60, vmin=20,
                reduce_C_function = np.median)

# %%
plt.hexbin(np.log(eachange_kndvi_use_ne.prec), eachange_kndvi_use_ne.fsc, 
                C= eachange_kndvi_use_ne.ea_change_kndvi, gridsize = 100, 
                vmax=-10, vmin=-60,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(eachange_kndvi_use_po.prec), eachange_kndvi_use_po.fsc, 
                C= eachange_kndvi_use_po.ea_change_kndvi, gridsize = 100, 
                vmax=50, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'EA change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_prec_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(eachange_kndvi_use_ne.prec), eachange_kndvi_use_ne.fsc, 
                C= eachange_kndvi_use_ne.ea_change_kndvi, gridsize = 100, 
                vmax=-10, vmin=-40,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'EA change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_prec_kndvi_ne.png', dpi = 600)

# %% [markdown]
# ### 2.4 gleam et

# %%
etchange_kndvi_q95 = etchange_kndvi_df.et_change_kndvi.quantile(0.95)
etchange_kndvi_q95

# %%
etchange_kndvi_q5 = etchange_kndvi_df.et_change_kndvi.quantile(0.05)
etchange_kndvi_q5

# %%
etchange_kndvi_use = etchange_kndvi_df[etchange_kndvi_df.et_change_kndvi < etchange_kndvi_q95]
etchange_kndvi_use = etchange_kndvi_use[etchange_kndvi_use.et_change_kndvi > etchange_kndvi_q5]
etchange_kndvi_use

# %%
plt.hexbin(x =etchange_kndvi_use.fsc, y= etchange_kndvi_use.et_change_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(etchange_kndvi_use.fsc.min(), etchange_kndvi_use.fsc.max())

# %%
etchange_kndvi_use['fsc_bins'] = pd.cut(etchange_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
etchange_kndvi_use[etchange_kndvi_use.et_change_kndvi>0].boxplot('et_change_kndvi', by='fsc_bins')

# %%
etchange_kndvi_use[etchange_kndvi_use.et_change_kndvi<0].boxplot('et_change_kndvi', by='fsc_bins')

# %%
etchange_kndvi_use_po = etchange_kndvi_use[etchange_kndvi_use.et_change_kndvi>0]
etchange_kndvi_use_ne = etchange_kndvi_use[etchange_kndvi_use.et_change_kndvi<0]

# %%
etchange_kndvi_use_po['et_change_log'] = np.log(etchange_kndvi_use_po.et_change_kndvi)
etchange_kndvi_use_ne['et_change_log'] = np.log(etchange_kndvi_use_ne.et_change_kndvi * -1) * -1

# %%
etchange_kndvi_use_po

# %%
etchange_kndvi_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_po.fsc_bins)
fsc_dfs = [etchange_kndvi_use_po.et_change_kndvi[etchange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po.et_change_kndvi[etchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po.et_change_kndvi[etchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 60, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,65)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change_fsc_kndvi_po.png', dpi = 600)


# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_po.fsc_bins)
fsc_dfs = [etchange_kndvi_use_po.et_change_log[etchange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po.et_change_log[etchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_po.et_change_log[etchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.2, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,4.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change_fsc_kndvi_po_log.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_ne.fsc_bins)
fsc_dfs = [etchange_kndvi_use_ne.et_change_kndvi[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne.et_change_kndvi[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne.et_change_kndvi[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -55, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-60,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change_fsc_kndvi_ne.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange_kndvi_use_ne.fsc_bins)
fsc_dfs = [etchange_kndvi_use_ne.et_change_log[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne.et_change_log[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange_kndvi_use_ne.et_change_log[etchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -4.2, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-4.5,0.9)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change_fsc_kndvi_ne_log.png', dpi = 600)

# %%
etchange_kndvi_use.biome.value_counts()

# %%
etchange_kndvi_fsc_bybi_po = etchange_kndvi_use_po.groupby("biome")
etchange_kndvi_fsc_bybi_ne = etchange_kndvi_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [65,65,65,60,60,65,50,50,65]

for i in range(9):
    bi_n = etchange_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change_csc_biome_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5,5,5,5,5,5,5,5,5]

for i in range(9):
    bi_n = etchange_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change_csc_biome_kndvi_po_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [60,60,60,60,60,60,30,30,30]

for i in range(9):
    bi_n = etchange_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change_csc_biome_kndvi_ne.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5]

for i in range(9):
    bi_n = etchange_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(etchange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change_csc_biome_kndvi_ne_log.png', dpi = 600)

# %%
plt.hexbin(np.log(etchange_kndvi_use_po.prec), etchange_kndvi_use_po.fsc, 
                C= etchange_kndvi_use_po.et_change_kndvi, gridsize = 100, 
                vmax=60, vmin=20,
                reduce_C_function = np.median)

# %%
plt.hexbin(np.log(etchange_kndvi_use_ne.prec), etchange_kndvi_use_ne.fsc, 
                C= etchange_kndvi_use_ne.et_change_kndvi, gridsize = 100, 
                vmax=-10, vmin=-60,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(etchange_kndvi_use_po.prec), etchange_kndvi_use_po.fsc, 
                C= etchange_kndvi_use_po.et_change_kndvi, gridsize = 100, 
                vmax=35, vmin=10,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'ET change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/et_change_fsc_prec_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(etchange_kndvi_use_ne.prec), etchange_kndvi_use_ne.fsc, 
                C= etchange_kndvi_use_ne.et_change_kndvi, gridsize = 100, 
                vmax=-5, vmin=-30,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'ET change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/et_change_fsc_prec_kndvi_ne.png', dpi = 600)

# %%
etchange2_kndvi_q95 = etchange2_kndvi_df.et_change2_kndvi.quantile(0.95)
etchange2_kndvi_q95
etchange2_kndvi_q5 = etchange2_kndvi_df.et_change2_kndvi.quantile(0.05)
etchange2_kndvi_q5
etchange2_kndvi_use = etchange2_kndvi_df[etchange2_kndvi_df.et_change2_kndvi < etchange2_kndvi_q95]
etchange2_kndvi_use = etchange2_kndvi_use[etchange2_kndvi_use.et_change2_kndvi > etchange2_kndvi_q5]
etchange2_kndvi_use
plt.hexbin(x =etchange2_kndvi_use.fsc, y= etchange2_kndvi_use.et_change2_kndvi, gridsize = 50,bins='log')
plt.colorbar()
print(etchange2_kndvi_use.fsc.min(), etchange2_kndvi_use.fsc.max())
etchange2_kndvi_use['fsc_bins'] = pd.cut(etchange2_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
etchange2_kndvi_use[etchange2_kndvi_use.et_change2_kndvi>0].boxplot('et_change2_kndvi', by='fsc_bins')
etchange2_kndvi_use[etchange2_kndvi_use.et_change2_kndvi<0].boxplot('et_change2_kndvi', by='fsc_bins')
etchange2_kndvi_use_po = etchange2_kndvi_use[etchange2_kndvi_use.et_change2_kndvi>0]
etchange2_kndvi_use_ne = etchange2_kndvi_use[etchange2_kndvi_use.et_change2_kndvi<0]
etchange2_kndvi_use_po['et_change2_log'] = np.log(etchange2_kndvi_use_po.et_change2_kndvi)
etchange2_kndvi_use_ne['et_change2_log'] = np.log(etchange2_kndvi_use_ne.et_change2_kndvi * -1) * -1
etchange2_kndvi_use_po
etchange2_kndvi_use_ne
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange2_kndvi_use_po.fsc_bins)
fsc_dfs = [etchange2_kndvi_use_po.et_change2_kndvi[etchange2_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_po.et_change2_kndvi[etchange2_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_po.et_change2_kndvi[etchange2_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 60, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,65)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change2_fsc_kndvi_po.png', dpi = 600)

fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(etchange2_kndvi_use_po.fsc_bins)
fsc_dfs = [etchange2_kndvi_use_po.et_change2_log[etchange2_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_po.et_change2_log[etchange2_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_po.et_change_log[etchange2_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 4.2, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,4.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change2_fsc_kndvi_po_log.png', dpi = 600)
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange2_kndvi_use_ne.fsc_bins)
fsc_dfs = [etchange2_kndvi_use_ne.et_change2_kndvi[etchange2_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_ne.et_change2_kndvi[etchange2_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_ne.et_change2_kndvi[etchange2_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -55, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-60,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change2_fsc_kndvi_ne.png', dpi = 600)
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(etchange2_kndvi_use_ne.fsc_bins)
fsc_dfs = [etchange2_kndvi_use_ne.et_change2_log[etchange2_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_ne.et_change2_log[etchange2_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(etchange2_kndvi_use_ne.et_change2_log[etchange2_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -4.2, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-4.5,0.9)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/et_change2_fsc_kndvi_ne_log.png', dpi = 600)
etchange2_kndvi_use.biome.value_counts()
etchange2_kndvi_fsc_bybi_po = etchange2_kndvi_use_po.groupby("biome")
etchange2_kndvi_fsc_bybi_ne = etchange2_kndvi_use_ne.groupby("biome")
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [65,65,65,60,60,65,50,50,65]

for i in range(9):
    bi_n = etchange2_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange2_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(etchange2_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change2_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change2_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change2_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change2_csc_biome_kndvi_po.png', dpi = 600)
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5,5,5,5,5,5,5,5,5]

for i in range(9):
    bi_n = etchange2_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange2_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(etchange2_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change2_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change2_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change2_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change2_csc_biome_kndvi_po_log.png', dpi = 600)
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [60,60,60,60,60,60,30,30,30]

for i in range(9):
    bi_n = etchange2_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange2_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(etchange2_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change2_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change2_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change2_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change2_csc_biome_kndvi_ne.png', dpi = 600)
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5,4.5]

for i in range(9):
    bi_n = etchange2_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = etchange2_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(etchange2_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.et_change2_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.et_change2_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.et_change2_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/et_change2_csc_biome_kndvi_ne_log.png', dpi = 600)

# %% [markdown]
# ### 2.5 gleam r_et_ea

# %%
retea_change_kndvi_q95 = retea_change_kndvi_df.ret_ea_change_kndvi.quantile(0.95)
retea_change_kndvi_q95

# %%
retea_change_kndvi_q5 = retea_change_kndvi_df.ret_ea_change_kndvi.quantile(0.05)
retea_change_kndvi_q5

# %%
retea_change_kndvi_use = retea_change_kndvi_df[retea_change_kndvi_df.ret_ea_change_kndvi < retea_change_kndvi_q95]
retea_change_kndvi_use = retea_change_kndvi_use[retea_change_kndvi_use.ret_ea_change_kndvi > retea_change_kndvi_q5]
retea_change_kndvi_use

# %%
plt.hexbin(x =retea_change_kndvi_use.fsc, y= retea_change_kndvi_use.ret_ea_change_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(retea_change_kndvi_use.fsc.min(), retea_change_kndvi_use.fsc.max())

# %%
retea_change_kndvi_use['fsc_bins'] = pd.cut(retea_change_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
retea_change_kndvi_use[retea_change_kndvi_use.ret_ea_change_kndvi>0].boxplot('ret_ea_change_kndvi', by='fsc_bins')

# %%
retea_change_kndvi_use[retea_change_kndvi_use.ret_ea_change_kndvi<0].boxplot('ret_ea_change_kndvi', by='fsc_bins')

# %%
retea_change_kndvi_use_po = retea_change_kndvi_use[retea_change_kndvi_use.ret_ea_change_kndvi>0]
retea_change_kndvi_use_ne = retea_change_kndvi_use[retea_change_kndvi_use.ret_ea_change_kndvi<0]

# %%
retea_change_kndvi_use_po['ret_ea_change_log'] = np.log(retea_change_kndvi_use_po.ret_ea_change_kndvi)
retea_change_kndvi_use_ne['ret_ea_change_log'] = np.log(retea_change_kndvi_use_ne.ret_ea_change_kndvi * -1) * -1

# %%
retea_change_kndvi_use_po

# %%
retea_change_kndvi_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(retea_change_kndvi_use_po.fsc_bins)
fsc_dfs = [retea_change_kndvi_use_po.ret_ea_change_kndvi[retea_change_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_po.ret_ea_change_kndvi[retea_change_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_po.ret_ea_change_kndvi[retea_change_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 100, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET/EA change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,110)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/retea_change_fsc_kndvi_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(retea_change_kndvi_use_po.fsc_bins)
fsc_dfs = [retea_change_kndvi_use_po.ret_ea_change_log[retea_change_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_po.ret_ea_change_log[retea_change_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_po.ret_ea_change_log[retea_change_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET/EA change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,5.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/retea_change_fsc_kndvi_po_log.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(retea_change_kndvi_use_ne.fsc_bins)
fsc_dfs = [retea_change_kndvi_use_ne.ret_ea_change_kndvi[retea_change_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_ne.ret_ea_change_kndvi[retea_change_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_ne.ret_ea_change_kndvi[retea_change_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -85, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET/EA change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-90,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/retea_change_fsc_kndvi_ne.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(retea_change_kndvi_use_ne.fsc_bins)
fsc_dfs = [retea_change_kndvi_use_ne.ret_ea_change_log[retea_change_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_ne.ret_ea_change_log[retea_change_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(retea_change_kndvi_use_ne.ret_ea_change_log[retea_change_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('ET/EA change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-5.5,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/retea_change_fsc_kndvi_ne_log.png', dpi = 600)

# %%
retea_change_kndvi_use.biome.value_counts()

# %%
retea_change_kndvi_fsc_bybi_po = retea_change_kndvi_use_po.groupby("biome")
retea_change_kndvi_fsc_bybi_ne = retea_change_kndvi_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [105,105,105,105,105,105,105,80,80]

for i in range(9):
    bi_n = retea_change_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = retea_change_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(retea_change_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ret_ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET/EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/retea_change_csc_biome_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5,5.5,5.5,5,5.5]

for i in range(9):
    bi_n = retea_change_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = retea_change_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(retea_change_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ret_ea_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET/EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/retea_change_csc_biome_kndvi_po_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [90,80,80,90,90,80,60,60,60]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = retea_change_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = retea_change_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(retea_change_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ret_ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET/EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/retea_change_csc_biome_kndvi_ne.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5, 5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]

for i in range(9):
    bi_n = retea_change_kndvi_use_ne.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = retea_change_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(retea_change_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ret_ea_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ret_ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('ET/EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/retea_change_csc_biome_kndvi_ne_log.png', dpi = 600)

# %%
plt.hexbin(np.log(retea_change_kndvi_use_po.prec), retea_change_kndvi_use_po.fsc, 
                C= retea_change_kndvi_use_po.ret_ea_change_kndvi, gridsize = 100, 
                vmax=60, vmin=20,
                reduce_C_function = np.median)

# %%
plt.hexbin(np.log(retea_change_kndvi_use_ne.prec), retea_change_kndvi_use_ne.fsc, 
                C= retea_change_kndvi_use_ne.ret_ea_change_kndvi, gridsize = 100, 
                vmax=-10, vmin=-60,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(retea_change_kndvi_use_po.prec), retea_change_kndvi_use_po.fsc, 
                C= retea_change_kndvi_use_po.ret_ea_change_kndvi, gridsize = 100, 
                vmax=50, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'EA change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/retea_change_fsc_prec_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(retea_change_kndvi_use_ne.prec), retea_change_kndvi_use_ne.fsc, 
                C= retea_change_kndvi_use_ne.ret_ea_change_kndvi, gridsize = 100, 
                vmax=-10, vmin=-40,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'EA change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/retea_change_fsc_prec_kndvi_ne.png', dpi = 600)

# %% [markdown]
# ## 3 sif 对应分析

# %% [markdown]
# ### 3.1 esa sm

# %%
smchange_sif_q90 = smchange_sif_df.smchange_sif.quantile(0.9)
smchange_sif_q90

# %%
smchange_sif_use = smchange_sif_df[smchange_sif_df.smchange_sif < smchange_sif_q90]
smchange_sif_use

# %%
plt.hexbin(x =smchange_sif_use.fsc, y= smchange_sif_use.smchange_sif, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(smchange_sif_use.fsc.min(), smchange_sif_use.fsc.max() )

# %%
smchange_sif_use['fsc_bins'] = pd.cut(smchange_sif_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
smchange_sif_use.boxplot('smchange_sif', by='fsc_bins')

# %%
smchange_sif_use['smchange_log'] = np.log(smchange_sif_use.smchange_sif)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smchange_sif_use.fsc_bins)
fsc_dfs = [smchange_sif_use.smchange_sif[smchange_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smchange_sif_use.smchange_sif[smchange_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smchange_sif_use.smchange_sif[smchange_sif_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 95, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,100)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_sif.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smchange_sif_use.fsc_bins)
fsc_dfs = [smchange_sif_use.smchange_log[smchange_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smchange_sif_use.smchange_log[smchange_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smchange_sif_use.smchange_log[smchange_sif_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0.5,5.5)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_sif_log.png', dpi = 600)

# %%
smchange_sif_use.biome.value_counts()

# %%
smchange_sif_fsc_bybi = smchange_sif_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [110,70,105,100,110,75,50,75,40]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = smchange_sif_use.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = smchange_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smchange_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smchange_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smchange_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smchange_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.3)
    axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = np.arange(9,11.4,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/sm_change_csc_biome_sif.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]


for i in range(9):
    bi_n = smchange_sif_use.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = smchange_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smchange_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smchange_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smchange_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smchange_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(1,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.3)
    axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.4,0.5),labels = np.arange(9,11.4,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/sm_change_csc_biome_sif_log.png', dpi = 600)

# %%
plt.hexbin(np.log(smchange_sif_use.prec), smchange_sif_use.fsc, 
                C= smchange_sif_use.smchange_sif, gridsize = 100, 
                vmax=80, vmin=25,
                reduce_C_function = np.median)

# %%
plt.hexbin(smchange_sif_use.prec, smchange_sif_use.fsc, 
                C= smchange_sif_use.smchange_sif, gridsize = 100, 
                vmax=70, vmin=15,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smchange_sif_use.prec), smchange_sif_use.fsc, 
                C= smchange_sif_use.smchange_sif, gridsize = 100, 
                vmax=60, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_prec_sif.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smchange_sif_use.prec), smchange_sif_use.fsc, 
                C= smchange_sif_use.smchange_log, gridsize = 100, 
                vmax=4, vmin=2.5,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change (log)', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/sm_change_fsc_prec_sif_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smchange_sif_use.ai), smchange_sif_use.fsc, 
                C= smchange_sif_use.smchange_sif, gridsize = 100, 
                vmax=70, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Aridity index')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([-3, 2])
axes.set_xticks(np.arange(-3,2))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

#plt.savefig('result_fig/sm_change_fsc_ai.png', dpi = 600)

# %% [markdown]
# ### 3.2 gleam smrz

# %%
smrzchange_sif_df.head()

# %%
smrzchange_sif_q90 = smrzchange_sif_df.smrz_change_sif.quantile(0.9)
smrzchange_sif_q90

# %%
smrzchange_sif_use = smrzchange_sif_df[smrzchange_sif_df.smrz_change_sif < smrzchange_sif_q90]
smrzchange_sif_use

# %%
plt.hexbin(x =smrzchange_sif_use.fsc, y= smrzchange_sif_use.smrz_change_sif, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(smrzchange_sif_use.fsc.min(), smrzchange_sif_use.fsc.max() )

# %%
smrzchange_sif_use['fsc_bins'] = pd.cut(smrzchange_sif_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
smrzchange_sif_use.boxplot('smrz_change_sif', by='fsc_bins')

# %%
smrzchange_sif_use['smrz_change_log'] = np.log(smrzchange_sif_use.smrz_change_sif)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smrzchange_sif_use.fsc_bins)
fsc_dfs = [smrzchange_sif_use.smrz_change_sif[smrzchange_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smrzchange_sif_use.smrz_change_sif[smrzchange_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smrzchange_sif_use.smrz_change_sif[smrzchange_sif_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 125, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,130)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_sif.png', dpi = 600)


# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(smrzchange_sif_use.fsc_bins)
fsc_dfs = [smrzchange_sif_use.smrz_change_log[smrzchange_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(smrzchange_sif_use.smrz_change_log[smrzchange_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(smrzchange_sif_use.smrz_change_log[smrzchange_sif_use.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0.4,5.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_sif_log.png', dpi = 600)

# %%
smrzchange_sif_use.biome.value_counts()

# %%
smrzchange_sif_fsc_bybi = smrzchange_sif_use.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [125,90,125,110,130,90,40,60,40]

for i in range(9):
    bi_n = smrzchange_sif_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smrz_change_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.9,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/smrz_change_csc_biome_sif.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]


for i in range(9):
    bi_n = smrzchange_sif_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = smrzchange_sif_fsc_bybi.get_group(bi_n)
    labels_fsc = np.unique(smrzchange_sif_fsc_bybi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.smrz_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.smrz_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
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

fig.savefig('result_figure/smrz_change_csc_biome_sif_log.png', dpi = 600)

# %%
plt.hexbin(np.log(smrzchange_sif_use.prec), smrzchange_sif_use.fsc, 
                C= smrzchange_sif_use.smrz_change_sif, gridsize = 100, 
                vmax=80, vmin=20,
                reduce_C_function = np.median)

# %%
plt.hexbin(smrzchange_sif_use.prec, smrzchange_sif_use.fsc, 
                C= smrzchange_sif_use.smrz_change_sif, gridsize = 100, 
                vmax=70, vmin=15,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))
im = axes.hexbin(np.log(smrzchange_sif_use.prec), smrzchange_sif_use.fsc, 
                C= smrzchange_sif_use.smrz_change_sif, gridsize = 100, 
                vmax=60, vmin=10,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_prec_sif.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smrzchange_sif_use.prec), smrzchange_sif_use.fsc, 
                C= smrzchange_sif_use.smrz_change_log, gridsize = 100, 
                vmax=4, vmin=2.5,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change (log)', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/smrz_change_fsc_prec_sif_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(smrzchange_sif_use.ai), smrzchange_sif_use.fsc, 
                C= smrzchange_sif_use.smrz_change_sif, gridsize = 100, 
                vmax=70, vmin=10,
                reduce_C_function = np.median)
axes.set_xlabel('Aridity index')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([-3, 2])
axes.set_xticks(np.arange(-3,2))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

#plt.savefig('result_fig/sm_change_fsc_ai.png', dpi = 600)

# %% [markdown]
# ### 3.3 gleam ea

# %%
eachange_sif_q95 = eachange_sif_df.ea_change_sif.quantile(0.95)
eachange_sif_q95

# %%
eachange_sif_q5 = eachange_sif_df.ea_change_sif.quantile(0.05)
eachange_sif_q5

# %%
eachange_sif_use = eachange_sif_df[eachange_sif_df.ea_change_sif < eachange_sif_q95]
eachange_sif_use = eachange_sif_use[eachange_sif_use.ea_change_sif > eachange_sif_q5]
eachange_sif_use

# %%
plt.hexbin(x =eachange_sif_use.fsc, y= eachange_sif_use.ea_change_sif, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(eachange_sif_use.fsc.min(), eachange_sif_use.fsc.max())

# %%
eachange_sif_use['fsc_bins'] = pd.cut(eachange_sif_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
eachange_sif_use[eachange_sif_use.ea_change_sif>0].boxplot('ea_change_sif', by='fsc_bins')

# %%
eachange_sif_use[eachange_sif_use.ea_change_sif<0].boxplot('ea_change_sif', by='fsc_bins')

# %%
eachange_sif_use_po = eachange_sif_use[eachange_sif_use.ea_change_sif>0]
eachange_sif_use_ne = eachange_sif_use[eachange_sif_use.ea_change_sif<0]
eachange_sif_use_po['ea_change_log'] = np.log(eachange_sif_use_po.ea_change_sif)
eachange_sif_use_ne['ea_change_log'] = np.log(eachange_sif_use_ne.ea_change_sif * -1) * -1

# %%
eachange_sif_use_po

# %%
eachange_sif_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(eachange_sif_use_po.fsc_bins)
fsc_dfs = [eachange_sif_use_po.ea_change_sif[eachange_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_sif_use_po.ea_change_sif[eachange_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_sif_use_po.ea_change_sif[eachange_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 110, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.5,120)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_sif_po.png', dpi = 600)


# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(eachange_sif_use_po.fsc_bins)
fsc_dfs = [eachange_sif_use_po.ea_change_log[eachange_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_sif_use_po.ea_change_log[eachange_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_sif_use_po.ea_change_log[eachange_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,5.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_sif_po_log.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(eachange_sif_use_ne.fsc_bins)
fsc_dfs = [eachange_sif_use_ne.ea_change_sif[eachange_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_sif_use_ne.ea_change_sif[eachange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_sif_use_ne.ea_change_sif[eachange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -110, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-120,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_sif_ne.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(eachange_sif_use_ne.fsc_bins)
fsc_dfs = [eachange_sif_use_ne.ea_change_log[eachange_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(eachange_sif_use_ne.ea_change_log[eachange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(eachange_sif_use_ne.ea_change_log[eachange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('EA change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-5.5,0.5)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_sif_ne_log.png', dpi = 600)

# %%
eachange_sif_use.biome.value_counts()

# %%
eachange_sif_fsc_bybi_po = eachange_sif_use_po.groupby("biome")
eachange_sif_fsc_bybi_ne = eachange_sif_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [120,120,120,120,120,120,100,80,120]

for i in range(9):
    bi_n = eachange_sif_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(eachange_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-0.5,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_sif_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5,5.5,5.5,5,5.5]

for i in range(9):
    bi_n = eachange_sif_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(eachange_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
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
        axes[i//3, i %3].set_ylabel('EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_sif_po_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [110,80,80,90,90,110,50,60,40]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = eachange_sif_use_ne.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(eachange_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_sif_ne.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5, 5.5,5.5,5.5,5.5,5.5,5.5,5.5,5.5]


for i in range(9):
    bi_n = eachange_sif_use_ne.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = eachange_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(eachange_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.ea_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(y_list[i]* -1, 0.5)
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('EA change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
axes[5//3, 5 %3].set_ylim(y_list[5]* -1, 1)
plt.tight_layout()

fig.savefig('result_figure/ea_change_csc_biome_sif_ne_log.png', dpi = 600)

# %%
plt.hexbin(np.log(eachange_sif_use_po.prec), eachange_sif_use_po.fsc, 
                C= eachange_sif_use_po.ea_change_sif, gridsize = 100, 
                vmax=60, vmin=20,
                reduce_C_function = np.median)

# %%
plt.hexbin(np.log(eachange_sif_use_ne.prec), eachange_sif_use_ne.fsc, 
                C= eachange_sif_use_ne.ea_change_sif, gridsize = 100, 
                vmax=-10, vmin=-60,
                reduce_C_function = np.median)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(eachange_sif_use_po.prec), eachange_sif_use_po.fsc, 
                C= eachange_sif_use_po.ea_change_sif, gridsize = 100, 
                vmax=50, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'EA change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_prec_sif_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(eachange_sif_use_ne.prec), eachange_sif_use_ne.fsc, 
                C= eachange_sif_use_ne.ea_change_sif, gridsize = 100, 
                vmax=-10, vmin=-40,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
#axes[0].set_title('(a)', loc='left', size = 14)


plt.colorbar(im,  extend='both', label = 'EA change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_figure/ea_change_fsc_prec_sif_ne.png', dpi = 600)

# %% [markdown]
# ## change log
# 1. 2025.10.28 重新画图看了干旱年份 土壤水分 和 蒸散发变化和 结构复杂度的关系

# %%


# %%


# %%



