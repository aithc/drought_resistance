# %% [markdown]
# # 结构对干旱抵抗力的影响
# 
# 只用中位数

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
# ### 1.1 干旱抵抗力数据

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance.nc') as data:
    kndvi_resistance_nt = data['kndvi_resistance']
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance.nc') as data:
    kndvi_resistance_sh = data['kndvi_resistance']

with xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance.nc') as data:
    sif_resistance_nt = data['sif_resistance']
with xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance.nc') as data:
    sif_resistance_sh = data['sif_resistance']

# %%
kndvi_resistance_nt

# %%
kndvi_resistance_sh

# %%
sif_resistance_nt

# %%
sif_resistance_sh

# %%
kndvi_resistance_nt_mid = kndvi_resistance_nt.median(dim='year')
kndvi_resistance_sh_mid = kndvi_resistance_sh.median(dim='year')
kndvi_resistance_mid = xr.concat([kndvi_resistance_nt_mid,kndvi_resistance_sh_mid], dim='lat').sortby('lat')
kndvi_resistance_mid

# %%
kndvi_resistance_mid.plot(vmax = 300)

# %%
sif_resistance_nt_mid = sif_resistance_nt.median(dim='year')
sif_resistance_sh_mid = sif_resistance_sh.median(dim='year')
sif_resistance_mid = xr.concat([sif_resistance_nt_mid,sif_resistance_sh_mid], dim='lat').sortby('lat')
sif_resistance_mid

# %%
sif_resistance_mid.plot(vmax = 150)

# %% [markdown]
# ### 1.2 干旱指标  可以用之前的

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
# ### 1.3 多样性和结构复杂度

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
fsc.plot()

# %%
fsc = fsc.interp_like(plant_richness, method='nearest')

# %%
fsc.plot()

# %% [markdown]
# ### 1.4 背景气候

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
# ### 1.5 biome

# %%
## biome
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

# %% [markdown]
# ### 1.9 合并数据

# %%
drought_all = xr.Dataset({
    'resistance_kndvi':kndvi_resistance_mid,
    'resistance_sif':sif_resistance_mid,
    'richness':plant_richness,
    'fsc':fsc,
    'drought_count':drought_count,
    'drought_duration':drought_duration,
    'drought_severity':drought_severity,
    'temp':annual_temp,
    'prec':annual_prec,
    'ai':ai_index,
    'biome':biome,
    'cec':soil_cec,
    'clay':soil_clay,
    'cti':cti,
    'sla':sla,
    'wood_den':wood_den
})
drought_all

# %%
drought_all_df = drought_all.to_dataframe()
drought_all_df.dropna(how='all')
drought_all_df.describe()

# %%
drought_resistance_kndvi_df = drought_all_df.drop(['resistance_sif'], axis=1)
drought_resistance_kndvi_df = drought_resistance_kndvi_df.dropna(how='any')
drought_resistance_kndvi_df.describe()

# %%
drought_resistance_sif_df = drought_all_df.drop(['resistance_kndvi'], axis=1)
drought_resistance_sif_df = drought_resistance_sif_df.dropna(how='any')
drought_resistance_sif_df.describe()

# %%
drought_resistance_kndvi_df.to_csv(r'E:/python_output/fsc_drought/drought_resistance_kndvi_df_csc.csv', index = False)
drought_resistance_sif_df.to_csv(r'E:/python_output/fsc_drought/drought_resilience_sif_df_csc.csv', index = False)

# %% [markdown]
# ## 2 kndvi 分析

# %% [markdown]
# ### 2.1 简单分析

# %% [markdown]
# #### 2.1.1  简单查看 数据

# %%
drought_resistance_kndvi_df.resistance_kndvi.quantile(0.9)

# %%
drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233].plot.scatter('fsc','resistance_kndvi')

# %%
drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233].plot.scatter('richness','resistance_kndvi')

# %%
drought_resistance_kndvi_df.resistance_kndvi[drought_resistance_kndvi_df.resistance_kndvi < 233].plot.hist()

# %%
plt.hexbin(x =drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233].fsc, 
            y= np.log(drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233].resistance_kndvi),
            gridsize = 50, bins='log')
plt.colorbar()

# %%
plt.hexbin(x =drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233].richness, 
            y= np.log(drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233].resistance_kndvi),
            gridsize = 50, bins='log')
plt.colorbar()

# %% [markdown]
# #### 2.1.2 简单分bins画图

# %%
drought_resistance_kndvi_df.fsc.max()

# %%
drought_resistance_kndvi_df.fsc.min()


# %%
drought_resistance_kndvi_df['fsc_bins'] = pd.cut(drought_resistance_kndvi_df.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
pd.unique(drought_resistance_kndvi_df['fsc_bins'])

# %%
drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 334].boxplot('resistance_kndvi', by='fsc_bins')

# %%
drought_resistance_kndvi_df.richness.max()

# %%
drought_resistance_kndvi_df['rich_bins'] = pd.cut(drought_resistance_kndvi_df.richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %%
drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 334].boxplot('resistance_kndvi', by='rich_bins')

# %%
drought_resistance_kndvi_df.boxplot('fsc', by='rich_bins')

# %% [markdown]
# #### 2.1.3 简单相关

# %%
drought_resistance_kndvi_use = drought_resistance_kndvi_df[drought_resistance_kndvi_df.resistance_kndvi < 233]

# %%
drought_resistance_kndvi_use['resistance_log'] = np.log(drought_resistance_kndvi_df['resistance_kndvi'] )

# %%
drought_resistance_kndvi_use['resistance_log'].plot.hist()

# %%
drought_resistance_kndvi_use['resistance_log'].min()

# %%
drought_resistance_kndvi_use.columns

# %%
drought_resistance_cor_kndvi = drought_resistance_kndvi_use.drop(['fsc_bins','rich_bins','biome','resistance_log'],axis=1).corr(method='spearman')
drought_resistance_cor_kndvi

# %%
var_name_raw = drought_resistance_cor_kndvi.index
var_name_raw

# %%
var_name = ['Drought resistance','Species richness','Forest structural complexity','Drought counts','Mean drought duration',
            'Mean drought severity','Mean annual temperature','Mean annual precipitation','Aridity index','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']

# %%
drought_resistance_np_kndvi = np.asarray(drought_resistance_cor_kndvi)
drought_resistance_np_kndvi

# %%
p_value_kndvi = np.full_like(drought_resistance_np_kndvi, fill_value=np.nan)
p_value_kndvi

# %%
from scipy.stats import spearmanr
for i in range(drought_resistance_np_kndvi.shape[0]):
    for j in range(drought_resistance_np_kndvi.shape[1]):
        
        p_value_kndvi[i,j] = spearmanr(drought_resistance_kndvi_use[var_name_raw[i]],drought_resistance_kndvi_use[var_name_raw[j]])[1]
p_value_kndvi

# %%
drought_resistance_np_kndvi = np.where(p_value_kndvi<0.001,drought_resistance_np_kndvi, np.nan)
for i in range(drought_resistance_np_kndvi.shape[0]):
    for j in range(drought_resistance_np_kndvi.shape[1]):
        if i <= j:
            drought_resistance_np_kndvi[i,j] = np.nan
drought_resistance_np_kndvi

# %%
drought_resistance_np_kndvi.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drought_resistance_np_kndvi[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(13))
ax.set_yticks(ticks=np.arange(13))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,14):
    for j in range(13):
        if ~np.isnan(drought_resistance_np_kndvi[i,j]):
            ax.text(j, i-1, str(round(drought_resistance_np_kndvi[i,j],3)), ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/cor_resistance_kndvi_csc_kndvi.png', dpi = 600)

# %% [markdown]
# ### 2.2 初步的 抵抗力-多样性-结构复杂度 关系

# %% [markdown]
# #### 2.2.1 所有数据

# %%
fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].hexbin(drought_resistance_kndvi_use.richness, drought_resistance_kndvi_use.fsc, 
                C= drought_resistance_kndvi_use.resistance_kndvi, gridsize = 100, 
                vmax=150, vmin=0,
                reduce_C_function = np.median)
axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Forest structural complexity')
axes[0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

axes[1].hexbin(np.log(drought_resistance_kndvi_use.prec), drought_resistance_kndvi_use.richness, 
                C= drought_resistance_kndvi_use.resistance_kndvi, gridsize = 100, 
                vmax=150, vmin=0,
                reduce_C_function = np.median)
axes[1].set_xlabel('Annual precipitation (log) / mm')
axes[1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

im = axes[2].hexbin(np.log(drought_resistance_kndvi_use.prec), drought_resistance_kndvi_use.fsc, 
                C= drought_resistance_kndvi_use.resistance_kndvi, gridsize = 100, 
                vmax=150, vmin=0,
                reduce_C_function = np.median)
axes[2].set_xlabel('Annual precipitation (log) / mm')
axes[2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

position=fig.add_axes([0.2,0.1,0.6,0.04])
plt.colorbar(im, position, extend='both', label = 'Resistance',orientation='horizontal')

plt.subplots_adjust(top=0.9, bottom=0.25, right=0.98, left=0.06, wspace=0.24)

plt.savefig('result_figure/resistance_richness_fsc_prec_kndvi_csc_mid.png', dpi = 600)

# %%
fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].hexbin(drought_resistance_kndvi_use.richness, drought_resistance_kndvi_use.fsc, 
                C= drought_resistance_kndvi_use.resistance_log, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Forest structural complexity')
axes[0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

axes[1].hexbin(np.log(drought_resistance_kndvi_use.prec), drought_resistance_kndvi_use.richness, 
                C= drought_resistance_kndvi_use.resistance_log, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[1].set_xlabel('Annual precipitation (log) / mm')
axes[1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

im = axes[2].hexbin(np.log(drought_resistance_kndvi_use.prec), drought_resistance_kndvi_use.fsc, 
                C= drought_resistance_kndvi_use.resistance_log, gridsize = 100, 
                vmax=5, vmin=1,
                reduce_C_function = np.median)
axes[2].set_xlabel('Annual precipitation (log) / mm')
axes[2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

position=fig.add_axes([0.2,0.1,0.6,0.04])
plt.colorbar(im, position, extend='both', label = 'Resistance (log)',orientation='horizontal')

plt.subplots_adjust(top=0.9, bottom=0.25, right=0.98, left=0.06, wspace=0.24)

plt.savefig('result_figure/resistance_richness_fsc_prec_kndvi_csc_mid_log.png', dpi = 600)

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

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(drought_resistance_kndvi_use.biome)
bi_dfs = [drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 220, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,240)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(drought_resistance_kndvi_use.biome)
bi_dfs = [drought_resistance_kndvi_use.richness[drought_resistance_kndvi_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.richness[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.richness[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(drought_resistance_kndvi_use.biome)
bi_dfs = [drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.5, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/resistance_biome_kndvi_csc_mid.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(drought_resistance_kndvi_use.biome)
bi_dfs = [drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15]
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
labels_bi = np.unique(drought_resistance_kndvi_use.biome)
bi_dfs = [drought_resistance_kndvi_use.richness[drought_resistance_kndvi_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.richness[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.richness[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(drought_resistance_kndvi_use.biome)
bi_dfs = [drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.biome == labels_bi_n]) > 15]
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

plt.savefig('result_figure/resistance_biome_kndvi_csc_mid_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(drought_resistance_kndvi_use.fsc_bins)
fsc_dfs = [drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 210, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,240)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(drought_resistance_kndvi_use.rich_bins)
rich_dfs = [drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.resistance_kndvi[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 210, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Resistance')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,240)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/richness_fsc_resistance_kndvi_csc_mid.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(drought_resistance_kndvi_use.fsc_bins)
fsc_dfs = [drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 6, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(drought_resistance_kndvi_use.rich_bins)
rich_dfs = [drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.resistance_log[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 6, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,6.5)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_kndvi_use.fsc[drought_resistance_kndvi_use.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/richness_fsc_resistance_kndvi_csc_mid_log.png', dpi = 600)

# %% [markdown]
# #### 2.2.2 分biome

# %%
drought_resistance_kndvi_use.biome.value_counts()

# %%
grouped_resistance_kndvi_by_bi = drought_resistance_kndvi_use.groupby("biome")
drought_resistance_kndvi_use.biome.value_counts().index

# %%
drought_resistance_kndvi_use.biome.value_counts().sort_values()

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
alpha_list = ['a','b','c','d','e','f','g','h','i']

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [245,245,180,150,180,240,60,60,60]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resistance_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resistance_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resistance_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
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

fig.savefig('result_figure/fsc_resistance_biome_kndvi_csc.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.5,6.5,6.5,6.5,6.5,6.5,6,5,5]
y_list_low = [0.5,0.5,0,0.5,0.5,0.5,0,0.5,0.5]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resistance_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resistance_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resistance_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
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

fig.savefig('result_figure/fsc_resistance_biome_kndvi_csc_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [245,125,170,100,150,220,35,60,50]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resistance_kndvi[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resistance_kndvi[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resistance_kndvi[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list[i]*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

plt.tight_layout()

fig.savefig('result_figure/richness_resistance_biome_kndvi_csc.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6.5,6.5,6.5,6.5,6.5,6.5,5,6.5,5]
y_list_low = [0.5,0.5,0.5,0,0.5,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_kndvi_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resistance_log[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resistance_log[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resistance_log[bi_df.rich_bins == labels_rich_n]) > 15]
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

fig.savefig('result_figure/richness_resistance_biome_kndvi_csc_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [12,11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5]
y_list_low = [9,8.5,9.5,9,8.5,8,8,8,8]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_kndvi_use.biome.value_counts().index[:-5].sort_values()[i]
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

fig.savefig('result_figure/richness_resistance_biome_fsc.png', dpi = 600)

# %% [markdown]
# #### 2.2.3 偏相关

# %%
import pingouin as pg
from scipy import stats

# %%
grouped_resistance_kndvi_by_bi = drought_resistance_kndvi_use.groupby("biome")

# %%
bi_cor_fsc= {}
bi_cor_rich = {}

# %%
for bi_n in drought_resistance_kndvi_use.biome.value_counts().index[:-5]:
    print(bi_n)
    bi_group_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['fsc'])[1]}
    bi_cor_rich[bi_n] = {'r':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['richness'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['richness'])[1]}

# %%
bi_cor_fsc

# %%
bi_cor_rich

# %%
stats.spearmanr(drought_resistance_kndvi_use['resistance_log'],drought_resistance_kndvi_use['fsc'])

# %%
stats.spearmanr(drought_resistance_kndvi_use['resistance_log'],drought_resistance_kndvi_use['richness'])

# %%
bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df

# %%
bi_pd_rich_cor_df = pd.DataFrame(bi_cor_rich).T
bi_pd_rich_cor_df['var'] = 'rich'
bi_pd_rich_cor_df = bi_pd_rich_cor_df.sort_index()
bi_pd_rich_cor_df

# %%
fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.404)
rich_cor_list = list(bi_pd_rich_cor_df.r)
rich_cor_list.append(0.400)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)

# %%

bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r

# %%
fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
rich_cor_list = list(bi_pd_rich_cor_df['p-val'])
rich_cor_list.append(0)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)

# %%

bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p

# %%
bi_pd_fsc_rich_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_rich_fsc_pcor = {}
bi_pd_rich_prec_pcor = {}
for bi_n in drought_resistance_kndvi_use.biome.value_counts().index[:-5]:
    print(bi_n)
    bi_group_df = grouped_resistance_kndvi_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='fsc',covar=['richness'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='fsc',covar=['prec'],method='spearman').round(4)
    rich_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='richness',covar=['fsc'],method='spearman').round(4)
    rich_prec_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='richness',covar=['prec'],method='spearman').round(4)
    bi_pd_fsc_rich_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_rich_fsc_pcor[bi_n] = {'r':rich_pcor['r'].values[0], 'p-val': rich_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_rich_prec_pcor[bi_n] = {'r':rich_prec_pcor['r'].values[0], 'p-val': rich_prec_pcor['p-val'].values[0]}

# %%
bi_pd_fsc_rich_pcor

# %%
bi_pd_fsc_prec_pcor

# %%
bi_pd_rich_fsc_pcor

# %%
bi_pd_rich_prec_pcor

# %%
pg.partial_corr(data = drought_resistance_kndvi_use,y='resistance_log',x='richness',covar=['prec'], method='spearman').round(3)

# %%
pg.partial_corr(data = drought_resistance_kndvi_use,y='resistance_log',x='richness',covar=['fsc'], method='spearman').round(3)

# %%
pg.partial_corr(data = drought_resistance_kndvi_use,y='resistance_log',x='fsc',covar=['prec'], method='spearman').round(3)

# %%
pg.partial_corr(data = drought_resistance_kndvi_use,y='resistance_log',x='fsc',covar=['richness'], method='spearman').round(3)

# %%
bi_pd_fsc_rich_pcor_df = pd.DataFrame(bi_pd_fsc_rich_pcor).T
bi_pd_fsc_rich_pcor_df['var'] = 'fsc_rich'
bi_pd_fsc_rich_pcor_df = bi_pd_fsc_rich_pcor_df.sort_index()
bi_pd_fsc_rich_pcor_df

# %%
bi_pd_rich_fsc_pcor_df = pd.DataFrame(bi_pd_rich_fsc_pcor).T
bi_pd_rich_fsc_pcor_df['var'] = 'rich_fsc'
bi_pd_rich_fsc_pcor_df = bi_pd_rich_fsc_pcor_df.sort_index()
bi_pd_rich_fsc_pcor_df

# %%
bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df

# %%
bi_pd_rich_prec_pcor_df = pd.DataFrame(bi_pd_rich_prec_pcor).T
bi_pd_rich_prec_pcor_df['var'] = 'rich_prec'
bi_pd_rich_prec_pcor_df = bi_pd_rich_prec_pcor_df.sort_index()
bi_pd_rich_prec_pcor_df

# %%
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df.r)
fsc_rich_list.append(0.434)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.284)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df.r)
rich_fsc_list.append(0.431)
rich_prec_list = list(bi_pd_rich_prec_pcor_df.r)
rich_prec_list.append(0.025)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)

# %%

bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pcor_df

# %%
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df['p-val'])
fsc_rich_list.append(0)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df['p-val'])
fsc_prec_list.append(0)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df['p-val'])
rich_fsc_list.append(0)
rich_prec_list = list(bi_pd_rich_prec_pcor_df['p-val'])
rich_prec_list.append(0.001)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)

# %%

bi_pval_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pval_df

# %%
bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
bi_pd_pcor_all

# %%
bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
bi_pd_pval_all

# %%
bi_pd_pcor_all.to_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_kndvi_mid.csv', index = False)
bi_pd_pval_all.to_csv('E:/python_output/fsc_drought/bi_pd_pval_all_kndvi_mid.csv', index = False)

# %%
draw_col = ['rich','rich_prec','rich_fsc','fsc','fsc_prec','fsc_rich']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(bi_pd_pcor_all[draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
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

plt.savefig('result_figure/pcor_biome_resistance_cor_pcor_kndvi_csc_mid.png', dpi = 600)

# %% [markdown]
# ## 3 sif 的分析

# %% [markdown]
# ### 3.1 简单分析

# %% [markdown]
# #### 3.1.1 简单查看数据

# %%
drought_resistance_sif_df.resistance_sif.quantile(0.9)

# %%
drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].plot.scatter('fsc','resistance_sif')

# %%
drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].plot.scatter('richness','resistance_sif')

# %%
drought_resistance_sif_df.resistance_sif[drought_resistance_sif_df.resistance_sif < 116].plot.hist()

# %%
plt.hexbin(x =drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].fsc, 
            y= np.log(drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].resistance_sif),
            gridsize = 50, bins='log')
plt.colorbar()

# %%
plt.hexbin(x =drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].richness, 
            y= np.log(drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].resistance_sif),
            gridsize = 50, bins='log')
plt.colorbar()

# %% [markdown]
# #### 2.1.2 简单分bins画图

# %%
print(drought_resistance_sif_df.fsc.max() ,drought_resistance_sif_df.fsc.min()) 

# %%
drought_resistance_sif_df['fsc_bins'] = pd.cut(drought_resistance_sif_df.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
pd.unique(drought_resistance_sif_df['fsc_bins'])

# %%
drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].boxplot('resistance_sif', by='fsc_bins')

# %%
drought_resistance_sif_df.richness.max()

# %%
drought_resistance_sif_df['rich_bins'] = pd.cut(drought_resistance_sif_df.richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])
drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116].boxplot('resistance_sif', by='rich_bins')

# %%
drought_resistance_sif_df.boxplot('fsc', by='rich_bins')

# %% [markdown]
# #### 3.1.3 简单相关

# %%
drought_resistance_sif_use = drought_resistance_sif_df[drought_resistance_sif_df.resistance_sif < 116]
drought_resistance_sif_use['resistance_log'] = np.log(drought_resistance_sif_df['resistance_sif'] )
drought_resistance_sif_use['resistance_log'].plot.hist()

# %%
drought_resistance_sif_use['resistance_log'].min()

# %%
drought_resistance_sif_use.columns

# %%
drought_resistance_cor_sif = drought_resistance_sif_use.drop(['fsc_bins','rich_bins','biome','resistance_log'],axis=1).corr(method='spearman')
drought_resistance_cor_sif

# %%
var_name_raw = drought_resistance_cor_sif.index
var_name_raw

# %%
var_name = ['Drought resistance','Species richness','Forest structural complexity','Drought counts','Mean drought duration',
            'Mean drought severity','Mean annual temperature','Mean annual precipitation','Aridity index','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']
drought_resistance_np_sif = np.asarray(drought_resistance_cor_sif)
drought_resistance_np_sif

# %%
p_value_sif = np.full_like(drought_resistance_np_sif, fill_value=np.nan)
p_value_sif

# %%
for i in range(drought_resistance_np_sif.shape[0]):
    for j in range(drought_resistance_np_sif.shape[1]):
        
        p_value_sif[i,j] = spearmanr(drought_resistance_sif_use[var_name_raw[i]],drought_resistance_sif_use[var_name_raw[j]])[1]
p_value_sif

# %%
drought_resistance_np_sif = np.where(p_value_sif<0.001,drought_resistance_np_sif, np.nan)
for i in range(drought_resistance_np_sif.shape[0]):
    for j in range(drought_resistance_np_sif.shape[1]):
        if i <= j:
            drought_resistance_np_sif[i,j] = np.nan
drought_resistance_np_sif

# %%
drought_resistance_np_sif.shape

# %%
fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drought_resistance_np_sif[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(13))
ax.set_yticks(ticks=np.arange(13))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,14):
    for j in range(13):
        if ~np.isnan(drought_resistance_np_sif[i,j]):
            ax.text(j, i-1, str(round(drought_resistance_np_sif[i,j],3)), ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_figure/cor_resistance_sif_csc_sif.png', dpi = 600)

# %% [markdown]
# ### 3.2 初步的 抵抗力-多样性-结构复杂度 关系

# %% [markdown]
# #### 3.2.1 所有数据

# %%
fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].hexbin(drought_resistance_sif_use.richness, drought_resistance_sif_use.fsc, 
                C= drought_resistance_sif_use.resistance_sif, gridsize = 100, 
                vmax=70, vmin=0,
                reduce_C_function = np.median)
axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Forest structural complexity')
axes[0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

axes[1].hexbin(np.log(drought_resistance_sif_use.prec), drought_resistance_sif_use.richness, 
                C= drought_resistance_sif_use.resistance_sif, gridsize = 100, 
                vmax=70, vmin=0,
                reduce_C_function = np.median)
axes[1].set_xlabel('Annual precipitation (log) / mm')
axes[1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

im = axes[2].hexbin(np.log(drought_resistance_sif_use.prec), drought_resistance_sif_use.fsc, 
                C= drought_resistance_sif_use.resistance_sif, gridsize = 100, 
                vmax=70, vmin=0,
                reduce_C_function = np.median)
axes[2].set_xlabel('Annual precipitation (log) / mm')
axes[2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

position=fig.add_axes([0.2,0.1,0.6,0.04])
plt.colorbar(im, position, extend='max', label = 'Resistance',orientation='horizontal')

plt.subplots_adjust(top=0.9, bottom=0.25, right=0.98, left=0.06, wspace=0.24)

plt.savefig('result_figure/resistance_richness_fsc_prec_sif_csc_mid.png', dpi = 600)

# %%
fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].hexbin(drought_resistance_sif_use.richness, drought_resistance_sif_use.fsc, 
                C= drought_resistance_sif_use.resistance_log, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Forest structural complexity')
axes[0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

axes[1].hexbin(np.log(drought_resistance_sif_use.prec), drought_resistance_sif_use.richness, 
                C= drought_resistance_sif_use.resistance_log, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[1].set_xlabel('Annual precipitation (log) / mm')
axes[1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

im = axes[2].hexbin(np.log(drought_resistance_sif_use.prec), drought_resistance_sif_use.fsc, 
                C= drought_resistance_sif_use.resistance_log, gridsize = 100, 
                vmax=4, vmin=1,
                reduce_C_function = np.median)
axes[2].set_xlabel('Annual precipitation (log) / mm')
axes[2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

position=fig.add_axes([0.2,0.1,0.6,0.04])
plt.colorbar(im, position, extend='both', label = 'Resistance (log)',orientation='horizontal')

plt.subplots_adjust(top=0.9, bottom=0.25, right=0.98, left=0.06, wspace=0.24)

plt.savefig('result_figure/resistance_richness_fsc_prec_sif_csc_mid_log.png', dpi = 600)

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

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(drought_resistance_sif_use.biome)
bi_dfs = [drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 110, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,120)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(drought_resistance_sif_use.biome)
bi_dfs = [drought_resistance_sif_use.richness[drought_resistance_sif_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.richness[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_sif_use.richness[drought_resistance_sif_use.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(drought_resistance_sif_use.biome)
bi_dfs = [drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 11.5, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(bi_dfs_names, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(8,12)
axes[2].set_xlim(0.2,13.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_figure/resistance_biome_sif_csc_mid.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(drought_resistance_sif_use.biome)
bi_dfs = [drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 5.4, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,6)
axes[0].set_xlim(0.2,13.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(drought_resistance_sif_use.biome)
bi_dfs = [drought_resistance_sif_use.richness[drought_resistance_sif_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.richness[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_sif_use.richness[drought_resistance_sif_use.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,13.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(drought_resistance_sif_use.biome)
bi_dfs = [drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
bi_dfs_names = [ biome_dic2[labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n]) > 15 ]
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.biome == labels_bi_n]) > 15]
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

plt.savefig('result_figure/resistance_biome_sif_csc_mid_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(drought_resistance_sif_use.fsc_bins)
fsc_dfs = [drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 115, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,125)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(drought_resistance_sif_use.rich_bins)
rich_dfs = [drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_sif_use.resistance_sif[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 115, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Resistance')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,125)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [drought_resistance_sif_use.fsc[drought_resistance_sif_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/richness_fsc_resistance_sif_csc_mid.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(drought_resistance_sif_use.fsc_bins)
fsc_dfs = [drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 5, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance (log)')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,6)
axes[1].set_xlim(8.7,11.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(drought_resistance_sif_use.rich_bins)
rich_dfs = [drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_sif_use.resistance_log[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 5, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Resistance (log)')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,6)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [drought_resistance_sif_use.fsc[drought_resistance_sif_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_sif_use.fsc[drought_resistance_sif_use.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.4, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 11.5, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(8,12)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)

fig.savefig(r'result_figure/richness_fsc_resistance_sif_csc_mid_log.png', dpi = 600)

# %% [markdown]
# #### 2.2.2 分biome

# %%
drought_resistance_sif_use.biome.value_counts()

# %%
grouped_resistance_sif_by_bi = drought_resistance_sif_use.groupby("biome")
drought_resistance_sif_use.biome.value_counts().index

# %%
drought_resistance_sif_use.biome.value_counts().sort_values()

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
alpha_list = ['a','b','c','d','e','f','g','h','i']
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [130,90,120,120,120,120,60,60,30]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_sif_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resistance_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resistance_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resistance_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
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

fig.savefig('result_figure/fsc_resistance_biome_sif_csc.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [6,6,6,6,6,6,5,5,5]
y_list_low = [0,0.5,0,0,0.5,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_sif_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resistance_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resistance_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resistance_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
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

fig.savefig('result_figure/fsc_resistance_biome_sif_csc_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [130,90,120,110,110,130,50,60,30]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_sif_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resistance_sif[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resistance_sif[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resistance_sif[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = y_list[i]*0.9, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,y_list[i])
    axes[i//3, i %3].set_xlim(0.5,5.5)
    axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(1,6,1),labels = np.arange(1,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

plt.tight_layout()

fig.savefig('result_figure/richness_resistance_biome_sif_csc.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [5.5,5.5,5.5,5.5,5.5,5.5,5,5.5,5]
y_list_low = [0.5,0.5,0.5,0,0.5,0,0,0,0]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_sif_use.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_sif_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resistance_log[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resistance_log[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resistance_log[bi_df.rich_bins == labels_rich_n]) > 15]
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

fig.savefig('result_figure/richness_resistance_biome_sif_csc_log.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [12,11.5,11.5,11.5,11.5,11.5,11.5,11.5,11.5]
y_list_low = [9,8.5,9.5,9,8.5,8.5,8,8,8]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(9):
    bi_n = drought_resistance_sif_use.biome.value_counts().index[:-5].sort_values()[i]
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

fig.savefig('result_figure/richness_resistance_biome_fsc.png', dpi = 600)

# %% [markdown]
# #### 2.2.3 偏相关

# %%
import pingouin as pg
from scipy import stats

# %%
grouped_resistance_sif_by_bi = drought_resistance_sif_use.groupby("biome")
bi_cor_fsc= {}
bi_cor_rich = {}

for bi_n in drought_resistance_sif_use.biome.value_counts().index[:-5]:
    print(bi_n)
    bi_group_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['fsc'])[1]}
    bi_cor_rich[bi_n] = {'r':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['richness'])[0],
                        'p-val':stats.spearmanr(bi_group_df['resistance_log'],bi_group_df['richness'])[1]}

# %%
bi_cor_fsc

# %%
bi_cor_rich

# %%
stats.spearmanr(drought_resistance_sif_use['resistance_log'],drought_resistance_sif_use['fsc'])

# %%
stats.spearmanr(drought_resistance_sif_use['resistance_log'],drought_resistance_sif_use['richness'])

# %%
bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df

# %%
bi_pd_rich_cor_df = pd.DataFrame(bi_cor_rich).T
bi_pd_rich_cor_df['var'] = 'rich'
bi_pd_rich_cor_df = bi_pd_rich_cor_df.sort_index()
bi_pd_rich_cor_df

# %%
fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.443)
rich_cor_list = list(bi_pd_rich_cor_df.r)
rich_cor_list.append(0.324)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)

# %%
bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r

# %%
fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
rich_cor_list = list(bi_pd_rich_cor_df['p-val'])
rich_cor_list.append(0)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)

# %%
bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p

# %%
bi_pd_fsc_rich_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_rich_fsc_pcor = {}
bi_pd_rich_prec_pcor = {}
for bi_n in drought_resistance_sif_use.biome.value_counts().index[:-5]:
    print(bi_n)
    bi_group_df = grouped_resistance_sif_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='fsc',covar=['richness'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='fsc',covar=['prec'],method='spearman').round(4)
    rich_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='richness',covar=['fsc'],method='spearman').round(4)
    rich_prec_pcor = pg.partial_corr(data = bi_group_df,y='resistance_log',x='richness',covar=['prec'],method='spearman').round(4)
    bi_pd_fsc_rich_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_rich_fsc_pcor[bi_n] = {'r':rich_pcor['r'].values[0], 'p-val': rich_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_rich_prec_pcor[bi_n] = {'r':rich_prec_pcor['r'].values[0], 'p-val': rich_prec_pcor['p-val'].values[0]}

# %%
bi_pd_fsc_rich_pcor

# %%
bi_pd_fsc_prec_pcor

# %%
bi_pd_rich_fsc_pcor

# %%
bi_pd_rich_prec_pcor

# %%
pg.partial_corr(data = drought_resistance_sif_use,y='resistance_log',x='richness',covar=['prec'], method='spearman').round(3)

# %%
pg.partial_corr(data = drought_resistance_sif_use,y='resistance_log',x='richness',covar=['fsc'], method='spearman').round(3)

# %%
pg.partial_corr(data = drought_resistance_sif_use,y='resistance_log',x='fsc',covar=['prec'], method='spearman').round(3)

# %%
pg.partial_corr(data = drought_resistance_sif_use,y='resistance_log',x='fsc',covar=['richness'], method='spearman').round(3)

# %%
bi_pd_fsc_rich_pcor_df = pd.DataFrame(bi_pd_fsc_rich_pcor).T
bi_pd_fsc_rich_pcor_df['var'] = 'fsc_rich'
bi_pd_fsc_rich_pcor_df = bi_pd_fsc_rich_pcor_df.sort_index()
bi_pd_fsc_rich_pcor_df

# %%
bi_pd_rich_fsc_pcor_df = pd.DataFrame(bi_pd_rich_fsc_pcor).T
bi_pd_rich_fsc_pcor_df['var'] = 'rich_fsc'
bi_pd_rich_fsc_pcor_df = bi_pd_rich_fsc_pcor_df.sort_index()
bi_pd_rich_fsc_pcor_df

# %%
bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df

# %%
bi_pd_rich_prec_pcor_df = pd.DataFrame(bi_pd_rich_prec_pcor).T
bi_pd_rich_prec_pcor_df['var'] = 'rich_prec'
bi_pd_rich_prec_pcor_df = bi_pd_rich_prec_pcor_df.sort_index()
bi_pd_rich_prec_pcor_df

# %%
fsc_rich_list = list(bi_pd_fsc_rich_pcor_df.r)
fsc_rich_list.append(0.442)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.314)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df.r)
rich_fsc_list.append(0.323)
rich_prec_list = list(bi_pd_rich_prec_pcor_df.r)
rich_prec_list.append(-0.09)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)

# %%

bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pcor_df

# %%
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

# %%

bi_pval_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pval_df

# %%
bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
bi_pd_pcor_all

# %%
bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
bi_pd_pval_all

# %%
bi_pd_pcor_all.to_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_sif_mid.csv', index = False)
bi_pd_pval_all.to_csv('E:/python_output/fsc_drought/bi_pd_pval_all_sif_mid.csv', index = False)

# %%
draw_col = ['rich','rich_prec','rich_fsc','fsc','fsc_prec','fsc_rich']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(bi_pd_pcor_all[draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(10))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
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

plt.savefig('result_figure/pcor_biome_resistance_cor_pcor_sif_csc_mid.png', dpi = 600)

# %% [markdown]
# ## change log
# 1. 2025.10.27  重新做了 干旱抵抗力和 多样性 结构复杂度的关系

# %%


# %%



