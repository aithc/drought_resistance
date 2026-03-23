# %% [markdown]
# # fsc 和 lst变化的关系

# %%
import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import rioxarray
import pandas as pd
import glob
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
# ### 1.3 干旱指数

# %%
with rioxarray.open_rasterio(r'D:/data/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif')  as data:
        ai_index= xr.DataArray(data.values[0], coords=[data.y, data.x], dims=['lat','lon'])
ai_index = ai_index.coarsen(lat=10,lon=10).mean()
ai_index

# %%
ai_index = ai_index * 0.0001

# %%
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
# ### 1.8 LST 变化

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_change_kndvi.nc') as data:
    lst_nt_change_kndvi = data['lst_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_change_kndvi.nc') as data:
    lst_sh_change_kndvi = data['lst_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_change_sif.nc') as data:
    lst_nt_change_sif = data['lst_change']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_change_sif.nc') as data:
    lst_sh_change_sif = data['lst_change']

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_zs_kndvi.nc') as data:
    lst_nt_zs_kndvi = data['lst_zs']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_zs_kndvi.nc') as data:
    lst_sh_zs_kndvi = data['lst_zs']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_zs_sif.nc') as data:
    lst_nt_zs_sif = data['lst_zs']
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_zs_sif.nc') as data:
    lst_sh_zs_sif = data['lst_zs']

# %%
lst_nt_zs_kndvi[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, center = 0, cmap = 'RdBu_r')

# %%
lst_nt_change_kndvi_mid = lst_nt_change_kndvi.median(dim='year')
lst_sh_change_kndvi_mid = lst_sh_change_kndvi.median(dim='year')

lst_nt_change_sif_mid = lst_nt_change_sif.median(dim='year')
lst_sh_change_sif_mid = lst_sh_change_sif.median(dim='year')

# %%
lst_nt_zs_kndvi_mid = lst_nt_zs_kndvi.median(dim='year')
lst_sh_zs_kndvi_mid = lst_sh_zs_kndvi.median(dim='year')

lst_nt_zs_sif_mid = lst_nt_zs_sif.median(dim='year')
lst_sh_zs_sif_mid = lst_sh_zs_sif.median(dim='year')

# %%
lst_nt_zs_kndvi_mid

# %%
lst_sh_zs_kndvi_mid.plot()

# %%
lst_nt_change_kndvi_mid.plot()

# %%
lst_change_kndvi = xr.concat([lst_nt_change_kndvi_mid,lst_sh_change_kndvi_mid], dim='lat').sortby('lat')
lst_change_kndvi

# %%
lst_change_kndvi.plot()

# %%
lst_change_sif = xr.concat([lst_nt_change_sif_mid,lst_sh_change_sif_mid], dim='lat').sortby('lat')
lst_change_sif

# %%
lst_change_sif.plot(vmax = 1000, vmin = -1000)

# %%
lst_zs_kndvi = xr.concat([lst_nt_zs_kndvi_mid,lst_sh_zs_kndvi_mid], dim='lat').sortby('lat')
lst_zs_kndvi

# %%
lst_zs_kndvi.plot()

# %%
lst_zs_sif = xr.concat([lst_nt_zs_sif_mid,lst_sh_zs_sif_mid], dim='lat').sortby('lat')
lst_zs_sif

# %%
lst_zs_sif.plot()

# %% [markdown]
# ### 1.9  合并

# %%
lstchange_all = xr.Dataset({
    'lstchange_kndvi':lst_change_kndvi,
    'lstchange_sif':lst_change_sif,
    'lstzs_kndvi':lst_zs_kndvi,
    'lstzs_sif':lst_zs_sif,
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
lstchange_all

# %%
lstchange_all = lstchange_all.drop('quantile')

# %%
lstchange_all_df = lstchange_all.to_dataframe()
lstchange_all_df.dropna(how='all')
lstchange_all_df.describe()

# %%
lstchange_kndvi_df = lstchange_all_df.drop(['lstchange_sif','lstzs_kndvi','lstzs_sif'], axis=1)
lstchange_kndvi_df = lstchange_kndvi_df.dropna(how='any')
lstchange_kndvi_df.describe()

# %%
lstchange_sif_df = lstchange_all_df.drop(['lstchange_kndvi','lstzs_kndvi','lstzs_sif'], axis=1)
lstchange_sif_df = lstchange_sif_df.dropna(how='any')
lstchange_sif_df.describe()

# %%
lstzs_kndvi_df = lstchange_all_df.drop(['lstchange_sif','lstchange_kndvi','lstzs_sif'], axis=1)
lstzs_kndvi_df = lstzs_kndvi_df.dropna(how='any')
lstzs_kndvi_df.describe()

# %%
lstzs_sif_df = lstchange_all_df.drop(['lstchange_sif','lstzs_kndvi','lstchange_kndvi'], axis=1)
lstzs_sif_df = lstzs_sif_df.dropna(how='any')
lstzs_sif_df.describe()

# %% [markdown]
# ## 2 kndvi 对应

# %% [markdown]
# ### 2.1 lst change

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
lstchange_kndvi_q95 = lstchange_kndvi_df.lstchange_kndvi.quantile(0.95)
lstchange_kndvi_q95

# %%
lstchange_kndvi_q5 = lstchange_kndvi_df.lstchange_kndvi.quantile(0.05)
lstchange_kndvi_q5

# %%
lstchange_kndvi_use = lstchange_kndvi_df[lstchange_kndvi_df.lstchange_kndvi < lstchange_kndvi_q95]
lstchange_kndvi_use = lstchange_kndvi_use[lstchange_kndvi_use.lstchange_kndvi > lstchange_kndvi_q5]
lstchange_kndvi_use

# %%
plt.hexbin(x =lstchange_kndvi_use.fsc, y= lstchange_kndvi_use.lstchange_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(lstchange_kndvi_use.fsc.min(), lstchange_kndvi_use.fsc.max())

# %%
lstchange_kndvi_use['fsc_bins'] = pd.cut(lstchange_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
lstchange_kndvi_use[lstchange_kndvi_use.lstchange_kndvi>0].boxplot('lstchange_kndvi', by='fsc_bins')

# %%
lstchange_kndvi_use[lstchange_kndvi_use.lstchange_kndvi<0].boxplot('lstchange_kndvi', by='fsc_bins')

# %%
lstchange_kndvi_use_po = lstchange_kndvi_use[lstchange_kndvi_use.lstchange_kndvi>0]
lstchange_kndvi_use_ne = lstchange_kndvi_use[lstchange_kndvi_use.lstchange_kndvi<0]

# %%
lstchange_kndvi_use_po['lst_change_log'] = np.log(lstchange_kndvi_use_po.lstchange_kndvi)
lstchange_kndvi_use_ne['lst_change_log'] = np.log(lstchange_kndvi_use_ne.lstchange_kndvi * -1) * -1

# %%
lstchange_kndvi_use_po

# %%
lstchange_kndvi_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_kndvi_use_po.fsc_bins)
fsc_dfs = [lstchange_kndvi_use_po.lstchange_kndvi[lstchange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_po.lstchange_kndvi[lstchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_po.lstchange_kndvi[lstchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 2100, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-5,2500)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_kndvi_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_kndvi_use_po.fsc_bins)
fsc_dfs = [lstchange_kndvi_use_po.lst_change_log[lstchange_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_po.lst_change_log[lstchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_po.lst_change_log[lstchange_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 7.9, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(4,8.2)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_kndvi_po_log.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_kndvi_use_ne.fsc_bins)
fsc_dfs = [lstchange_kndvi_use_ne.lstchange_kndvi[lstchange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_ne.lstchange_kndvi[lstchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_ne.lstchange_kndvi[lstchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -1600, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-1700,5)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_kndvi_ne.png', dpi = 600)


# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_kndvi_use_ne.fsc_bins)
fsc_dfs = [lstchange_kndvi_use_ne.lst_change_log[lstchange_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_ne.lst_change_log[lstchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_kndvi_use_ne.lst_change_log[lstchange_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -7.7, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-8,-3)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_kndvi_ne_log.png', dpi = 600)

# %%
lstchange_kndvi_use.biome.value_counts()

# %%
lstchange_kndvi_fsc_bybi_po = lstchange_kndvi_use_po.groupby("biome")
lstchange_kndvi_fsc_bybi_ne = lstchange_kndvi_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [2500,2000,1500,2000,2000,2100,1000,1000,1000]

for i in range(9):
    bi_n = lstchange_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstchange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstchange_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstchange_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstchange_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-50,y_list[i])
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

fig.savefig('result_figure/lst_change_csc_biome_kndvi_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [8.5,8.5,8.5,8.5,8.5,8.5,8.5,8.5,8.5]

for i in range(9):
    bi_n = lstchange_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstchange_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(4,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/lst_change_csc_biome_kndvi_po_log.png', dpi = 600)

# %%
lstchange_kndvi_use_ne.biome.value_counts()

# %%

fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [1800,1800,1800]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstchange_kndvi_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstchange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstchange_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstchange_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstchange_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 0.5)
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

fig.savefig('result_figure/lst_change_csc_biome_kndvi_ne.png', dpi = 600)


# %%

fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [8,8,8]


for i in range(3):
    bi_n = lstchange_kndvi_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstchange_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = (y_list[i]-2)*0.95* -1 - 2, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, -2)
    axes[i].set_xlim(8.7,11.3)
    #axes[i].set_xticks(np.arange(9,11.3,0.5),labels = [])
    axes[i].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i].set_xlabel('Forest structural complexity')
        axes[i].set_xticks(np.arange(9,11.3,0.5),labels = np.arange(9,11.3,0.5))
    if (i%3) == 0 :
        axes[i].set_ylabel('LST change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/lst_change_csc_biome_kndvi_ne_log.png', dpi = 600)

# %% [markdown]
# ### 2.2 lst zs

# %%
lstzs_kndvi_df

# %%
lstzs_kndvi_q95 = lstzs_kndvi_df.lstzs_kndvi.quantile(0.95)
lstzs_kndvi_q95

# %%
lstzs_kndvi_q5 = lstzs_kndvi_df.lstzs_kndvi.quantile(0.05)
lstzs_kndvi_q5

# %%
lstzs_kndvi_use = lstzs_kndvi_df[lstzs_kndvi_df.lstzs_kndvi < lstzs_kndvi_q95]
lstzs_kndvi_use = lstzs_kndvi_use[lstzs_kndvi_use.lstzs_kndvi > lstzs_kndvi_q5]
lstzs_kndvi_use

# %%
plt.hexbin(x =lstzs_kndvi_use.fsc, y= lstzs_kndvi_use.lstzs_kndvi, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(lstzs_kndvi_use.fsc.min(), lstzs_kndvi_use.fsc.max())

# %%
lstzs_kndvi_use['fsc_bins'] = pd.cut(lstzs_kndvi_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
lstzs_kndvi_use[lstzs_kndvi_use.lstzs_kndvi>0].boxplot('lstzs_kndvi', by='fsc_bins')

# %%
lstzs_kndvi_use[lstzs_kndvi_use.lstzs_kndvi<0].boxplot('lstzs_kndvi', by='fsc_bins')

# %%
lstzs_kndvi_use_po = lstzs_kndvi_use[lstzs_kndvi_use.lstzs_kndvi>0]
lstzs_kndvi_use_ne = lstzs_kndvi_use[lstzs_kndvi_use.lstzs_kndvi<0]

# %%
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
fsc_dfs = [lstzs_kndvi_use_po.lstzs_kndvi[lstzs_kndvi_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_po.lstzs_kndvi[lstzs_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_po.lstzs_kndvi[lstzs_kndvi_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 2.4, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.2,2.6)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_zs_fsc_kndvi_po.png', dpi = 600)


# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_kndvi_use_ne.fsc_bins)
fsc_dfs = [lstzs_kndvi_use_ne.lstzs_kndvi[lstzs_kndvi_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_ne.lstzs_kndvi[lstzs_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_kndvi_use_ne.lstzs_kndvi[lstzs_kndvi_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -0.6, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.65,0.05)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_zs_fsc_kndvi_ne.png', dpi = 600)


# %%
lstzs_kndvi_use.biome.value_counts()

# %%
lstzs_kndvi_fsc_bybi_po = lstzs_kndvi_use_po.groupby("biome")
lstzs_kndvi_fsc_bybi_ne = lstzs_kndvi_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5]

for i in range(9):
    bi_n = lstzs_kndvi_use_po.biome.value_counts().index[:-5].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstzs_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstzs_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstzs_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
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

fig.savefig('result_figure/lst_zs_csc_biome_kndvi_po.png', dpi = 600)

# %%
lstzs_kndvi_use_ne.biome.value_counts()

# %%

fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [0.7,0.7,0.7]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstzs_kndvi_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_kndvi_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstzs_kndvi_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstzs_kndvi[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstzs_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstzs_kndvi[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 0.05)
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

fig.savefig('result_figure/lst_zs_csc_biome_kndvi_ne.png', dpi = 600)

# %% [markdown]
# ## 3 sif 对应

# %% [markdown]
# ### 2.1 lst change

# %%
lstchange_sif_q95 = lstchange_sif_df.lstchange_sif.quantile(0.95)
lstchange_sif_q95

# %%
lstchange_sif_q5 = lstchange_sif_df.lstchange_sif.quantile(0.05)
lstchange_sif_q5

# %%
lstchange_sif_use = lstchange_sif_df[lstchange_sif_df.lstchange_sif < lstchange_sif_q95]
lstchange_sif_use = lstchange_sif_use[lstchange_sif_use.lstchange_sif > lstchange_sif_q5]
lstchange_sif_use

# %%
plt.hexbin(x =lstchange_sif_use.fsc, y= lstchange_sif_use.lstchange_sif, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(lstchange_sif_use.fsc.min(), lstchange_sif_use.fsc.max())

# %%
lstchange_sif_use['fsc_bins'] = pd.cut(lstchange_sif_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
lstchange_sif_use[lstchange_sif_use.lstchange_sif>0].boxplot('lstchange_sif', by='fsc_bins')

# %%
lstchange_sif_use[lstchange_sif_use.lstchange_sif<0].boxplot('lstchange_sif', by='fsc_bins')

# %%
lstchange_sif_use_po = lstchange_sif_use[lstchange_sif_use.lstchange_sif>0]
lstchange_sif_use_ne = lstchange_sif_use[lstchange_sif_use.lstchange_sif<0]
lstchange_sif_use_po['lst_change_log'] = np.log(lstchange_sif_use_po.lstchange_sif)
lstchange_sif_use_ne['lst_change_log'] = np.log(lstchange_sif_use_ne.lstchange_sif * -1) * -1

# %%
lstchange_sif_use_po

# %%
lstchange_sif_use_ne

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_sif_use_po.fsc_bins)
fsc_dfs = [lstchange_sif_use_po.lstchange_sif[lstchange_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_sif_use_po.lstchange_sif[lstchange_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_sif_use_po.lstchange_sif[lstchange_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 2100, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-5,2300)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_sif_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_sif_use_po.fsc_bins)
fsc_dfs = [lstchange_sif_use_po.lst_change_log[lstchange_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_sif_use_po.lst_change_log[lstchange_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_sif_use_po.lst_change_log[lstchange_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 8.1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(4,8.4)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_sif_po_log.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_sif_use_ne.fsc_bins)
fsc_dfs = [lstchange_sif_use_ne.lstchange_sif[lstchange_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_sif_use_ne.lstchange_sif[lstchange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_sif_use_ne.lstchange_sif[lstchange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -2450, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-2600,50)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_sif_ne.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstchange_sif_use_ne.fsc_bins)
fsc_dfs = [lstchange_sif_use_ne.lst_change_log[lstchange_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstchange_sif_use_ne.lst_change_log[lstchange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstchange_sif_use_ne.lst_change_log[lstchange_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -8, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change (log)')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-8.3,-3.5)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_change_fsc_sif_ne_log.png', dpi = 600)

# %%
lstchange_sif_use.biome.value_counts()

# %%
lstchange_sif_use_po.biome.value_counts().index[:-5]

# %%
lstchange_sif_fsc_bybi_po = lstchange_sif_use_po.groupby("biome")
lstchange_sif_fsc_bybi_ne = lstchange_sif_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [2500,2500,1500,2000,2500,2300,1000,1000,1200]

for i in range(9):
    bi_n = lstchange_sif_use_po.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstchange_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstchange_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstchange_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstchange_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = y_list[i]*0.93, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(-50,y_list[i])
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

fig.savefig('result_figure/lst_change_csc_biome_sif_po.png', dpi = 600)

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [8.5,8.5,8.5,8.5,8.5,8.5,8.5,8.5,8.5]

for i in range(9):
    bi_n = lstchange_sif_use_po.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstchange_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = (y_list[i] - 4)*0.9 + 4, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(4,y_list[i])
    axes[i//3, i %3].set_xlim(8.7,11.8)
    axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(9,11.8,0.5),labels = np.arange(9,11.8,0.5))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('LST change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/lst_change_csc_biome_sif_po_log.png', dpi = 600)

# %%
lstchange_sif_use_ne.biome.value_counts()

# %%
fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [2600,2600,2600]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstchange_sif_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstchange_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstchange_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstchange_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstchange_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 100)
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

fig.savefig('result_figure/lst_change_csc_biome_sif_ne.png', dpi = 600)


# %%
fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [8.3,8.3,8.3]


for i in range(3):
    bi_n = lstchange_sif_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstchange_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstchange_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lst_change_log[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = (y_list[i]-4)*0.95* -1 - 4, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, -4)
    axes[i].set_xlim(8.7,11.3)
    #axes[i].set_xticks(np.arange(9,11.3,0.5),labels = [])
    axes[i].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i].set_xlabel('Forest structural complexity')
        axes[i].set_xticks(np.arange(9,11.3,0.5),labels = np.arange(9,11.3,0.5))
    if (i%3) == 0 :
        axes[i].set_ylabel('LST change (log)')

#axes[1, 2].set_xlabel('Forest structural complexity')
#axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
#axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_figure/lst_change_csc_biome_sif_ne_log.png', dpi = 600)

# %% [markdown]
# ### 3.2 lst zs

# %%
lstzs_sif_df

# %%
lstzs_sif_q95 = lstzs_sif_df.lstzs_sif.quantile(0.95)
lstzs_sif_q95

# %%
lstzs_sif_q5 = lstzs_sif_df.lstzs_sif.quantile(0.05)
lstzs_sif_q5

# %%
lstzs_sif_use = lstzs_sif_df[lstzs_sif_df.lstzs_sif < lstzs_sif_q95]
lstzs_sif_use = lstzs_sif_use[lstzs_sif_use.lstzs_sif > lstzs_sif_q5]
lstzs_sif_use

# %%
plt.hexbin(x =lstzs_sif_use.fsc, y= lstzs_sif_use.lstzs_sif, gridsize = 50,bins='log')
plt.colorbar()

# %%
print(lstzs_sif_use.fsc.min(), lstzs_sif_use.fsc.max())

# %%
lstzs_sif_use['fsc_bins'] = pd.cut(lstzs_sif_use.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])

# %%
lstzs_sif_use[lstzs_sif_use.lstzs_sif>0].boxplot('lstzs_sif', by='fsc_bins')
lstzs_sif_use[lstzs_sif_use.lstzs_sif<0].boxplot('lstzs_sif', by='fsc_bins')

# %%
lstzs_sif_use_po = lstzs_sif_use[lstzs_sif_use.lstzs_sif>0]
lstzs_sif_use_ne = lstzs_sif_use[lstzs_sif_use.lstzs_sif<0]

# %%
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
fsc_dfs = [lstzs_sif_use_po.lstzs_sif[lstzs_sif_use_po.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_sif_use_po.lstzs_sif[lstzs_sif_use_po.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_sif_use_po.lstzs_sif[lstzs_sif_use_po.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 2.8, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.2,3)
ax.set_xlim(8.7,11.8)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_zs_fsc_sif_po.png', dpi = 600)

# %%
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(lstzs_sif_use_ne.fsc_bins)
fsc_dfs = [lstzs_sif_use_ne.lstzs_sif[lstzs_sif_use_ne.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(lstzs_sif_use_ne.lstzs_sif[lstzs_sif_use_ne.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(lstzs_sif_use_ne.lstzs_sif[lstzs_sif_use_ne.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.25, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = -0.7, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('LST change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(-0.75,0.05)
ax.set_xlim(8.7,11.3)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_figure/lst_zs_fsc_sif_ne.png', dpi = 600)

# %%
lstzs_sif_use.biome.value_counts()

# %%
lstzs_sif_fsc_bybi_po = lstzs_sif_use_po.groupby("biome")
lstzs_sif_fsc_bybi_ne = lstzs_sif_use_ne.groupby("biome")

# %%
fig, axes = plt.subplots(3,3, figsize=(14,10))

y_list = [3,3,3,3,3,3,3,3,3]

for i in range(9):
    bi_n = lstzs_sif_use_po.biome.value_counts().index[:-4].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_sif_fsc_bybi_po.get_group(bi_n)
    labels_fsc = np.unique(lstzs_sif_fsc_bybi_po.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstzs_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstzs_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstzs_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
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

fig.savefig('result_figure/lst_zs_csc_biome_sif_po.png', dpi = 600)

# %%
lstzs_sif_use_ne.biome.value_counts()

fig, axes = plt.subplots(1,3, figsize=(14,4))

y_list = [0.75,0.75,0.75]

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(3):
    bi_n = lstzs_sif_use_ne.biome.value_counts().index[:3].sort_values()[i]
    #print(ld_n)
    bi_df = lstzs_sif_fsc_bybi_ne.get_group(bi_n)
    labels_fsc = np.unique(lstzs_sif_fsc_bybi_ne.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.lstzs_sif[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.lstzs_sif[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.lstzs_sif[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.25, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i].text(x =labels_fsc_use[j], y = y_list[i]*0.95* -1, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i].set_ylim(y_list[i]* -1, 0.05)
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

fig.savefig('result_figure/lst_zs_csc_biome_sif_ne.png', dpi = 600)

# %% [markdown]
# ## change log
# 1. 总体来说  结构越复杂的森林，干旱时 lst变化更偏 低温， 但是有个问题，就是可能是本来就更湿润的地方，结构更复杂，所以 可以更好的保持蒸散发，所以lst偏低

# %%



