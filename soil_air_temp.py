#!/usr/bin/env python
# coding: utf-8

# ## soil-air-temp-diff

import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import rioxarray
import glob
import pandas as pd

soil_temp_path = glob.glob(r'data/soil_temp/*0_5cm*tif')
soil_temp_path

air_temp_path = glob.glob(r'data/chelsa/CHELSA_*.tif')
air_temp_path

with rioxarray.open_rasterio(r'D:/data/forest_structural_complexity/Potential_structural_complexity_Map.tif') as data:
    fsc = data.where(data>0).coarsen(x=6, y =6, boundary='pad', side='right').mean()
fsc = xr.DataArray(fsc[0].values, coords=[fsc.y, fsc.x], dims=['lat','lon'])
fsc

with xr.open_dataset(r'result_data/landcover_005_use.nc') as data:
    ld = data['modis_landcover'].interp_like(fsc, method='nearest')
ld


soil_air_temp_all = []
for i in range(1,13):
    print(r'data/soil_temp/soilT_{}_0_5cm.tif'.format(i))
    with rioxarray.open_rasterio(r'D:/data/soil_temp/soilT_{}_0_5cm.tif'.format(i)) as data:
        soil_temp =  data.where(data > -5000).coarsen(x=6, y =6, boundary='pad', side='right').mean()
        soil_temp = soil_temp.interp_like(fsc, method='nearest')
        soil_temp = soil_temp.where(fsc>0)
    print(air_temp_path[i-1])
    with rioxarray.open_rasterio(air_temp_path[i-1]) as data:
        air_temp =  data.where(data > -5000).coarsen(x=6, y =6, boundary='pad', side='right').mean()
        air_temp =  air_temp.interp_like(fsc, method='nearest')
        air_temp = air_temp.where(fsc>0)
    
    soil_air_temp = soil_temp -air_temp

    soil_air_temp = xr.DataArray(soil_air_temp[0].values, coords=[soil_air_temp.y, soil_air_temp.x], dims=['lat','lon'])
    soil_air_temp_all.append(soil_air_temp)

soil_air_temp_all = xr.concat(soil_air_temp_all, dim='month')

soil_air_temp_all = xr.DataArray(soil_air_temp_all.values, coords=[np.arange(1,13),soil_air_temp_all.lat,soil_air_temp_all.lon], dims=['month','lat','lon'])
soil_air_temp_all

soil_air_temp_all.name = 'soil_air_temp_offset'
soil_air_temp_all.to_netcdf(r'D:/data/soil_air_temp_all.nc')

fsc = xr.DataArray(fsc[0].values, coords=[fsc.y, fsc.x], dims=['lat','lon'])
fsc

soil_air_fsc = xr.Dataset({'temp_offset':soil_air_temp_all, 'fsc':fsc, 'ld':ld})
soil_air_fsc

soil_air_fsc_df = soil_air_fsc.to_dataframe()
soil_air_fsc_df = soil_air_fsc_df.dropna(how='any')

soil_air_fsc_df.to_csv(r'result_data/soil_air_fsc_df.csv')


# ### fsc - temp-diff

from scipy import stats

## biome
with rioxarray.open_rasterio(r'D:/data/official_teow/biome.tif')  as data:
    biome = data

biome = biome.where(biome>0)
biome = biome.where(biome<90)
biome = xr.DataArray(biome[0].values, coords=[biome.y, biome.x], dims=['lat','lon'])

biome = biome.interp_like(soil_air_temp_all.sel(month = 1), method='nearest')
biome

soil_air_temp_all['fsc'] = fsc

soil_air_temp_all['biome'] = biome
soil_air_temp_all


soil_air_fsc_df = soil_air_temp_all.to_dataframe().reset_index()
soil_air_fsc_df = soil_air_fsc_df.dropna(how='any')
soil_air_fsc_df.describe()

soil_air_fsc_size = soil_air_fsc_df.groupby(['biome']).size().reset_index()
soil_air_fsc_size.columns = ['biome','size']
soil_air_fsc_size


soil_air_fsc_grouped = soil_air_fsc_df.groupby(['biome','month'])

biome_use = []

month_all = []
bi_all = []
cor_all = []
pval_all = []
for name,group_n in soil_air_fsc_grouped:
    print(name)
    
    cor_all.append( stats.pearsonr(group_n['soil_air_temp_offset'],group_n['fsc'])[0] )
    pval_all.append( stats.pearsonr(group_n['soil_air_temp_offset'],group_n['fsc'])[1] )
    bi_all.append(name[0])
    month_all.append(name[1])

soil_air_fsc_cor = pd.DataFrame({'month':month_all, 'biome':bi_all, 'cor':cor_all})
soil_air_fsc_cor = soil_air_fsc_cor.pivot(index='biome', columns='month', values='cor')
soil_air_fsc_cor


biome_short_dic = {1: 'Trop.&Subtrop. Moist Broad. Forests',
 2: 'Trop.&Subtrop. Dry Broad. Forests',
 3: 'Trop.&Subtrop. Coni. Forests',
 4: 'Temp. Broad.&Mixed Forests',
 5: 'Temp. Coni. Forests',
 6: 'Boreal Forests',
 7: 'Trop.&Subtrop. Gra.,Sav.&Shrub.',
 8: 'Temp. Gra.,Sav.&Shrub.',
 9: 'Flooded Gra.&Sav',
 10: 'Montane Gra.&Shrub.',
 11: 'Tundra',
 12: 'Mediter. Forests, Woodlands & Scrub',
 13: 'Deserts & Xeric Shrublands',
 14: 'Mangroves',
          0:'All'}



plt.rc('font',family='Times New Roman', size = 15)
fig, ax = plt.subplots(figsize=(14,6))

im = ax.imshow(soil_air_fsc_cor, vmin=-0.75, vmax=0.75, cmap='PiYG_r', aspect=0.6)

for i in range(12):
    for j in range(14):
        ax.text(i,j,round( soil_air_fsc_cor.iloc[j,i],3),size = 12, ha='center', va = 'center')

for k in range(14):
    ax.text(11.7,k, int(soil_air_fsc_size['size'][k]/12),size = 12, ha='left', va = 'center')

ax.set_xlabel('Month')
ax.set_xticks(ticks=np.arange(12))
ax.set_yticks(ticks=np.arange(14))
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in soil_air_fsc_cor.index])
ax.set_xticklabels(np.arange(1,13))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8, pad = 0.1)
cb.set_label(label='Correlation Coefficients')
cb.set_ticks(np.arange(-0.75,0.8,0.25))
cb.set_ticklabels(np.arange(-0.75,0.8,0.25))
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_fig/cor_fsc_soil_air_temp_biome.png', dpi = 600)


soil_air_fsc_df['sphere'] = ['N'  if x>0 else 'S' for x in soil_air_fsc_df.lat]

soil_air_fsc_size = soil_air_fsc_df.groupby(['biome','sphere']).size().reset_index()
soil_air_fsc_size.columns = ['biome','sphere','size']
soil_air_fsc_size

soil_air_fsc_grouped = soil_air_fsc_df.groupby(['biome','sphere','month'])


month_all = []
sphere_all = []
bi_all = []
cor_all = []
pval_all = []
for name,group_n in soil_air_fsc_grouped:
    print(name)
    if group_n.shape[0]>100:
        cor_all.append( stats.pearsonr(group_n['soil_air_temp_offset'],group_n['fsc'])[0] )
        pval_all.append( stats.pearsonr(group_n['soil_air_temp_offset'],group_n['fsc'])[1] )
        bi_all.append(name[0])
        sphere_all.append(name[1])
        month_all.append(name[2])

soil_air_fsc_cor = pd.DataFrame({'month':month_all, 'sphere':sphere_all,'biome':bi_all, 'cor':cor_all})
soil_air_fsc_cor = soil_air_fsc_cor.pivot(index=['sphere','biome'], columns='month', values='cor').reset_index()
soil_air_fsc_cor


soil_air_fsc_size['n_pix'] = soil_air_fsc_size['size']/12
soil_air_fsc_size

soil_air_fsc_cor = pd.merge(soil_air_fsc_cor, soil_air_fsc_size[['biome','sphere','n_pix']], on=['biome','sphere'], how='left')
soil_air_fsc_cor

plt.rc('font',family='Times New Roman', size = 15)
fig, ax = plt.subplots(figsize=(15,11))

im = ax.imshow(soil_air_fsc_cor.iloc[:,2:14], vmin=-0.75, vmax=0.75, cmap='PiYG_r', aspect=0.6)

for i in range(12):
    for j in range(24):
        ax.text(i,j,round( soil_air_fsc_cor.iloc[j,i+2],3),size = 12, ha='center', va = 'center')

for k in range(24):
    ax.text(11.7,k, int(soil_air_fsc_cor['n_pix'][k]),size = 12, ha='left', va = 'center')

ax.set_xlabel('Month')
ax.set_xticks(ticks=np.arange(12))
ax.set_yticks(ticks=np.arange(24))
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in soil_air_fsc_cor.biome])
ax.set_xticklabels(np.arange(1,13))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.75, pad = 0.075)
cb.set_label(label='Correlation Coefficients')
cb.set_ticks(np.arange(-0.75,0.8,0.25))
cb.set_ticklabels(np.arange(-0.75,0.8,0.25))
cb.outline.set_linewidth(0.05)

ax.axhline(y=13.5, color='black', linestyle='--', linewidth=2)
#ax.axhline(y=16.5, color='black', linestyle='--', linewidth=2)

plt.subplots_adjust( right=0.998, top=0.98, bottom=0.07)
#plt.tight_layout()

plt.savefig('result_fig/cor_fsc_soil_air_temp_biome_NS.png', dpi = 600)


# ### moving window

with xr.open_dataset(r'D:/data/soil_air_temp_all.nc')['soil_air_temp_offset'] as data:
    soil_air_temp_all = data

soil_air_temp_all

## 15*15
soil_rolling = soil_air_temp_all.sel(month=1).rolling(lon=15, lat = 15, center=True)
soil_rolling = soil_rolling.construct(window_dim={'lon':'lon1','lat':'lat1'})

soil_rolling = soil_rolling.values.reshape([2549,7200,225])


import pandas as pd

def cor_fsc_soil(fsc_data, fsc_roll, soli_roll):
    if np.isnan(fsc_data):
        return  (np.nan, np.nan)
    elif np.isnan(fsc_roll).sum() > 150:
        return (np.nan, np.nan)
    else:
        pd_data = pd.DataFrame({'fsc':fsc_roll,'soil':soli_roll})
        pd_data = pd_data.dropna(how='any')
        cor_re = stats.pearsonr(pd_data.fsc,pd_data.soil)[0] 
        p_re = stats.pearsonr(pd_data.fsc,pd_data.soil)[1] 
        return (cor_re, p_re)

fsc = fsc.where(ld>0)

cor_result_all = []
cor_p_result_all = []

for month_n in range(1,13):
    print(month_n)
    
    soil_rolling = soil_air_temp_all.sel(month=month_n).rolling(lon=15, lat = 15, center=True)
    soil_rolling = soil_rolling.construct(window_dim={'lon':'lon1','lat':'lat1'})
    soil_rolling = soil_rolling.values.reshape([2549,7200,225])

    cor_result = xr.apply_ufunc(cor_fsc_soil,
                            fsc,
                            fsc_rolling,
                            soil_rolling,
                            input_core_dims=[[],['win'],['win']],
                            output_core_dims=[[],[]],
                            vectorize= True)
    
    cor_result_all.append(cor_result[0])
    cor_p_result_all.append(cor_result[1])

del soil_rolling, fsc_rolling

cor_result_all = xr.concat(cor_result_all, dim='month')

cor_result_all.shape

cor_result_all = xr.DataArray(cor_result_all.values, coords=[np.arange(1,13), cor_result_all.lat, cor_result_all.lon],
                              dims=['month','lat','lon'])
cor_result_all

cor_p_result_all = xr.concat(cor_p_result_all, dim='month')
cor_p_result_all.shape


cor_p_result_all = xr.DataArray(cor_p_result_all.values, coords=[np.arange(1,13), cor_p_result_all.lat, cor_p_result_all.lon],
                              dims=['month','lat','lon'])
cor_p_result_all


cor_result_all.name = 'cor'
cor_p_result_all.name = 'p_value'


cor_result_all.to_netcdf(r'result_data/cor_soil_air_temp_offset_and_fsc.nc')
cor_p_result_all.to_netcdf(r'result_data/pvalue_soil_air_temp_offset_and_fsc.nc')


