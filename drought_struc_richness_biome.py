#!/usr/bin/env python
# coding: utf-8


import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import rioxarray
import pandas as pd
import glob

plt.rc('font',family='Times New Roman', size = 15)


drought_resistance_df = pd.read_csv(r'result_data/drought_resistance_df_add.csv') 
drought_resistance_df.head()

drought_resistance_df['fsc_bins'] = pd.cut(drought_resistance_df.fsc, bins = [0,2.5,3.5,4.5,5.5,6.5,7.5,15], labels= [2,3,4,5,6,7,8])
drought_resistance_df['rich_bins'] = pd.cut(drought_resistance_df.richness, bins = [0,1,2,3,4,5,6], labels= [0.5,1.5,2.5,3.5,4.5,5.5])

drought_resistance_use = drought_resistance_df[drought_resistance_df.resistance < 300]



fig, axes = plt.subplots(1,3, figsize=(12,6))

axes[0].hexbin(drought_resistance_use.richness, drought_resistance_use.fsc, 
                C= drought_resistance_use.resistance, gridsize = 100, 
                vmax=300, vmin=0,
                reduce_C_function = np.median)
axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Forest structural complexity')
axes[0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

axes[1].hexbin(np.log(drought_resistance_use.prec), drought_resistance_use.richness, 
                C= drought_resistance_use.resistance, gridsize = 100, 
                vmax=300, vmin=0,
                reduce_C_function = np.median)
axes[1].set_xlabel('Annual precipitation (log) / mm')
axes[1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

im = axes[2].hexbin(np.log(drought_resistance_use.prec), drought_resistance_use.fsc, 
                C= drought_resistance_use.resistance, gridsize = 100, 
                vmax=300, vmin=0,
                reduce_C_function = np.median)
axes[2].set_xlabel('Annual precipitation (log) / mm')
axes[2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

position=fig.add_axes([0.2,0.1,0.6,0.04])
plt.colorbar(im, position, extend='both', label = 'Resistance',orientation='horizontal')

plt.subplots_adjust(top=0.9, bottom=0.25, right=0.98, left=0.06, hspace=0)

plt.savefig('result_fig/resistance_richness_fsc_prec.png', dpi = 600)


biome_dic = {1:'Tropical & Subtropical Moist Broadleaf Forests',
          2:'Tropical & Subtropical Dry Broadleaf Forests',
          3:'Tropical & Subtropical Coniferous Forests',
          4:'Temperate Broadleaf & Mixed Forests',
          5:'Temperate Conifer Forests',
          6:'Boreal Forests/Taiga',
          7:'Tropical & Subtropical Grasslands, Savannas & Shrublands',
          8:'Temperate Grasslands, Savannas & Shrublands',
          9:'Flooded Grasslands & Savannas',
          10:'Montane Grasslands & Shrublands',
          11:'Tundra',
          12:'Mediterranean Forests, Woodlands & Scrub',
          13:'Deserts & Xeric Shrublands',
          14:'Mangroves',
          0:'All'}



## biomes resistance

biome_list = ['Trop.&Subtrop. Moist Broad. Forests', 'Trop.&Subtrop. Dry Broad. Forests', 'Trop.&Subtrop. Coni. Forests', 'Temp. Broad.&Mix Forests',
          'Temp. Coni. Forests','Boreal Forests','Trop.&Subtrop. Gra.,Sav.&Shrub.','Temp. Gra.,Sav.&Shrub.','Flooded Gra.&Sav',
          'Montane Gra.&Shrub.','Tundra','Mediter. Forests, Woodlands & Scrub','Deserts & Xeric Shrublands','Mangroves']

fig, axes = plt.subplots(3,1, figsize=(10,15))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  biome
labels_bi = np.unique(drought_resistance_use.biome)
bi_dfs = [drought_resistance_use.resistance[drought_resistance_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_use.resistance[drought_resistance_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_use.resistance[drought_resistance_use.biome == labels_bi_n]) > 15]
axes[0].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[0].text(x = j+1, y = 310, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[0].set_xlabel('IGCP Landcover')
axes[0].set_ylabel('Resistance')
axes[0].set_xticklabels([])
axes[0].set_ylim(0,330)
axes[0].set_xlim(0.2,14.8)
axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0].set_title('(a)', loc='left', size = 14)

## richness  vs  ld
labels_bi = np.unique(drought_resistance_use.biome)
bi_dfs = [drought_resistance_use.richness[drought_resistance_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_use.richness[drought_resistance_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_use.richness[drought_resistance_use.biome == labels_bi_n]) > 15]
axes[1].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[1].text(x = j+1, y = 6, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

#axes[1].set_xlabel('IGCP Landcover')
axes[1].set_ylabel('Species richness')
axes[1].set_xticklabels([])
axes[1].set_ylim(0,6.5)
axes[1].set_xlim(0.2,14.8)
axes[1].grid(c = 'lightgray', alpha = 0.3)
axes[1].set_title('(b)', loc='left', size = 14)

## fsc  vs  ld
labels_bi = np.unique(drought_resistance_use.biome)
bi_dfs = [drought_resistance_use.fsc[drought_resistance_use.biome == labels_bi_n] for labels_bi_n in labels_bi if len(drought_resistance_use.fsc[drought_resistance_use.biome == labels_bi_n]) > 15 ]
bi_dfs_len = np.asarray([ len(bi_dfs_n) for bi_dfs_n in bi_dfs if len(bi_dfs_n) >15])
#print(fsc_dfs_len)
labels_bi_use = [  labels_bi_n for labels_bi_n in labels_bi if len(drought_resistance_use.fsc[drought_resistance_use.biome == labels_bi_n]) > 15]
axes[2].boxplot(bi_dfs, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_bi_use)):
    axes[2].text(x = j+1, y = 12, s = 'n='+ str(bi_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Biome')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_xticklabels(biome_list, rotation=-90, rotation_mode="anchor", ha="left", va='center')
axes[2].set_ylim(0,13.5)
axes[2].set_xlim(0.2,14.8)
axes[2].grid(c = 'lightgray', alpha = 0.3)
axes[2].set_title('(c)', loc='left', size = 14)

fig.align_labels()
plt.tight_layout()

plt.savefig('result_fig/resistance_biome.png', dpi = 600)


fig, axes = plt.subplots(3,1, figsize=(8,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(drought_resistance_use.fsc_bins)
fsc_dfs = [drought_resistance_use.resistance[drought_resistance_use.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(drought_resistance_use.resistance[drought_resistance_use.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(drought_resistance_use.resistance[drought_resistance_use.fsc_bins == labels_fsc_n]) > 15]
axes[1].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    axes[1].text(x =labels_fsc_use[j], y = 310, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[1].set_xlabel('Forest structural complexity')
axes[1].set_ylabel('Resistance')
axes[1].set_title('(b)', loc='left', size = 14)
axes[1].set_ylim(0,330)
axes[1].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  resistance
labels_rich = np.unique(drought_resistance_use.rich_bins)
rich_dfs = [drought_resistance_use.resistance[drought_resistance_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_use.resistance[drought_resistance_use.rich_bins == labels_rich_n]) > 15 ]
rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_use.resistance[drought_resistance_use.rich_bins == labels_rich_n]) > 15]
axes[0].boxplot(rich_dfs, positions = labels_rich_use, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[0].text(x =labels_rich_use[j], y = 310, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)

axes[0].set_xlabel('Species richness')
axes[0].set_ylabel('Resistance')
axes[0].set_title('(a)', loc='left', size = 14)
axes[0].set_ylim(0,330)
axes[0].grid(c = 'lightgray', alpha = 0.3)

## richness  vs  fsc

rich_fsc = [drought_resistance_use.fsc[drought_resistance_use.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(drought_resistance_use.fsc[drought_resistance_use.rich_bins == labels_rich_n]) > 15 ]
rich_fsc_len = np.asarray([ len(rich_fsc_n) for rich_fsc_n in rich_fsc if len(rich_fsc_n) >15])

labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(drought_resistance_use.fsc[drought_resistance_use.rich_bins == labels_rich_n]) > 15]
axes[2].boxplot(rich_fsc, positions = labels_rich_use, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_rich_use)):
    axes[2].text(x =labels_rich_use[j], y = 15, s = 'n='+ str(rich_fsc_len[j]) ,horizontalalignment='center', size = 12)

axes[2].set_xlabel('Species richness')
axes[2].set_ylabel('Forest structural complexity')
axes[2].set_title('(c)', loc='left', size = 14)
axes[2].set_ylim(0,16)
axes[2].grid(c = 'lightgray', alpha = 0.3)

fig.align_labels()
fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1)




import pingouin as pg

drought_resistance = drought_resistance_use.drop(['lat','lon','landcover','fsc_bins','rich_bins','biome','resis_log'],axis=1).corr(method='spearman')
drought_resistance

var_name = drought_resistance.index
var_name

var_name = ['Drought resistance','Species richness','Forest structural complexity','Drought counts','Mean drought duration',
            'Mean drought severity','Mean annual temperature','Mean annual precipitation','Aridity index','Cation exchange capacity',
            'Clay content','Compound topographic index','Specific leaf area','Wood density']

drought_resistance = np.asarray(drought_resistance)
drought_resistance


for i in range(drought_resistance.shape[0]):
    for j in range(drought_resistance.shape[1]):
        if i <= j:
            drought_resistance[i,j] = np.nan


fig, ax = plt.subplots(figsize=(12,7))

im = ax.imshow(drought_resistance[1:,:-1], vmin=-1, vmax=1, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(13))
ax.set_yticks(ticks=np.arange(13))
ax.set_xticklabels(var_name[:-1], rotation=45, rotation_mode="anchor", ha="right")
ax.set_yticklabels(var_name[1:])

for i in range(1,14):
    for j in range(13):
        if ~np.isnan(drought_resistance[i,j]):
            ax.text(j, i-1, str(round(drought_resistance[i,j],3)), ha='center', va = 'center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.8)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()

plt.savefig('result_fig/cor_resistance_add.png', dpi = 600)


# ####  biome

drought_resistance_use.biome.value_counts()

grouped_resistance_by_bi = drought_resistance_use.groupby("biome")



biome_list = ['Trop.&Subtrop. Moist Broad. Forests', 'Trop.&Subtrop. Dry Broad. Forests', 'Trop.&Subtrop. Coni. Forests', 'Temp. Broad.&Mixed Forests',
          'Temp. Coni. Forests','Boreal Forests','Trop.&Subtrop. Gra.,Sav.&Shrub.','Temp. Gra.,Sav.&Shrub.','Flooded Gra.&Sav',
          'Montane Gra.&Shrub.','Tundra','Mediter. Forests, Woodlands & Scrub','Deserts & Xeric Shrublands','Mangroves']

biome_short_dic = {1: 'Trop.&Subtrop. Moist Broad. Forests',
 2: 'Trop.&Subtrop. Dry Broad. Forests',
 3: 'Trop.&Subtrop. Coni. Forests',
 4: 'Temp. Broad.&Mixed Forests',
 5: 'Temp. Coni. Forests',
 6: 'Boreal Forests',
 7: 'Trop.&Subtrop. Gra.,Sav.&Shrub.',
 8: 'Temp. Gra.,Sav.&Shrub.',
 9: 'Flooded Gra.&Sav.',
 10: 'Montane Gra.&Shrub.',
 11: 'Tundra',
 12: 'Mediter. Forests, Woodlands & Scrub',
 13: 'Deserts & Xeric Shrublands',
 14: 'Mangroves',
          0:'All'}


biome_use = [1,4,5,6,7,8,10,11]
alpha_list = ['a','b','c','d','e','f','g','h']

fig, axes = plt.subplots(3,3, figsize=(14,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(8):
    bi_n = drought_resistance_use.biome.value_counts().index[:-6].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_by_bi.get_group(bi_n)
    labels_fsc = np.unique(grouped_resistance_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_fsc)
    fsc_dfs = [bi_df.resistance[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.resistance[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(fsc_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.resistance[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_fsc_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = 310, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' +biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,330)
    axes[i//3, i %3].set_xlim(2.5,8.5)
    axes[i//3, i %3].set_xticks(np.arange(3,9),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(3,9),labels = np.arange(3,9))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

axes[1, 2].set_xlabel('Forest structural complexity')
axes[1,2].set_xticks(np.arange(3,9),labels = np.arange(3,9))
axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_fig/fsc_resistance_biome.png', dpi = 600)

fig, axes = plt.subplots(3,3, figsize=(14,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(8):
    bi_n = drought_resistance_use.biome.value_counts().index[:-6].sort_values()[i]
    #print(ld_n)
    bi_df = grouped_resistance_by_bi.get_group(bi_n)
    labels_rich = np.unique(grouped_resistance_by_bi.get_group(bi_n).rich_bins)
    #print(labels_rich)
    rich_dfs = [bi_df.resistance[bi_df.rich_bins == labels_rich_n] for labels_rich_n in labels_rich if len(bi_df.resistance[bi_df.rich_bins == labels_rich_n]) > 15 ]
    rich_dfs_len = np.asarray([ len(rich_dfs_n) for rich_dfs_n in rich_dfs if len(rich_dfs_n) >15])
    #print(rich_dfs_len)
    labels_rich_use = [  labels_rich_n for labels_rich_n in labels_rich if len(bi_df.resistance[bi_df.rich_bins == labels_rich_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(rich_dfs, positions = labels_rich_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_rich_use)):
        axes[i//3, i %3].text(x =labels_rich_use[j], y = 310, s = 'n='+ str(rich_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    #axes[i//3, i %3].set_title('('+alpha_list[i]+')', loc='left', size = 14)
    axes[i//3, i %3].set_ylim(0,330)
    axes[i//3, i %3].set_xlim(0,6)
    axes[i//3, i %3].set_xticks(np.arange(0.5,6,1),labels =[])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Tree species richness')
        axes[i//3, i %3].set_xticks(np.arange(0.5,6,1),labels = np.arange(0.5,6,1))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Resistance')

axes[1, 2].set_xlabel('Tree species richness')
axes[1,2].set_xticks(np.arange(0.5,6,1),labels = np.arange(0.5,6,1))
axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_fig/richness_resistance_biome.png', dpi = 600)


# #### pcor

from scipy import stats


# ##### cor


bi_cor_fsc= {}
bi_cor_rich = {}

for bi_n in drought_resistance_use.biome.value_counts().index[:-6]:
    print(bi_n)
    bi_group_df = grouped_resistance_by_bi.get_group(bi_n)
    bi_cor_fsc[bi_n] = {'r':stats.pearsonr(bi_group_df['resis_log'],bi_group_df['fsc'])[0],
                        'p-val':stats.pearsonr(bi_group_df['resis_log'],bi_group_df['fsc'])[1]}
    bi_cor_rich[bi_n] = {'r':stats.pearsonr(bi_group_df['resis_log'],bi_group_df['richness'])[0],
                        'p-val':stats.pearsonr(bi_group_df['resis_log'],bi_group_df['richness'])[1]}



stats.pearsonr(drought_resistance_use['resis_log'],drought_resistance_use['fsc'])
stats.pearsonr(drought_resistance_use['resis_log'],drought_resistance_use['richness'])

bi_pd_fsc_cor_df = pd.DataFrame(bi_cor_fsc).T
bi_pd_fsc_cor_df['var'] = 'fsc'
bi_pd_fsc_cor_df = bi_pd_fsc_cor_df.sort_index()
bi_pd_fsc_cor_df


bi_pd_rich_cor_df = pd.DataFrame(bi_cor_rich).T
bi_pd_rich_cor_df['var'] = 'rich'
bi_pd_rich_cor_df = bi_pd_rich_cor_df.sort_index()
bi_pd_rich_cor_df


fsc_cor_list = list(bi_pd_fsc_cor_df.r)
fsc_cor_list.append(0.344)
rich_cor_list = list(bi_pd_rich_cor_df.r)
rich_cor_list.append(0.340)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)

bi_cor_pd_r = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_r


fsc_cor_list = list(bi_pd_fsc_cor_df['p-val'])
fsc_cor_list.append(0)
rich_cor_list = list(bi_pd_rich_cor_df['p-val'])
rich_cor_list.append(0)
bi_cor_list = list(bi_pd_rich_cor_df.index)
bi_cor_list.append(0)

bi_cor_pd_p = pd.DataFrame({'fsc':fsc_cor_list, 'rich':rich_cor_list, 'biome':bi_cor_list})
bi_cor_pd_p

bi_pd_fsc_rich_pcor = {}
bi_pd_fsc_prec_pcor = {}
bi_pd_rich_fsc_pcor = {}
bi_pd_rich_prec_pcor = {}
for bi_n in drought_resistance_use.biome.value_counts().index[:-6]:
    print(bi_n)
    bi_group_df = grouped_resistance_by_bi.get_group(bi_n)
    fsc_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['richness'],method='spearman').round(4)
    fsc_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='fsc',covar=['prec'],method='spearman').round(4)
    rich_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='richness',covar=['fsc'],method='spearman').round(4)
    rich_prec_pcor = pg.partial_corr(data = bi_group_df,y='resis_log',x='richness',covar=['prec'],method='spearman').round(4)
    bi_pd_fsc_rich_pcor[bi_n] = {'r':fsc_pcor['r'].values[0], 'p-val': fsc_pcor['p-val'].values[0]}
    bi_pd_rich_fsc_pcor[bi_n] = {'r':rich_pcor['r'].values[0], 'p-val': rich_pcor['p-val'].values[0]}
    bi_pd_fsc_prec_pcor[bi_n] = {'r':fsc_prec_pcor['r'].values[0], 'p-val': fsc_prec_pcor['p-val'].values[0]}
    bi_pd_rich_prec_pcor[bi_n] = {'r':rich_prec_pcor['r'].values[0], 'p-val': rich_prec_pcor['p-val'].values[0]}


pg.partial_corr(data = drought_resistance_use,y='resis_log',x='richness',covar=['prec'], method='spearman').round(3)
pg.partial_corr(data = drought_resistance_use,y='resis_log',x='richness',covar=['fsc'], method='spearman').round(3)

pg.partial_corr(data = drought_resistance_use,y='resis_log',x='fsc',covar=['prec'], method='spearman').round(3)
pg.partial_corr(data = drought_resistance_use,y='resis_log',x='fsc',covar=['richness'], method='spearman').round(3)

bi_pd_fsc_rich_pcor_df = pd.DataFrame(bi_pd_fsc_rich_pcor).T
bi_pd_fsc_rich_pcor_df['var'] = 'fsc_rich'
bi_pd_fsc_rich_pcor_df = bi_pd_fsc_rich_pcor_df.sort_index()
bi_pd_fsc_rich_pcor_df

bi_pd_rich_fsc_pcor_df = pd.DataFrame(bi_pd_rich_fsc_pcor).T
bi_pd_rich_fsc_pcor_df['var'] = 'rich_fsc'
bi_pd_rich_fsc_pcor_df = bi_pd_rich_fsc_pcor_df.sort_index()
bi_pd_rich_fsc_pcor_df

bi_pd_fsc_prec_pcor_df = pd.DataFrame(bi_pd_fsc_prec_pcor).T
bi_pd_fsc_prec_pcor_df['var'] = 'fsc_prec'
bi_pd_fsc_prec_pcor_df = bi_pd_fsc_prec_pcor_df.sort_index()
bi_pd_fsc_prec_pcor_df

bi_pd_rich_prec_pcor_df = pd.DataFrame(bi_pd_rich_prec_pcor).T
bi_pd_rich_prec_pcor_df['var'] = 'rich_prec'
bi_pd_rich_prec_pcor_df = bi_pd_rich_prec_pcor_df.sort_index()
bi_pd_rich_prec_pcor_df

fsc_rich_list = list(bi_pd_fsc_rich_pcor_df.r)
fsc_rich_list.append(0.199)
fsc_prec_list = list(bi_pd_fsc_prec_pcor_df.r)
fsc_prec_list.append(0.144)
rich_fsc_list = list(bi_pd_rich_fsc_pcor_df.r)
rich_fsc_list.append(0.255)
rich_prec_list = list(bi_pd_rich_prec_pcor_df.r)
rich_prec_list.append(0.121)
bi_pcor_list = list(bi_pd_rich_prec_pcor_df.index)
bi_pcor_list.append(0)

bi_pcor_df = pd.DataFrame({'fsc_prec':fsc_prec_list, 'fsc_rich':fsc_rich_list,'rich_prec':rich_prec_list, 'rich_fsc':rich_fsc_list, 'biome':bi_pcor_list})
bi_pcor_df

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


bi_pd_pcor_all = pd.merge(bi_cor_pd_r,bi_pcor_df,on='biome',how='left')
bi_pd_pcor_all

bi_pd_pval_all = pd.merge(bi_cor_pd_p,bi_pval_df,on='biome',how='left')
bi_pd_pval_all


draw_col = ['rich','rich_prec','rich_fsc','fsc','fsc_prec','fsc_rich']
fig, ax = plt.subplots(figsize=(16,10))

im = ax.imshow(bi_pd_pcor_all[draw_col], vmin=-0.3, vmax=0.3, cmap='PiYG_r', aspect=0.6)
ax.set_xticks(ticks=np.arange(6))
ax.set_yticks(ticks=np.arange(9))
ax.set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
ax.set_yticklabels([biome_short_dic[bi_n] for bi_n in bi_pd_pcor_all.biome])

for i in [5,4,3,2,1,0]:
    for j in range(9):
        if bi_pd_pval_all[draw_col].iloc[j,i] < 0.001:
            ax.text(i,j, '***', ha='center', va = 'center')
        elif bi_pd_pval_all[draw_col].iloc[j,i] < 0.01:
            ax.text(i,j, '**', ha='center', va = 'center')
        elif bi_pd_pval_all[draw_col].iloc[j,i] < 0.05:
            ax.text(i,j, '*', ha='center', va = 'center')
for k in range(8):
    ax.text(6.5,k, '('+str(round(ai_biome[bi_pd_pcor_all.biome[k]],3))+ ' ' + str(round(ai_bi_std[bi_pd_pcor_all.biome[k]],3)) + ')' , ha='right', va = 'center', color = 'blue')

ax.text(6.5, 8, 'Aridity Index',ha='right', va = 'center', color = 'blue')
ax.text(6.5, 7.5, '(Mean    Std)',ha='right', va = 'center', color = 'blue')

ax.text(1, 9.1, 'Tree species richness',ha='center', va = 'center', color = 'black', size=18)
ax.text(4, 9.1, 'Forest structural complexity',ha='center', va = 'center', color = 'black',size=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

cb = plt.colorbar(im, shrink=0.7, pad = 0.15)
cb.set_label(label='Correlation Coefficients')
cb.outline.set_linewidth(0.05)

plt.tight_layout()
plt.savefig('result_fig/pcor_biome_resistance_cor_pcor.png', dpi = 600)


# ### 4 sm change 
sm_change_fsc_df = pd.read_csv(r'result_data/sm_change_fsc.csv')

sm_change_fsc_df['fsc_bins'] = pd.cut(sm_change_fsc_df.fsc, bins = [0,2.5,3.5,4.5,5.5,6.5,7.5,8.5,15], labels= [2,3,4,5,6,7,8,9])
fig, ax = plt.subplots(1, figsize=(8,5))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

## resistance  vs  fsc
labels_fsc = np.unique(sm_change_fsc_df.fsc_bins)
fsc_dfs = [sm_change_fsc_df.sm_change[sm_change_fsc_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(sm_change_fsc_df.sm_change[sm_change_fsc_df.fsc_bins == labels_fsc_n]) > 15 ]
fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
#print(fsc_dfs_len)
labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(sm_change_fsc_df.sm_change[sm_change_fsc_df.fsc_bins == labels_fsc_n]) > 15]
ax.boxplot(fsc_dfs, positions = labels_fsc_use, flierprops=out_values, widths =0.5, boxprops = boxprops, medianprops=medianprops, patch_artist =True)

for j in range(len(labels_fsc_use)):
    ax.text(x =labels_fsc_use[j], y = 170, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)

ax.set_xlabel('Forest structural complexity')
ax.set_ylabel('Soil moisture change')
#axes[0].set_title('(a)', loc='left', size = 14)
ax.set_ylim(0,180)
ax.grid(c = 'lightgray', alpha = 0.3)

plt.tight_layout()

plt.savefig('result_fig/sm_change_fsc.png', dpi = 600)



alpha_list = ['a','b','c','d','e','f','g','h','i']
sm_change_fsc_by_bi = sm_change_fsc_df.groupby("biome")


fig, axes = plt.subplots(3,3, figsize=(14,10))

out_values = dict(markersize = 2, marker='D', markerfacecolor = 'gray', markeredgecolor = 'none' )
boxprops = dict( color='aquamarine', facecolor = 'aquamarine')
medianprops = dict(linewidth=2.5, color='forestgreen')

for i in range(8):
    bi_n = sm_change_fsc_df.biome.value_counts().index[:-6].sort_values()[i]
    #print(ld_n)
    bi_df = sm_change_fsc_by_bi.get_group(bi_n)
    labels_fsc = np.unique(sm_change_fsc_by_bi.get_group(bi_n).fsc_bins)
    #print(labels_rich)
    fsc_dfs = [bi_df.sm_change[bi_df.fsc_bins == labels_fsc_n] for labels_fsc_n in labels_fsc if len(bi_df.sm_change[bi_df.fsc_bins == labels_fsc_n]) > 15 ]
    fsc_dfs_len = np.asarray([ len(fsc_dfs_n) for fsc_dfs_n in fsc_dfs if len(fsc_dfs_n) >15])
    #print(rich_dfs_len)
    labels_fsc_use = [  labels_fsc_n for labels_fsc_n in labels_fsc if len(bi_df.sm_change[bi_df.fsc_bins == labels_fsc_n]) > 15]
    #print(labels_rich_use)
    f = axes[i//3, i %3].boxplot(fsc_dfs, positions = labels_fsc_use, boxprops = boxprops, flierprops=out_values, medianprops=medianprops, widths =0.5, patch_artist =True)

    for j in range(len(labels_fsc_use)):
        axes[i//3, i %3].text(x =labels_fsc_use[j], y = 170, s = 'n='+ str(fsc_dfs_len[j]) ,horizontalalignment='center', size = 12)
    axes[i//3, i %3].set_title('('+alpha_list[i]+')  ' + biome_short_dic[bi_n], loc= 'left', size = 14)
    axes[i//3, i %3].set_ylim(0,180)
    axes[i//3, i %3].set_xlim(2.5,9.5)
    axes[i//3, i %3].set_xticks(np.arange(3,10),labels = [])
    axes[i//3, i %3].grid(c = 'lightgray', alpha = 0.3)

    if i > 5 :
        axes[i//3, i %3].set_xlabel('Forest structural complexity')
        axes[i//3, i %3].set_xticks(np.arange(3,10),labels = np.arange(3,10))
    if (i%3) == 0 :
        axes[i//3, i %3].set_ylabel('Soil moisture change')

axes[1, 2].set_xlabel('Forest structural complexity')
axes[1,2].set_xticks(np.arange(3,10),labels = np.arange(3,10))
axes[2,2].axis('off')
plt.tight_layout()

fig.savefig('result_fig/sm_change_fsc_biome.png', dpi = 600)


fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(sm_change_fsc_df.prec), sm_change_fsc_df.fsc, 
                C= sm_change_fsc_df.sm_change, gridsize = 100, 
                vmax=90, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Annual precipitation (log) / m')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([5, 9.5])
axes.set_xticks(np.arange(5,10))

plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')

plt.tight_layout()

plt.savefig('result_fig/sm_change_fsc_prec.png', dpi = 600)

fig, axes = plt.subplots(figsize=(8,6))

im = axes.hexbin(np.log(sm_change_fsc_df.ai), sm_change_fsc_df.fsc, 
                C= sm_change_fsc_df.sm_change, gridsize = 100, 
                vmax=90, vmin=15,
                reduce_C_function = np.median)
axes.set_xlabel('Aridity index')
axes.set_ylabel('Forest structural complexity')
axes.set_xlim([-3, 2])
axes.set_xticks(np.arange(-3,2))


plt.colorbar(im,  extend='both', label = 'Soil moisture change', shrink = 0.6, orientation='horizontal')
plt.tight_layout()
plt.savefig('result_fig/sm_change_fsc_ai.png', dpi = 600)






