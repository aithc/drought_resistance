#!/usr/bin/env python
# coding: utf-8


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, ScalarFormatter, FuncFormatter
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



my_projn = ccrs.EqualEarth(central_longitude=0)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'bold'



def maps(grid, my_projn, var, vmin, vmax, cmap_use, cb_label, cb_or, cb_extend, cb_shrink, cb_pad):
    map_ax = plt.subplot(grid, projection=my_projn)
    im = plt.pcolormesh(var.lon, var.lat,  var.values, transform=ccrs.PlateCarree(), cmap= cmap_use,
                    vmin=vmin,vmax=vmax, shading="auto")
    map_ax.coastlines(linewidth=.3,zorder=20)
    map_ax.add_feature(cfeature.LAND, facecolor="gainsboro")
    gls = map_ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5
                   )


    cb = plt.colorbar(im, extend = cb_extend, orientation = cb_or, shrink = cb_shrink, pad = cb_pad)
    cb.set_label(label=cb_label)
    cb.outline.set_linewidth(0.05)
    
    return map_ax, im

def degree_formatter(x, pos):

    if x > 0:
        return f"{int(x)}°N"
    elif x < 0:
         return f"{abs(int(x))}°S"
    else:
        return f"{abs(int(x))}°"

def lats(var, grid, colors, ymin, ymax, xmin, xmax,labels):
    plant_lat = var.mean('lon')
    plant_lat_std = var.std('lon')
    lat_mean = plt.subplot(grid)
    lat_mean.plot(plant_lat.values, plant_lat.lat,  color =colors, label=labels)
    lat_mean.fill_betweenx(plant_lat.lat, plant_lat.values - plant_lat_std.values,
                      plant_lat.values + plant_lat_std.values, color = colors, alpha = 0.5)
    
    lat_mean.set_xlabel(labels)
    lat_mean.set_ylim(ymin, ymax)
    lat_mean.set_xlim(xmin, xmax,) 
    lat_mean.yaxis.set_major_locator(MultipleLocator(30))
    # lat_mean.tick_params(axis='y', which='major', labelrotation=90)
    lat_mean.yaxis.set_major_formatter(FuncFormatter(degree_formatter))

    lat_mean.spines['top'].set_color('none')
    lat_mean.spines['right'].set_color('none')
    return lat_mean

def small_map(fig, extent, var, bins_use, c1):
    hist_ax = fig.add_axes(extent)

    hist_ax.hist(var.values.flatten(), bins = bins_use, color =c1, alpha =0.5)

    hist_ax.set_xticks(np.arange(0,150,25))
    hist_ax.set_yticks([0,5000,10000])

    hist_ax.tick_params(axis='both', which='both', color='gray', labelsize=9,labelcolor='gray')
    hist_ax.spines['bottom'].set_color('gray')
    hist_ax.spines['left'].set_color('gray')

    hist_ax.spines['top'].set_color('none')
    hist_ax.spines['right'].set_color('none')
    hist_ax.set_facecolor('none')

    hist_ax.yaxis.set_tick_params(labelsize=8, colors='gray')

    hist_ax.yaxis.offsetText.set(size=8)
    return hist_ax



with xr.open_dataset(r'data/spei/spei06.nc')  as data:
    spei = data['spei'][-1]
spei


spei = xr.DataArray(spei.values, coords=[spei.lat, spei.lon], dims=['lat','lon'])

with xr.open_dataset(r'D:/data/modis_landcover/modis_IGBP_2001_2022.nc') as data:
    ld_mask = data['modis_landcover'][-1]


ld_mask = xr.DataArray(ld_mask, coords=[ld_mask.lat, ld_mask.lon], dims=['lat','lon'])
ld_mask = ld_mask.where(ld_mask >0)
ld_mask = ld_mask.where(ld_mask <15)
ld_mask = ld_mask.interp_like(spei, method='nearest')

ld_mask

## drought chars

with xr.open_dataset(r'result_data/drought_chars_1982_2022_new.nc') as data:
    drought_severity = data['severity']
    drought_count =  data['count']     
    drought_duration = data['duration']

drought_severity = drought_severity.where(spei > -10)
drought_count =  drought_count.where(spei > -10)
drought_duration = drought_duration.where(spei > -10)

drought_severity = drought_severity.where(ld_mask > 0)
drought_count =  drought_count.where(ld_mask > 0)
drought_duration = drought_duration.where(ld_mask > 0)


fig = plt.figure(figsize = (8,8))  
grid = plt.GridSpec(6,8, )

map_ds, im1 = maps(grid[0:2, 0:6], my_projn, drought_severity,    -2.5, -0.5, cmap_use='magma_r', cb_label= 'Severity', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dc, im1 = maps(grid[2:4, 0:6], my_projn, drought_count,    5, 35, cmap_use='magma', cb_label= 'Count', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dd, im1 = maps(grid[4:6, 0:6], my_projn, drought_duration,    2, 6, cmap_use='magma', cb_label= 'Duration', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)

lat_ds = lats(drought_severity,    grid[0:2, 6:8],    'orangered', -60, 90, -2.5, 0, 'Severity')
lat_dc = lats(drought_count, grid[2:4, 6:8],    'orangered', -60, 90, 5, 35, 'Count')
lat_dd = lats(drought_duration,    grid[4:6, 6:8],    'orangered', -60, 90, 1, 6.5, 'Duration')

map_ds.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_ds.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
map_dc.set_title('(c)', loc='left', fontsize=11, fontweight='bold')
lat_dc.set_title('(d)', loc='left',fontsize=11, fontweight='bold')   
map_dd.set_title('(e)', loc='left', fontsize=11, fontweight='bold')  
lat_dd.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.01)

plt.savefig(r'result_fig/global_drought_char.png', dpi = 600)


# ### 2 resistance

with xr.open_dataset(r'result_data/gimms_resistance.nc')  as data:
    resistance = data['gimms_resistance']

resistance_mean = resistance.mean(dim='year')



fig = plt.figure(figsize = (8,4))   
grid = plt.GridSpec(5,16, ) 

map_rs, im1 = maps(grid[0:5, 0:14], my_projn, resistance_mean,  0, 300, cmap_use='summer_r', cb_label= 'Drought resistance',
                    cb_extend='max', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

lat_rs = lats( resistance_mean.where( resistance_mean<350),    grid[0:4, 14:16],    'green', -40, 75, -20, 250, 'Drought resistance')

his_rs= small_map(fig, [0.15, 0.4, 0.12, 0.15],   resistance_mean,  bins_use= [0,50,100,150,200,250,300] ,  c1='springgreen')  

map_rs.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.9)

plt.savefig(r'result_fig/global_drought_resistance.png', dpi = 600)


# ### 3 threshold


with xr.open_dataset(r'result_data/spei_annual_threshold.nc')['spei'] as data:
    spei_threshold = data.where(ld_mask > 0)

fig = plt.figure(figsize = (8,4))   
grid = plt.GridSpec(5,16, ) 

map_th, im1 = maps(grid[0:5, 0:14], my_projn, spei_threshold,  -1.5, 0, cmap_use='inferno_r', cb_label= 'Drought threshold',
                    cb_extend='min', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

lat_th = lats( spei_threshold,    grid[0:4, 14:16],    'red', -60, 75, -1.5, 0, 'Drought threshold')

his_th= small_map(fig, [0.15, 0.4, 0.12, 0.15],   spei_threshold,  bins_use= [-2,-1.5,-1,-0.5,0] ,  c1='orange')  

map_th.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_th.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.9)

plt.savefig(r'result_fig/global_drought_threshold.png', dpi = 600)


# ### 4 trends

with xr.open_dataset(r'result_data/drought_count_trend.nc') as data:
    drought_count_trend = data.where(ld_mask > 0)

with xr.open_dataset(r'result_data/drought_dura_trend.nc') as data:
    drought_dura_trend = data.where(ld_mask > 0)

with xr.open_dataset(r'result_data/drought_sev_trend.nc') as data:
    drought_sev_trend = data.where(ld_mask > 0)

fig = plt.figure(figsize = (6,8))   
grid = plt.GridSpec(6,6, )

map_ds, im1 = maps(grid[0:2, 0:6], my_projn, drought_sev_trend['trend'],    -0.02, 0.02, cmap_use='BrBG', cb_label= 'Severity trend', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dc, im1 = maps(grid[2:4, 0:6], my_projn, drought_count_trend['trend'],    -1, 1, cmap_use='BrBG_r', cb_label= 'Count trend', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dd, im1 = maps(grid[4:6, 0:6], my_projn, drought_dura_trend['trend'],    -0.3, 0.3, cmap_use='BrBG_r', cb_label= 'Duration trend', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)

map_ds.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
map_dc.set_title('(b)', loc='left', fontsize=11, fontweight='bold')
map_dd.set_title('(c)', loc='left', fontsize=11, fontweight='bold')  


fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05)

plt.savefig(r'result_fig/global_drought_char_trend.png', dpi = 600)


# ### 5 fsc richness landcover

with xr.open_dataset(r'result_data/fsc_05.nc') as data:
    fsc = data['fsc']
fsc


with xr.open_dataset(r'result_data/plant_richness_log_05.nc') as data:
    plant_richness = data['richness']
plant_richness


fig = plt.figure(figsize = (8,7))  
grid = plt.GridSpec(4,8, )

map_rich, im = maps(grid[0:2, 0:8], my_projn, plant_richness,    0.5, 5, cmap_use='viridis', cb_label= 'Species richness (log)', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_fsc, im = maps(grid[2:4, 0:8], my_projn, fsc.where(spei > -10),    2, 12, cmap_use='viridis', cb_label= 'Forest structural complexity', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)

map_rich.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
map_fsc.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  


fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05)

plt.savefig(r'result_fig/global_richness_fsc.png', dpi = 600)


with xr.open_dataset(r'result_data/landcover_005_use.nc') as data:    
    ld = data['modis_landcover'].interp_like(fsc, method='nearest')
ld

veget_cmap = ['olivedrab','darkolivegreen','forestgreen','limegreen','lightgreen','orange',
              'moccasin','sienna','peru','yellow','dodgerblue']

fig = plt.figure(figsize=[10,4])
ax = plt.subplot(projection=my_projn)

map_ld = (ld+0.5).plot.imshow(ax = ax, transform = ccrs.PlateCarree(), colors = veget_cmap, levels = 11 , add_colorbar =False)
ax.coastlines(linewidth=.3,zorder=20)
ax.add_feature(cfeature.LAND, facecolor="gainsboro")
gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5)

cbar_ld = fig.colorbar(map_ld, label = 'IGBP Landcover', ticks = np.arange(1.5,12,1), shrink = 0.8)
cbar_ld.set_ticklabels(['Evergreen Needleleaf Forests','Evergreen Broadleaf Forests',
                        'Deciduous Needleleaf Forests','Deciduous Broadleaf Forests',
                        'Mixed Forests','Closed Shrublands',
                        'Open Shrublands','Woody Savannas',
                        'Savannas','Grasslands','Permanent Wetlands'])

fig.tight_layout()

plt.savefig(r'result_fig/global_landcover.png', dpi = 600)


# ### 6 fsc-richness

fsc_richness = xr.Dataset({'fsc':fsc, 'richness':plant_richness})

fsc_richness_df = fsc_richness.to_dataframe()
fsc_richness_df.head() 

fsc_richness_df = fsc_richness_df.dropna(how='any')
fsc_richness_df = fsc_richness_df.reset_index()
fsc_richness_df.head()

import mapclassify


fsc_bin = mapclassify.Quantiles(y=fsc_richness_df['fsc'],k=10)
fsc_bin

richness_bin = mapclassify.Quantiles(y=fsc_richness_df['richness'],k=10)
richness_bin


fsc_richness_df.loc[:,'fsc_Class'] = fsc_bin.yb
fsc_richness_df.loc[:,'richness_Class'] = richness_bin.yb
fsc_richness_df.head()

from matplotlib.colors import rgb2hex,hex2color


cp1 = np.linspace(0,1,10)
cp2 = np.linspace(0,1,10)

Cp1, Cp2 = np.meshgrid(cp1,cp2)
C0 = np.zeros_like(Cp1)
Legend = np.dstack((Cp1,Cp2, C0+0.5))
plt.imshow(Legend, origin="lower", extent=[0,1,0,1])
plt.show()


bivariate_palette = {}
n_categories =10
for j in range(n_categories):
    for i in range(n_categories):
        x = Legend[i][j]
        bivariate_palette[(i, j)] = rgb2hex(x)

fsc_richness_df.loc[:,"color"] = [bivariate_palette[(j, i)] for i,j in             zip(fsc_richness_df["fsc_Class"],fsc_richness_df["richness_Class"])]
fsc_richness_df.head()

fig = plt.figure(figsize=(12,6),facecolor="w")
ax = fig.add_subplot(projection=my_projn)
ax.coastlines(linewidth=0.5,zorder=20)
ax.add_feature(cfeature.LAND, facecolor='gainsboro')
gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5)

map_fsc_rich = ax.scatter( x=fsc_richness_df['lon'],y=fsc_richness_df['lat'], c=fsc_richness_df['color'] , transform = ccrs.PlateCarree(), s=0.1)


fig = plt.figure(figsize=(12,6),facecolor="w")
ax = fig.add_subplot(projection=my_projn)
ax.coastlines(linewidth=0.5,zorder=20)
ax.add_feature(cfeature.LAND, facecolor='gainsboro')

map_fsc_rich = ax.scatter( x=fsc_richness_df['lon'],y=fsc_richness_df['lat'], c=fsc_richness_df['color'] , transform = ccrs.PlateCarree(), s=1.5, edgecolor='none')
ax.scatter( x=-180,y=0, c='white' , transform = ccrs.PlateCarree(), s=0.1)  

gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5)

cax = fig.add_axes([0.05,0.2,0.25,0.25])

cax.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) 
cax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) 
cax.axis("off")
cax.imshow(Legend, origin="lower", extent=[0,1,0,1])
cax.text(s='Species richness (log)', x=0, y=-0.18,size=12) 
cax.text(s='Structural complexity', x=-0.18, y=0, size=12,rotation=90)

plt.tight_layout()
fig.savefig(r'result_fig/fsc_richness_bi_try.png', dpi=600)




def interpolate_color_from_4corners(bl, br, tl, tr, nx, ny):
    import numpy as np
    from matplotlib.colors import hex2color
    bl = np.array(hex2color(bl))*255
    br = np.array(hex2color(br))*255
    tl = np.array(hex2color(tl))*255
    tr = np.array(hex2color(tr))*255
    new_colors = np.zeros((ny,nx,3)).astype(np.uint8)
    for irgb in np.arange(3):
        for iy in np.arange(ny):
            for ix in np.arange(nx):
                u0, v0 = ix / nx, iy / ny
                u1, v1 = 1 - u0, 1 - v0
                new_colors[iy, ix, irgb] = np.floor(0.5 + u1*v1*bl[irgb] 
                                                    + u0*v1*br[irgb] 
                                                    + u1*v0*tl[irgb] 
                                                    + u0*v0*tr[irgb])
    return new_colors


bl = "#FFB7B8"
br = "#7F00FF" 
tl = "#80FF00"
tr = "#004847"
nx, ny = 10, 10


colors_100 = interpolate_color_from_4corners(bl, br, tl, tr, nx, ny)


plt.imshow(colors_100, origin='lower')



# ### 8 soil moisture

with xr.open_dataset(r'result_data/sm_change_mean.nc')  as data:
    sm_change = data['sm_change']

sm_change

fig = plt.figure(figsize = (8,4))  
grid = plt.GridSpec(5,16, ) 

map_sm, im1 = maps(grid[0:5, 0:14], my_projn, sm_change,  0, 100, cmap_use='gist_earth_r', cb_label= 'Soil moisture change',
                    cb_extend='max', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

lat_sm = lats( sm_change,    grid[0:4, 14:16],    'lightseagreen', -40, 75, -20, 120, 'Soil moisture\n' + 'change')

his_sm= small_map(fig, [0.15, 0.4, 0.12, 0.15],   sm_change,  bins_use= [0,25,50,75,100,125] ,  c1='aquamarine')  

map_sm.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_sm.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.9)

plt.savefig(r'result_fig/global_sm_change.png', dpi = 600)


# ### 9 temp-diff

with xr.open_dataset(r'result_data/cor_soil_air_temp_offset_and_fsc.nc')['cor'] as data:
    cor_result_all = data
cor_result_all

with  xr.open_dataset(r'result_data/pvalue_soil_air_temp_offset_and_fsc.nc')['p_value']  as data:
    p_result_all = data
p_result_all

my_projn = ccrs.Mollweide(central_longitude=0)

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


fig, axes = plt.subplots(4,3,figsize = (12,9), subplot_kw={'projection':my_projn})   # 画布

for i in range(12):
    axes[i//3,i%3].axis('off')
    plot_data = cor_result_all.sel(month= (i+1))
    axes[i//3,i%3].set_extent([-180, 180, -50, 85], crs=ccrs.PlateCarree())
    im = axes[i//3,i%3].pcolormesh(plot_data.lon, plot_data.lat,  plot_data.values, transform=ccrs.PlateCarree(), cmap= 'RdBu_r',
                    vmin=-1,vmax=1, shading="auto")
    #axes[i//3,i%3].coastlines(linewidth=.3,zorder=20)
    axes[i//3,i%3].set_title(months[i], loc='left', size = 15)
    axes[i//3,i%3].add_feature(cfeature.LAND, facecolor="whitesmoke")

position=fig.add_axes([0.2,0.07,0.6,0.02])
cb = plt.colorbar(im, position, extend='neither',orientation='horizontal',shrink = 0.8)
cb.set_label('Correlation Coefficients',fontdict={'size':15})
cb.outline.set_linewidth(0.05)

plt.subplots_adjust(top=0.95, bottom=0.12, right=0.98, left=0.04, hspace=0, wspace=0)

plt.savefig('result_fig/cor_map_fsc_soil_air_temp.png', dpi = 600)


# ### 10 biome

import rioxarray

with rioxarray.open_rasterio(r'data/official_teow/biome.tif')  as data:
    biome = data
biome = biome.where(biome>0)
biome = biome.where(biome<90)

biome = xr.DataArray(biome[0].values, coords=[biome.y, biome.x], dims=['lat','lon'])
biome

biome_list = ['Trop.&Subtrop. Moist Broad. Forests', 'Trop.&Subtrop. Dry Broad. Forests', 'Trop.&Subtrop. Coni. Forests', 'Temp. Broad.&Mixed Forests',
          'Temp. Coni. Forests','Boreal Forests',
          'Trop.&Subtrop. Gra.,Sav.&Shrub.','Temp. Gra.,Sav.&Shrub.','Flooded Gra.&Sav.',
          'Montane Gra.&Shrub.','Tundra','Mediter. Forests, Woodlands & Scrub.','Deserts & Xeric Shrublands','Mangroves']

biome_cmap = ['#aafe00','#00a985','#9fd6c3','#38a900','#98e601','#267200','#ffab00',
              '#ffff72','#015be7','#a97000','#bee8fe','#cdf67a','#ffebbf','#a900e7']


fig = plt.figure(figsize=[10,4])
ax = plt.subplot(projection=my_projn)

map_ld = (biome+0.5).plot.imshow(ax = ax, transform = ccrs.PlateCarree(), colors = biome_cmap, levels = 14 , add_colorbar =False)
ax.coastlines(linewidth=.3,zorder=20)
ax.add_feature(cfeature.LAND, facecolor="gainsboro")
gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5)

cbar_ld = fig.colorbar(map_ld, label = 'Biome', ticks = np.arange(1.5,15,1), shrink = 0.8)
cbar_ld.set_ticklabels(biome_list)

fig.tight_layout()

plt.savefig(r'result_fig/global_biomes.png', dpi = 600)



