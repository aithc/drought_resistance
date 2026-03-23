# %% [markdown]
# ## 干旱抵抗力 新的结果画图
# 
# 所有图的范围都要改到csc的范围
# 
# 加上机器学习的结果
# 
# 抵抗力要画所有数据计算的结果  kndvi 和 sif放在正文  ndvi和vod放在附件

# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, ScalarFormatter, FuncFormatter
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn  as sns

# %%
my_projn = ccrs.EqualEarth(central_longitude=0)
# 设置字体格式
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'bold'

# %% [markdown]
# ### 定义好的 画图函数  cch

# %%
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
    """
    自定义度数格式化函数，将纬度数值转换为度数表示，如90°S。
    """
    if x > 0:
        return f"{int(x)}°N"
    elif x < 0:
         return f"{abs(int(x))}°S"
    else:
        return f"{abs(int(x))}°"

# 定义纬向平均绘图函数
def lats(var, grid, colors, ymin, ymax, xmin, xmax,labels, x_ticks):
    plant_lat = var.mean('lon')
    plant_lat_std = var.std('lon')
    lat_mean = plt.subplot(grid)
    lat_mean.plot(plant_lat.values, plant_lat.lat,  color =colors, label=labels)
    lat_mean.fill_betweenx(plant_lat.lat, plant_lat.values - plant_lat_std.values,
                      plant_lat.values + plant_lat_std.values, color = colors, alpha = 0.5)
    
    lat_mean.set_xlabel(labels)
    lat_mean.set_ylim(ymin, ymax)
    lat_mean.set_xlim(xmin, xmax,) 
    lat_mean.set_xticks(x_ticks)
    lat_mean.yaxis.set_major_locator(MultipleLocator(30))
    # lat_mean.tick_params(axis='y', which='major', labelrotation=90)
    lat_mean.yaxis.set_major_formatter(FuncFormatter(degree_formatter))

    lat_mean.spines['top'].set_color('none')
    lat_mean.spines['right'].set_color('none')
    return lat_mean

def small_map(fig, extent, var, bins_use, c1, x_ticks, y_ticks = [0,3000,6000]):
    """
        fig:画布
        var：变量
        hist_ax：子图
        xmin，xmax: x轴的最大值最小值
        bw：柱状图的宽度
        ymin，ymax：y轴的最大最小值
        yloc：y轴的坐标间隔
    
    """
    # 在底图中添加小图
    hist_ax = fig.add_axes(extent)
        # 绘制直方图
    hist_ax.hist(var.values.flatten(), bins = bins_use, color =c1, alpha =0.5)
    # 设置x/y范围，刻度间隔，以及科学技术法
    #hist_ax.set_xlim(xmin,xmax)
    #hist_ax.set_ylim(ymin,ymax)
    #hist_ax.xaxis.set_major_locator(MultipleLocator(1))
    # hist_ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    # 科学计数法
    #y_formatter = ScalarFormatter(useMathText=True)
    #y_formatter.set_powerlimits((-2, 2))  # 可选：设置指数的显示范围
    #hist_ax.yaxis.set_major_formatter(y_formatter)
    #hist_ax.yaxis.set_minor_locator(MultipleLocator(yloc))
    hist_ax.set_xticks(x_ticks)
    hist_ax.set_yticks(y_ticks)
    # 将横纵轴的颜色调淡（轴，轴刻度和刻度标签）
    hist_ax.tick_params(axis='both', which='both', color='gray', labelsize=9,labelcolor='gray')
    hist_ax.spines['bottom'].set_color('gray')
    hist_ax.spines['left'].set_color('gray')
    # 不显示上部和右部的框线
    hist_ax.spines['top'].set_color('none')
    hist_ax.spines['right'].set_color('none')
    hist_ax.set_facecolor('none')

    hist_ax.yaxis.set_tick_params(labelsize=8, colors='gray')
    # 调整 x 轴刻度标签上指数的字体大小
    hist_ax.yaxis.offsetText.set(size=8)
    return hist_ax


# %% [markdown]
# ## 0 准备工作 
# 
# 读取一些数据来确定画图的范围  csc 和 ld
# 
# 读取一些数据

# %%
with xr.open_dataset(r'D:/data/fsc_from_su/data/global_forest_csc/global_forest_csc.tif')  as  data:
    fsc_ex = data['band_data'][0].drop(['spatial_ref','band'])
    fsc_ex = fsc_ex.rename({'x':'lon','y':'lat'})
    fsc_ex = fsc_ex.coarsen(lat = 20, lon=20).mean()
fsc_ex

# %%
fsc_ex.plot()

# %%
fsc_ex.sel(lat = slice(52,-52)).plot()

# %%
fsc_ex = fsc_ex.sel(lat = slice(52,-52))

# %%
with xr.open_dataarray(r'../result_data/ld_mask_05.nc') as data:
    ld_mask = data
ld_mask

# %%
ld_mask.plot()

# %%
ld_mask = ld_mask.interp_like(fsc_ex)
ld_mask

# %%
ld_mask.plot()

# %% [markdown]
# ### 0.1  读取 抵抗力和fsc的 csv

# %%
import pandas as pd

# %%
drought_resistance_mid_df_kndvi = pd.read_csv(r'E:/python_output/fsc_drought/drought_resistance_kndvi_df_csc.csv') 
drought_resistance_mid_df_kndvi.head()

# %%
drought_resistance_mid_df_kndvi.describe()

# %%
drought_resistance_mid_df_sif = pd.read_csv(r'E:/python_output/fsc_drought/drought_resistance_sif_df_csc.csv') 
drought_resistance_mid_df_sif.head()

# %%
drought_resistance_mid_df_sif.describe()

# %%
drou_mid_use_kndvi = drought_resistance_mid_df_kndvi[drought_resistance_mid_df_kndvi.resistance_kndvi < 233]
drou_mid_use_kndvi['fsc_bins'] = pd.cut(drou_mid_use_kndvi.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
drou_mid_use_kndvi['rich_bins'] = pd.cut(drou_mid_use_kndvi.richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %%
drou_mid_use_sif = drought_resistance_mid_df_sif[drought_resistance_mid_df_sif.resistance_sif < 116]
drou_mid_use_sif['fsc_bins'] = pd.cut(drou_mid_use_sif.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
drou_mid_use_sif['rich_bins'] = pd.cut(drou_mid_use_sif.richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %%
drou_sif_draw = drou_mid_use_sif[['fsc_bins','rich_bins','resistance_sif']].rename(columns={'fsc_bins':'fsc','rich_bins':'richness','resistance_sif':'resistance'})
drou_kndvi_draw = drou_mid_use_kndvi[['fsc_bins','rich_bins','resistance_kndvi']].rename(columns={'fsc_bins':'fsc','rich_bins':'richness','resistance_kndvi':'resistance'})

# %%
drou_sif_draw['data_type'] = 'SIF'
drou_kndvi_draw['data_type'] = 'kNDVI'
drought_resistance_mid_draw = pd.concat([drou_sif_draw,drou_kndvi_draw])
drought_resistance_mid_draw.head()

# %%
drought_resistance_mid_draw['resistance_log'] = np.log(drought_resistance_mid_draw['resistance'])

# %%
## 换成 基于事件的
df_all_kndvi = pd.read_csv(r'E:/python_output/fsc_drought/df_all_kndvi_events_data.csv')
df_all_kndvi_after2000 = pd.read_csv(r'E:/python_output/fsc_drought/df_all_kndvi_events_data_after2000.csv')
df_all_sif = pd.read_csv(r'E:/python_output/fsc_drought/df_all_sif_events_data.csv')
df_all_kndvi['fsc_bins'] = pd.cut(df_all_kndvi.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
df_all_kndvi['rich_bins'] = pd.cut(df_all_kndvi.plant_richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])
df_all_kndvi_after2000['fsc_bins'] = pd.cut(df_all_kndvi_after2000.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
df_all_kndvi_after2000['rich_bins'] = pd.cut(df_all_kndvi_after2000.plant_richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])
df_all_sif['fsc_bins'] = pd.cut(df_all_sif.fsc, bins = [0,9.25,9.75,10.25,10.75,11.25,13], labels= [9,9.5,10,10.5,11,11.5])
df_all_sif['rich_bins'] = pd.cut(df_all_sif.plant_richness, bins = [0,1.5,2.5,3.5,4.5,5.5], labels= [1,2,3,4,5])

# %%
df_all_kndvi.head()

# %%
df_all_kndvi['kndvi2_log'] = np.log(df_all_kndvi['kndvi2']/(1-df_all_kndvi['kndvi2']))
df_all_kndvi_after2000['kndvi2_log'] = np.log(df_all_kndvi_after2000['kndvi2']/(1-df_all_kndvi_after2000['kndvi2']))
df_all_sif['sif2_log'] = np.log(df_all_sif['sif2']/(1-df_all_sif['sif2']))

# %%
drou_sif_draw = df_all_sif[['fsc_bins','rich_bins','resis_log','sif2','sif2_log']].rename(columns={'fsc_bins':'fsc','rich_bins':'richness','resis_log':'resistance','sif2':'resis_r','sif2_log':'resis_r_log'})
drou_kndvi_draw = df_all_kndvi[['fsc_bins','rich_bins','resis_log','kndvi2','kndvi2_log']].rename(columns={'fsc_bins':'fsc','rich_bins':'richness','resis_log':'resistance','kndvi2':'resis_r','kndvi2_log':'resis_r_log'})
drou_kndvi_draw_after2000 = df_all_kndvi_after2000[['fsc_bins','rich_bins','resis_log','kndvi2','kndvi2_log']].rename(columns={'fsc_bins':'fsc','rich_bins':'richness','resis_log':'resistance','kndvi2':'resis_r','kndvi2_log':'resis_r_log'})
drou_sif_draw['data_type'] = 'SIF'
drou_kndvi_draw['data_type'] = 'kNDVI'
drou_kndvi_draw_after2000['data_type'] = 'kNDVI2000'
drought_resistance_mid_draw = pd.concat([drou_sif_draw,drou_kndvi_draw,drou_kndvi_draw_after2000])
drought_resistance_mid_draw.head()

# %% [markdown]
# ## 1 干旱指标地图

# %% [markdown]
# ### 1.1 读取数据

# %%
with xr.open_dataset(r'result_data/drought_chars_1982_2022_new.nc') as data:
    drought_severity = data['severity']
    drought_count =  data['count']     
    drought_duration = data['duration']

# %%
drought_severity = drought_severity.interp_like(ld_mask)
drought_count = drought_count.interp_like(ld_mask)
drought_duration = drought_duration.interp_like(ld_mask)

# %%
drought_severity.where(ld_mask > 0).plot(center =False)

# %%
drought_count.where(ld_mask > 0).plot()

# %%
drought_duration.where(ld_mask > 0).plot()

# %%
drought_severity = drought_severity.where(ld_mask > 0)
drought_count =  drought_count.where(ld_mask > 0)
drought_duration = drought_duration.where(ld_mask > 0)

# %%
fig = plt.figure(figsize = (8,8))   # 画布
grid = plt.GridSpec(6,8, ) # 子图网格

# 全球恢复力分布
map_ds, im1 = maps(grid[0:2, 0:6], my_projn, drought_severity,    -2.5, -0.5, cmap_use='magma_r', cb_label= 'Severity', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dc, im1 = maps(grid[2:4, 0:6], my_projn, drought_count,    5, 35, cmap_use='magma', cb_label= 'Count', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dd, im1 = maps(grid[4:6, 0:6], my_projn, drought_duration,    2, 6, cmap_use='magma', cb_label= 'Duration', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)

# 纬向平均图
lat_ds = lats(drought_severity,    grid[0:2, 6:8],    'orangered', -60, 60, -2.4, -0.8, 'Severity', x_ticks=[-2,-1])
lat_dc = lats(drought_count, grid[2:4, 6:8],    'orangered', -60, 60, 10, 35, 'Count', x_ticks=[15,30])
lat_dd = lats(drought_duration,    grid[4:6, 6:8],    'orangered', -60, 60, 1.5, 5.5, 'Duration', x_ticks=[2,4])

map_ds.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_ds.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
map_dc.set_title('(c)', loc='left', fontsize=11, fontweight='bold')
lat_dc.set_title('(d)', loc='left',fontsize=11, fontweight='bold')   
map_dd.set_title('(e)', loc='left', fontsize=11, fontweight='bold')  
lat_dd.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.01)

plt.savefig(r'result_fig_new/global_drought_char.png', dpi = 600)

# %% [markdown]
# ## 2 干旱抵抗力地图
# 
# 变成两个 sif 和 kndvi

# %% [markdown]
# ### 2.1 读取数据

# %%
## 之前的抵抗力指标
## kndvi
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance.nc')  as data:
    resis_kndvi_nt = np.log(data['kndvi_resistance'])

with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance.nc')  as data:
    resis_kndvi_sh = np.log(data['kndvi_resistance'])

## kndvi after2000
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance_after2000.nc')  as data:
    resis_kndvi_nt_after2000 = np.log(data['kndvi_resistance'])

with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance_after2000.nc')  as data:
    resis_kndvi_sh_after2000 = np.log(data['kndvi_resistance'])

# sif
with xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance.nc')  as data:
    resis_sif_nt = np.log(data['sif_resistance'])

with xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance.nc')  as data:
    resis_sif_sh = np.log(data['sif_resistance'])

## Ydrou/Ymean
## kndvi
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2.nc')  as data:
    resis_kndvi2_nt = data['kndvi_resistance']

with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2.nc')  as data:
    resis_kndvi2_sh = data['kndvi_resistance']

## kndvi after2000
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2_after2000.nc')  as data:
    resis_kndvi2_nt_after2000 = data['kndvi_resistance']

with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2_after2000.nc')  as data:
    resis_kndvi2_sh_after2000 = data['kndvi_resistance']

# sif
with xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance2.nc')  as data:
    resis_sif2_nt = data['sif_resistance']

with xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance2.nc')  as data:
    resis_sif2_sh = data['sif_resistance']

## Ydrou/Ymean log
## kndvi
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2.nc')  as data:
    resis_kndvi2_log_nt = np.log(data['kndvi_resistance'] / (1-data['kndvi_resistance']) )

with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2.nc')  as data:
    resis_kndvi2_log_sh = np.log(data['kndvi_resistance'] / (1-data['kndvi_resistance']) )

## kndvi after2000
with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_nt_resistance2_after2000.nc')  as data:
    resis_kndvi2_log_nt_after2000 = np.log(data['kndvi_resistance'] / (1-data['kndvi_resistance']) )

with xr.open_dataset(r'E:/python_output/fsc_drought/kndvi_sh_resistance2_after2000.nc')  as data:
    resis_kndvi2_log_sh_after2000 = np.log(data['kndvi_resistance'] / (1-data['kndvi_resistance']) )

# sif
with xr.open_dataset(r'E:/python_output/fsc_drought/sif_nt_resistance2.nc')  as data:
    resis_sif2_log_nt = np.log(data['sif_resistance'] / (1-data['sif_resistance']) )

with xr.open_dataset(r'E:/python_output/fsc_drought/sif_sh_resistance2.nc')  as data:
    resis_sif2_log_sh = np.log(data['sif_resistance'] / (1-data['sif_resistance']) )

# %%
resis_kndvi_nt_mid = resis_kndvi_nt.median(dim='year')
resis_kndvi_sh_mid = resis_kndvi_sh.median(dim='year')

resis_kndvi_mid = xr.concat([resis_kndvi_nt_mid,resis_kndvi_sh_mid], dim='lat')
resis_kndvi_mid = resis_kndvi_mid.sortby('lat')
resis_kndvi_mid = resis_kndvi_mid.interp_like(ld_mask)
resis_kndvi_mid

# %%
resis_kndvi_mid.quantile([0.01,0.1,0.5,0.9,0.99])

# %%
resis_kndvi_mid.plot(vmax = 6, vmin = 1)

# %%
resis_sif_nt_mid = resis_sif_nt.median(dim='year')
resis_sif_sh_mid = resis_sif_sh.median(dim='year')

resis_sif_mid = xr.concat([resis_sif_nt_mid,resis_sif_sh_mid], dim='lat')
resis_sif_mid = resis_sif_mid.sortby('lat')
resis_sif_mid = resis_sif_mid.interp_like(ld_mask)
resis_sif_mid

# %%
resis_sif_mid.plot(vmax = 5, vmin = 1)

# %%
resis_kndvi_nt_mid_after2000 = resis_kndvi_nt_after2000.median(dim='year')
resis_kndvi_sh_mid_after2000= resis_kndvi_sh_after2000.median(dim='year')

resis_kndvi_mid_after2000 = xr.concat([resis_kndvi_nt_mid_after2000,resis_kndvi_sh_mid_after2000], dim='lat')
resis_kndvi_mid_after2000 = resis_kndvi_mid_after2000.sortby('lat')
resis_kndvi_mid_after2000 = resis_kndvi_mid_after2000.interp_like(ld_mask)
resis_kndvi_mid_after2000

# %%
resis_kndvi2_nt_mid = resis_kndvi2_nt.median(dim='year')
resis_kndvi2_sh_mid = resis_kndvi2_sh.median(dim='year')

resis_kndvi2_mid = xr.concat([resis_kndvi2_nt_mid,resis_kndvi2_sh_mid], dim='lat')
resis_kndvi2_mid = resis_kndvi2_mid.sortby('lat')
resis_kndvi2_mid = resis_kndvi2_mid.interp_like(ld_mask)
resis_kndvi2_mid

# %%
resis_sif2_nt_mid = resis_sif2_nt.median(dim='year')
resis_sif2_sh_mid = resis_sif2_sh.median(dim='year')

resis_sif2_mid = xr.concat([resis_sif2_nt_mid,resis_sif2_sh_mid], dim='lat')
resis_sif2_mid = resis_sif2_mid.sortby('lat')
resis_sif2_mid = resis_sif2_mid.interp_like(ld_mask)
resis_sif2_mid

# %%
resis_kndvi2_nt_mid_after2000 = resis_kndvi2_nt_after2000.median(dim='year')
resis_kndvi2_sh_mid_after2000= resis_kndvi2_sh_after2000.median(dim='year')

resis_kndvi2_mid_after2000 = xr.concat([resis_kndvi2_nt_mid_after2000,resis_kndvi2_sh_mid_after2000], dim='lat')
resis_kndvi2_mid_after2000 = resis_kndvi2_mid_after2000.sortby('lat')
resis_kndvi2_mid_after2000 = resis_kndvi2_mid_after2000.interp_like(ld_mask)
resis_kndvi2_mid_after2000

# %%
resis_kndvi2_log_nt_mid = resis_kndvi2_log_nt.median(dim='year')
resis_kndvi2_log_sh_mid = resis_kndvi2_log_sh.median(dim='year')

resis_kndvi2_log_mid = xr.concat([resis_kndvi2_log_nt_mid,resis_kndvi2_log_sh_mid], dim='lat')
resis_kndvi2_log_mid = resis_kndvi2_log_mid.sortby('lat')
resis_kndvi2_log_mid = resis_kndvi2_log_mid.interp_like(ld_mask)
resis_kndvi2_log_mid

# %%
resis_sif2_log_nt_mid = resis_sif2_log_nt.median(dim='year')
resis_sif2_log_sh_mid = resis_sif2_log_sh.median(dim='year')

resis_sif2_log_mid = xr.concat([resis_sif2_log_nt_mid,resis_sif2_log_sh_mid], dim='lat')
resis_sif2_log_mid = resis_sif2_log_mid.sortby('lat')
resis_sif2_log_mid = resis_sif2_log_mid.interp_like(ld_mask)
resis_sif2_log_mid

# %%
resis_kndvi2_log_nt_mid_after2000 = resis_kndvi2_log_nt_after2000.median(dim='year')
resis_kndvi2_log_sh_mid_after2000= resis_kndvi2_log_sh_after2000.median(dim='year')

resis_kndvi2_log_mid_after2000 = xr.concat([resis_kndvi2_log_nt_mid_after2000,resis_kndvi2_log_sh_mid_after2000], dim='lat')
resis_kndvi2_log_mid_after2000 = resis_kndvi2_log_mid_after2000.sortby('lat')
resis_kndvi2_log_mid_after2000 = resis_kndvi2_log_mid_after2000.interp_like(ld_mask)
resis_kndvi2_log_mid_after2000

# %%
fig = plt.figure(figsize = (12,6))   # 画布

grid = plt.GridSpec(20,27, ) # 子图网格

# 全球恢复力分布
map_rs_kndvi, im1 = maps(grid[0:8, 0:14], my_projn, resis_kndvi_mid,  1, 6, cmap_use='summer_r', cb_label= 'Drought resistance (kNDVI)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)
map_rs_sif, im1 = maps(grid[10:18, 0:14], my_projn, resis_sif_mid,  1, 5, cmap_use='summer_r', cb_label= 'Drought resistance (SIF)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_rs_kndvi = lats( resis_kndvi_mid.where( resis_kndvi_mid<500),    grid[0:7, 14:17],    'green', -60, 60, -0.5, 7.5, 'Drought resistance (kNDVI)',x_ticks=[0,6])
lat_rs_sif = lats( resis_sif_mid.where( resis_sif_mid<350),    grid[10:17, 14:17],    'green', -60, 60, -0.5, 6, 'Drought resistance (SIF)',x_ticks=[0,5])

# 直方图
his_rs_kndvi= small_map(fig, [0.1, 0.72, 0.05, 0.1],   resis_kndvi_mid,  bins_use= [0,1,2,3,4,5,6,7] ,  c1='springgreen',x_ticks=[0,6,3])  
his_rs_sif= small_map(fig, [0.1, 0.265, 0.05, 0.1],   resis_sif_mid,  bins_use= [0,1,2,3,4,5,6,7] ,  c1='springgreen',x_ticks=[0,6,3])  

## 抵抗力和 fsc 多样性的关系
fsc_ax = plt.subplot(grid[0:7, 19:27])
rich_ax = plt.subplot(grid[10:17, 19:27])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI2000'], x = 'fsc', y = 'resistance', hue = 'data_type', width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI': '#03A9F4'}, ax = fsc_ax)
fsc_ax.legend(title="", frameon = None, loc = 'upper right', ncol = 2,  bbox_to_anchor=(0.95, 1.2))
fsc_ax.set_xlabel('Forest structural complexity')
fsc_ax.set_ylabel('Drought resistance')
fsc_ax.set_yticks([0,1,2,3,4,5,6])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI2000'], x = 'richness', y = 'resistance', hue = 'data_type', width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI': '#03A9F4'}, ax = rich_ax)
rich_ax.legend_.remove()
rich_ax.set_xlabel('Tree species richness')
rich_ax.set_ylabel('Drought resistance')
rich_ax.set_yticks([0,1,2,3,4,5,6])

map_rs_kndvi.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_kndvi.set_title('(c)', loc='left',fontsize=11, fontweight='bold')  
fsc_ax.set_title('(e)', loc='left',fontsize=11, fontweight='bold') 
map_rs_sif.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_sif.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 
rich_ax.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 



fig.subplots_adjust(top=0.92, bottom=0.02, right=0.97, left=0.01)

plt.savefig(r'result_figure/figure_use_20260105/global_drought_resistance_kndvi&sif_event.png', dpi = 600)

# %%
fig = plt.figure(figsize = (12,6))   # 画布

grid = plt.GridSpec(20,27, ) # 子图网格

# 全球恢复力分布
map_rs_kndvi, im1 = maps(grid[0:8, 0:14], my_projn, resis_kndvi_mid_after2000,  1, 6, cmap_use='summer_r', cb_label= 'Drought resistance (kNDVI)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)
map_rs_sif, im1 = maps(grid[10:18, 0:14], my_projn, resis_sif_mid,  1, 5, cmap_use='summer_r', cb_label= 'Drought resistance (SIF)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_rs_kndvi = lats( resis_kndvi_mid_after2000.where( resis_kndvi_mid_after2000<500),    grid[0:7, 14:17],    'green', -60, 60, -0.5, 7.5, 'Drought resistance (kNDVI)',x_ticks=[0,6])
lat_rs_sif = lats( resis_sif_mid.where( resis_sif_mid<350),    grid[10:17, 14:17],    'green', -60, 60, -0.5, 6, 'Drought resistance (SIF)',x_ticks=[0,5])

# 直方图
his_rs_kndvi= small_map(fig, [0.1, 0.72, 0.05, 0.1],   resis_kndvi_mid_after2000,  bins_use= [0,1,2,3,4,5,6,7] ,  c1='springgreen',x_ticks=[0,6,3],y_ticks=[0,2500,5000])  
his_rs_sif= small_map(fig, [0.1, 0.265, 0.05, 0.1],   resis_sif_mid,  bins_use= [0,1,2,3,4,5,6,7] ,  c1='springgreen',x_ticks=[0,6,3],y_ticks=[0,2500,5000])  

## 抵抗力和 fsc 多样性的关系
fsc_ax = plt.subplot(grid[0:7, 19:27])
rich_ax = plt.subplot(grid[10:17, 19:27])

box_1 = sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI'], x = 'fsc', y = 'resistance', hue = 'data_type', hue_order=['SIF', 'kNDVI2000'], width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI2000': '#03A9F4'}, ax = fsc_ax)

handles, original_labels = box_1.get_legend_handles_labels()

fsc_ax.legend(title="",handles=handles, labels= ['SIF','kNDVI'], frameon = None, loc = 'upper right', ncol = 2,  bbox_to_anchor=(0.95, 1.2))
fsc_ax.set_xlabel('Forest structural complexity')
fsc_ax.set_ylabel('Drought resistance')
fsc_ax.set_yticks([0,1,2,3,4,5,6])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI'], x = 'richness', y = 'resistance', hue = 'data_type', hue_order=['SIF', 'kNDVI2000'],width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI2000': '#03A9F4'}, ax = rich_ax)
rich_ax.legend_.remove()
rich_ax.set_xlabel('Tree species richness')
rich_ax.set_ylabel('Drought resistance')
rich_ax.set_yticks([0,1,2,3,4,5,6])

map_rs_kndvi.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_kndvi.set_title('(c)', loc='left',fontsize=11, fontweight='bold')  
fsc_ax.set_title('(e)', loc='left',fontsize=11, fontweight='bold') 
map_rs_sif.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_sif.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 
rich_ax.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 



fig.subplots_adjust(top=0.92, bottom=0.02, right=0.97, left=0.01)

plt.savefig(r'result_figure/figure_use_20260105/global_drought_resistance_kndvi_after2000&sif_event.png', dpi = 600)

# %%
fig = plt.figure(figsize = (12,6))   # 画布

grid = plt.GridSpec(20,27, ) # 子图网格

# 全球恢复力分布
map_rs_kndvi, im1 = maps(grid[0:8, 0:14], my_projn, resis_kndvi2_mid,  0.6, 1, cmap_use='summer_r', cb_label= 'Drought resistance (kNDVI)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)
map_rs_sif, im1 = maps(grid[10:18, 0:14], my_projn, resis_sif2_mid,  0.6, 1, cmap_use='summer_r', cb_label= 'Drought resistance (SIF)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_rs_kndvi = lats( resis_kndvi2_mid,    grid[0:7, 14:17],    'green', -60, 60, 0.5, 1.1, 'Drought resistance (kNDVI)',x_ticks=[0.5,1])
lat_rs_sif = lats( resis_sif2_mid,    grid[10:17, 14:17],    'green', -60, 60, 0.5, 1.1, 'Drought resistance (SIF)',x_ticks=[0.5,1])

# 直方图
his_rs_kndvi= small_map(fig, [0.1, 0.72, 0.05, 0.1],   resis_kndvi2_mid,  bins_use= np.arange(0.4,1.1,0.1) ,  c1='springgreen',x_ticks=[0.5,1],y_ticks=[0,5000,10000])  
his_rs_sif= small_map(fig, [0.1, 0.265, 0.05, 0.1],   resis_sif2_mid,  bins_use= np.arange(0.4,1.1,0.1) ,  c1='springgreen',x_ticks=[0.5,1],y_ticks=[0,5000,10000])  

## 抵抗力和 fsc 多样性的关系
fsc_ax = plt.subplot(grid[0:7, 19:27])
rich_ax = plt.subplot(grid[10:17, 19:27])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI2000'], x = 'fsc', y = 'resis_r', hue = 'data_type', width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI': '#03A9F4'}, ax = fsc_ax)
fsc_ax.legend(title="", frameon = None, loc = 'upper right', ncol = 2,  bbox_to_anchor=(0.95, 1.2))
fsc_ax.set_xlabel('Forest structural complexity')
fsc_ax.set_ylim(0.3,1.05)
fsc_ax.set_ylabel('Drought resistance')
fsc_ax.set_yticks([0.3,0.6,0.9])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI2000'], x = 'richness', y = 'resis_r', hue = 'data_type', width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI': '#03A9F4'}, ax = rich_ax)
rich_ax.legend_.remove()
rich_ax.set_xlabel('Tree species richness')
rich_ax.set_ylim(0.3,1.05)
rich_ax.set_ylabel('Drought resistance')
rich_ax.set_yticks([0.3,0.6,0.9])

map_rs_kndvi.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_kndvi.set_title('(c)', loc='left',fontsize=11, fontweight='bold')  
fsc_ax.set_title('(e)', loc='left',fontsize=11, fontweight='bold') 
map_rs_sif.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_sif.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 
rich_ax.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 



fig.subplots_adjust(top=0.92, bottom=0.02, right=0.97, left=0.01)

plt.savefig(r'result_figure/figure_use_20260105/global_drought_resistance2_kndvi&sif_event.png', dpi = 600)

# %%
fig = plt.figure(figsize = (12,6))   # 画布

grid = plt.GridSpec(20,27, ) # 子图网格

# 全球恢复力分布
map_rs_kndvi, im1 = maps(grid[0:8, 0:14], my_projn, resis_kndvi2_mid_after2000,  0.6, 1, cmap_use='summer_r', cb_label= 'Drought resistance (kNDVI)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)
map_rs_sif, im1 = maps(grid[10:18, 0:14], my_projn, resis_sif2_mid,  0.6, 1, cmap_use='summer_r', cb_label= 'Drought resistance (SIF)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_rs_kndvi = lats( resis_kndvi2_mid_after2000,    grid[0:7, 14:17],    'green', -60, 60, 0.5, 1.1, 'Drought resistance (kNDVI)',x_ticks=[0.5,1])
lat_rs_sif = lats( resis_sif2_mid,    grid[10:17, 14:17],    'green', -60, 60, 0.5, 1.1, 'Drought resistance (SIF)',x_ticks=[0.5,1])

# 直方图
his_rs_kndvi= small_map(fig, [0.1, 0.72, 0.05, 0.1],   resis_kndvi2_mid_after2000,  bins_use= np.arange(0.4,1.1,0.1) ,  c1='springgreen',x_ticks=[0.5,1],y_ticks=[0,5000,10000])  
his_rs_sif= small_map(fig, [0.1, 0.265, 0.05, 0.1],   resis_sif2_mid,  bins_use= np.arange(0.4,1.1,0.1) ,  c1='springgreen',x_ticks=[0.5,1],y_ticks=[0,5000,10000])  

## 抵抗力和 fsc 多样性的关系
fsc_ax = plt.subplot(grid[0:7, 19:27])
rich_ax = plt.subplot(grid[10:17, 19:27])

box_1 = sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI'], x = 'fsc', y = 'resis_r', hue = 'data_type', hue_order=['SIF', 'kNDVI2000'], width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI2000': '#03A9F4'}, ax = fsc_ax)

handles, original_labels = box_1.get_legend_handles_labels()

fsc_ax.legend(title="",handles=handles, labels= ['SIF','kNDVI'], frameon = None, loc = 'upper right', ncol = 2,  bbox_to_anchor=(0.95, 1.2))
fsc_ax.set_xlabel('Forest structural complexity')
fsc_ax.set_ylabel('Drought resistance')
fsc_ax.set_ylim(0.28,1.05)
fsc_ax.set_yticks([0.3,0.6,0.9])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI'], x = 'richness', y = 'resis_r', hue = 'data_type', hue_order=['SIF', 'kNDVI2000'],width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI2000': '#03A9F4'}, ax = rich_ax)
rich_ax.legend_.remove()
rich_ax.set_xlabel('Tree species richness')
rich_ax.set_ylabel('Drought resistance')
rich_ax.set_ylim(0.28,1.05)
rich_ax.set_yticks([0.3,0.6,0.9])

map_rs_kndvi.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_kndvi.set_title('(c)', loc='left',fontsize=11, fontweight='bold')  
fsc_ax.set_title('(e)', loc='left',fontsize=11, fontweight='bold') 
map_rs_sif.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_sif.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 
rich_ax.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 


fig.subplots_adjust(top=0.92, bottom=0.02, right=0.97, left=0.01)

plt.savefig(r'result_figure/figure_use_20260105/global_drought_resistance2_kndvi_after2000&sif_event.png', dpi = 600)

# %%
drought_resistance_mid_draw.head()

# %%
fig = plt.figure(figsize = (12,6))   # 画布

grid = plt.GridSpec(20,27, ) # 子图网格

# 全球恢复力分布
map_rs_kndvi, im1 = maps(grid[0:8, 0:14], my_projn, resis_kndvi2_log_mid,  0, 6, cmap_use='summer_r', cb_label= 'Drought resistance (kNDVI)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)
map_rs_sif, im1 = maps(grid[10:18, 0:14], my_projn, resis_sif2_log_mid,  0, 6, cmap_use='summer_r', cb_label= 'Drought resistance (SIF)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_rs_kndvi = lats( resis_kndvi2_log_mid,    grid[0:7, 14:17],    'green', -60, 60, 0, 6.5, 'Drought resistance (kNDVI)',x_ticks=[0,3,6])
lat_rs_sif = lats( resis_sif2_log_mid,    grid[10:17, 14:17],    'green', -60, 60, 0, 6, 'Drought resistance (SIF)',x_ticks=[0,3,6])

# 直方图
his_rs_kndvi= small_map(fig, [0.1, 0.72, 0.05, 0.1],   resis_kndvi2_log_mid,  bins_use= np.arange(-1,6.1,1) ,  c1='springgreen',x_ticks=[0,3,6],y_ticks=[0,2500,5000])  
his_rs_sif= small_map(fig, [0.1, 0.265, 0.05, 0.1],   resis_sif2_log_mid,  bins_use= np.arange(-1,6.1,1) ,  c1='springgreen',x_ticks=[0,3,6],y_ticks=[0,2500,5000])  

## 抵抗力和 fsc 多样性的关系
fsc_ax = plt.subplot(grid[0:7, 19:27])
rich_ax = plt.subplot(grid[10:17, 19:27])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI2000'], x = 'fsc', y = 'resis_r_log', hue = 'data_type', width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI': '#03A9F4'}, ax = fsc_ax)
fsc_ax.legend(title="", frameon = None, loc = 'upper right', ncol = 2,  bbox_to_anchor=(0.95, 1.2))
fsc_ax.set_xlabel('Forest structural complexity')
fsc_ax.set_ylim(-1,7)
fsc_ax.set_ylabel('Drought resistance')
fsc_ax.set_yticks([0,3,6])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI2000'], x = 'richness', y = 'resis_r_log', hue = 'data_type', width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI': '#03A9F4'}, ax = rich_ax)
rich_ax.legend_.remove()
rich_ax.set_xlabel('Tree species richness')
rich_ax.set_ylim(-1.2,6.8)
rich_ax.set_ylabel('Drought resistance')
rich_ax.set_yticks([0,3,6])

map_rs_kndvi.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_kndvi.set_title('(c)', loc='left',fontsize=11, fontweight='bold')  
fsc_ax.set_title('(e)', loc='left',fontsize=11, fontweight='bold') 
map_rs_sif.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_sif.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 
rich_ax.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 



fig.subplots_adjust(top=0.92, bottom=0.02, right=0.97, left=0.01)

plt.savefig(r'result_figure/figure_use_20260105/global_drought_resistance2_log_kndvi&sif_event.png', dpi = 600)

# %%
fig = plt.figure(figsize = (12,6))   # 画布

grid = plt.GridSpec(20,27, ) # 子图网格

# 全球恢复力分布
map_rs_kndvi, im1 = maps(grid[0:8, 0:14], my_projn, resis_kndvi2_log_mid_after2000,  0, 6, cmap_use='summer_r', cb_label= 'Drought resistance (kNDVI)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)
map_rs_sif, im1 = maps(grid[10:18, 0:14], my_projn, resis_sif2_log_mid,  0, 6, cmap_use='summer_r', cb_label= 'Drought resistance (SIF)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_rs_kndvi = lats( resis_kndvi2_log_mid_after2000,    grid[0:7, 14:17],    'green', -60, 60, -1, 6.5, 'Drought resistance (kNDVI)',x_ticks=[0,3,6])
lat_rs_sif = lats( resis_sif2_log_mid,    grid[10:17, 14:17],    'green', -60, 60, -1, 6, 'Drought resistance (SIF)',x_ticks=[0,3,6])

# 直方图
his_rs_kndvi= small_map(fig, [0.1, 0.72, 0.05, 0.1],   resis_kndvi2_log_mid_after2000,  bins_use= np.arange(-1,6.1, 1) ,  c1='springgreen',x_ticks=[0,3,6],y_ticks=[0,2500,5000])  
his_rs_sif= small_map(fig, [0.1, 0.265, 0.05, 0.1],   resis_sif2_log_mid,  bins_use= np.arange(-1,6.1, 1) ,  c1='springgreen',x_ticks=[0,3,6],y_ticks=[0,2500,5000])  

## 抵抗力和 fsc 多样性的关系
fsc_ax = plt.subplot(grid[0:7, 19:27])
rich_ax = plt.subplot(grid[10:17, 19:27])

box_1 = sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI'], x = 'fsc', y = 'resis_r_log', hue = 'data_type', hue_order=['SIF', 'kNDVI2000'], width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI2000': '#03A9F4'}, ax = fsc_ax)

handles, original_labels = box_1.get_legend_handles_labels()

fsc_ax.legend(title="",handles=handles, labels= ['SIF','kNDVI'], frameon = None, loc = 'upper right', ncol = 2,  bbox_to_anchor=(0.95, 1.2))
fsc_ax.set_xlabel('Forest structural complexity')
fsc_ax.set_ylabel('Drought resistance')
fsc_ax.set_ylim(-1.2,6.8)
fsc_ax.set_yticks([0,3,6])

sns.boxplot(drought_resistance_mid_draw[drought_resistance_mid_draw['data_type'] != 'kNDVI'], x = 'richness', y = 'resis_r_log', hue = 'data_type', hue_order=['SIF', 'kNDVI2000'],width = 0.5, gap = 0.2, 
            flierprops={"marker": "."},palette={'SIF': '#4CAF50', 'kNDVI2000': '#03A9F4'}, ax = rich_ax)
rich_ax.legend_.remove()
rich_ax.set_xlabel('Tree species richness')
rich_ax.set_ylabel('Drought resistance')
rich_ax.set_ylim(-1.5,6.5)
rich_ax.set_yticks([0,3,6])

map_rs_kndvi.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_kndvi.set_title('(c)', loc='left',fontsize=11, fontweight='bold')  
fsc_ax.set_title('(e)', loc='left',fontsize=11, fontweight='bold') 
map_rs_sif.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
lat_rs_sif.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 
rich_ax.set_title('(f)', loc='left',fontsize=11, fontweight='bold') 


fig.subplots_adjust(top=0.92, bottom=0.02, right=0.97, left=0.01)

plt.savefig(r'result_figure/figure_use_20260105/global_drought_resistance2_log_kndvi_after2000&sif_event.png', dpi = 600)

# %% [markdown]
# ## 3 干旱阈值 

# %%
with xr.open_dataset(r'result_data/spei_nt_annual_threshold.nc')['spei'] as data:
    spei_nt_threshold = data

with xr.open_dataset(r'result_data/spei_sh_annual_threshold.nc') as data:
    spei_sh_threshold = data

# %%
spei_nt_threshold

# %%
spei_sh_threshold

# %%
spei_sh_threshold = spei_sh_threshold['__xarray_dataarray_variable__']
spei_sh_threshold.name = 'spei'
spei_sh_threshold

# %%
spei_threshold = xr.concat([spei_sh_threshold,spei_nt_threshold],dim='lat')
spei_threshold = spei_threshold.sortby('lat')
spei_threshold

# %%
spei_threshold = spei_threshold.interp_like(ld_mask)
spei_threshold.plot()

# %%
spei_threshold = spei_threshold.where(ld_mask > 0)
spei_threshold.plot()

# %%
spei_threshold.mean()

# %%
spei_threshold.median()

# %%
spei_threshold.plot.hist()

# %%
fig = plt.figure(figsize = (10,4))   # 画布
grid = plt.GridSpec(5,18, ) # 子图网格

# 全球恢复力分布
map_th, im1 = maps(grid[0:5, 0:14], my_projn, spei_threshold,  -1.5, 0, cmap_use='inferno_r', cb_label= 'Drought threshold',
                    cb_extend='min', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_th = lats( spei_threshold,    grid[0:4, 15:18],    'red', -60, 60, -1.5, 0, 'Drought threshold', x_ticks=[-1,0])

# 直方图
his_th= small_map(fig, [0.13, 0.4, 0.12, 0.15],   spei_threshold,  bins_use= [-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0] ,  c1='orange', x_ticks=[-2,-1,0])  

map_th.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_th.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  


fig.subplots_adjust(top=0.9, bottom=0.1, right=0.95,left=0.05)

plt.savefig(r'result_fig_new/global_drought_threshold.png', dpi = 600)

# %% [markdown]
# ## 4 干旱特征变化趋势

# %%
with xr.open_dataset(r'result_data/drought_count_trend.nc') as data:
    drought_count_trend = data.where(ld_mask > 0)

with xr.open_dataset(r'result_data/drought_dura_trend.nc') as data:
    drought_dura_trend = data.where(ld_mask > 0)

with xr.open_dataset(r'result_data/drought_sev_trend.nc') as data:
    drought_sev_trend = data.where(ld_mask > 0)

# %%
drought_dura_trend['trend'].plot.hist(bins = [-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5])

# %%
drought_dura_trend['trend'].where(drought_dura_trend['trend'] < 0 ).plot.hist(bins = [-0.5,-0.4, -0.3,-0.2,-0.1,0])

# %%
drought_dura_trend['trend'].where(drought_dura_trend['trend'] > 0 ).plot.hist(bins = np.arange(0,0.55, 0.1))

# %%
(~np.isnan(drought_dura_trend['trend'].where(drought_dura_trend['trend'] > 0 ).values.flatten() )).sum() / (~np.isnan(drought_dura_trend['trend'].where(drought_dura_trend['trend'] < 0 ).values.flatten() )).sum()

# %%
drought_count_trend['trend'].plot.hist(bins = np.arange(-1.2,1.3,0.2))

# %%
drought_sev_trend['trend'].plot.hist(bins = np.arange(-0.04,0.05,0.01))

# %%
fig = plt.figure(figsize = (6,8))   # 画布
grid = plt.GridSpec(6,6, ) # 子图网格

# 全球恢复力分布
map_ds, im1 = maps(grid[0:2, 0:6], my_projn, drought_sev_trend['trend'],    -0.02, 0.02, cmap_use='BrBG', cb_label= 'Severity trend', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dc, im1 = maps(grid[2:4, 0:6], my_projn, drought_count_trend['trend'],    -1, 1, cmap_use='BrBG_r', cb_label= 'Count trend', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_dd, im1 = maps(grid[4:6, 0:6], my_projn, drought_dura_trend['trend'],    -0.3, 0.3, cmap_use='BrBG_r', cb_label= 'Duration trend', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)

map_ds.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
map_dc.set_title('(b)', loc='left', fontsize=11, fontweight='bold')
map_dd.set_title('(c)', loc='left', fontsize=11, fontweight='bold')  


fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05)

plt.savefig(r'result_fig/global_drought_char_trend.png', dpi = 600)

# %% [markdown]
# ## 5 结构复杂度 物种多样性  和 landcover

# %%
with xr.open_dataset(r'D:/data/fsc_from_su/data/global_forest_csc/global_forest_csc.tif')  as  data:
    fsc = data['band_data'][0].drop(['spatial_ref','band'])
    fsc = fsc.rename({'x':'lon','y':'lat'})
    fsc = fsc.coarsen(lat = 20, lon=20).mean()
    fsc = fsc.sel(lat = slice(52,-52))
fsc

# %%
fsc.plot()

# %%
with xr.open_dataset(r'result_data/plant_richness_log_05.nc') as data:
    plant_richness = data['richness']
plant_richness

# %%
plant_richness = plant_richness.interp_like(fsc)

# %%
fig = plt.figure(figsize = (8,6))   # 画布
grid = plt.GridSpec(4,8, ) # 子图网格

# 全球恢复力分布
map_rich, im = maps(grid[0:2, 0:8], my_projn, plant_richness,    0.5, 5, cmap_use='viridis', cb_label= 'Species richness (log)', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)
map_fsc, im = maps(grid[2:4, 0:8], my_projn, fsc,    9, 11, cmap_use='viridis', cb_label= 'Forest structural complexity', cb_extend='both', cb_or='vertical', cb_shrink=1, cb_pad=0.1)

#lat_rich = lats(plant_richness, grid[0:2, 8:10],    'green', -60, 90, 0, 6, 'Species richness (log)')
#lat_fsc = lats(fsc,   grid[2:4, 8:10],    'green', -60, 90, 0, 12, 'Forest structural complexity')

map_rich.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
map_fsc.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
#lat_rich.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  
#lat_fsc.set_title('(d)', loc='left',fontsize=11, fontweight='bold') 

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.05)

plt.savefig(r'result_fig_new/global_richness_fsc.png', dpi = 600)

# %%
with xr.open_dataset(r'result_data/landcover_005_use.nc') as data:    ## 这里用的landcover和其他数据分析里用的不一样
    ld = data['modis_landcover'].interp_like(fsc, method='nearest')
ld

# %%
veget_cmap = ['olivedrab','darkolivegreen','forestgreen','limegreen','lightgreen','orange',
              'moccasin','sienna','peru','yellow','dodgerblue']

# %%
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

plt.savefig(r'result_fig_new/global_landcover.png', dpi = 600)

# %% [markdown]
# #### 换数据画一下  landcover

# %%
with xr.open_dataset(r'D:/data/modis_landcover/landcover_gimms_mask.nc').modis_landcover as data:
    ld_gimms = xr.DataArray(data.values, coords=[data.lat, data.lon] , dims=['lat','lon'])
ld_gimms

# %%
ld_gimms = ld_gimms.interp_like(plant_richness, method='nearest')

# %%
fig = plt.figure(figsize=[10,4])
ax = plt.subplot(projection=my_projn)

map_ld = (ld_gimms+0.5).plot.imshow(ax = ax, transform = ccrs.PlateCarree(), colors = veget_cmap, levels = 11 , add_colorbar =False)
ax.set_extent([-185,185 ,-52, 52], crs=ccrs.PlateCarree())
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

plt.savefig(r'result_fig_new/global_landcover_new.png', dpi = 600)

# %% [markdown]
# ## 6 用双变量图 画结构复杂度和树种多样性

# %%
fsc_richness = xr.Dataset({'fsc':fsc, 'richness':plant_richness})

# %%
fsc_richness_df = fsc_richness.to_dataframe()
fsc_richness_df.head() 

# %%
fsc_richness_df = fsc_richness_df.dropna(how='any')
fsc_richness_df = fsc_richness_df.reset_index()
fsc_richness_df.head()

# %%
import mapclassify

# %%
fsc_bin = mapclassify.Quantiles(y=fsc_richness_df['fsc'],k=10)
fsc_bin

# %%
richness_bin = mapclassify.Quantiles(y=fsc_richness_df['richness'],k=10)
richness_bin

# %%
fsc_richness_df.loc[:,'fsc_Class'] = fsc_bin.yb
fsc_richness_df.loc[:,'richness_Class'] = richness_bin.yb
fsc_richness_df.head()

# %% [markdown]
# #### 6.1  先试试 代码

# %%
from matplotlib.colors import rgb2hex,hex2color

# %%
#对数据进行处理
cp1 = np.linspace(0,1,10)
cp2 = np.linspace(0,1,10)

Cp1, Cp2 = np.meshgrid(cp1,cp2)
C0 = np.zeros_like(Cp1)
Legend = np.dstack((Cp1,Cp2, C0+0.5))
plt.imshow(Legend, origin="lower", extent=[0,1,0,1])
plt.show()

# %%
C0

# %%
Legend

# %%
Legend.shape

# %%
bivariate_palette = {}
n_categories =10
for j in range(n_categories):
    for i in range(n_categories):
        x = Legend[i][j]
        bivariate_palette[(i, j)] = rgb2hex(x)

# %%
rgb2hex(x)

# %%
bivariate_palette

# %%
fsc_richness_df.loc[:,"color"] = [bivariate_palette[(j, i)] for i,j in \
            zip(fsc_richness_df["fsc_Class"],fsc_richness_df["richness_Class"])]
fsc_richness_df.head()

# %%
fig = plt.figure(figsize=(12,6),facecolor="w")
ax = fig.add_subplot(projection=my_projn)
ax.coastlines(linewidth=0.5,zorder=20)
ax.add_feature(cfeature.LAND, facecolor='gainsboro')
gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5)

map_fsc_rich = ax.scatter( x=fsc_richness_df['lon'],y=fsc_richness_df['lat'], c=fsc_richness_df['color'] , transform = ccrs.PlateCarree(), s=0.1)

# %%
fig = plt.figure(figsize=(12,6),facecolor="w")
ax = fig.add_subplot(projection=my_projn)
ax.coastlines(linewidth=0.5,zorder=20)
ax.add_feature(cfeature.LAND, facecolor='gainsboro')

map_fsc_rich = ax.scatter( x=fsc_richness_df['lon'],y=fsc_richness_df['lat'], c=fsc_richness_df['color'] , transform = ccrs.PlateCarree(), s=1.5, edgecolor='none')
ax.scatter( x=-180,y=0, c='white' , transform = ccrs.PlateCarree(), s=0.1)  ## 加一个点 让地图完整

gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=1,alpha=.5)

#添加图例
cax = fig.add_axes([0.05,0.2,0.25,0.25])
#对数据进行处理

cax.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) # draw arrow for x 
cax.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1)) # draw arrow for 
cax.axis("off")
cax.imshow(Legend, origin="lower", extent=[0,1,0,1])
cax.text(s='Species richness (log)', x=0, y=-0.18,size=12) 
cax.text(s='Structural complexity', x=-0.18, y=0, size=12,rotation=90)

plt.tight_layout()
fig.savefig(r'result_fig_new/fsc_richness_bi_try.png', dpi=600)

# %% [markdown]
# #### 6.2 调整颜色

# %%
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

# %%
bl = "#FFB7B8"
br = "#7F00FF" 
tl = "#80FF00"
tr = "#004847"
nx, ny = 10, 10

# %%
colors_100 = interpolate_color_from_4corners(bl, br, tl, tr, nx, ny)

# %%
colors_100

# %%
plt.imshow(colors_100, origin='lower')

# %%


# %% [markdown]
# ## 7 试着画一个双色的散点分布图

# %%
import pandas as pd

# %%
## 简单生成几个点
exam_pd = pd.DataFrame( {'lat':  30 + np.random.rand(10)* 60,
                         'lon':  60 + np.random.rand(10)* 60,
                         'cor_1':np.random.rand(10),
                         'cor_2':np.random.rand(10)  })
exam_pd

# %%
from matplotlib.markers import MarkerStyle

# %%
fig = plt.figure(figsize=(12,6),facecolor="w")
ax = fig.add_subplot(projection=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5,zorder=20)
ax.add_feature(cfeature.LAND, facecolor='gainsboro')
ax.set_extent([28, 122, 28, 90], crs=ccrs.PlateCarree())

ax.scatter( x=exam_pd['lon'],y=exam_pd['lat'], c=exam_pd['cor_1'] , marker=MarkerStyle('o', 'left'), cmap='PiYG',
           vmax=1, vmin=0,transform = ccrs.PlateCarree(), s=250, edgecolor='none')
im = ax.scatter( x=exam_pd['lon'],y=exam_pd['lat'], c=exam_pd['cor_2'] , marker=MarkerStyle('o', 'right'), cmap='PiYG',
           vmax=1, vmin=0,transform = ccrs.PlateCarree(), s=250, edgecolor='none')


gls = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), 
                   color='k', linestyle='dashed', linewidth=0.3, 
                   y_inline=False,zorder=10,alpha=.5)

plt.colorbar(im)

# %% [markdown]
# ## 8 土壤水分变化率地图

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/sm_nt_change_kndvi.nc')  as data:
    sm_nt_change = data['sm_change']
    sm_nt_change_mid = sm_nt_change.median('year')

with xr.open_dataset(r'E:/python_output/fsc_drought/sm_sh_change_kndvi.nc')  as data:
    sm_sh_change = data['sm_change']
    sm_sh_change_mid = sm_sh_change.median('year')

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_nt_change_kndvi.nc')  as data:
    smrz_nt_change = data['sm_change']
    smrz_nt_change_mid = smrz_nt_change.median('year')

with xr.open_dataset(r'E:/python_output/fsc_drought/smrz_sh_change_kndvi.nc')  as data:
    smrz_sh_change = data['sm_change']
    smrz_sh_change_mid = smrz_sh_change.median('year')

# %%
sm_change_mid = xr.concat([sm_nt_change_mid, sm_sh_change_mid], dim='lat')
sm_change_mid = sm_change_mid.sortby('lat')
sm_change_mid

# %%
sm_change_mid = sm_change_mid.interp_like(fsc_ex)
sm_change_mid.plot()

# %%
sm_change_mid = sm_change_mid.where(fsc_ex > 0)
sm_change_mid.plot()

# %%
sm_change_mid = np.log(sm_change_mid)

# %%
fig = plt.figure(figsize = (10,4))   # 画布
grid = plt.GridSpec(5,18, ) # 子图网格

# 全球恢复力分布
map_sm, im1 = maps(grid[0:5, 0:14], my_projn, sm_change_mid,  1, 5, cmap_use='BrBG', cb_label= 'Soil moisture change (log)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_sm = lats( sm_change_mid.where(sm_change_mid<10),    grid[0:4, 15:18],    'lightseagreen', -60, 60, 2, 6, 'Soil moisture\n' + 'change (log)', x_ticks=[2,5])

# 直方图
his_sm= small_map(fig, [0.15, 0.4, 0.1, 0.15],   sm_change_mid,  bins_use= [0,1,2,3,4,5,6] ,  c1='aquamarine',x_ticks=[1,5], y_ticks=[0,1000,2000,3000])  

map_sm.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_sm.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.95,left=0.05)

plt.savefig(r'result_figure/figure_use/global_sm_change_mid.png', dpi = 600)

# %%
smrz_change_mid = xr.concat([smrz_nt_change_mid, smrz_sh_change_mid], dim='lat')
smrz_change_mid = smrz_change_mid.sortby('lat')
smrz_change_mid

# %%
smrz_change_mid = smrz_change_mid.interp_like(fsc_ex)
smrz_change_mid.plot()

# %%
smrz_change_mid = smrz_change_mid.where(fsc_ex > 0)
smrz_change_mid.plot()

# %%
smrz_change_mid = np.log(smrz_change_mid)

# %%
fig = plt.figure(figsize = (10,4))   # 画布
grid = plt.GridSpec(5,18, ) # 子图网格

# 全球恢复力分布
map_sm, im1 = maps(grid[0:5, 0:14], my_projn, smrz_change_mid,  1, 5, cmap_use='BrBG', cb_label= 'Soil moisture change (log)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_sm = lats( smrz_change_mid.where(smrz_change_mid<10),    grid[0:4, 15:18],    'lightseagreen', -60, 60, 1, 6, 'Soil moisture\n' + 'change (log)', x_ticks=[1,5])

# 直方图
his_sm= small_map(fig, [0.15, 0.4, 0.1, 0.15],   sm_change_mid,  bins_use= [0,1,2,3,4,5,6] ,  c1='aquamarine',x_ticks=[1,5], y_ticks=[0,1000,2000,3000])  

map_sm.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_sm.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.95,left=0.05)

plt.savefig(r'result_figure/figure_use/global_smrz_change_mid.png', dpi = 600)


# %% [markdown]
# ## 9 土壤-大气温度差 和 fsc的相关  空间地图

# %%
## 数据
with xr.open_dataset(r'result_data/cor_soil_air_temp_offset_and_csc.nc')['cor'] as data:
    cor_result_all = data.sel(lat = slice(52,-52))
cor_result_all

# %%
with  xr.open_dataset(r'result_data/pvalue_soil_air_temp_offset_and_csc.nc')['p_value']  as data:
    p_result_all = data.sel(lat = slice(52,-52))
p_result_all

# %%
my_projn = ccrs.Mollweide(central_longitude=0)

# %%
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# %%
fig, axes = plt.subplots(4,3,figsize = (12,9), subplot_kw={'projection':my_projn})   # 画布

for i in range(12):
    axes[i//3,i%3].axis('off')
    plot_data = cor_result_all.sel(month= (i+1))
    axes[i//3,i%3].set_extent([-180, 180, -52, 52], crs=ccrs.PlateCarree())
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

plt.savefig('result_fig_new/cor_map_fsc_soil_air_temp.png', dpi = 600)

# %% [markdown]
# ### 9.2 生长季的温度差 地图

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/temp_offset_005.nc') as data:
    temp_offset = data['temp_offset']
temp_offset

# %%
temp_offset.plot(vmin = -1, vmax = 1)

# %%
temp_offset_05 = temp_offset.interp_like(fsc_ex)
temp_offset_05.plot(vmin = -1, vmax = 1)

# %%
temp_offset_05.where(fsc_ex > 0).plot(vmin = -1, vmax = 1)

# %%
temp_offset_05 = temp_offset_05.where(fsc_ex > 0) * 0.1

# %%
fig = plt.figure(figsize = (10,4))   # 画布
grid = plt.GridSpec(5,18, ) # 子图网格

map_to, im1 = maps(grid[0:5, 0:14], my_projn, temp_offset_05,  -2, 2, cmap_use='coolwarm', cb_label= 'Temperature offset (℃)',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_to = lats( temp_offset_05,    grid[0:4, 15:18],    'blue', -60, 60, -4.5,4.5, 'Temperature offset (℃)', x_ticks=[-4,0,4])

# 直方图
his_th= small_map(fig, [0.13, 0.4, 0.12, 0.15],   temp_offset_05,  bins_use= np.arange(-4,4.1,0.5) ,  c1='blue', x_ticks=[-4,0,4])  
his_th.set_ylim(0, 2000)
his_th.set_yticks([0,1000,2000])

map_to.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_to.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  


fig.subplots_adjust(top=0.9, bottom=0.1, right=0.95,left=0.05)

plt.savefig(r'result_figure/figure_use/global_temperature_offset.png', dpi = 600)

# %% [markdown]
# ## 10 biome

# %%
import rioxarray

# %%
with rioxarray.open_rasterio(r'D:/data/official_teow/biome.tif')  as data:
    biome = data
biome = biome.where(biome>0)
biome = biome.where(biome<90)

biome = xr.DataArray(biome[0].values, coords=[biome.y, biome.x], dims=['lat','lon'])
biome

# %%
biome_list = ['Trop.&Subtrop. Moist Broad. Forests', 'Trop.&Subtrop. Dry Broad. Forests', 'Trop.&Subtrop. Coni. Forests', 'Temp. Broad.&Mixed Forests',
          'Temp. Coni. Forests','Boreal Forests',
          'Trop.&Subtrop. Gra.,Sav.&Shrub.','Temp. Gra.,Sav.&Shrub.','Flooded Gra.&Sav.',
          'Montane Gra.&Shrub.','Tundra','Mediter. Forests, Woodlands & Scrub.','Deserts & Xeric Shrublands','Mangroves']

# %%
biome_cmap = ['#aafe00','#00a985','#9fd6c3','#38a900','#98e601','#267200','#ffab00',
              '#ffff72','#015be7','#a97000','#bee8fe','#cdf67a','#ffebbf','#a900e7']

# %%
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

plt.savefig(r'result_fig_new/global_biomes.png', dpi = 600)

# %% [markdown]
# ## 11 lst 变化的地图

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_nt_zs_kndvi.nc') as data:
    lst_zs_nt = data['lst_zs']
    lst_zs_nt_mid = lst_zs_nt.median(dim='year')
with xr.open_dataset(r'E:/python_output/fsc_drought/lst_sh_zs_kndvi.nc') as data:
    lst_zs_sh = data['lst_zs']
    lst_zs_sh_mid = lst_zs_sh.median(dim='year')

# %%
lst_zs_mid = xr.concat([lst_zs_nt_mid, lst_zs_sh_mid], dim='lat')
lst_zs_mid = lst_zs_mid.sortby('lat')
lst_zs_mid

# %%
lst_zs_mid = lst_zs_mid.interp_like(fsc_ex)
lst_zs_mid.plot()

# %%
lst_zs_mid = lst_zs_mid.where(fsc_ex > 0)
lst_zs_mid.plot()

# %%
fig = plt.figure(figsize = (10,4))   # 画布
grid = plt.GridSpec(5,18, ) # 子图网格

# 全球恢复力分布
map_sm, im1 = maps(grid[0:5, 0:14], my_projn, lst_zs_mid,  -4, 4, cmap_use='coolwarm', cb_label= 'Land surface temperature change',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_sm = lats( lst_zs_mid.where(lst_zs_mid<10),    grid[0:4, 15:18],    'tomato', -60, 60, -1, 3.5, 'Land surface temperature\n' + 'change', x_ticks=[0,2])

# 直方图
his_sm= small_map(fig, [0.15, 0.4, 0.1, 0.15],   lst_zs_mid,  bins_use= [-2,-1,0,1,2,3,4] ,  c1='tomato',x_ticks=[0,5], y_ticks=[0,2000,4000])  

map_sm.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_sm.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.95,left=0.05)

plt.savefig(r'result_figure/figure_use/global_lst_zs_mid.png', dpi = 600)

# %% [markdown]
# ## 12 et 变化的地图

# %%
with xr.open_dataset(r'E:/python_output/fsc_drought/et_nt_change2_kndvi.nc') as data:
    et_change_nt = data['et_change2']
    et_change_nt_mid = et_change_nt.median(dim='year')
with xr.open_dataset(r'E:/python_output/fsc_drought/et_sh_change2_kndvi.nc') as data:
    et_change_sh = data['et_change2']
    et_change_sh_mid = et_change_sh.median(dim='year')

# %%
et_change_mid = xr.concat([et_change_nt_mid, et_change_sh_mid], dim='lat')
et_change_mid = et_change_mid.sortby('lat')
et_change_mid

# %%
et_change_mid = et_change_mid.interp_like(fsc_ex)
et_change_mid.plot()

# %%
fig = plt.figure(figsize = (10,4))   # 画布
grid = plt.GridSpec(5,18, ) # 子图网格

# 全球恢复力分布
map_sm, im1 = maps(grid[0:5, 0:14], my_projn, et_change_mid,  -0.4, 0.4, cmap_use='coolwarm', cb_label= 'Transpiration change',
                    cb_extend='both', cb_or='horizontal', cb_shrink = 0.6, cb_pad= 0.05)

# 纬向平均图
lat_sm = lats( et_change_mid,    grid[0:4, 15:18],    'dodgerblue', -60, 60, -0.5, 0.4, 'Transpiration change', x_ticks=[-0.3,0,0.3])

# 直方图
his_sm= small_map(fig, [0.15, 0.4, 0.1, 0.15],   et_change_mid,  bins_use= [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3] ,  c1='dodgerblue',x_ticks=[-0.3,0,0.3], y_ticks=[0,3000,6000])  

map_sm.set_title('(a)', loc='left',fontsize=11, fontweight='bold')  
lat_sm.set_title('(b)', loc='left',fontsize=11, fontweight='bold')  

fig.tight_layout()
fig.subplots_adjust(top=0.9, bottom=0.1, right=0.95,left=0.05)

plt.savefig(r'result_figure/figure_use/global_et_change_mid.png', dpi = 600)

# %% [markdown]
# ## 13 合并偏相关的结果

# %% [markdown]
# ### 13.1  读取所有的数据

# %%
bi_pd_pcor_all_kndvi_mid = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_kndvi_mid.csv')
bi_pd_pval_all_kndvi_mid = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pval_all_kndvi_mid.csv')

bi_pd_pcor_all_sif_mid = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pcor_all_sif_mid.csv')
bi_pd_pval_all_sif_mid = pd.read_csv('E:/python_output/fsc_drought/bi_pd_pval_all_sif_mid.csv')

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
draw_col = ['rich','rich_prec','rich_fsc','fsc','fsc_prec','fsc_rich']
fig, axes = plt.subplots(1, 2, figsize=(16,7))

im = axes[0].imshow(bi_pd_pcor_all_kndvi_mid[draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[0].set_xticks(ticks=np.arange(6))
axes[0].set_yticks(ticks=np.arange(10))
axes[0].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[0].set_yticklabels([biome_short_dic[bi_n] for bi_n in bi_pd_pcor_all_kndvi_mid.biome])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if bi_pd_pval_all_kndvi_mid[draw_col].iloc[j,i] < 0.001:
            axes[0].text(i,j, '***', ha='center', va = 'center')
        elif bi_pd_pval_all_kndvi_mid[draw_col].iloc[j,i] < 0.01:
            axes[0].text(i,j, '**', ha='center', va = 'center')
        elif bi_pd_pval_all_kndvi_mid[draw_col].iloc[j,i] < 0.05:
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
axes[1].imshow(bi_pd_pcor_all_sif_mid[draw_col], vmin=-0.5, vmax=0.5, cmap='PiYG_r', aspect=0.6)
axes[1].set_xticks(ticks=np.arange(6))
axes[1].set_yticks(ticks=np.arange(10))
axes[1].set_xticklabels(['[ ]','PREC','FSC','[ ]','PREC','TSR'])
axes[1].set_yticklabels([])

for i in [5,4,3,2,1,0]:
    for j in range(10):
        if bi_pd_pval_all_sif_mid[draw_col].iloc[j,i] < 0.001:
            axes[1].text(i,j, '***', ha='center', va = 'center')
        elif bi_pd_pval_all_sif_mid[draw_col].iloc[j,i] < 0.01:
            axes[1].text(i,j, '**', ha='center', va = 'center')
        elif bi_pd_pval_all_sif_mid[draw_col].iloc[j,i] < 0.05:
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

plt.savefig('result_figure/figure_use/pcor_richness_fsc_prec_kndvi&sif_csc_mid.png', dpi = 600)


# %% [markdown]
# ## 14 干旱抵抗力 hexplot

# %%
drou_mid_use_kndvi['resistance_log'] = np.log(drou_mid_use_kndvi.resistance_kndvi)
drou_mid_use_sif['resistance_log'] = np.log(drou_mid_use_sif.resistance_sif)

# %%
##  kndvi 和 sif
fig, axes = plt.subplots(2,3, figsize=(12,8))

axes[0,0].hexbin(drou_mid_use_kndvi.richness, drou_mid_use_kndvi.fsc, 
                C= drou_mid_use_kndvi.resistance_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('Forest structural complexity')
axes[0,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,0].set_title('(a)', loc='left', size = 14)

axes[0,1].hexbin(np.log(drou_mid_use_kndvi.prec), drou_mid_use_kndvi.richness, 
                C= drou_mid_use_kndvi.resistance_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,1].set_title('(b)', loc='left', size = 14)

im = axes[0,2].hexbin(np.log(drou_mid_use_kndvi.prec), drou_mid_use_kndvi.fsc, 
                C= drou_mid_use_kndvi.resistance_log, gridsize = 100, 
                vmax=5, vmin=2,
                reduce_C_function = np.median)
axes[0,2].set_xlabel('')
axes[0,2].set_ylabel('Forest structural complexity')
#axes[2].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[0,2].set_title('(c)', loc='left', size = 14)

position1=fig.add_axes([0.9,0.57,0.01,0.3])
plt.colorbar(im, position1, extend='both', label = 'Resistance (kNDVI)',orientation='vertical',ticks=[2,3,4,5])

axes[1,0].hexbin(drou_mid_use_sif.richness, drou_mid_use_sif.fsc, 
                C= drou_mid_use_sif.resistance_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,0].set_xlabel('Species richness')
axes[1,0].set_ylabel('Forest structural complexity')
axes[1,0].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,0].set_title('(d)', loc='left', size = 14)

axes[1,1].hexbin(np.log(drou_mid_use_sif.prec), drou_mid_use_sif.richness, 
                C= drou_mid_use_sif.resistance_log, gridsize = 100, 
                vmax=4, vmin=2,
                reduce_C_function = np.median)
axes[1,1].set_xlabel('Annual precipitation (log) / mm')
axes[1,1].set_ylabel('Species richness')
#axes[1].set_xticks(np.arange(6))
#axes[0].grid(c = 'lightgray', alpha = 0.3)
axes[1,1].set_title('(e)', loc='left', size = 14)

im = axes[1,2].hexbin(np.log(drou_mid_use_sif.prec), drou_mid_use_sif.fsc, 
                C= drou_mid_use_sif.resistance_log, gridsize = 100, 
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

plt.savefig('result_figure/figure_use/resistance_richness_fsc_prec_kndvi&sif_csc_mid.png', dpi = 600)


# %%


# %% [markdown]
# ## change log
# 1. 2025.02.20  画了 干旱特征，抵抗力， 多样性， 结构复杂度 和 landcover的图
# 2. 2025.02.24  画了 ndvi和vod的干旱抵抗力地图  补充了抵抗力中位数地图

# %%



