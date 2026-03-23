import semopy as sem
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
import rioxarray


# %%
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_nt_resistance.nc') as data:
    kndvi_nt_resistance = data['kndvi_resistance']
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_sh_resistance.nc') as data:
    kndvi_sh_resistance = data['kndvi_resistance']

# %%
kndvi_nt_resistance

# %%
kndvi_sh_resistance

# %%
kndvi_nt_resistance[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_nt_resistance2.nc') as data:
    kndvi_nt_resistance2 = data['kndvi_resistance']
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_sh_resistance2.nc') as data:
    kndvi_sh_resistance2 = data['kndvi_resistance']

# %%
kndvi_nt_resistance2

# %%
kndvi_sh_resistance2

# %%
kndvi_nt_resistance2[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')


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
spei_nt_drought_use_kndvi[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
spei_sh_drought_use_kndvi[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')


# %%
era_temp_path = glob.glob(r'./data/era5_land_annual/era5_land_mean*.nc')
era_prec_path = glob.glob(r'./data/era5_land_annual/era5_land_pre*.nc')

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


with rioxarray.open_rasterio(r'./data/Global-AI_ET0_annual_v3/Global-AI_ET0_v3_annual/ai_v3_yr.tif')  as data:
        ai_index= xr.DataArray(data.values[0], coords=[data.y, data.x], dims=['lat','lon'])
ai_index = ai_index.coarsen(lat=10,lon=10).mean()
ai_index

# %%
ai_index = ai_index * 0.0001
ai_index.where(ai_index<1.5).plot()


with xr.open_dataset(r'../result_data/plant_richness_log_05.nc') as data:
    plant_richness = data['richness']
plant_richness

plant_richness.plot()

with xr.open_dataset(r'./data/global_forest_csc.tif')  as  data:
    fsc = data['band_data'][0].drop(['spatial_ref','band'])
fsc = fsc.rename({'x':'lon','y':'lat'})
fsc = fsc.coarsen(lat = 20, lon=20).mean()
fsc


fsc = fsc.interp_like(plant_richness, method='nearest')
fsc.plot()


with xr.open_dataset(r'./python_output/fsc_drought/smrz_nt_change_kndvi.nc') as data:
    smrz_nt_change_kndvi = data['sm_change']
with xr.open_dataset(r'./python_output/fsc_drought/smrz_sh_change_kndvi.nc') as data:
    smrz_sh_change_kndvi = data['sm_change']

# %%
smrz_nt_change_kndvi

# %%
smrz_nt_change_kndvi[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
smrz_sh_change_kndvi

# %%
smrz_sh_change_kndvi[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

with xr.open_dataset(r'./python_output/fsc_drought/et_nt_change_kndvi.nc') as data:
    et_nt_change_kndvi = data['et_change']
with xr.open_dataset(r'./python_output/fsc_drought/et_sh_change_kndvi.nc') as data:
    et_sh_change_kndvi = data['et_change']

# %%
et_nt_change_kndvi

# %%
et_nt_change_kndvi[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -5, vmax = 5)

# %%
et_sh_change_kndvi

# %%
et_sh_change_kndvi[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -5, vmax = 5)


with xr.open_dataset(r'./python_output/fsc_drought/et_nt_change2_kndvi.nc') as data:
    et_nt_change2_kndvi = data['et_change2']
with xr.open_dataset(r'./python_output/fsc_drought/et_sh_change2_kndvi.nc') as data:
    et_sh_change2_kndvi = data['et_change2']

et_nt_change2_kndvi


et_nt_change2_kndvi[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)

et_sh_change2_kndvi


et_sh_change2_kndvi[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)


with xr.open_dataset(r'./python_output/fsc_drought/lst_nt_change_kndvi.nc') as data:
    lst_nt_change_kndvi = data['lst_change']
with xr.open_dataset(r'./python_output/fsc_drought/lst_sh_change_kndvi.nc') as data:
    lst_sh_change_kndvi = data['lst_change']

# %%
lst_nt_change_kndvi

# %%
lst_nt_change_kndvi[0:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)

# %%
lst_sh_change_kndvi

# %%
lst_sh_change_kndvi[0:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)


with xr.open_dataset(r'./python_output/fsc_drought/lst_nt_zs_kndvi.nc') as data:
    lst_nt_zs_kndvi = data['lst_zs']
with xr.open_dataset(r'./python_output/fsc_drought/lst_sh_zs_kndvi.nc') as data:
    lst_sh_zs_kndvi = data['lst_zs']

# %%
lst_nt_zs_kndvi[:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)


with xr.open_dataset(r'./python_output/fsc_drought/temp_offset_005.nc')  as data:
    temp_offset = data['temp_offset']
temp_offset

# %%
temp_offset.plot()

# %%
temp_offset = temp_offset.coarsen(lat = 10, lon=10).mean()
temp_offset = temp_offset.interp_like(fsc, method='nearest')
temp_offset

# %%
temp_offset.plot()


with xr.open_dataset(r'./data/official_teow/biome.tif')  as data:
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


# %%
annual_temp = annual_temp.coarsen(lat = 5, lon=5, boundary='trim').mean()
annual_prec = annual_prec.coarsen(lat =5, lon=5, boundary='trim').mean()
ai_index = ai_index.coarsen(lat = 6, lon = 6).mean()
annual_temp = annual_temp.interp_like(fsc, method='nearest')
annual_prec = annual_prec.interp_like(fsc, method='nearest')
ai_index = ai_index.interp_like(fsc, method='nearest')

# %%
dataset_nt = xr.Dataset({
        'kndvi': kndvi_nt_resistance,
        'kndvi2': kndvi_nt_resistance2,
        'spei': spei_nt_drought_use_kndvi,
        'sm_change': smrz_nt_change_kndvi,
        'et_change': et_nt_change_kndvi,
        'et_change2': et_nt_change2_kndvi,
        'lst_change': lst_nt_change_kndvi,
        'lst_zs': lst_nt_zs_kndvi}).drop('quantile')
dataset_nt

# %%
dataset_sh = xr.Dataset({
        'kndvi': kndvi_sh_resistance,
        'kndvi2': kndvi_sh_resistance2,
        'spei': spei_sh_drought_use_kndvi,
        'sm_change': smrz_sh_change_kndvi,
        'et_change': et_sh_change_kndvi,
        'et_change2': et_sh_change2_kndvi,
        'lst_change': lst_sh_change_kndvi,
        'lst_zs': lst_sh_zs_kndvi}).drop('quantile')
dataset_sh

# %%
df_nt = dataset_nt.to_dataframe().reset_index()
df_sh = dataset_sh.to_dataframe().reset_index()
df_nt = df_nt.dropna()
df_sh = df_sh.dropna()

# %%
df_nt.head()

# %%
df_sh.head()

# %%
df_all = pd.concat([df_nt, df_sh])
df_all.index = np.arange(df_all.shape[0])
df_all.head()

# %%
other_factor = xr.Dataset({
        'fsc': fsc,
        'annual_temp': annual_temp,
        'annual_prec': annual_prec,
        'ai_index': ai_index,
        'plant_richness': plant_richness,
        'temp_offset': temp_offset,
        'biome': biome
    }).to_dataframe().reset_index()
other_factor.head()

# %%
df_all = pd.merge(df_all, other_factor, on = ['lat','lon'])
df_all.head()

# %%
df_all = df_all.dropna()
df_all.index = np.arange(df_all.shape[0])
df_all.head()

# %%
df_all.describe()

# %%
df_all.to_csv(r'./python_output/df_all_kndvi_sem_20260108.csv', index = False)


kndvi_q1, kndvi_q99 = df_all['kndvi'].quantile([0.01,0.99])
print(kndvi_q1, kndvi_q99)

# %%
sm_change_q1, sm_change_q99 = df_all['sm_change'].quantile([0.01,0.99])
print(sm_change_q1, sm_change_q99)

# %%
et_change_q1, et_change_q99 = df_all['et_change'].quantile([0.01,0.99])
print(et_change_q1, et_change_q99)

# %%
lst_change_q1, lst_change_q99 = df_all['lst_change'].quantile([0.01,0.99])
print(lst_change_q1, lst_change_q99)

# %%
df_use = df_all[(df_all['kndvi'] > kndvi_q1) & (df_all['kndvi'] < kndvi_q99) ]
df_use = df_use[(df_use['sm_change'] > sm_change_q1) & (df_use['sm_change'] < sm_change_q99) ]
df_use = df_use[(df_use['et_change'] > et_change_q1) & (df_use['et_change'] < et_change_q99) ]
df_use = df_use[(df_use['lst_change'] > lst_change_q1) & (df_use['lst_change'] < lst_change_q99) ]
df_use.head()

# %%
df_use.index = np.arange(df_use.shape[0])

# %%
df_use.describe()

# %%
df_use.to_csv(r'./python_output/fsc_drought/df_use.csv', index = False)


df_use_log = df_use.copy()
df_use_log[['kndvi','sm_change', 'annual_prec']] = np.log(df_use_log[['kndvi','sm_change', 'annual_prec']])
df_use_log.head()

# %%
df_use_log['kndvi2_log'] = np.log(df_use_log['kndvi2']/(1-df_use_log['kndvi2']))

# %%
def log_trans(x):
    if x > 0:
        return np.log(x)
    elif x < 0:
        return -np.log(-x)
    else:
        return 0

# %%
df_use_log[['et_change','lst_change']] = df_use_log.transform({'et_change': log_trans, 'lst_change': log_trans})
df_use_log.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# %%
df_use_log[['kndvi', 'kndvi2_log','spei', 'sm_change', 'et_change','et_change2','lst_change', 'lst_zs', 'fsc', 'annual_temp', 'annual_prec', 'ai_index','plant_richness','temp_offset']] = scaler.fit_transform(
    df_use_log[['kndvi', 'kndvi2_log','spei', 'sm_change', 'et_change','et_change2','lst_change', 'lst_zs','fsc', 'annual_temp', 'annual_prec', 'ai_index','plant_richness','temp_offset']])
df_use_log.head()

# %%
df_use_log.to_csv(r'./python_output/fsc_drought/df_use_log.csv', index = False)

# %%
desc = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+ spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc +temp_offset+ sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc+temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
kndvi ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""

# %%
mod_raw = sem.Model(desc)
res = mod_raw.fit(df_use_log)
print(res)

# %%
ins_raw = mod_raw.inspect()
ins_raw

# %%
ins_raw.to_csv(r'./python_output/fsc_drought/ins_raw_tempoffset_20251216.csv')

# %%
stats_raw = sem.calc_stats(mod_raw)
print(stats_raw.T)

desc = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+ spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc +temp_offset+ sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc+temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
kndvi2 ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""
mod_raw_kndvi2 = sem.Model(desc)
res_kndvi2 = mod_raw_kndvi2.fit(df_use_log)
print(res_kndvi2)

# %%
ins_raw_kndvi2 = mod_raw_kndvi2.inspect()
ins_raw_kndvi2

# %%
ins_raw_kndvi2.to_csv(r'./python_output/fsc_drought/ins_raw_kndvi2_tempoffset_20260105.csv')

# %%
stats_raw_kndvi2 = sem.calc_stats(mod_raw_kndvi2)
print(stats_raw_kndvi2.T)


desc = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+ spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc +temp_offset+ sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc+temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
kndvi2_log ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""

# %%
mod_raw_kndvi2_log = sem.Model(desc)
res_kndvi2_log = mod_raw_kndvi2_log.fit(df_use_log)
print(res_kndvi2_log)

# %%
ins_raw_kndvi2_log = mod_raw_kndvi2_log.inspect()
ins_raw_kndvi2_log

# %%
ins_raw_kndvi2_log.to_csv(r'./python_output/fsc_drought/ins_raw_kndvi2_log_tempoffset_20260105.csv')

# %%
stats_raw_kndvi2_log = sem.calc_stats(mod_raw_kndvi2_log)
print(stats_raw_kndvi2_log.T)


# %%
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_nt_resistance_after2000.nc') as data:
    kndvi_nt_resistance_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_sh_resistance_after2000.nc') as data:
    kndvi_sh_resistance_after2000 = data['kndvi_resistance']
kndvi_nt_resistance_after2000

kndvi_sh_resistance_after2000

# %%
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_nt_resistance2_after2000.nc') as data:
    kndvi_nt_resistance2_after2000 = data['kndvi_resistance']
with  xr.open_dataset(r'./python_output/fsc_drought/kndvi_sh_resistance2_after2000.nc') as data:
    kndvi_sh_resistance2_after2000 = data['kndvi_resistance']
kndvi_nt_resistance2_after2000

# %%
kndvi_sh_resistance2_after2000


spei_nt_drought_use_kndvi_after2000 = spei_nt_annual_drought.sel(year = slice(2000, 2021))
spei_sh_drought_use_kndvi_after2000 = spei_sh_annual_drought.sel(year = slice(2000, 2020))



with xr.open_dataset(r'./python_output/fsc_drought/smrz_nt_change_kndvi_after2000.nc') as data:
    smrz_nt_change_kndvi_after2000 = data['sm_change']
with xr.open_dataset(r'./python_output/fsc_drought/smrz_sh_change_kndvi_after2000.nc') as data:
    smrz_sh_change_kndvi_after2000 = data['sm_change']
smrz_nt_change_kndvi_after2000


with xr.open_dataset(r'./python_output/fsc_drought/et_nt_change_kndvi_after2000.nc') as data:
    et_nt_change_kndvi_after2000 = data['et_change']
with xr.open_dataset(r'./python_output/fsc_drought/et_sh_change_kndvi_after2000.nc') as data:
    et_sh_change_kndvi_after2000 = data['et_change']
et_nt_change_kndvi_after2000


with xr.open_dataset(r'./python_output/fsc_drought/et_nt_change2_kndvi_after2000.nc') as data:
    et_nt_change2_kndvi_after2000 = data['et_change2']
with xr.open_dataset(r'./python_output/fsc_drought/et_sh_change2_kndvi_after2000.nc') as data:
    et_sh_change2_kndvi_after2000 = data['et_change2']
et_nt_change2_kndvi_after2000


with xr.open_dataset(r'./python_output/fsc_drought/lst_nt_change_kndvi_after2000.nc') as data:
    lst_nt_change_kndvi_after2000 = data['lst_change']
with xr.open_dataset(r'./python_output/fsc_drought/lst_sh_change_kndvi_after2000.nc') as data:
    lst_sh_change_kndvi_after2000 = data['lst_change']
lst_nt_change_kndvi_after2000


with xr.open_dataset(r'./python_output/fsc_drought/lst_nt_zs_kndvi_after2000.nc') as data:
    lst_nt_zs_kndvi_after2000 = data['lst_zs']
with xr.open_dataset(r'./python_output/fsc_drought/lst_sh_zs_kndvi_after2000.nc') as data:
    lst_sh_zs_kndvi_after2000 = data['lst_zs']


lst_nt_zs_kndvi_after2000

dataset_nt_after2000 = xr.Dataset({
        'kndvi': kndvi_nt_resistance_after2000,
        'kndvi2': kndvi_nt_resistance2_after2000,
        'spei': spei_nt_drought_use_kndvi_after2000,
        'sm_change': smrz_nt_change_kndvi_after2000,
        'et_change': et_nt_change_kndvi_after2000,
        'et_change2': et_nt_change2_kndvi_after2000,
        'lst_change': lst_nt_change_kndvi_after2000,
        'lst_zs': lst_nt_zs_kndvi_after2000}).drop('quantile')
dataset_nt_after2000

# %%
dataset_sh_after2000 = xr.Dataset({
        'kndvi': kndvi_sh_resistance_after2000,
        'kndvi2': kndvi_sh_resistance2_after2000,
        'spei': spei_sh_drought_use_kndvi_after2000,
        'sm_change': smrz_sh_change_kndvi_after2000,
        'et_change': et_sh_change_kndvi_after2000,
        'et_change2': et_sh_change2_kndvi_after2000,
        'lst_change': lst_sh_change_kndvi_after2000,
        'lst_zs': lst_sh_zs_kndvi_after2000}).drop('quantile')
dataset_sh_after2000

# %%
df_nt_after2000 = dataset_nt_after2000.to_dataframe().reset_index()
df_sh_after2000 = dataset_sh_after2000.to_dataframe().reset_index()
df_nt_after2000 = df_nt_after2000.dropna()
df_sh_after2000 = df_sh_after2000.dropna()
df_nt_after2000.head()

# %%
df_sh_after2000.head()

# %%
df_all_after2000 = pd.concat([df_nt_after2000, df_sh_after2000])
df_all_after2000.index = np.arange(df_all_after2000.shape[0])
df_all_after2000.head()

# %%
df_all_after2000 = pd.merge(df_all_after2000, other_factor, on = ['lat','lon'])
df_all_after2000.head()

# %%
df_all_after2000 = df_all_after2000.dropna()
df_all_after2000.index = np.arange(df_all_after2000.shape[0])
df_all_after2000.head()

# %%
df_all_after2000.describe()

# %%
df_all_after2000.to_csv(r'./python_output/fsc_drought/df_all_kndvi_sem_after2000_20260105.csv', index = False)

kndvi_q1_after2000, kndvi_q99_after2000 = df_all_after2000['kndvi'].quantile([0.01,0.99])
print(kndvi_q1_after2000, kndvi_q99_after2000)

# %%
sm_change_q1_after2000, sm_change_q99_after2000 = df_all_after2000['sm_change'].quantile([0.01,0.99])
print(sm_change_q1_after2000, sm_change_q99_after2000)

# %%
et_change_q1_after2000, et_change_q99_after2000 = df_all_after2000['et_change'].quantile([0.01,0.99])
print(et_change_q1_after2000, et_change_q99_after2000)

# %%
lst_change_q1_after2000, lst_change_q99_after2000 = df_all_after2000['lst_change'].quantile([0.01,0.99])
print(lst_change_q1_after2000, lst_change_q99_after2000)

# %%
df_use_after2000 = df_all_after2000[(df_all_after2000['kndvi'] > kndvi_q1_after2000) & (df_all_after2000['kndvi'] < kndvi_q99_after2000) ]
df_use_after2000 = df_use_after2000[(df_use_after2000['sm_change'] > sm_change_q1_after2000) & (df_use_after2000['sm_change'] < sm_change_q99_after2000) ]
df_use_after2000 = df_use_after2000[(df_use_after2000['et_change'] > et_change_q1_after2000) & (df_use_after2000['et_change'] < et_change_q99_after2000) ]
df_use_after2000 = df_use_after2000[(df_use_after2000['lst_change'] > lst_change_q1_after2000) & (df_use_after2000['lst_change'] < lst_change_q99_after2000) ]
df_use_after2000.head()

# %%
df_use_after2000.index = np.arange(df_use_after2000.shape[0])
df_use_after2000.describe()

# %%
df_use_log_after2000 = df_use_after2000.copy()
df_use_log_after2000[['kndvi','sm_change', 'annual_prec']] = np.log(df_use_log_after2000[['kndvi','sm_change', 'annual_prec']])
df_use_log_after2000.head()

# %%
df_use_log_after2000['kndvi2_log'] = np.log(df_use_log_after2000['kndvi2']/(1 - df_use_log_after2000['kndvi2']))
df_use_log_after2000.head()

# %%
df_use_log_after2000[['et_change','lst_change']] = df_use_log_after2000.transform({'et_change': log_trans, 'lst_change': log_trans})
df_use_log_after2000.head()

# %%
df_use_log_after2000[['kndvi', 'kndvi2_log','spei', 'sm_change', 'et_change','et_change2','lst_change', 'lst_zs', 'fsc', 'annual_temp', 'annual_prec', 'ai_index','plant_richness','temp_offset']] = scaler.fit_transform(
    df_use_log_after2000[['kndvi', 'kndvi2_log','spei', 'sm_change', 'et_change','et_change2','lst_change', 'lst_zs','fsc', 'annual_temp', 'annual_prec', 'ai_index','plant_richness','temp_offset']])
df_use_log_after2000.head()


desc = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+ spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc +temp_offset+ sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc+temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
kndvi ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""


mod_raw_after2000 = sem.Model(desc)
res_after2000 = mod_raw_after2000.fit(df_use_log_after2000)
print(res_after2000)

# %%
ins_raw_after2000 = mod_raw_after2000.inspect()
ins_raw_after2000

# %%
ins_raw_after2000.to_csv(r'./python_output/fsc_drought/ins_raw_tempoffset_after2000_20260105.csv')

# %%
stats_raw_after2000 = sem.calc_stats(mod_raw_after2000)
print(stats_raw_after2000.T)


desc = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+ spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc +temp_offset+ sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc+temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
kndvi2 ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""
mod_raw_kndvi2_after2000 = sem.Model(desc)
res_kndvi2_after2000 = mod_raw_kndvi2_after2000.fit(df_use_log_after2000)
print(res_kndvi2_after2000)

# %%
ins_raw_kndvi2_after2000 = mod_raw_kndvi2_after2000.inspect()
ins_raw_kndvi2_after2000

# %%
ins_raw_kndvi2_after2000.to_csv(r'./python_output/fsc_drought/ins_raw_kndvi2_tempoffset_after2000_20260105.csv')

# %%
stats_raw_kndvi2_after2000 = sem.calc_stats(mod_raw_kndvi2_after2000)
print(stats_raw_kndvi2_after2000.T)


desc = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+ spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc +temp_offset+ sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc+temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
kndvi2_log ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""
mod_raw_kndvi2_log_after2000 = sem.Model(desc)

# %%
res_kndvi2_log_after2000 = mod_raw_kndvi2_log_after2000.fit(df_use_log_after2000)
print(res_kndvi2_log_after2000)

# %%
ins_raw_kndvi2_log_after2000 = mod_raw_kndvi2_log_after2000.inspect()
ins_raw_kndvi2_log_after2000

# %%
ins_raw_kndvi2_log_after2000.to_csv(r'./python_output/fsc_drought/ins_raw_kndvi2_log_tempoffset_after2000_20260105.csv')

# %%
stats_raw_kndvi2_log_after2000 = sem.calc_stats(mod_raw_kndvi2_log_after2000)
print(stats_raw_kndvi2_log_after2000.T)


with  xr.open_dataset(r'./python_output/fsc_drought/sif_nt_resistance.nc') as data:
    sif_nt_resistance = data['sif_resistance']
with  xr.open_dataset(r'./python_output/fsc_drought/sif_sh_resistance.nc') as data:
    sif_sh_resistance = data['sif_resistance']

# %%
sif_nt_resistance

# %%
sif_sh_resistance

# %%
sif_nt_resistance[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
with  xr.open_dataset(r'./python_output/fsc_drought/sif_nt_resistance2.nc') as data:
    sif_nt_resistance2 = data['sif_resistance']
with  xr.open_dataset(r'./python_output/fsc_drought/sif_sh_resistance2.nc') as data:
    sif_sh_resistance2 = data['sif_resistance']
sif_nt_resistance2

# %%
sif_sh_resistance2

# %%
sif_nt_resistance2[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')


with xr.open_dataset(r'../result_data/spei_nt_annual_drought.nc') as data:
    spei_nt_annual_drought = data['spei']

with xr.open_dataset(r'../result_data/spei_sh_annual_drought.nc') as data:
    spei_sh_annual_drought = data['__xarray_dataarray_variable__']
    spei_sh_annual_drought.name = 'spei'

# %%
spei_nt_drought_use_sif = spei_nt_annual_drought.sel(year = slice(2000, 2022))
spei_sh_drought_use_sif = spei_sh_annual_drought.sel(year = slice(2000, 2021))

# %%
spei_nt_drought_use_sif[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

# %%
spei_sh_drought_use_sif[0:4].plot(x = 'lon', y = 'lat', col = 'year', col_wrap = 4, cmap = 'RdYlGn')

with xr.open_dataset(r'./python_output/fsc_drought/smrz_nt_change_sif.nc') as data:
    smrz_nt_change_sif = data['sm_change']
with xr.open_dataset(r'./python_output/fsc_drought/smrz_sh_change_sif.nc') as data:
    smrz_sh_change_sif = data['sm_change']

# %%
smrz_nt_change_sif

# %%
smrz_nt_change_sif[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

# %%
smrz_sh_change_sif

# %%
smrz_sh_change_sif[0:4].plot(x = 'lon',y = 'lat', col = 'year', col_wrap = 4)

with xr.open_dataset(r'./python_output/fsc_drought/et_nt_change_sif.nc') as data:
    et_nt_change_sif = data['et_change']
with xr.open_dataset(r'./python_output/fsc_drought/et_sh_change_sif.nc') as data:
    et_sh_change_sif = data['et_change']

# %%
et_nt_change_sif

# %%
et_nt_change_sif[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -5, vmax = 5)

# %%
et_sh_change_sif

# %%
et_sh_change_sif[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -5, vmax = 5)

with xr.open_dataset(r'./python_output/fsc_drought/et_nt_change2_sif.nc') as data:
    et_nt_change2_sif = data['et_change2']
with xr.open_dataset(r'./python_output/fsc_drought/et_sh_change2_sif.nc') as data:
    et_sh_change2_sif = data['et_change2']

# %%
et_nt_change2_sif

# %%
et_nt_change2_sif[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)

# %%
et_sh_change2_sif

# %%
et_sh_change2_sif[0:4].plot(x = 'lon',y='lat',col = 'year', col_wrap = 4, vmin = -0.5, vmax = 0.5)


with xr.open_dataset(r'./python_output/fsc_drought/lst_nt_change_sif.nc') as data:
    lst_nt_change_sif = data['lst_change']
with xr.open_dataset(r'./python_output/fsc_drought/lst_sh_change_sif.nc') as data:
    lst_sh_change_sif = data['lst_change']

lst_nt_change_sif

# %%
lst_nt_change_sif[0:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)

# %%
lst_sh_change_sif

# %%
lst_sh_change_sif[0:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)


with xr.open_dataset(r'./python_output/fsc_drought/lst_nt_zs_sif.nc') as data:
    lst_nt_zs_sif = data['lst_zs']
with xr.open_dataset(r'./python_output/fsc_drought/lst_sh_zs_sif.nc') as data:
    lst_sh_zs_sif = data['lst_zs']


lst_nt_zs_sif[:4].plot(x = 'lon',y ='lat', col = 'year', col_wrap = 4, vmin = -1, vmax = 1)


dataset_nt_sif = xr.Dataset({
        'sif': sif_nt_resistance,
        'sif2': sif_nt_resistance2,
        'spei': spei_nt_drought_use_sif,
        'sm_change': smrz_nt_change_sif,
        'et_change': et_nt_change_sif,
        'et_change2': et_nt_change2_sif,
        'lst_change': lst_nt_change_sif,
        'lst_zs': lst_nt_zs_sif}).drop('quantile')
dataset_nt_sif

dataset_sh_sif = xr.Dataset({
        'sif': sif_sh_resistance,
        'sif2': sif_sh_resistance2,
        'spei': spei_sh_drought_use_sif,
        'sm_change': smrz_sh_change_sif,
        'et_change': et_sh_change_sif,
        'et_change2': et_sh_change2_sif,
        'lst_change': lst_sh_change_sif,
        'lst_zs': lst_sh_zs_sif}).drop('quantile')
dataset_sh_sif

df_nt_sif = dataset_nt_sif.to_dataframe().reset_index()
df_sh_sif = dataset_sh_sif.to_dataframe().reset_index()
df_nt_sif = df_nt_sif.dropna()
df_sh_sif = df_sh_sif.dropna()


df_nt_sif.head()

df_sh_sif.head()

df_all_sif = pd.concat([df_nt_sif, df_sh_sif])
df_all_sif.index = np.arange(df_all_sif.shape[0])
df_all_sif.head()

df_all_sif.describe()

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
df_all_sif.to_csv(r'./python_output/fsc_drought/df_all_sif.csv', index = False)


kndvi_q1_sif, kndvi_q99_sif = df_all_sif['sif'].quantile([0.01,0.99])
print(kndvi_q1_sif, kndvi_q99_sif)

# %%
sm_change_q1_sif, sm_change_q99_sif = df_all_sif['sm_change'].quantile([0.01,0.99])
print(sm_change_q1_sif, sm_change_q99_sif)

# %%
et_change_q1_sif, et_change_q99_sif = df_all_sif['et_change'].quantile([0.01,0.99])
print(et_change_q1_sif, et_change_q99_sif)

# %%
lst_change_q1_sif, lst_change_q99_sif = df_all_sif['lst_change'].quantile([0.01,0.99])
print(lst_change_q1_sif, lst_change_q99_sif)

# %%
df_use_sif = df_all_sif[(df_all_sif['sif'] > kndvi_q1_sif) & (df_all_sif['sif'] < kndvi_q99_sif) ]
df_use_sif = df_use_sif[(df_use_sif['sm_change'] > sm_change_q1_sif) & (df_use_sif['sm_change'] < sm_change_q99_sif) ]
df_use_sif = df_use_sif[(df_use_sif['et_change'] > et_change_q1_sif) & (df_use_sif['et_change'] < et_change_q99_sif) ]
df_use_sif = df_use_sif[(df_use_sif['lst_change'] > lst_change_q1_sif) & (df_use_sif['lst_change'] < lst_change_q99_sif) ]
df_use_sif.head()

# %%
df_use_sif.index = np.arange(df_use_sif.shape[0])
df_use_sif.describe()

# %%
df_use_sif.to_csv(r'./python_output/fsc_drought/df_use_sif.csv', index = False)


df_use_sif_log = df_use_sif.copy()
df_use_sif_log[['sif','sm_change', 'annual_prec']] = np.log(df_use_sif_log[['sif','sm_change', 'annual_prec']])
df_use_sif_log.head()

# %%
df_use_sif_log['sif2_log'] = np.log(df_use_sif_log['sif2']/(1-df_use_sif_log['sif2']))
df_use_sif_log.head()

# %%
df_use_sif_log[['et_change','lst_change']] = df_use_sif_log.transform({'et_change': log_trans, 'lst_change': log_trans})
df_use_sif_log.head()

# %%
df_use_sif_log[['sif', 'sif2_log', 'spei', 'sm_change', 'et_change','et_change2','lst_change', 'lst_zs', 'fsc', 'annual_temp', 'annual_prec','ai_index', 'plant_richness','temp_offset']] = scaler.fit_transform(
    df_use_sif_log[['sif', 'sif2_log','spei', 'sm_change', 'et_change','et_change2','lst_change', 'lst_zs','fsc', 'annual_temp', 'annual_prec','ai_index', 'plant_richness','temp_offset']])
df_use_sif_log.head()

# %%
df_use_sif_log.to_csv(r'./python_output/fsc_drought/df_use_sif_log.csv', index = False)

desc_sif = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + temp_offset+ fsc + spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
sif ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""

mod_raw_sif = sem.Model(desc_sif)
res_sif = mod_raw_sif.fit(df_use_sif_log)
print(res_sif)

# %%
ins_raw_sif = mod_raw_sif.inspect()
ins_raw_sif

ins_raw_sif.to_csv(r'./python_output/fsc_drought/ins_raw_tempoffset_sif_20260108.csv')

stats_raw_sif = sem.calc_stats(mod_raw_sif)
print(stats_raw_sif.T)

desc_sif = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + temp_offset+ fsc + spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
sif2 ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""
mod_raw_sif2 = sem.Model(desc_sif)
res_sif2 = mod_raw_sif2.fit(df_use_sif_log)
print(res_sif2)

# %%
ins_raw_sif2 = mod_raw_sif2.inspect()
ins_raw_sif2

# %%
ins_raw_sif2.to_csv(r'./python_output/fsc_drought/ins_raw_tempoffset_sif2_20260108.csv')

# %%
stats_raw_sif2 = sem.calc_stats(mod_raw_sif2)
print(stats_raw_sif2.T)

desc_sif = """
fsc ~ annual_prec + annual_temp + plant_richness
plant_richness ~ annual_prec + annual_temp
sm_change ~ annual_prec + annual_temp + plant_richness + temp_offset+ fsc + spei
lst_zs ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset+sm_change + spei
et_change2 ~ annual_prec + annual_temp + plant_richness + fsc + temp_offset + sm_change + lst_zs + spei
temp_offset ~ annual_prec + annual_temp + plant_richness + fsc
sif2_log ~ annual_prec + annual_temp + plant_richness + fsc + spei + sm_change + lst_zs + et_change2 + temp_offset
"""
mod_raw_sif2_log = sem.Model(desc_sif)
res_sif2_log = mod_raw_sif2_log.fit(df_use_sif_log)
print(res_sif2_log)

# %%
ins_raw_sif2_log = mod_raw_sif2_log.inspect()
ins_raw_sif2_log

# %%
ins_raw_sif2_log.to_csv(r'./python_output/fsc_drought/ins_raw_tempoffset_sif2_log_20260108.csv')

# %%
stats_raw_sif2_log = sem.calc_stats(mod_raw_sif2_log)
print(stats_raw_sif2_log.T)
