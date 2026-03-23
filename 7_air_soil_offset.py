
import xarray as xr 
import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as ccrs
import rioxarray
import glob
import pandas as pd


soil_temp_path = glob.glob(r'./data/soil_temp/*0_5cm*tif')
soil_temp_path

# %%
air_temp_path = glob.glob(r'./data/chelsa/CHELSA_*.tif')
air_temp_path

# %%
with xr.open_dataset(r'../result_data/landcover_005_use.nc') as data:
    ld = data['modis_landcover']
ld

ld.plot()

# %%
soil_air_temp_all = []
for i in range(1,13):
    print(r'./data/soil_temp/soilT_{}_0_5cm.tif'.format(i))
    with rioxarray.open_rasterio(r'./data/soil_temp/soilT_{}_0_5cm.tif'.format(i)) as data:
        soil_temp =  data.where(data > -5000).coarsen(x=6, y =6, boundary='pad', side='right').mean()
        soil_temp =  soil_temp.rename({'x':'lon','y':'lat'})
        soil_temp = soil_temp.interp_like(ld, method='nearest')
        soil_temp = soil_temp.where(ld>0)
    print(air_temp_path[i-1])
    with rioxarray.open_rasterio(air_temp_path[i-1]) as data:
        air_temp =  data.where(data > -5000).coarsen(x=6, y =6, boundary='pad', side='right').mean()
        air_temp =  air_temp.rename({'x':'lon','y':'lat'})
        air_temp =  air_temp.interp_like(ld, method='nearest')
        air_temp = air_temp.where(ld>0)
    
    soil_air_temp = soil_temp -air_temp
    soil_air_temp = soil_air_temp.drop('spatial_ref')

    soil_air_temp_all.append(soil_air_temp)

# %%
soil_air_temp_all

# %%
soil_air_temp_all = xr.concat(soil_air_temp_all, dim='band')
soil_air_temp_all

# %%
soil_air_temp_all = xr.DataArray(soil_air_temp_all.values, coords=[np.arange(1,13),soil_air_temp_all.lat,soil_air_temp_all.lon], dims=['month','lat','lon'])
soil_air_temp_all

# %%
soil_air_temp_all[0].plot()


soil_air_temp_nh = soil_air_temp_all.sel(lat = slice(23.5,90), month = slice(5,9)).mean(dim='month')
soil_air_temp_sh = soil_air_temp_all.sel(lat = slice(-90,-23.5), month = soil_air_temp_all.month.isin([11,12,1,2,3])).mean(dim='month')
soil_air_temp_tr = soil_air_temp_all.sel(lat = slice(-23.5,23.5)).mean(dim = 'month')

# %%
soil_air_temp_nh.plot()

# %%
soil_air_temp_sh.plot()

# %%
soil_air_temp_tr.plot()

# %%
soil_air_temp = xr.concat([soil_air_temp_nh,soil_air_temp_sh,soil_air_temp_tr], dim='lat').sortby('lat')
soil_air_temp

# %%
soil_air_temp.plot()

# %%
soil_air_temp.name = 'temp_offset'

# %%
soil_air_temp.to_netcdf(r'./python_output/fsc_drought/temp_offset_005.nc')


