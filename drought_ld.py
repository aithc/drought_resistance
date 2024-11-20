#!/usr/bin/env python
# coding: utf-8

# ## modis landcover 



import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import dask
from pyhdf.SD import SD, SDC



lat_list = np.arange(-89.975, 90, 0.05)
lon_list = np.arange(-179.975, 180, 0.05)


# ### read data

def read_hdfs(path):
    modis_ld = SD(path, SDC.READ)
    IGBP = modis_ld.select('Majority_Land_Cover_Type_1')
    IGBP_data = IGBP[:,:].astype(np.float64)
    
    year = int(path[28:32])
    IGBP_xr = xr.DataArray([IGBP_data[::-1,]], coords=[[year], lat_list, lon_list], dims=['year','lat', 'lon'])
    
    return IGBP_xr



modis_ld_path = glob.glob('modis_landcover/*.hdf')
modis_ld_path


modis_ld_all = []
for modis_path_n in modis_ld_path:
    print(modis_path_n)
    IGBP_n = read_hdfs(modis_path_n)
    modis_ld_all.append(IGBP_n)
    
modis_ld_all = xr.concat(modis_ld_all, dim='year')



modis_ld_all.name = 'modis_landcover'
modis_ld_all


modis_ld_all = modis_ld_all.where(modis_ld_all>0)
modis_ld_all = modis_ld_all.where(modis_ld_all<12)


def ld_unchanged(ld_st):
    if np.isnan(ld_st).all():
        change = 2
    else:
        ld_len = len(np.unique(ld_st))
        if ld_len > 1:
            change = 1
        else:
            change = 0
    return change


ld_change =  xr.apply_ufunc(
    ld_unchanged,
    modis_ld_all,
    input_core_dims=[['year']],
    output_core_dims=[[]],
    vectorize= True
)


ld_change


modis_ld_filtered = modis_ld_all.where(ld_change == 0)
modis_ld_filtered


modis_ld_use = xr.DataArray(modis_ld_filtered[0].values, coords = [modis_ld_filtered.lat, modis_ld_filtered.lon], dims=['lat','lon'])
modis_ld_use.name = 'modis_landcover'
modis_ld_use


modis_ld_use.to_netcdf(r'result_data/landcover_005_use.nc')




