# -*- coding: utf-8 -*-
"""
modified from Parker's code
(c) Elizabeth Brasseale 11/18/2022

"""
import os
import sys
#%% ****************** CASE-SPECIFIC CODE *****************

import netCDF4 as nc
import numpy as np
from datetime import datetime, timedelta


# name output file
dir0='/dataSIO/ebrasseale/'
#grid file
grid_fn = dir0+'NADB_Entero/Input/GRID_SDTJRE_LV4_ROTATE_rx020_hplus020_DK_4river_otaymk.nc'
gds = nc.Dataset(grid_fn)
#example file
ref_fn = dir0+'Codes_XWu/Model_BCs/BC_LV4_20171117_20180615_Nz10_dye.nc'
ds1 = nc.Dataset(ref_fn, mode='r')
#output file
ini_fn = dir0+'Codes_XWu/Model_BCs/BC_LV4_20171117_20180615_Nz10_dye_entero.nc'

# get rid of the old version, if it exists
try:
    os.remove(ini_fn)
except OSError:
    pass # assume error was because the file did not exist
# ds2 = nc.Dataset(ini_fn, 'w', format='NETCDF3_64BIT_OFFSET')
ds2 = nc.Dataset(ini_fn, 'w', format='NETCDF4')

# Copy dimensions
for dname, the_dim in ds1.dimensions.items():
    if 'time' in dname:
        ds2.createDimension(dname, None)
    else:
        ds2.createDimension(dname, len(the_dim) if not the_dim.isunlimited() else None)


# copy all variables for forcing file
for v_name, varin in ds1.variables.items():
    # if v_name in var_list:
    outVar = ds2.createVariable(v_name, varin.datatype, varin.dimensions)
    # Copy variable attributes, {} is a dict comprehension, cool!
    outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
    outVar[:] = varin[:]
    # if varin.ndim > 1:
    #     outVar[:] = varin[0,:]
    # else:
    #     outVar[:] = varin[0]

vv = ds2.createVariable('Entero_north', float, ('dye_time', 's_rho', 'xi_rho'))
vv.long_name = 'Enterococcus flux across north boundary'
vv.units = 'meter-2' # unitless. Use of empty string is "strongly discouraged"
vv[:] = 0.0

vv = ds2.createVariable('Entero_south', float, ('dye_time', 's_rho', 'xi_rho'))
vv.long_name = 'Enterococcus flux across south boundary'
vv.units = 'meter-2' # unitless. Use of empty string is "strongly discouraged"
vv[:] = 0.0

vv = ds2.createVariable('Entero_west', float, ('dye_time', 's_rho', 'eta_rho'))
vv.long_name = 'Enterococcus flux across west boundary'
vv.units = 'meter-2' # unitless. Use of empty string is "strongly discouraged"
vv[:] = 0.0

ds1.close()
ds2.close()

