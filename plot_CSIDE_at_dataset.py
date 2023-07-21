#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare CSIDE output with NOAA tide gauge
"""

# setup
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean as cmo
import netCDF4 as nc
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.transforms as mtrans
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import wqfun
import pytz
import utide

c10 = plt.get_cmap('tab10',10)
plt.close('all')
dir0 = '/Users/elizabethbrasseale/Projects/Water Quality/WQ_data/validation/'


# choose which station to extract
print(30*'*' + ' plot_CSIDE_at_dataset.py ' + 30*'*')
print('\n%s\n' % '** Choose extraction to plot (return for \'NOAA tide gauge/CSIDE_2018_at_NOAA.p\') **')
fn_list = ['NOAA tide gauge/CSIDE_2018_at_NOAA.p','NOAA tide gauge/CSIDE_2019_at_NOAA.p', 'CDIP/CSIDE_2018_at_CDIP.p','CDIP/CSIDE_2019_at_CDIP.p','South Bay mooring/CSIDE_2018_at_SBOO_z.p','South Bay mooring/CSIDE_2019_at_SBOO_z.p']
Nfn = len(fn_list)
fn_dict = dict(zip(range(Nfn), fn_list))
for nfn in range(Nfn):
    print(str(nfn) + ': ' + fn_list[nfn])
my_nfn = input('-- Input number -- ')
if len(my_nfn)==0:
    data_fn = dir0+'NOAA tide gauge/CSIDE_2018_at_NOAA.p'
else:
    data_fn = dir0+fn_dict[int(my_nfn)]

D = pickle.load(open(data_fn,'rb'))

for var_name in D.keys():
    locals()[var_name] = D[var_name]

fig=plt.figure(figsize=(12,8))

nvar = len(data_dict['var_list'])

if data_dict['station']=='SBOO':
    z_list = [key for key in D['data_dict'][D['data_dict']['var_list'][0]].keys()]
    nz = len(z_list)
    # nvar = nvar*nz
    nax = nvar*nz
else:
    nax = nvar

if data_dict['station']=='NOAA':
    #add plot for frequency decomposition
    nax+=3
    #demean data
    for data in data_dict, CSIDE:
        data['SSH (m)'] = data['SSH (m)']-data['SSH (m)'].mean()

# if data_dict['station']=='SBOO':
    # data_dict['time'] = data_dict['time'].tz_localize(pytz.timezone("America/Los_Angeles"),ambiguous=False).tz_convert(pytz.utc)

gs = GridSpec(nax,2)

# plot location of tide gauge
ax_map = fig.add_subplot(gs[:,1])
ax_map.contour(lon_rho,lat_rho,mask_rho,levels=[0.5],colors='gray',label=None)
ax_map.plot(data_dict['lon'],data_dict['lat'],marker='o',color='none',markersize=10,mec='k',mfc='None',label=data_dict['dataset_name'])
ax_map.plot(lon_rho[jref,iref],lat_rho[jref,iref],marker='*',color='none',markersize=15,mec='k',mfc='yellow',label='Extraction point in model')
# ax_map.plot(lon_rho[jref,iref]-0.01,lat_rho[jref,iref],marker='*',markersize=15,mec='k',mfc='yellow')
ax_map.set_xlabel('longitude')
ax_map.set_ylabel('latitude')
dl = 0.1
xlim0 = np.max([data_dict['lon']-dl,lon_rho.min()])
xlim1 = np.min([data_dict['lon']+dl,lon_rho.max()])
ylim0 = np.max([data_dict['lat']-dl,lat_rho.min()])
ylim1 = np.min([data_dict['lat']+dl,lat_rho.max()])
ax_map.axis([xlim0,xlim1,ylim0,ylim1])
wqfun.dar(ax_map)
ax_map.legend()

# time series
lw0 = 1.0
# lw1 = 1.0

CSIDE['dataset_name'] = 'model'
for vv in range(nvar):
    varname = data_dict['var_list'][vv]

    if data_dict['station']=='SBOO':
        
        for zz in range(nz):
            z = z_list[zz]
            ax = fig.add_subplot(gs[vv*4+zz,0])
            dc = 0
            for data in CSIDE, data_dict:
                s = 1-0.5*dc
                ax.scatter(data['time'],data[varname][z],label=data['dataset_name'],s=s,color=c10(dc),alpha=0.5)
                if data==data_dict:
                    ax.set_ylabel(data[varname][z].name)
                dc+=1
                
            if (vv*4+zz)<(nax-1):
                ax.set_xlabel('')
                ax.set_xticklabels([''])
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Y"))
                plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
                ax.set_xlabel('Date')
                    
            ax.grid()
            ax.set_xlim([CSIDE['time'][0],CSIDE['time'][-1]])
            

    else:
        print('recognized we are not at SBOO')

        ax = fig.add_subplot(gs[vv,0])
        # ax.plot(CSIDE['time_list'],CSIDE[varname],label='model',lw=lw0)
        # ax.plot(data_dict['time'],data_dict[varname],label=data_dict['dataset_name'],lw=lw0)
    
        dc = 0
        for data in CSIDE, data_dict:
            s = 1-0.5*dc
            if data_dict['station']=='NOAA':
                ax.plot(data['time'],data[varname],label=data['dataset_name'],lw=0.5,alpha=0.75)
            else:
                ax.scatter(data['time'],data[varname],label=data['dataset_name'],s=s,alpha=0.5)
            dc+=1
        ax.set_ylabel(varname)
        if vv<nvar-1:
            ax.set_xlabel('')
            ax.set_xticklabels([''])
            # ax.get_xaxis().set_visible(False)
        if vv==nvar-1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d %Y"))
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
            ax.set_xlabel('Date')

        ax.grid()
        ax.set_xlim([CSIDE['time'][0],CSIDE['time'][-1]])
        if vv==0:
            ax.legend()

if data_dict['station']=='NOAA':
    constit_list = ['O1','K1','N2','M2','S2']
    #open the additional axis to plot frequency decomposition
    ax_fft = fig.add_subplot(gs[-3,0])
    ax_utide_amp = fig.add_subplot(gs[-2,0])
    ax_utide_pha = fig.add_subplot(gs[-1,0])
    dc = 0
    for data in CSIDE,data_dict:
        #build x axis by counting time steps in a day
        fft_frac = (data['time'][1]-data['time'][0])/timedelta(days=1)
        nt = len(data['time'])
        nt2 = int(nt*0.5)
        freq = np.fft.fftfreq(nt,d=fft_frac)
        
        #now do fft of data
        data['SSH_fft'] = np.fft.fft(data['SSH (m)'])/nt
        
        #plot fft along with x-axis you built above
        ax_fft.plot(freq[:nt2],data['SSH_fft'][:nt2],label=data['dataset_name'],lw=lw0,color=c10(dc))
        
        #now calculate and plot tidal constituent amplitudes
        time = np.asarray([(t-datetime(1999,1,1)).total_seconds()/(24*3600) for t in data['time']])
        coef = utide.solve(t=time,u=data['SSH (m)'],v=None,epoch=datetime(1999,1,1),lat=data_dict['lat'],method='ols')
        constit_ind_list = [np.where(coef['name']==constit)[0][0] for constit in constit_list]
        amp_list = [coef['A'][constit_ind] for constit_ind in constit_ind_list]
        pha_list = [coef['g'][constit_ind] for constit_ind in constit_ind_list]
        #bar plot code
        X = np.arange(len(constit_list))+dc*0.25
        ax_utide_amp.bar(X,amp_list,color=c10(dc),width=0.25)
        ax_utide_pha.bar(X,pha_list,color=c10(dc),width=0.25)
        
        dc+=1
        
    ax_utide_amp.set_ylabel('Amplitude (m)')
    ax_utide_pha.set_ylabel('Phase (deg)')
    
    for ax_utide in ax_utide_amp,ax_utide_pha:
        ax_utide.set_xticks(np.arange(len(constit_list)))
        ax_utide.set_xticklabels(constit_list)
        
    
    ax_fft.set_xlabel('Frequency (times per day)')
    # ax.set_ylabel('Amplitude (m)')
    ax_fft.set_xlim([0,2.1])
    plt.subplots_adjust(hspace=0.4)
        

# plt.tight_layout()

plt.show(block=False)
plt.pause(0.1)
# out_fn = '/data0/ebrasseale/WQ_plots/CSIDE_2018_vs_NOAA_tide_gauge.png'
# plt.savefig(out_fn)
