#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results of a particle tracking experime.
"""

# setup

import wqfun
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cmocean as cmo
import pickle
import numpy as np
import netCDF4 as nc
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.transforms as mtrans
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
import skill_metrics as sm
import string

c10 = plt.get_cmap('tab10')
atoz = string.ascii_lowercase

plt.rc('font', size=14)
plt.close('all')

home = '/Users/elizabethbrasseale/Projects/Water quality/'

data_fn = home+'WQ_data/ddPCR data Condensed_ for Elizabeth.xlsx'
df = pd.read_excel(data_fn,sheet_name='ddPCR data Condensed_ for Sen')

site_to_ll = {
            'MX1':{'Lat':32.43786,'Long':-117.10189,'Description':'Beach station; South of SADB WWTP'},
            'MX2':{'Lat':32.44804,'Long':-117.10527,'Description':'SADB WWTP Outfall'},
            'MX3':{'Lat':32.447652,'Long':-117.108627,'Description':'Beach station; Directly upcoast of SADB WWTP (mixing zone)'},
            'MX4':{'Lat':32.502769,'Long':-117.123575,'Description':'Beach station; Upcoast of SADB WWTP; Southern end of Playas'},
            'MX5':{'Lat':32.527417,'Long':-117.1245,'Description':'Beach station; Upcoast of SADB WWTP'},
            'SD1':{'Lat':32.5434,'Long':-117.125,'Description':'Beach station; Border Field State Park'},
            'SD2':{'Lat':32.55299,'Long':-117.12776,'Description':'Tijuana River Estuary'},
            'TJRM':{'Lat':32.55235,'Long':-117.1277,'Description':'Tijuana Rivermouth; Mixing Zone'},
            'SD3':{'Lat':32.561,'Long':-117.132,'Description':'Beach station; Tijuana Slough National Wildlife Refuge'},
            'IB1':{'Lat':32.57887,'Long':-117.133,'Description':'Beach station; Imperial Beach'},
            'IB2':{'Lat':32.58028,'Long':-117.133,'Description':'Beach station; Imperial Beach'},
            'SD4':{'Lat':32.5847,'Long':-117.133,'Description':'Beach station; Imperial Beach'},
            'SD5':{'Lat':32.6296,'Long':-117.141,'Description':'Beach station; N Silver Strand State Beach'}
        }

#convert from lat/lon to grid-following coordinate r
fname = home+'WQ_data/extractions2017-2019/nearshore_variables_wavebuoy_5m_2017–2019.p'
D = pickle.load(open(fname,'rb'))
beach_name_list = ['PB']
beach_list = wqfun.get_beach_location(beach_name_list)
for key in site_to_ll.keys():
    site = site_to_ll[key]
    # find rshore by nearest index on latshore
    y_ind = np.argmin(np.abs(D['latshore']-site['Lat']))
    site['y_ind'] = y_ind
    site['r'] = D['rshore'][y_ind]-beach_list['PB']['r']
    
# my_event = 4
# if my_event==1:
#     year=2018
#     date_str = '201810'
# else:
#     year=2019
#     date_str = '201910'
# my_df = df[df['Event']==my_event]

# load in models
# colorscale plotting parameters
dyemax = 5e-2
dyemin = 1e-4
vmax = dyemax
vmin = dyemin
colmap = cmo.cm.matter
normal = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)

model_name_list = ['U_ibn155_201810','U_ibn155_201910']
D = wqfun.get_shoreline_models(model_name_list)
y = D['U_ibn155_201810']['y']-beach_list['PB']['r']
for year in [2018,2019]:
    date_str = f'{year:}10'
    # get datetime for Oct 2018 CDIP wave buoy data
    buoy_fn = home+f'WQ_data/ImperialBeachNearshoreBuoy155_{date_str}.csv'
    df0 = pd.read_csv(buoy_fn,header=0,skiprows=[1,-1],sep='\s+')
    # columns : ['<pre>YEAR', 'MO', 'DY', 'HR', 'MN', 'Hs', 'Tp', 'Dp', 'Depth', 'Ta',
    #       'Pres', 'Wspd', 'Wdir', 'Temp', 'Temp.1']
    df0 = df0[:-1]
    df0['datetime'] = pd.to_datetime(dict(year=df0['<pre>YEAR'],month=df0.MO,day=df0.DY,hour=df0.HR,minute=df0.MN))
    D[f'U_ibn155_{date_str:}']['t']=df0.loc[:,'datetime']

fig=plt.figure(figsize=(12,9))
gs = GridSpec(2,3)
# ax00 = fig.add_subplot(gs[0,:2])
# ax10 = gs[1,0]
ax01 = fig.add_subplot(gs[0,:])
# ax11 = gs[1,1]
ax2a = fig.add_subplot(gs[1,0])
ax2b = fig.add_subplot(gs[1,1])
ax2c = fig.add_subplot(gs[1,2])
# ax2d = fig.add_subplot(gs[1,3])
ax2_list = [ax2a,ax2b,ax2c]#,ax2d]

# count=0
# for year,ax in [[2018,ax00],[2019,ax01]]:
#     date_str = f'{year:}10'
#     # get datetime for Oct 2018 CDIP wave buoy data
#     # buoy_fn = home+f'WQ_data/ImperialBeachNearshoreBuoy155_{date_str}.csv'
#     # df = pd.read_csv(buoy_fn,header=0,skiprows=[1,-1],sep='\s+')
#     # # columns : ['<pre>YEAR', 'MO', 'DY', 'HR', 'MN', 'Hs', 'Tp', 'Dp', 'Depth', 'Ta',
#     # #       'Pres', 'Wspd', 'Wdir', 'Temp', 'Temp.1']
#     # df = df[:-1]
#     # df['datetime'] = pd.to_datetime(dict(year=df['<pre>YEAR'],month=df.MO,day=df.DY,hour=df.HR,minute=df.MN))
#     # ax = fig.add_subplot(gs[3:,count])
#     pv = ax.pcolormesh(D[f'U_ibn155_{date_str:}']['t'][1:],0.001*y,np.transpose(D[f'U_ibn155_{date_str:}']['dye']),norm=normal,cmap=colmap,shading='nearest',zorder=20)
#     ax.contour(D[f'U_ibn155_{date_str:}']['t'][1:],0.001*y,np.transpose(D[f'U_ibn155_{date_str:}']['dye']),levels=[5e-4],linestyles=['solid'],linewidths=[0.2],colors=['k'],zorder=25)
#     ax.set_xlim([datetime(year,10,1),datetime(year,10,31)])
#     ylim = ax.get_ylim()
#     ax.set_ylim([2,ylim[-1]])
#     ax.set_ylabel('km from PB')
#     ax.text(0.2,0.9,'1D nearshore model',transform=ax.transAxes)
#     ax.set_zorder(1000)
#
#     ax.set_xlabel(f'{year:} Date')
#     ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
#     plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
#     count+=1

# count=0
# for year,ax in [[2018,ax00],[2019,ax01]]:
year = 2019
ax = ax01
date_str = f'{year:}10'
# get datetime for Oct 2018 CDIP wave buoy data
# buoy_fn = home+f'WQ_data/ImperialBeachNearshoreBuoy155_{date_str}.csv'
# df = pd.read_csv(buoy_fn,header=0,skiprows=[1,-1],sep='\s+')
# # columns : ['<pre>YEAR', 'MO', 'DY', 'HR', 'MN', 'Hs', 'Tp', 'Dp', 'Depth', 'Ta',
# #       'Pres', 'Wspd', 'Wdir', 'Temp', 'Temp.1']
# df = df[:-1]
# df['datetime'] = pd.to_datetime(dict(year=df['<pre>YEAR'],month=df.MO,day=df.DY,hour=df.HR,minute=df.MN))
# ax = fig.add_subplot(gs[3:,count])
pv = ax.pcolormesh(D[f'U_ibn155_{date_str:}']['t'][1:],0.001*y,np.transpose(D[f'U_ibn155_{date_str:}']['dye']),norm=normal,cmap=colmap,shading='nearest')
ax.contour(D[f'U_ibn155_{date_str:}']['t'][1:],0.001*y,np.transpose(D[f'U_ibn155_{date_str:}']['dye']),levels=[5e-4],linestyles=['solid'],linewidths=[0.2],colors=['k'])
ax.set_xlim([datetime(year,10,1),datetime(year,10,31)])
ylim = ax.get_ylim()
ax.set_ylim([2,ylim[-1]])
ax.set_ylabel('km from PB')
ax.text(0.1,0.9,'a) 1D nearshore model',transform=ax.transAxes)


ax.set_xlabel(f'{year:} Date')
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right",rotation_mode='anchor')
# count+=1

TP = 0
TN = 0
FP = 0
FN = 0
dye_list = []
ddPCR_list = []
target_count=0
# target_list = ['HF183','Lachno3','dENT','cENT']
target_list = ['HF183','Lachno3','dENT']

# event_list = [1,3,4]
event_list=[3,4]
m_list = ['o','^','s']
for my_target in target_list:
    #start by plotting one variable for one event
    # my_target = 'Lachno3'
    my_df = df[df['Target']==my_target]
    cutoff=1
    if my_target=='cENT':
        cutoff=5
    if my_target=='dENT':
        my_target='Enterococcus'
    ax2 = ax2_list[target_count]
    count=0
    for event in event_list:
        g = my_df[my_df['Event']==event]
        r_list = [site_to_ll[ss]['r'] for ss in g['Site2']]
        g['r'] = r_list
        year = g.Datetime.dt.year.min()
        dataset = D[f'U_ibn155_{year:}10']
        nr,nc = g.shape
        dye_list0 = np.zeros((nr))
        for i in range(nr):
            #only look at points a little ways north of PB
            if g.iloc[i].r<2500:
                dye_list0[i]=np.nan
            else:
                r_ind = np.argmin(np.abs(y-g.iloc[i].r))
                t_ind = np.argmin(np.abs(dataset['t']-g.iloc[i].Datetime))
                dye_list0[i] = dataset['dye'][t_ind,r_ind]
                if dye_list0[i]<1e-8:
                    dye_list0[i]=4e-4
        if year==2018:
            ax = ax00
        else:
            ax = ax01
        # p=ax.scatter(g['Datetime'],0.001*g['r'],c=g['qty_LOQ'],marker=m_list[count],norm=matplotlib.colors.LogNorm(vmax=1e5),cmap = cmo.cm.matter)
        p=ax.scatter(g['Datetime'],0.001*g['r'],c=c10(count),marker=m_list[count],s=50,edgecolors='k')
    
        ax2.scatter(dye_list0,g['qty_LOQ'],marker=m_list[count],alpha=[0.5],c=c10(count),s=50,label=f'Event {event:}')
    
        ddPCR = g.qty_LOQ.values
        ddPCR = ddPCR[~np.isnan(dye_list0)]
        dye_list0 = dye_list0[~np.isnan(dye_list0)]
    
    
        # count detects vs nondetects
        model_detects = dye_list0>5e-4
        sampling_detects = ddPCR>cutoff
    
        #calculate sensitivity and specificity
        TP += np.sum(model_detects & sampling_detects)
        FP += np.sum(model_detects & ~sampling_detects)
        TN += np.sum(~model_detects & ~sampling_detects)
        FN += np.sum(~model_detects & sampling_detects)
    
        # sensitivity = TP/(TP+FN)
        # specificity = TN/(TN+FP)
    
        #for fit, ignore SCCWRP nondetects
        dye_list0 = dye_list0[ddPCR>cutoff]
        ddPCR = ddPCR[ddPCR>cutoff]
        #for fit, ignore model nondetects
        ddPCR = ddPCR[dye_list0>5e-4]
        dye_list0 = dye_list0[dye_list0>5e-4]
    
        p = np.polyfit(np.log(dye_list0),np.log(ddPCR),1)
        xx = np.array([dye_list0.min(),dye_list0.max()])
        yy = xx**p[0]*np.exp(p[1])
        ax2.plot(xx,yy,linestyle='dashed',color=c10(count))

        wss = wqfun.willmott(ddPCR,dye_list0)
        r = np.corrcoef(ddPCR,dye_list0)[0,1]
        rsquared = r**2
        rmsd = sm.rmsd(ddPCR,dye_list0)
        print(f'{my_target}, Event {event:}')
        print(f'R2={rsquared:.3}'+'\n '+f'slope = {p[0]:.2}'+'\n '+f'intercept = {p[1]:.2}')
        # ax2.text(0.1,0.8,r'$R^{2}$'+f'={rsquared:.3}'+'\n '+f'slope = {p[0]:.2}'+'\n '+f'intercept = {p[1]:.2}',transform=ax2.transAxes,ha='left',va='center')
        # ax2.text(0.1,0.6,f'Sensitivity={sensitivity:.3}'+'\n '+f'Specificity = {specificity:.3}',transform=ax2.transAxes,ha='left',va='center')
        if my_target=='cENT':
            ax2.set_ylabel(f'{my_target:} MPN/100mL')
        else:
            ax2.set_ylabel(f'Copies/100 mL')
        if count==(len(event_list)-1):
            ax2.set_xlabel('Model dye')
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        # ax2.legend()
        # ax2.text(0.1,0.9,f'Event {event:}',color='k',fontsize=14,fontweight='bold',transform=ax2.transAxes,ha='left',va='center')
    
        # dye_list= np.append(dye_list,dye_list0,axis=0)
        # ddPCR_list = np.append(ddPCR_list,g.qty_LOQ.values,axis=0)
        count+=1
    
    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()

    ax2.axis([xlim2[0],xlim2[1],ylim2[0],ylim2[1]])

    # add model nondetects
    line1 = 1e-5 # vertical x = 3
    line2 = 5e-4 # vertical x = 5

    ax2.axvspan(line1, line2, alpha=.2, color='gray')
    

    # add sampling nondetects
    line1 = 0.001 # vertical x = 3
    line2 = cutoff+1 # vertical x = 5

    ax2.axhspan(line1, line2, alpha=.2, color='gray')
    
    ax2.text(0.15,0.95,atoz[target_count+1]+f') {my_target}',color='k',fontsize=16,transform=ax2.transAxes,ha='left',va='top')
    if target_count==0:
        # ax2.text(0.15,0.85,'Event 1',color=c10(0),transform=ax2.transAxes,ha='left',va='top')
        ax2.text(0.15,0.88,'Campaign 1',color=c10(0),transform=ax2.transAxes,ha='left',va='top')
        ax2.text(0.15,0.82,'Campaign 2',color=c10(1),transform=ax2.transAxes,ha='left',va='top')
        ax2.text(0.1,0.68,'Model nondetects',transform=ax2.transAxes,rotation=90,ha='center',va='center')
        ax2.text(0.5,0.1,'Sampling nondetects',transform=ax2.transAxes,rotation=0,ha='center',va='center')
    
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    # ax2c.text(0.1,0.6,f'Sensitivity={sensitivity:.3}'+'\n '+f'Specificity = {specificity:.3}',transform=ax2c.transAxes,ha='left',va='center')
    print(f'{my_target}'+'\n'+f'Sensitivity={sensitivity:.3}'+'\n '+f'Specificity = {specificity:.3}')
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    target_count+=1
# ax00.set_ylabel('km from PB')
# # ax00.text(0.2,0.9,my_target+' Event 1',transform=ax00.transAxes)
# xlim = [datetime(2018,10,1),datetime(2018,10,31)]
# # plt.setp(ax00.get_xticklabels(), visible=False)
# ax00.set_xlim(xlim)
# ylim = ax00.get_ylim()
# ax00.set_ylim([2,ylim[-1]])
# ax00.grid(zorder=1000)

# consistent axes for RHS axes
# xlim0 = np.min([ax2a.get_xlim(),ax2b.get_xlim(),ax2c.get_xlim()])
# xlim1 = np.max([ax2a.get_xlim(),ax2b.get_xlim(),ax2c.get_xlim()])
# ylim0 = np.min([ax2a.get_ylim(),ax2b.get_ylim(),ax2c.get_ylim()])
# ylim1 = np.max([ax2a.get_ylim(),ax2b.get_ylim(),ax2c.get_ylim()])
    
# g = my_df[my_df['Event']==3]
# r_list = [site_to_ll[ss]['r'] for ss in g['Site2']]
# g['r'] = r_list
# p=ax01.scatter(g['Datetime'],0.001*g['r'],c=g['qty_LOQ'],norm=matplotlib.colors.LogNorm(vmax=1e5),cmap = cmo.cm.matter)
# g = my_df[my_df['Event']==4]
# r_list = [site_to_ll[ss]['r'] for ss in g['Site2']]
# g['r'] = r_list
# p=ax01.scatter(g['Datetime'],0.001*g['r'],c=g['qty_LOQ'],norm=matplotlib.colors.LogNorm(vmax=1e5),cmap = cmo.cm.matter)
# ax01.set_ylabel('km from PB')
# ax01.text(0.2,0.9,my_target+' Events 3 and 4',transform=ax01.transAxes)
xlim = [datetime(2019,10,1),datetime(2019,10,31)]
# plt.setp(ax01.get_xticklabels(), visible=False)
# plt.setp(ax01.get_yticklabels(), visible=False)
ax01.set_xlim(xlim)
# ylim = ax01.get_ylim()
# ax01.grid(zorder=1000)
ax01.set_ylim([2,ylim[-1]])
ax01.text(datetime(2019,10,3,12),25,'Campaign 1',color='k',va='center',ha='center')

ax01.text(datetime(2019,10,28,12),25,'Campaign 2',color='k',va='center',ha='center')# ddPCR_list = ddPCR_list[~np.isnan(dye_list)]
# dye_list = dye_list[~np.isnan(dye_list)]
# p = np.polyfit(np.log(dye_list),np.log(ddPCR_list),1)
# xx = np.array([dye_list.min(),dye_list.max()])
# yy = xx**p[0]*np.exp(p[1])
# ax2.plot(xx,yy,linestyle='dashed',color='k')
#
# wss = wqfun.willmott(ddPCR_list,dye_list)
# r = np.corrcoef(ddPCR_list,dye_list)[0,1]
# rsquared = r**2
# rmsd = sm.rmsd(ddPCR_list,dye_list)
# ax2.text(0.1,0.8,r'$R^{2}$'+f'={rsquared:.3}'+'\n '+f'slope = {p[0]:.2}',transform=ax2.transAxes,ha='left',va='center')
# ax2.set_ylabel('SCCWRP ddPCR HF183 (conc)')
# ax2.set_xlabel('Tracer shoreline model run w/ CDIP buoy (conc)')
# ax2.set_xscale("log")
# ax2.set_yscale("log")
# ax2.legend()



cbaxes = inset_axes(ax01, width="30%", height="4%", loc='upper right',bbox_transform=ax01.transAxes,bbox_to_anchor=(-0.25,0.,1,1))
cb = fig.colorbar(pv, cax=cbaxes, orientation='horizontal')
# cbaxes.set_zorder(1000)
cbaxes.set_xlabel('Model dye')

plt.subplots_adjust(hspace=0.25,wspace=0.4,top=0.98)
plt.show(block=False)
plt.pause(0.1)
# fig_fn = home+f'WQ_plots/compare_SCCWRP_buoymodel_alltargets_B2.png'
# plt.savefig(fig_fn)