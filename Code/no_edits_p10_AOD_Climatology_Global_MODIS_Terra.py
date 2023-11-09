#Cell 1
#!/usr/bin/env python
import numpy  as np
import pandas as pd
import xarray as xr
git 
from xmac import (api, xstats, plot)

#Cell 2
# ======================| import lib |=========================================
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from   matplotlib.patches import BoxStyle
from   matplotlib.colors import BoundaryNorm
from   matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from   cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

mpl.rcParams['figure.dpi'] = 200

#Cell 3
def map_conf(ax):
    minLon = -180; maxLon = 180
    minLat = -90;  maxLat = 90

    # ax.stock_img()
    ax.set_global()
    # ax.set_extent([minLon, maxLon, minLat, maxLat], data_proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.2, )#zorder=0, alpha=1)
    ax.add_feature(cfeature.LAND,      color='silver')#'oldlace')#'whitesmoke')
    ax.add_feature(cfeature.OCEAN,     color='silver')#'aliceblue')#'whitesmoke')
    # ax.add_feature(cfeature.BORDERS,   linewidth=0.2, zorder=1, alpha=1)
    # ax.add_feature(cfeature.LAKES,     linewidth=0.2, zorder=1)
    # ax.add_feature(cfeature.STATES,    linewidth=0.2, zorder=1, alpha=1)
    # ax.outline_patch.set_linewidth(0.5)

    # lon_formatter = LongitudeFormatter(zero_direction_label=False)
    # lat_formatter = LatitudeFormatter()
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
    # plt.setp(ax.get_xminorticklabels(), visible=False)
    # plt.setp(ax.get_yminorticklabels(), visible=False)
    # equator = ax.gridlines(ylocs=[0],draw_labels=False,linewidth=0.1,linestyle=(0,(5,10)), color='k', )
    # green   = ax.gridlines(xlocs=[0],draw_labels=False,linewidth=0.1,linestyle=(0,(5,10)), color='k', )#edgecolor='k')

    # # ax.set_xticklabels([])
    # # ax.set_yticklabels([])
    # ax.xaxis.set_visible(True)
    # ax.yaxis.set_visible(True)

# =================| color bar|===============================================
def color_bar(map, cax=None, ax=None, label='color bar label', orientation='horizontal', tickstep=1):
    if orientation=='horizontal':
        rotation=0
        pad=0.02
        labelpad=10
        labelsize=14
    elif orientation=='vertical':
        rotation=-90
        labelpad=25

    cbar  = fig.colorbar(mappable=map,
                         cax=cax,
                         ax=ax,
                         orientation=orientation,
                        #  location='right',
                        #  extend='both',
                         extendfrac=0.05,
                         extendrect=False,
                         fraction=0.15,
                         aspect=25,
                         pad=pad,
                        )
    cbar.set_label(label,
                   size=14,
                   rotation=rotation,
                   labelpad=labelpad,
                #, position=(0., 0.),
                   )
    cbar.ax.tick_params(axis='both',
                        which='both',
                        labelsize=labelsize,
                        direction='out',
                        top=False,
                        bottom=True,
                        pad=5,
                        )

    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_ticks(levs[::tickstep])
    #cbar.minorticks_on()
    #   cbar.minorticks_on()
    #   cbar.ax.tick_params(axis='x', which='both', direction='out', left=True, right=True) #, labelsize=12)
    #   # cbar.ax.locator_params(nbins=9) # max number
    #   xtick_value = np.arange(0,5+0.1,1)/10.
    #   xtick_label = [ str(tmp) for tmp in xtick_value]
    #   cbar.ax.set_xticks(xtick_value)
    #   cbar.ax.set_xticklabels(xtick_label, fontsize=11)
    #   # cbar.ax.set_xlabel('jinqinjian', size=12, labelpad=-45, position=(0.48, 0.), )
    #   cbar.ax.set_yticklabels('', fontsize=10)
    #   cbar.set_label('MODIS DOD (2003–2020)', size=12, rotation=0, labelpad=2)#, position=(0., 0.), )
    return cbar

#Cell 4
# =================| dir |====================================================
dir_data = '/DATA/satellite/04_modis/v61/MOD08_M3/aod/'
fil_list = api.file_list( dir_data + 'MOD08_M3.{2000..2021}????.061.*.nc' )
api.file_size(fil_list)

# =================| fil |====================================================
ds = xr.open_mfdataset(fil_list)

# =================| ssn |====================================================
ssn = xstats.month_to_season_mean(ds)

globe_average = xstats.area_mean(ssn)

varN = 'aod'
data = ssn[varN]
globe_average = globe_average[varN]

#Cell 5
proj_map  = ccrs.Robinson()
proj_data = ccrs.PlateCarree()

fig, axs = plt.subplots(3,2,
                        subplot_kw={'projection': proj_map},
                        gridspec_kw={'hspace': 0.01, 'wspace': 0.01},
                        **{'figsize': [8, 6], 'constrained_layout': False},
                        )
print(axs)

# =================| contour |=================================================
axs = axs.flat
axs[0].remove()
axs = axs[1:]

cf_list = []
title = ['(a) Annual', '(b) DJF', '(c) MAM', '(d) JJA', '(e) SON', ]
cmap=plot.get_cmap('WhiteBlueGreenYellowRed')

levs = np.linspace(0, 1, 10*5+1)
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='max')
for idx, ax in enumerate(axs):
    map_conf(ax)
    # cf_list += [ax.contourf(data.lon, data.lat, data[idx, :, :],
    #                        levels=np.linspace(0., 0.5, 5*5+1),
    #                        cmap=cmap,
    #                        extend='max',
    #                        transform=proj_data,
    #                        )]
    ax.set_title(title[idx], fontsize=12, pad=5)

    cf_list += [ax.pcolormesh(data.lon, data.lat, data[idx, :, :],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    if idx <= 2:
            ax.set_xticklabels([])
    if idx == 2 or idx==4:
            ax.set_yticklabels([])

    ax.text(0, -55, '{:0.3f}'.format(globe_average[idx].values),
            fontsize=12, color='blue',
            verticalalignment='center', horizontalalignment='center',
            transform=proj_data,
            )

fig.subplots_adjust(bottom=0.02, top=0.98, left=0.1, right=0.9, wspace=0.02, hspace=0.05)
cb_ax = fig.add_axes([0.2, -0.0 , 0.6, 0.02]) # x, y, and width, height # add an axes, measured in figure coordinate
color_bar(cf_list[4], cb_ax, label='MODIS-Terra AOD$_{550}$', orientation='horizontal', tickstep=10)

# =================| subplot adjust |==========================================
box = axs[0].get_position()
box.x0 = box.x0 - 0.205
box.x1 = box.x1 - 0.205
axs[0].set_position(box)

plt.show()

