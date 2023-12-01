# Cell 1.
'''
this is called a shebang line and is typically found at the beginning of Unix/Linux  scrips.
It tells the system to use the python interpreter located in the user's environment to run the script.
'''
#!/usr/bin/env python

'''
the line below imports the numpy library and gives it the alias 'np.' Numpy is usually used for
#numerical and mathematical operations in Python.
'''
import numpy  as np

'''
imports pandas and gives it the alias pd. Pandas is popular for data manipulation and mathematical
operations in Python.
'''
import pandas as pd

'''
xarray is a library used for working with labeled multi-dimensional arrays, particularly used
for scientific data analysis.
'''
import xarray as xr

'''
A package/library Qinjian made with certain functions and classes for specific tasks.
'''
from xmac import (api, xstats, plot)


# Cell 2.
# ======================| import lib |=========================================
'''
imports matplotlib library. Matplotlib is a popular Python library for creating static, animated,
and interactive visualizations in Python
'''
import matplotlib as mpl

'''
Imports the pyplot module from Matplotlib. The pyplot module provides a simple interface for
creating and customizing plots.
'''
import matplotlib.pyplot as plt

'''
Imports the gridspec module from Matplotlib. gridspec allows you to create complex grid layouts
of subplots within a single figure.
'''
import matplotlib.gridspec as gridspec

'''
Imports the BoxStyle class from Matplotlib's patches module. This class is used to define custom
box styles for text boxes or shapes in plots.
'''
from   matplotlib.patches import BoxStyle

'''
Imports the BoundaryNorm class from Matplotlib's colors module. BoundaryNorm is used for mapping
data values to colors within discrete intervals or boundaries.
'''
from   matplotlib.colors import BoundaryNorm

'''
Imports the ListedColormap class from Matplotlib's colors module. ListedColormap is used to create
a colormap from a list of specified colors.
'''
from   matplotlib.colors import ListedColormap

'''
Imports the Cartopy library. Cartopy is a library for cartographic (drawing maps) projections and
geospatial data visualization.
'''
import cartopy.crs as ccrs

'''
Imports the cfeature module from Cartopy. cfeature provides access to various geospatial features
like coastlines, land, rivers, and more.
'''
import cartopy.feature as cfeature

'''
Imports the LongitudeFormatter and LatitudeFormatter classes from Cartopy's mpl.ticker module.
These classes are used for custom formatting of longitude and latitude axis tick labels.
'''
from   cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

'''
Sets the Matplotlib figure DPI (dots per inch) to 200. This line of code configures the resolution
of figures produced by Matplotlib, making them higher resolution when displayed or saved.
'''
mpl.rcParams['figure.dpi'] = 200


# Cell 3.
'''
This function is desinged to configure the settings of a Cartopy map.
'''
def map_conf(ax):
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 31

    # ax.stock_img()
    '''
    It sets up a map with global coverage (entire Earth's surface) using ax.set_global()
    '''
    ax.set_global()
    ax.set_extent([minLon, maxLon, minLat, maxLat]) #this can make figure zoom in...sometimes this may need to be moved to end.
    '''
    Several features are added including coastlines, land masses, and oceans.
    '''
    ax.add_feature(cfeature.COASTLINE, linewidth=0.2, )#zorder=0, alpha=1)
    ax.add_feature(cfeature.LAND,      color='silver')#'oldlace')#'whitesmoke')
    ax.add_feature(cfeature.OCEAN,     color='silver')#'aliceblue')#'whitesmoke')

# =================| color bar|===============================================
'''
This function is used for creating a color bar legend for a map. It takes several parameters,
including the map (the color-mapped object you want to create a legend for), cax (the color bar
axis), ax (the axis where the color bar will be drawn), label (the label for the color bar),
orientation (horizontal or vertical), and tickstep (spacing between color bar ticks).

The function sets up the color bar with appropriate labels, size, rotation, and tick parameters
based on the specified orientation.

At the end, the created color bar object is retured.
'''
def color_bar(fig, map, cax=None, ax=None, label='color bar label', orientation='horizontal', tickstep=1):
    if orientation=='horizontal':
        rotation=0
        pad=0.02
        labelpad=15
        labelsize=8
    elif orientation=='vertical':
        rotation=-90
        labelpad=25

    cbar  = fig.colorbar(mappable=map,
                         cax=cax,
                         ax=ax,
                         orientation=orientation,
                        #  location='right',
                         extend='both',
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
    return cbar


# Cell 4.
# =================| dir |====================================================

'''
dir_data is a variable storing the path to a directory containing satellite data files relating
MODIS aerosol optical depth (AOD)
'''
# dir_data = '/DATA/satellite/05_viirs/SNPP/v002/L2/aod/D010' #make this the path from the email.
# #!ls $dir_data #running just this line prints all the data files corresponding to the above path.

# dir_data2 = '/DATA2/reanalysis/EAC4/2D'
# #!ls $dir_data2

# dir_data3 = '/DATA2/reanalysis/MERRA2/tavg1_2d_aer_Nx/raw'
# #!ls $dir_data3

# dir_data4 = '/DATA2/reanalysis/NAAPS/6h'
# #!ls $dir_data3

meera2_base = '/DATA/tmp/kaden/data/10_pyFillGrid_VIIRS_MERRA2'
#! ls /DATA/tmp/kaden/data/10_pyFillGrid_VIIRS_MERRA2
meera2_reanalysis = '/DATA/tmp/kaden/data/00_match_VIIRS_MERRA2_aod'
#! ls /DATA/tmp/kaden/data/00_match_VIIRS_MERRA2_aod

eac4_base = '/DATA/tmp/kaden/data/11_pyFillGrid_VIIRS_EAC4'
#! ls /DATA/tmp/kaden/data/11_pyFillGrid_VIIRS_EAC4
eac4_reanalysis = '/DATA/tmp/kaden/data/01_match_VIIRS_EAC4_aod'
#! ls /DATA/tmp/kaden/data/01_match_VIIRS_EAC4_aod

naaps_base = '/DATA/tmp/kaden/data/12_pyFillGrid_VIIRS_NAAPS'
#! ls /DATA/tmp/kaden/data/12_pyFillGrid_VIIRS_NAAPS
naaps_reanalysis = '/DATA/tmp/kaden/data/02_match_VIIRS_NAAPS_aod'
#! ls /DATA/tmp/kaden/data/02_match_VIIRS_NAAPS_aod


# 00_match_VIIRS_MERRA2_aod	02_match_VIIRS_NAAPS_aod_best
# 00_match_VIIRS_MERRA2_aod_best	10_pyFillGrid_VIIRS_MERRA2
# 01_match_VIIRS_EAC4_aod		11_pyFillGrid_VIIRS_EAC4
# 01_match_VIIRS_EAC4_aod_best	12_pyFillGrid_VIIRS_NAAPS
# 02_match_VIIRS_NAAPS_aod




'''
fil_list is a variable that uses the api.file_list() function from the xmac package to create a
list of file paths. These paths correspond to MODIS AOD data files from the years 2000 to 2021.
The ??? and * in the file pattern are likely used as placeholders for specific file naming
conventions
'''
# fil_list = api.file_list( f'{dir_data}/AERDB_L2_VIIRS_SNPP.202006{{14..28}}*' ) #the part after {dir_data} is the file name until the date...the second set of curly braces houses the days of data.
# #fil_list #running just this single prints the data specified above in fil_list.

# fil_list2 = api.file_list( f'{dir_data2}/EAC4_D_202006{{14..28}}*' ) #the part after {dir_data} is the file name until the date...the second set of curly braces houses the days of data.

# fil_list3 = api.file_list( f'{dir_data3}/MERRA2_400.tavg1_2d_aer_Nx.202006{{14..28}}*' )

# fil_list4 = api.file_list( f'{dir_data4}/202006{{14..28}}12*' ) #added 12 for the noon hour

meera2_base_fil_list =  api.file_list( f'{meera2_base}/AERDB_L2_VIIRS_SNPP.202006{{14..28}}*' )
meera2_reanalysis_fil_list = api.file_list( f'{meera2_reanalysis}/MERRA2_400.tavg1_2d_aer_Nx.202006{{14..28}}*' )

eac4_base_fil_list =  api.file_list( f'{eac4_base}/AERDB_L2_VIIRS_SNPP.202006{{14..28}}*' )
eac4_reanalysis_fil_list = api.file_list( f'{eac4_reanalysis}/EAC4_D_202006{{14..28}}*' )

naaps_base_fil_list =  api.file_list( f'{naaps_base}/AERDB_L2_VIIRS_SNPP.202006{{14..28}}*' )
naaps_reanalysis_fil_list = api.file_list( f'{naaps_reanalysis}/202006{{14..28}}*' )

'''
This line appears to calculate and possibly print the sizes of the files listed in fil_list. It
likely uses a function provided by the api module in the xmac package.
'''
# api.file_size(fil_list)
# api.file_size(fil_list2)
# api.file_size(fil_list3)
# api.file_size(fil_list4)

api.file_size(meera2_base_fil_list)
api.file_size(meera2_reanalysis_fil_list)

api.file_size(eac4_base_fil_list)
api.file_size(eac4_reanalysis_fil_list)

api.file_size(naaps_base_fil_list)
api.file_size(naaps_reanalysis_fil_list)


# =================| fil |====================================================
'''
xr.open_mfdataset(fil_list) is used to open multiple NetCDF (Network Common Data Form) files
specified in fil_list as a single xarray dataset (ds). This is a common operation for handling
multi-file datasets.
'''
# ds = xr.open_mfdataset(fil_list)
# #print(ds) #allows me to see what variables are in the dataset
# ds2 = xr.open_mfdataset(fil_list2)
# #print(ds2)
# ds3 = xr.open_mfdataset(fil_list3)
# #print(ds3)

# ds4 = xr.open_mfdataset(fil_list4, combine='nested', concat_dim='time') #the file couldn't be accessed like in the above commands so we had to add parameters to aquire the proper days and time.

# date = [fileName.split("_")[0].split("/")[-1] for fileName in fil_list4] #split the file so that we could access the time.
# ds4['time'] = pd.to_datetime(date, format='%Y%m%d%H')
# ds4['lat']  = ds4['lat'] - 89 #changed the location that appears in the subplots.
# ds4['lon']  = ds4['lon'] - 179

ds5 = xr.open_mfdataset(meera2_base_fil_list)
#print(ds5)
ds6 = xr.open_mfdataset(meera2_reanalysis_fil_list)
#print(ds6)

ds7 = xr.open_mfdataset(eac4_base_fil_list)
#print(ds7)
ds8 = xr.open_mfdataset(eac4_reanalysis_fil_list)
#print(ds8)

ds9 = xr.open_mfdataset(naaps_base_fil_list)
#print(ds9)
ds10 = xr.open_mfdataset(naaps_reanalysis_fil_list)
#print(ds10)


# Cell 5.
# MEERA2 satellite data
data = ds5['aod']

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,

                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
plt.subplots_adjust(hspace= -0.18, wspace=0.06)


#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.transpose().flat

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 15','June 16','June 17','June 18','June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26','June 27','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
cmap.set_under('w') # changes the background of anything that is not AOD to white (10.30.23)
#cmap

#An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # Add longitude tick marks to the left side of the first column
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 50

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)


    ax.text(-107, 3, title[idx], fontsize=10) #reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])



#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height. The first two values represent x and y of the
#entire color bar, while the next x and y represent the actual gradient bar.
cb_ax = fig.add_axes([0.22, 0.07, 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
color_bar(fig, cf_list[4], cb_ax, label='MEERA2 Satellite (aod)', orientation='horizontal', tickstep=1)

plt.show()


# Cell 6.
# MEERA2 reanalysis data

data = ds6['TOTEXTTAU']

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.transpose().flat

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 15','June 16','June 17','June 18','June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26','June 27','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
cmap.set_under('w') # changes the background of anything that is not AOD to white (10.30.23)
#cmap

#An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # Add longitude tick marks to the left side of the first column
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 50

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    ax.text(-107, 3, title[idx], fontsize=10) #reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])



#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height. The first two values represent x and y of the
#entire color bar, while the next x and y represent the actual gradient bar.
cb_ax = fig.add_axes([0.22, 0.07, 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
color_bar(fig, cf_list[4], cb_ax, label='MEERA2 Reanalysis (TOTEXTTAU)', orientation='horizontal', tickstep=1)

plt.show()


# Cell 7.
# Absolute difference between MERRA2 reanalysis and satellite

fig, axs = plt.subplots(5, 3,
                        subplot_kw={'projection': proj_map},
                        figsize=[10, 8],
                        constrained_layout=False)
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

# Convert axs to a 1D array
axs = axs.transpose().flat

# An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

# A list of subplot titles.
title = ['June 14', 'June 15', 'June 16', 'June 17', 'June 18', 'June 19', 'June 20', 'June 21', 'June 22', 'June 23', 'June 24', 'June 25', 'June 26', 'June 27', 'June 28']
time_values = ds6['TOTEXTTAU']['time'].values

# A color map (colormap) for visualizing data.
cmap_difference = plt.get_cmap('coolwarm')
difference_levs = np.arange(-1, 1.1, 0.2)
difference_norm = BoundaryNorm(difference_levs, ncolors=cmap_difference.N, extend='both')

step = 1  # alters the resolution
for idx, ax in enumerate(axs):
    # Add longitude tick marks to the left side of the first column
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 50

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    # Use .sel() with 'method' set to 'nearest' to select the nearest available date
    ds6_data = ds6['TOTEXTTAU'].sel(time=pd.to_datetime(time_values[idx]), method='nearest')
    ds5_data = ds5['aod'].sel(time=pd.to_datetime(time_values[idx]), method='nearest')
    difference = ds6_data - ds5_data
    # Configures the map settings for the current subplot.
    map_conf(ax)
    ax.text(-107, 3, title[idx], fontsize=10)  # Reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(difference.lon[::step], difference.lat[::step], difference[::step, ::step],
                              shading='auto',
                              norm=difference_norm,
                              cmap=cmap_difference,
                              transform=proj_data,
                              )]

# A color bar axis (cb_ax) is added to the figure.
cb_ax = fig.add_axes([0.21, 0.07, 0.6, 0.02])
# Specify the desired tick steps for the difference colorbar
difference_ticks = np.arange(-1, 1.1, 0.2)

# Create and customize a color bar based on the pseudocolor plots in cf_list[4].
cb = plt.colorbar(cf_list[4], cax=cb_ax, orientation='horizontal', ticks=difference_ticks, extend='both')
cb.set_label("Difference Between MEERA2 Reanalysis and MEERA2 Satellite", fontsize=14, labelpad=15)
cb.ax.tick_params(labelsize=8)  # Adjusts the font size

plt.show()


# Cell 8.
# EAC4 satellite data

data = ds7['aod']

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.transpose().flat

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 15','June 16','June 17','June 18','June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26','June 27','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
cmap.set_under('w') # changes the background of anything that is not AOD to white (10.30.23)
#cmap

#An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)


    ax.text(-107, 3, title[idx], fontsize=10) #reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])



#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height. The first two values represent x and y of the
#entire color bar, while the next x and y represent the actual gradient bar.
cb_ax = fig.add_axes([0.22, 0.07, 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
color_bar(fig, cf_list[4], cb_ax, label='EAC4 Satellite (aod)', orientation='horizontal', tickstep=1)

plt.show()


# Cell 9.
# EAC4 reanalysis data

# Rename latitude and longitude to avoid dimension errors.
ds8 = ds8.rename({'longitude': 'lon', 'latitude': 'lat'})
data = ds8['aod550']

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.transpose().flat

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 15','June 16','June 17','June 18','June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26','June 27','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
cmap.set_under('w') # changes the background of anything that is not AOD to white (10.30.23)
#cmap

#An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    ax.text(-107, 3, title[idx], fontsize=10) #reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])



#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height. The first two values represent x and y of the
#entire color bar, while the next x and y represent the actual gradient bar.
cb_ax = fig.add_axes([0.22, 0.07, 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
color_bar(fig, cf_list[4], cb_ax, label='EAC4 Reanalysis (aod550)', orientation='horizontal', tickstep=1)

plt.show()


# Cell 10.
# Absolute difference between EAC4 reanalysis and satellite

fig, axs = plt.subplots(5, 3,
                        subplot_kw={'projection': proj_map},
                        figsize=[10, 8],
                        constrained_layout=False)
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

# Convert axs to a 1D array
axs = axs.transpose().flat

# An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

# A list of subplot titles.
title = ['June 14', 'June 15', 'June 16', 'June 17', 'June 18', 'June 19', 'June 20', 'June 21', 'June 22', 'June 23', 'June 24', 'June 25', 'June 26', 'June 27', 'June 28']

# A color map (colormap) for visualizing data.
cmap_difference = plt.get_cmap('coolwarm')
difference_levs = np.arange(-1, 1.1, 0.2)
difference_norm = BoundaryNorm(difference_levs, ncolors=cmap_difference.N, extend='both')

step = 1  # alters the resolution
for idx, ax in enumerate(axs):
    # Add longitude tick marks to the left side of the first column
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 50

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    # Use .sel() with 'method' set to 'nearest' to select the nearest available date
    ds8_data = ds8['aod550'].sel(time=pd.to_datetime(time_values[idx]), method='nearest')
    ds7_data = ds7['aod'].sel(time=pd.to_datetime(time_values[idx]), method='nearest')
    difference = ds8_data - ds7_data
    # Configures the map settings for the current subplot.
    map_conf(ax)
    ax.text(-107, 3, title[idx], fontsize=10)  # Reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(difference.lon[::step], difference.lat[::step], difference[::step, ::step],
                              shading='auto',
                              norm=difference_norm,
                              cmap=cmap_difference,
                              transform=proj_data,
                              )]

    # Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
        ax.set_xticklabels([])

    # Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx == 4:
        ax.set_yticklabels([])

# A color bar axis (cb_ax) is added to the figure.
cb_ax = fig.add_axes([0.21, 0.07, 0.6, 0.02])

# Create and customize a color bar based on the pseudocolor plots in cf_list[4].
cb = plt.colorbar(cf_list[4], cax=cb_ax, orientation='horizontal', ticks=difference_ticks, extend='both')
cb.set_label("Difference Between EAC4 Reanalysis and EAC4 Satellite", fontsize=14, labelpad=15)
cb.ax.tick_params(labelsize=8)  # Adjusts the font size

plt.show()


# Cell 11.
# NAAPS satellite data

data = ds9['aod']

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.transpose().flat

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 15','June 16','June 17','June 18','June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26','June 27','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
cmap.set_under('w') # changes the background of anything that is not AOD to white (10.30.23)
#cmap

#An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    ax.text(-107, 3, title[idx], fontsize=10) #reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])

#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height. The first two values represent x and y of the
#entire color bar, while the next x and y represent the actual gradient bar.
cb_ax = fig.add_axes([0.22, 0.07, 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
color_bar(fig, cf_list[4], cb_ax, label='NAAPS Satellite (aod)', orientation='horizontal', tickstep=1)

plt.show()


# Cell 12.
# NAAPS reanalysis data

# Rename latitude and longitude to avoid dimension errors.
data = ds10['total_aod']

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.transpose().flat

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 15','June 16','June 17','June 18','June 19','June 20','June 21','June 22','June 23','June 24','June 25','June 26','June 27','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
cmap.set_under('w') # changes the background of anything that is not AOD to white (10.30.23)
#cmap

#An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    ax.text(-107, 3, title[idx], fontsize=10) #reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])

#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height. The first two values represent x and y of the
#entire color bar, while the next x and y represent the actual gradient bar.
cb_ax = fig.add_axes([0.22, 0.07, 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
color_bar(fig, cf_list[4], cb_ax, label='NAAPS Reanalysis (total_aod)', orientation='horizontal', tickstep=1)

plt.show()


# Cell 13.
# Absolute difference between NAAPS reanalysis and satellite

fig, axs = plt.subplots(5, 3,
                        subplot_kw={'projection': proj_map},
                        figsize=[10, 8],
                        constrained_layout=False)
plt.subplots_adjust(hspace= -0.18, wspace=0.06)

# Convert axs to a 1D array
axs = axs.transpose().flat

# An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

# A list of subplot titles.
title = ['June 14', 'June 15', 'June 16', 'June 17', 'June 18', 'June 19', 'June 20', 'June 21', 'June 22', 'June 23', 'June 24', 'June 25', 'June 26', 'June 27', 'June 28']

# A color map (colormap) for visualizing data.
cmap_difference = plt.get_cmap('coolwarm')
difference_levs = np.arange(-1, 1.1, 0.2)
difference_norm = BoundaryNorm(difference_levs, ncolors=cmap_difference.N, extend='both')

step = 1  # alters the resolution
for idx, ax in enumerate(axs):
    # Add longitude tick marks to the left side of the first column
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 50

    # Y-axis
    if idx == 0:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 1:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 2:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 3:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)
    if idx == 4:
        ax.set_yticks(np.arange(minLat, maxLat + 1, 5))  # Adjust the tick interval as needed
        ax.set_yticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lat in range(minLat, maxLat + 1, 10):
            if lat == 0:
                ax.text(minLon - 2, lat, f'{lat} ', ha='right', va='center', fontsize=9)
            else:
                ax.text(minLon - 2, lat, f'{lat}°N ', ha='right', va='center', fontsize=9)

    # X-axis
    if idx == 4:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 9:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)
    if idx == 14:
        ax.set_xticks(np.arange(minLon, maxLon + 1, 10))  # Adjust the tick interval as needed
        ax.set_xticklabels([])  # Remove tick labels to avoid overlap with the plot
        # Add text to the left of tick marks
        for lon in range(-90, 1, 30):
            if lon == 0:
                ax.text(lon, minLat - 4, f'{lon} ', ha='center', va='top', fontsize=9)
                break
            ax.text(lon, minLat - 4, f'{abs(lon)}°W ', ha='center', va='top', fontsize=9)

    # Use .sel() with 'method' set to 'nearest' to select the nearest available date
    ds10_data = ds10['total_aod'].sel(time=pd.to_datetime(time_values[idx]), method='nearest')
    ds9_data = ds9['aod'].sel(time=pd.to_datetime(time_values[idx]), method='nearest')
    difference = ds10_data - ds9_data
    # Configures the map settings for the current subplot.
    map_conf(ax)
    ax.text(-107, 3, title[idx], fontsize=10)  # Reformats where the data shows up on the figure.
    cf_list += [ax.pcolormesh(difference.lon[::step], difference.lat[::step], difference[::step, ::step],
                              shading='auto',
                              norm=difference_norm,
                              cmap=cmap_difference,
                              transform=proj_data,
                              )]

    # Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
        ax.set_xticklabels([])

    # Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx == 4:
        ax.set_yticklabels([])

# A color bar axis (cb_ax) is added to the figure.
cb_ax = fig.add_axes([0.21, 0.07, 0.6, 0.02])

# Create and customize a color bar based on the pseudocolor plots in cf_list[4].
cb = plt.colorbar(cf_list[4], cax=cb_ax, orientation='horizontal', ticks=difference_ticks, extend='both')
cb.set_label("Difference Between NAAPS Reanalysis and NAAPS Satellite", fontsize=14, labelpad=15)
cb.ax.tick_params(labelsize=8)  # Adjusts the font size

plt.show()
