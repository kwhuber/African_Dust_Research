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
# MEERA data to be used in difference comparison .
fil = '/DATA/tmp/kaden/data/00_match_VIIRS_MERRA2_aod/MERRA2*20200615*'
ds0 = xr.open_mfdataset(fil)
ds0['TOTEXTTAU'][0].plot()

# Cell 3.
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

# Cell 4. 
'''
This function is desinged to configure the settings of a Cartopy map.
'''
def map_conf(ax):
    minLon = -110; maxLon = 10
    minLat = 0;  maxLat = 40

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
'''
This function is used for creating a color bar legend for a map. It takes several parameters,
including the map (the color-mapped object you want to create a legend for), cax (the color bar
axis), ax (the axis where the color bar will be drawn), label (the label for the color bar),
orientation (horizontal or vertical), and tickstep (spacing between color bar ticks).

The function sets up the color bar with appropriate labels, size, rotation, and tick parameters
based on the specified orientation.

At the end, the created color bar object is retured.
'''
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

# Cell 5.
# =================| dir |====================================================

'''
dir_data is a variable storing the path to a directory containing satellite data files relating
MODIS aerosol optical depth (AOD)
'''
dir_data = '/DATA/satellite/05_viirs/SNPP/v002/L2/aod/D010' #make this the path from the email.
#!ls $dir_data #running just this line prints all the data files corresponding to the above path.

dir_data2 = '/DATA2/reanalysis/EAC4/2D'
#!ls $dir_data2

dir_data3 = '/DATA2/reanalysis/MERRA2/tavg1_2d_aer_Nx/raw'
#!ls $dir_data3

dir_data4 = '/DATA2/reanalysis/NAAPS/6h'
#!ls $dir_data3

'''
fil_list is a variable that uses the api.file_list() function from the xmac package to create a
list of file paths. These paths correspond to MODIS AOD data files from the years 2000 to 2021.
The ??? and * in the file pattern are likely used as placeholders for specific file naming
conventions
'''
fil_list = api.file_list( f'{dir_data}/AERDB_L2_VIIRS_SNPP.202006{{14..28}}*' ) #the part after {dir_data} is the file name until the date...the second set of curly braces houses the days of data.
#fil_list #running just this single prints the data specified above in fil_list.

fil_list2 = api.file_list( f'{dir_data2}/EAC4_D_202006{{14..28}}*' ) #the part after {dir_data} is the file name until the date...the second set of curly braces houses the days of data.

fil_list3 = api.file_list( f'{dir_data3}/MERRA2_400.tavg1_2d_aer_Nx.202006{{14..28}}*' )

fil_list4 = api.file_list( f'{dir_data4}/202006{{14..28}}12*' ) #added 12 for the noon hour

'''
This line appears to calculate and possibly print the sizes of the files listed in fil_list. It
likely uses a function provided by the api module in the xmac package.
'''
api.file_size(fil_list)
api.file_size(fil_list2)
api.file_size(fil_list3)
api.file_size(fil_list4)


# =================| fil |====================================================
'''
xr.open_mfdataset(fil_list) is used to open multiple NetCDF (Network Common Data Form) files
specified in fil_list as a single xarray dataset (ds). This is a common operation for handling
multi-file datasets.
'''
ds = xr.open_mfdataset(fil_list)
print(ds)
ds2 = xr.open_mfdataset(fil_list2)
print(ds2) #allows me to see what variables are in the dataset
ds3 = xr.open_mfdataset(fil_list3)
print(ds3)
ds4 = xr.open_mfdataset(fil_list4, combine='nested', concat_dim='time') #the file couldn't be accessed like in the above commands so we had to add parameters to aquire the proper days and time.

date = [fileName.split("_")[0].split("/")[-1] for fileName in fil_list4] #split the file so that we could access the time.
ds4['time'] = pd.to_datetime(date, format='%Y%m%d%H')
ds4['lat']  = ds4['lat'] - 89 #changed the location that appears in the subplots.
ds4['lon']  = ds4['lon'] - 179
print(ds4)

#difference1 = ds['aod'] - ds2['aod'] #VIIRS - MEERA2 differnece...plot each data set (2 figures) then their difference (figure 3)...do this for the other 2 comparisons (so 9 figure total)


# =================| ssn |====================================================
'''
xstats.month_to_season_mean(ds) calculates seasonal means from the dataset ds. It appears to
aggregate (sum together) the data from monthly to seasonal values. ssn could stand for Seasonal
Mean.
'''
#ssn = xstats.month_to_season_mean(ds)

'''
xstats.area_mean(ssn) is used to calculate the global average of the data stored in the ssn dataset.
This likely means averaging the data values over the entire globe.
'''
#globe_average = xstats.area_mean(ssn)

#a variable that holds the name of aerosol optical depth.
#varN = 'aod'

#extracts the variable specified by varN from the ssn dataset, so it can be manipulated.
#data = ssn[varN]

'''
Reassigns globe_average to be the extracted variable, so you can work with it separately from
original dataset.
'''
#globe_average = globe_average[varN]

# Cell 6 (first dataset).
data = ds['aod'] #can use the variable aod_best for different resolution of data.

'''
Defines a variable proj_map and assigns it the map projection Robinson from the Cartopy library
(ccrs). The Robinson projection is a world map projection often used for global visualizations.
'''
#proj_map  = ccrs.Robinson()
proj_map = ccrs.PlateCarree()

'''
Defines a variable proj_data and assigns it to the PlateCarree map projection. This projection
represents a simple, equidistant cylindrical projection often used for displaying data on maps.

'''
proj_data = ccrs.PlateCarree()


#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig, axs = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.01},#When is adjusted wspace it did not make the subplots come closer together widthwise???
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs)

# =================| contour |=================================================

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs = axs.flat

#The first subplot (at index 0) is removed with axs[0].remove(). This is likely done to create a specific layout for the subplots.
#axs[0].remove()

#The resulting axs now contains references to the remaining subplots (excluding the first one).
#axs = axs[1:]

#The following five lines configure the subplots and are used to customize appearance of the subplots.

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list = []

#A list of subplot titles.
title = ['June 14','June 19','June 24','June 15','June 20','June 25','June 16','June 21','June 26','June 17','June 22','June 27','June 18','June 23','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
#cmap

#An array of levels for data contouring.
levs = np.linspace(0, 1, 10*5+1)

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='max')

'''
This line initiates a loop over the subplots, where idx represents the index of the subplot in the
array, and ax is the reference to the current subplot being processed.
'''

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx, ax in enumerate(axs):
    #Configures the maps settings for the current subplot.
    map_conf(ax)

    # cf_list += [ax.contourf(data.lon, data.lat, data[idx, :, :],
    #                        levels=np.linspace(0., 0.5, 5*5+1),
    #                        cmap=cmap,
    #                        extend='max',
    #                        transform=proj_data,
    #                        )]

    #Sets the title for the current subplot based on the title list, with the title text and font size customized.
    # ax.set_title(title[idx], fontsize=5, pad=5)
    ax.text(-110, 5, title[idx], fontsize=8) #reformats where the data shows up on the figure.

    '''
    For each subplot, this code creates a pseudocolor plot (pcolormesh) using data from the variable
    data. The data is plotted on the subplot's map projection (proj_data). Parameters like shading,
    norm, and cmap are used to configure the appearance of the pseudocolor plot. The resulting
    pseudocolor plot object is added to the cf_list.
    '''
    cf_list += [ax.pcolormesh(data.lon[::step], data.lat[::step], data[idx, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    '''
    These lines customize the appearance of the subplots further.
    '''
    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx <= 2:
            ax.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx == 2 or idx==4:
            ax.set_yticklabels([])

    '''
    Text annotations are added to the bottom of each subplot to display numeric values
    (likely global averages) with blue color and customized alignment.
    '''
    #ax.text(0, -55, '{:0.3f}'.format(globe_average[idx].values),
            #fontsize=12, color='blue',
            #verticalalignment='center', horizontalalignment='center',
            #transform=proj_data,
            #)

#This line adjusts the layout and spacing of the subplots within the figure, setting properties such as margin sizes and spacing between subplots.
#fig.subplots_adjust(bottom=0.02, top=0.98, left=0.1, right=0.8, wspace=0.02, hspace=0.05)

#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height.
#cb_ax = fig.add_axes([0.15, -0.05 , 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
#color_bar(cf_list[4], cb_ax, label='MODIS-Terra AOD$_{550}$', orientation='horizontal', tickstep=10)

# =================| subplot adjust |==========================================
'''
This line extracts the position information of the first subplot (axs[0]), and assigns it to the
variable box. The get_position() method retrieves the position of the subplot within the figure.
'''
#box = axs[0].get_position()

#Shifts the position of the first subplot to the left by 0.205 units, changing its horizontal position in the figure.
#box.x0 = box.x0 - 0.205
#box.x1 = box.x1 - 0.205

#Sets the position of the first subplot to the modified box.
#axs[0].set_position(box)

#Displays the updated figure with the adjusted subplot layout using Matplotlib's plt.show() funtion.
plt.show()

# Cell 7 (second dataset).
data2 = ds2['aod550']
#print(data2) #allows me to see variables like lon/lat or longitude/latitude

'''
Defines a variable proj_map and assigns it the map projection Robinson from the Cartopy library
(ccrs). The Robinson projection is a world map projection often used for global visualizations.
'''
#proj_map  = ccrs.Robinson()
proj_map = ccrs.PlateCarree()

'''
Defines a variable proj_data and assigns it to the PlateCarree map projection. This projection
represents a simple, equidistant cylindrical projection often used for displaying data on maps.

'''
proj_data = ccrs.PlateCarree()

#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig2, axs2 = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.01},#When is adjusted wspace it did not make the subplots come closer together widthwise???
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs2)

# =================| contour |=================================================

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs2 = axs2.flat

#The first subplot (at index 0) is removed with axs[0].remove(). This is likely done to create a specific layout for the subplots.
#axs[0].remove()

#The resulting axs now contains references to the remaining subplots (excluding the first one).
#axs = axs[1:]

#The following five lines configure the subplots and are used to customize appearance of the subplots.

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list2 = []

#A list of subplot titles.
title = ['June 14','June 19','June 24','June 15','June 20','June 25','June 16','June 21','June 26','June 17','June 22','June 27','June 18','June 23','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
#cmap

#An array of levels for data contouring.
levs = np.linspace(0, 1, 10*5+1)

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='max')

'''
This line initiates a loop over the subplots, where idx represents the index of the subplot in the
array, and ax is the reference to the current subplot being processed.
'''

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.


for idx2, ax2 in enumerate(axs2):
    #Configures the maps settings for the current subplot.
    map_conf(ax2)

    # cf_list += [ax.contourf(data.lon, data.lat, data[idx, :, :],
    #                        levels=np.linspace(0., 0.5, 5*5+1),
    #                        cmap=cmap,
    #                        extend='max',
    #                        transform=proj_data,
    #                        )]

    #Sets the title for the current subplot based on the title list, with the title text and font size customized.
    # ax.set_title(title[idx], fontsize=5, pad=5)
    ax2.text(-110, 5, title[idx2], fontsize=8) #reformats where the data shows up on the figure.

    '''
    For each subplot, this code creates a pseudocolor plot (pcolormesh) using data from the variable
    data. The data is plotted on the subplot's map projection (proj_data). Parameters like shading,
    norm, and cmap are used to configure the appearance of the pseudocolor plot. The resulting
    pseudocolor plot object is added to the cf_list.
    '''
    cf_list2 += [ax2.pcolormesh(data2.longitude[::step], data2.latitude[::step], data2[idx2, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    '''
    These lines customize the appearance of the subplots further.
    '''
    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx2 <= 2:
            ax2.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx2 == 2 or idx2==4:
            ax2.set_yticklabels([])

    '''
    Text annotations are added to the bottom of each subplot to display numeric values
    (likely global averages) with blue color and customized alignment.
    '''
    #ax.text(0, -55, '{:0.3f}'.format(globe_average[idx].values),
            #fontsize=12, color='blue',
            #verticalalignment='center', horizontalalignment='center',
            #transform=proj_data,
            #)

#This line adjusts the layout and spacing of the subplots within the figure, setting properties such as margin sizes and spacing between subplots.
#fig.subplots_adjust(bottom=0.02, top=0.98, left=0.1, right=0.8, wspace=0.02, hspace=0.05)

#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height.
#cb_ax = fig.add_axes([0.15, -0.05 , 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
#color_bar(cf_list[4], cb_ax, label='MODIS-Terra AOD$_{550}$', orientation='horizontal', tickstep=10)

# =================| subplot adjust |==========================================
'''
This line extracts the position information of the first subplot (axs[0]), and assigns it to the
variable box. The get_position() method retrieves the position of the subplot within the figure.
'''
#box = axs[0].get_position()

#Shifts the position of the first subplot to the left by 0.205 units, changing its horizontal position in the figure.
#box.x0 = box.x0 - 0.205
#box.x1 = box.x1 - 0.205

#Sets the position of the first subplot to the modified box.
#axs[0].set_position(box)

#Displays the updated figure with the adjusted subplot layout using Matplotlib's plt.show() funtion.
plt.show()

# Cell 8 (third dataset).
data3 = ds3['TOTEXTTAU']

'''
Defines a variable proj_map and assigns it the map projection Robinson from the Cartopy library
(ccrs). The Robinson projection is a world map projection often used for global visualizations.
'''
#proj_map  = ccrs.Robinson()
proj_map = ccrs.PlateCarree()

'''
Defines a variable proj_data and assigns it to the PlateCarree map projection. This projection
represents a simple, equidistant cylindrical projection often used for displaying data on maps.

'''
proj_data = ccrs.PlateCarree()

#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig3, axs3 = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.01},#When is adjusted wspace it did not make the subplots come closer together widthwise???
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs3)

# =================| contour |=================================================

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs3 = axs3.flat

#The first subplot (at index 0) is removed with axs[0].remove(). This is likely done to create a specific layout for the subplots.
#axs[0].remove()

#The resulting axs now contains references to the remaining subplots (excluding the first one).
#axs = axs[1:]

#The following five lines configure the subplots and are used to customize appearance of the subplots.

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list3 = []

#A list of subplot titles.
title = ['June 14','June 19','June 24','June 15','June 20','June 25','June 16','June 21','June 26','June 17','June 22','June 27','June 18','June 23','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
#cmap

#An array of levels for data contouring.
levs = np.linspace(0, 1, 10*5+1)

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='max')

'''
This line initiates a loop over the subplots, where idx represents the index of the subplot in the
array, and ax is the reference to the current subplot being processed.
'''

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx3, ax3 in enumerate(axs3):
    #Configures the maps settings for the current subplot.
    map_conf(ax3)

    # cf_list += [ax.contourf(data.lon, data.lat, data[idx, :, :],
    #                        levels=np.linspace(0., 0.5, 5*5+1),
    #                        cmap=cmap,
    #                        extend='max',
    #                        transform=proj_data,
    #                        )]

    #Sets the title for the current subplot based on the title list, with the title text and font size customized.
    # ax.set_title(title[idx], fontsize=5, pad=5)
    ax3.text(-110, 5, title[idx3], fontsize=8) #reformats where the data shows up on the figure.

    '''
    For each subplot, this code creates a pseudocolor plot (pcolormesh) using data from the variable
    data. The data is plotted on the subplot's map projection (proj_data). Parameters like shading,
    norm, and cmap are used to configure the appearance of the pseudocolor plot. The resulting
    pseudocolor plot object is added to the cf_list.
    '''
    cf_list3 += [ax3.pcolormesh(data3.lon[::step], data3.lat[::step], data3[idx3, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    '''
    These lines customize the appearance of the subplots further.
    '''
    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx3 <= 2:
            ax3.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx3 == 2 or idx3==4:
            ax3.set_yticklabels([])

    '''
    Text annotations are added to the bottom of each subplot to display numeric values
    (likely global averages) with blue color and customized alignment.
    '''
    #ax.text(0, -55, '{:0.3f}'.format(globe_average[idx].values),
            #fontsize=12, color='blue',
            #verticalalignment='center', horizontalalignment='center',
            #transform=proj_data,
            #)

#This line adjusts the layout and spacing of the subplots within the figure, setting properties such as margin sizes and spacing between subplots.
#fig.subplots_adjust(bottom=0.02, top=0.98, left=0.1, right=0.8, wspace=0.02, hspace=0.05)

#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height.
#cb_ax = fig.add_axes([0.15, -0.05 , 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
#color_bar(cf_list[4], cb_ax, label='MODIS-Terra AOD$_{550}$', orientation='horizontal', tickstep=10)

# =================| subplot adjust |==========================================
'''
This line extracts the position information of the first subplot (axs[0]), and assigns it to the
variable box. The get_position() method retrieves the position of the subplot within the figure.
'''
#box = axs[0].get_position()

#Shifts the position of the first subplot to the left by 0.205 units, changing its horizontal position in the figure.
#box.x0 = box.x0 - 0.205
#box.x1 = box.x1 - 0.205

#Sets the position of the first subplot to the modified box.
#axs[0].set_position(box)

#Displays the updated figure with the adjusted subplot layout using Matplotlib's plt.show() funtion.
plt.show()

# Cell 9 (fourth dataset).
data4 = ds4['total_aod']

'''
Defines a variable proj_map and assigns it the map projection Robinson from the Cartopy library
(ccrs). The Robinson projection is a world map projection often used for global visualizations.
'''
#proj_map  = ccrs.Robinson()
proj_map = ccrs.PlateCarree()

'''
Defines a variable proj_data and assigns it to the PlateCarree map projection. This projection
represents a simple, equidistant cylindrical projection often used for displaying data on maps.

'''
proj_data = ccrs.PlateCarree()
#The following block creates a figure (fig) containing a 3x2 grid of subplots (axs).
fig4, axs4 = plt.subplots(5,3,
                        #Specifies that each subplot should use the proj_map projection defined earler.
                        subplot_kw={'projection': proj_map},
                        #sets the spacing between the subplots with a small horizontal space (wsapce) and vertical space (hspace). More parameters like figsize and constrained_layout are set to customize size and layout.
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.01},#When is adjusted wspace it did not make the subplots come closer together widthwise???
                        **{'figsize': [10, 8], 'constrained_layout': False},
                        )
#Prints the axs varialbe which contains references to the individual subplots creaetd in the previous step.
print(axs4)

# =================| contour |=================================================

#These lines convert the axs object, which is initially a 2D array of subplots, into a 1D array using axs.flat.
axs4 = axs4.flat

#The first subplot (at index 0) is removed with axs[0].remove(). This is likely done to create a specific layout for the subplots.
#axs[0].remove()

#The resulting axs now contains references to the remaining subplots (excluding the first one).
#axs = axs[1:]

#The following five lines configure the subplots and are used to customize appearance of the subplots.

#An empty list that will store contourf or pseudocolor objects for each subplot.
cf_list4 = []

#A list of subplot titles.
title = ['June 14','June 19','June 24','June 15','June 20','June 25','June 16','June 21','June 26','June 17','June 22','June 27','June 18','June 23','June 28', ]

#A color map (colormap) for visualizing data.
cmap=plot.get_cmap('MPL_YlOrBr')
#cmap

#An array of levels for data contouring.
levs = np.linspace(0, 1, 10*5+1)

#A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='max')

'''
This line initiates a loop over the subplots, where idx represents the index of the subplot in the
array, and ax is the reference to the current subplot being processed.
'''

step = 1 #alters the resolution...the lower the number, the better the resolution...this variable appears in cf_list variable down below.
for idx4, ax4 in enumerate(axs4):
    #Configures the maps settings for the current subplot.
    map_conf(ax4)

    # cf_list += [ax.contourf(data.lon, data.lat, data[idx, :, :],
    #                        levels=np.linspace(0., 0.5, 5*5+1),
    #                        cmap=cmap,
    #                        extend='max',
    #                        transform=proj_data,
    #                        )]

    #Sets the title for the current subplot based on the title list, with the title text and font size customized.
    # ax.set_title(title[idx], fontsize=5, pad=5)
    ax4.text(-110, 5, title[idx4], fontsize=8) #reformats where the data shows up on the figure.

    '''
    For each subplot, this code creates a pseudocolor plot (pcolormesh) using data from the variable
    data. The data is plotted on the subplot's map projection (proj_data). Parameters like shading,
    norm, and cmap are used to configure the appearance of the pseudocolor plot. The resulting
    pseudocolor plot object is added to the cf_list.
    '''
    cf_list4 += [ax4.pcolormesh(data4.lon[::step], data4.lat[::step], data4[idx4, ::step, ::step],
                              shading='auto',
                              norm=norm,
                              cmap=cmap,
                              transform=proj_data,
                            )]

    '''
    These lines customize the appearance of the subplots further.
    '''
    #Subplots with idx less than or equal to 2 have their x-axis tick labels removed.
    if idx4 <= 2:
            ax4.set_xticklabels([])

    #Subplots with idx equal to 2 or 4 have their y-axis tick labels removed.
    if idx4 == 2 or idx4==4:
            ax4.set_yticklabels([])

    '''
    Text annotations are added to the bottom of each subplot to display numeric values
    (likely global averages) with blue color and customized alignment.
    '''
    #ax.text(0, -55, '{:0.3f}'.format(globe_average[idx].values),
            #fontsize=12, color='blue',
            #verticalalignment='center', horizontalalignment='center',
            #transform=proj_data,
            #)

#This line adjusts the layout and spacing of the subplots within the figure, setting properties such as margin sizes and spacing between subplots.
#fig.subplots_adjust(bottom=0.02, top=0.98, left=0.1, right=0.8, wspace=0.02, hspace=0.05)

#A color bar axis (cb_ax) is added to the figure. This axis is positioned at specific coordinates within the figure and has a defined width and height.
#cb_ax = fig.add_axes([0.15, -0.05 , 0.6, 0.02])

#The color_bar function is called to create and customize a color bar based on the pseudocolor plots in cf_list[4]. The color bar is configured with a label and horizontal orientation, and tick steps are specified.
#color_bar(cf_list[4], cb_ax, label='MODIS-Terra AOD$_{550}$', orientation='horizontal', tickstep=10)

# =================| subplot adjust |==========================================
'''
This line extracts the position information of the first subplot (axs[0]), and assigns it to the
variable box. The get_position() method retrieves the position of the subplot within the figure.
'''
#box = axs[0].get_position()

#Shifts the position of the first subplot to the left by 0.205 units, changing its horizontal position in the figure.
#box.x0 = box.x0 - 0.205
#box.x1 = box.x1 - 0.205

#Sets the position of the first subplot to the modified box.
#axs[0].set_position(box)

#Displays the updated figure with the adjusted subplot layout using Matplotlib's plt.show() funtion.
plt.show()

