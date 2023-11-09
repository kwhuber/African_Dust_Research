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
def color_bar(fig, map, cax=None, ax=None, label='color bar label', orientation='horizontal', tickstep=1):
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

# Cell 4.
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
fil_list = api.file_list( f'{dir_data}/AERDB_L2_VIIRS_SNPP.202006{{14..28}}*' ) #the part after {dir_data} is the file name until the date...the second set of curly braces houses the days of data.
#fil_list #running just this single prints the data specified above in fil_list.

fil_list2 = api.file_list( f'{dir_data2}/EAC4_D_202006{{14..28}}*' ) #the part after {dir_data} is the file name until the date...the second set of curly braces houses the days of data.

fil_list3 = api.file_list( f'{dir_data3}/MERRA2_400.tavg1_2d_aer_Nx.202006{{14..28}}*' )

fil_list4 = api.file_list( f'{dir_data4}/202006{{14..28}}12*' ) #added 12 for the noon hour

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
api.file_size(fil_list)
api.file_size(fil_list2)
api.file_size(fil_list3)
api.file_size(fil_list4)

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
ds = xr.open_mfdataset(fil_list)
#print(ds) #allows me to see what variables are in the dataset
ds2 = xr.open_mfdataset(fil_list2)
#print(ds2)
ds3 = xr.open_mfdataset(fil_list3)
#print(ds3)

ds4 = xr.open_mfdataset(fil_list4, combine='nested', concat_dim='time') #the file couldn't be accessed like in the above commands so we had to add parameters to aquire the proper days and time.

date = [fileName.split("_")[0].split("/")[-1] for fileName in fil_list4] #split the file so that we could access the time.
ds4['time'] = pd.to_datetime(date, format='%Y%m%d%H')
ds4['lat']  = ds4['lat'] - 89 #changed the location that appears in the subplots.
ds4['lon']  = ds4['lon'] - 179

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
# Plotting the difference between MEERA2 Reanalysis and MEERA2 Satellite

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()

# Configure cmap for ds5 and ds6 datasets
cmap = plt.get_cmap('YlOrBr')
cmap.set_under('w')

# Configure cmap for difference
cmap_difference=plt.get_cmap('coolwarm')

# An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]
difference_levs = [-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0]

# A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')
difference_norm = BoundaryNorm(difference_levs, ncolors=cmap_difference.N, extend='both')

# Get the time dimension from the dataset
time_values = ds6['TOTEXTTAU']['time'].values

# Calculate the number of rows and columns
num_rows = len(time_values)
num_columns = 3

# Create a figure with subplots
fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows), subplot_kw={'projection': proj_map})

# Iterate over each day
for i, date_str in enumerate(pd.to_datetime(time_values)):
    # Use .sel() with 'method' set to 'nearest' to select the nearest available date
    ds6_data = ds6['TOTEXTTAU'].sel(time=date_str, method='nearest')
    ds5_data = ds5['aod'].sel(time=date_str, method='nearest')
    difference = ds6_data - ds5_data

    # Configure map settings for the difference subplot (Left)
    map_conf(axs[i, 0])
    cax = axs[i, 0].pcolormesh(difference.lon, difference.lat, difference, cmap=cmap_difference, transform=proj_data, norm=difference_norm)
    axs[i, 0].set_title(f'Difference on {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Configure map settings for ds5 (Middle)
    map_conf(axs[i, 1])
    cax_ds5 = axs[i, 1].pcolormesh(ds5_data.lon, ds5_data.lat, ds5_data, shading='auto', norm=norm, cmap=cmap, transform=proj_data)
    axs[i, 1].set_title(f'MEERA2 Satellite\n {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Configure map settings for ds6 (Right)
    map_conf(axs[i, 2])
    cax_ds6 = axs[i, 2].pcolormesh(ds6_data.lon, ds6_data.lat, ds6_data, shading='auto', norm=norm, cmap=cmap, transform=proj_data)
    axs[i, 2].set_title(f'MEERA2 Reanalysis\n {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Add a colorbar for the difference subplot
    cb = plt.colorbar(cax, orientation='horizontal', pad=0.1, ax=axs[i, 0], extend='both')
    cb.set_label("Difference")

    # Add a colorbar for the MEERA2 Satellite (ds5) subplot
    cb2 = plt.colorbar(cax_ds5, orientation='horizontal', pad=0.1, ax=axs[i, 1])
    cb2.set_label("AOD")

    # Add a colorbar for the MEERA2 Reanalysis (ds6) subplot
    cb3 = plt.colorbar(cax_ds6, orientation='horizontal', pad=0.1, ax=axs[i, 2])
    cb3.set_label("TOTEXTTAU")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# Cell 6. 
# Plotting the difference between EAC4 Reanalysis and EAC4 Satellite

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()

# Configure cmap for ds5 and ds6 datasets
cmap = plt.get_cmap('YlOrBr')
cmap.set_under('w')

# Configure cmap for difference
cmap_difference=plt.get_cmap('coolwarm')

# An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]
difference_levs = [-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0]

# A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')
difference_norm = BoundaryNorm(difference_levs, ncolors=cmap_difference.N, extend='both')

# Get the time dimension from the dataset
time_values = ds8['aod550']['time'].values

# Calculate the number of rows and columns
num_rows = len(time_values)
num_columns = 3

# Create a figure with subplots
fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows), subplot_kw={'projection': proj_map})

# Rename latitude and longitude to avoid dimension errors.
ds8 = ds8.rename({'longitude': 'lon', 'latitude': 'lat'})

# Iterate over each day
for i, date_str in enumerate(pd.to_datetime(time_values)):
    # Use .sel() with 'method' set to 'nearest' to select the nearest available date
    ds8_data = ds8['aod550'].sel(time=date_str, method='nearest')
    ds7_data = ds7['aod'].sel(time=date_str, method='nearest')
    difference = ds8_data - ds7_data

    # Configure map settings for the difference subplot (Left)
    map_conf(axs[i, 0])
    cax = axs[i, 0].pcolormesh(difference.lon, difference.lat, difference, cmap=cmap_difference, transform=proj_data, norm=difference_norm)
    axs[i, 0].set_title(f'Difference on {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Configure map settings for ds5 (Middle)
    map_conf(axs[i, 1])
    cax_ds7 = axs[i, 1].pcolormesh(ds7_data.lon, ds7_data.lat, ds7_data, shading='auto', norm=norm, cmap=cmap, transform=proj_data)
    axs[i, 1].set_title(f'EAC4 Satellite\n {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Configure map settings for ds6 (Right)
    map_conf(axs[i, 2])
    cax_ds8 = axs[i, 2].pcolormesh(ds8_data.lon, ds8_data.lat, ds8_data, shading='auto', norm=norm, cmap=cmap, transform=proj_data)
    axs[i, 2].set_title(f'EAC4 Reanalysis\n {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Add a colorbar for the difference subplot
    cb = plt.colorbar(cax, orientation='horizontal', pad=0.1, ax=axs[i, 0], extend='both')
    cb.set_label("Difference")

    # Add a colorbar for the MEERA2 Satellite (ds5) subplot
    cb2 = plt.colorbar(cax_ds7, orientation='horizontal', pad=0.1, ax=axs[i, 1])
    cb2.set_label("AOD")

    # Add a colorbar for the MEERA2 Reanalysis (ds6) subplot
    cb3 = plt.colorbar(cax_ds8, orientation='horizontal', pad=0.1, ax=axs[i, 2])
    cb3.set_label("AOD550")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Cell 7. 
# Plotting the difference between NAAPS Reanalysis and NAAPS Satellite

proj_map = ccrs.PlateCarree()
proj_data = ccrs.PlateCarree()

# Configure cmap for ds5 and ds6 datasets
cmap = plt.get_cmap('YlOrBr')
cmap.set_under('w')

# Configure cmap for difference
cmap_difference=plt.get_cmap('coolwarm')

# An array of levels for data contouring.
levs = [0.4, 0.7, 1.0, 1.5, 2, 2.5, 3]
difference_levs = [-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0]

# A normalization object that maps data values to colors based on the defined levels.
norm = BoundaryNorm(levs, ncolors=cmap.N, extend='both')
difference_norm = BoundaryNorm(difference_levs, ncolors=cmap_difference.N, extend='both')

# Get the time dimension from the dataset
time_values = ds10['total_aod']['time'].values

# Calculate the number of rows and columns
num_rows = len(time_values)
num_columns = 3

# Create a figure with subplots
fig, axs = plt.subplots(num_rows, num_columns, figsize=(15, 5 * num_rows), subplot_kw={'projection': proj_map})

# Iterate over each day
for i, date_str in enumerate(pd.to_datetime(time_values)):
    # Use .sel() with 'method' set to 'nearest' to select the nearest available date
    ds10_data = ds10['total_aod'].sel(time=date_str, method='nearest')
    ds9_data = ds9['aod'].sel(time=date_str, method='nearest')
    difference = ds10_data - ds9_data

    # Configure map settings for the difference subplot (Left)
    map_conf(axs[i, 0])
    cax = axs[i, 0].pcolormesh(difference.lon, difference.lat, difference, cmap=cmap_difference, transform=proj_data, norm=difference_norm)
    axs[i, 0].set_title(f'Difference on {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Configure map settings for ds5 (Middle)
    map_conf(axs[i, 1])
    cax_ds9 = axs[i, 1].pcolormesh(ds9_data.lon, ds9_data.lat, ds9_data, shading='auto', norm=norm, cmap=cmap, transform=proj_data)
    axs[i, 1].set_title(f'NAAPS Satellite\n {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Configure map settings for ds6 (Right)
    map_conf(axs[i, 2])
    cax_ds10 = axs[i, 2].pcolormesh(ds10_data.lon, ds10_data.lat, ds10_data, shading='auto', norm=norm, cmap=cmap, transform=proj_data)
    axs[i, 2].set_title(f'NAAPS Reanalysis\n {date_str.strftime("%Y-%m-%d")}', fontsize=10)

    # Add a colorbar for the difference subplot
    cb = plt.colorbar(cax, orientation='horizontal', pad=0.1, ax=axs[i, 0], extend='both')
    cb.set_label("Difference")

    # Add a colorbar for the MEERA2 Satellite (ds5) subplot
    cb2 = plt.colorbar(cax_ds9, orientation='horizontal', pad=0.1, ax=axs[i, 1])
    cb2.set_label("AOD")

    # Add a colorbar for the MEERA2 Reanalysis (ds6) subplot
    cb3 = plt.colorbar(cax_ds10, orientation='horizontal', pad=0.1, ax=axs[i, 2])
    cb3.set_label("TOTAL_AOD")

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
