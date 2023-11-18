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
    ax.add_feature(cfeature.COASTLINE, linewidth=0.2, )
    ax.add_feature(cfeature.LAND,      color='silver')
    ax.add_feature(cfeature.OCEAN,     color='silver')

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
    return cbar

# Cell 4. 
# =================| dir |====================================================

'''
dir_data is a variable storing the path to a directory containing satellite data files relating
MODIS aerosol optical depth (AOD)
'''
dir_data = '/DATA/satellite/05_viirs/SNPP/v002/L2/aod/D010' #make this the path from the email.
#!ls $dir_data #running just this line prints all the data files corresponding to the above path.

dir_data2 = '/DATA/tmp/kaden/data/01_match_VIIRS_EAC4_aod'
#!ls $dir_data2

dir_data3 = '/DATA/tmp/kaden/data/00_match_VIIRS_MERRA2_aod'
#!ls $dir_data3

dir_data4 = '/DATA/tmp/kaden/data/02_match_VIIRS_NAAPS_aod'
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

fil_list4 = api.file_list( f'{dir_data4}/202006{{14..28}}*' ) #added 12 for the noon hour

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
#print(ds)
ds2 = xr.open_mfdataset(fil_list2)
ds2 = ds2.rename({'longitude': 'lon', 'latitude': 'lat'})
#print(ds2) #allows me to see what variables are in the dataset
ds3 = xr.open_mfdataset(fil_list3)
#print(ds3)
ds4 = xr.open_mfdataset(fil_list4) #the file couldn't be accessed like in the above commands so we had to add parameters to aquire the proper days and time.
#print(ds4)

# Cell 5. 
# mdates module imported for date and time formatting. I use the DateFormatter class to specify format for date and time below.
import matplotlib.dates as mdates

def area_weighted_average(ds, var_name):
    weights = np.cos(np.deg2rad(ds['lat']))
    weighted_var = ds[var_name].weighted(weights)
    return weighted_var.mean(dim=('lat', 'lon'))

regions = {
    'Western North Africa': {'lat_range': (8, 25), 'lon_range': (-18, 0)},
    'Tropical North Atlantic': {'lat_range': (8, 25), 'lon_range': (-90, -18)}
}

first_region = list(regions.keys())[0]
second_region = list(regions.keys())[1]

# Create a figure with two subplots, sharing the x-axis
fig, axs = plt.subplots(2, 1, sharex=True)

# Remove vertical space between axes
fig.subplots_adjust(hspace=0.15)

# Remove x-axis for the first subplot
axs[0].xaxis.set_visible(False)

# Set ticks at every day
axs[1].xaxis.set_major_locator(mdates.DayLocator())
# Set major tick format to display month and day
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
# Rotate the tick labels for better readability
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)
# Set the fontsize for the tick labels
axs[1].xaxis.set_tick_params(labelsize=7)

# Set the y-axis range of the first subplot
axs[0].set_ylim(0.3, 1.7)
# Set the y-axis tick step of the first subplot
axs[0].set_yticks(np.arange(0.4, 1.7, 0.4))
# Set the fontsize of y-axis tick labels for the first plot
axs[0].tick_params(axis='y', labelsize=7)

# Set the y-axis range of the second subplot
axs[1].set_ylim(0.3, 1.0)
# Set the y-axis tick step of the second subplot
axs[1].set_yticks(np.arange(0.3, 1.0, 0.2))
# Set the fontsize of y-axis tick labels for the second plot
axs[1].tick_params(axis='y', labelsize=7)

for i, (region_name, region_coords) in enumerate(regions.items()):
    # Select the region
    ds_region = ds.sel(lat=slice(region_coords['lat_range'][0], region_coords['lat_range'][1]),
                       lon=slice(region_coords['lon_range'][0], region_coords['lon_range'][1]))

    ds2_region = ds2.sel(lat=slice(region_coords['lat_range'][0], region_coords['lat_range'][1]),
                         lon=slice(region_coords['lon_range'][0], region_coords['lon_range'][1]))

    ds3_region = ds3.sel(lat=slice(region_coords['lat_range'][0], region_coords['lat_range'][1]),
                         lon=slice(region_coords['lon_range'][0], region_coords['lon_range'][1]))

    ds4_region = ds4.sel(lat=slice(region_coords['lat_range'][0], region_coords['lat_range'][1]),
                         lon=slice(region_coords['lon_range'][0], region_coords['lon_range'][1]))

    # Align time coordinates for ds2
    ds2_region = ds2_region.sel(time=ds_region['time'])

    # Calculate area weights
    weights = (ds_region['aod'][0].notnull().astype(float))
    weights /= weights.sum(dim=('lat', 'lon'))

    # Calculate area-averaged AOD
    aod_mean = area_weighted_average(ds_region, 'aod')
    aod2_mean = area_weighted_average(ds2_region, 'aod550')
    aod3_mean = area_weighted_average(ds3_region, 'TOTEXTTAU')
    aod4_mean = area_weighted_average(ds4_region, 'total_aod')

    # Plot each graph on the corresponding subplot
    axs[i].plot(aod_mean['time'], aod_mean, label='VIIRS', color='black', linewidth=4.0)
    axs[i].plot(aod2_mean['time'], aod2_mean, label='EAC4')
    axs[i].plot(aod3_mean['time'], aod3_mean, label='MEERA2')
    axs[i].plot(aod4_mean['time'], aod4_mean, label='NAAPS')

    # Set titles and labels
    axs[0].set_title(f'AOD in {first_region}',fontsize=8)
    axs[1].set_title(f'AOD in {second_region}',fontsize=8)
    axs[i].set_ylabel('AOD')
    axs[1].set_xlabel('Date (Year of 2020)')
    plt.xlim(pd.Timestamp('2020-06-14'), pd.Timestamp('2020-06-28'))

    axs[i].legend(fontsize=9)

# Show the plot
plt.show()
