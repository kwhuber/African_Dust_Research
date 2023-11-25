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

# date = [fileName.split("_")[0].split("/")[-1] for fileName in fil_list4] #split the file so that we could access the time.
# ds4['time'] = pd.to_datetime(date, format='%Y%m%d%H')
# ds4['lat']  = ds4['lat'] - 89 #changed the location that appears in the subplots.
# ds4['lon']  = ds4['lon'] - 179
#print(ds4)


# Cell 4. 
from scipy.stats import norm, linregress

# Function to create histograms
def create_histogram(dataset, variable_name, dataset_name, custom_x_axis=None, fill_intervals=None, degree=2):
    plt.figure(figsize=(10, 6))

    # Histogram with specified intervals for filling
    n, bins, patches = plt.hist(dataset[variable_name].values.flatten(), bins=50, range=(min(custom_x_axis), max(custom_x_axis)), color='skyblue', edgecolor='black', histtype='bar', alpha=0.7)

    # Filling the bars with different colors for specified intervals
    if fill_intervals is not None:
        for interval, color in zip(fill_intervals, ['green', 'orange', 'yellow', 'red', 'purple']):
            plt.axvspan(interval[0], interval[1], alpha=0.3, color=color)

    # Add a curve of best fit using polynomial regression
    coefficients = np.polyfit((bins[:-1] + bins[1:]) / 2, n, degree)
    polynomial = np.poly1d(coefficients)
    x_vals = np.linspace(bins[0], bins[-1], 100)
    y_vals = polynomial(x_vals)
    plt.plot(x_vals, y_vals, '--', label=f'Best Fit (Degree {degree})')

    plt.title(f'Histogram of {variable_name} in {dataset_name}')
    plt.xlabel(variable_name)
    plt.ylabel('Frequency')

    if custom_x_axis is not None:
        plt.xticks(custom_x_axis)

    plt.legend()
    plt.grid(True)
    plt.show()

# Define custom x-axes for each dataset
custom_x_axis_ds = [i * 0.01 for i in range(0, 21)]  # Adjust the range and step based on your data
fill_intervals_ds = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2)] # Defines intervals that I want to fill with different colors.

# Create histograms with custom x-axis for each dataset
create_histogram(ds, 'aod', 'VIIRS', custom_x_axis_ds, fill_intervals_ds, degree=2)
create_histogram(ds2, 'aod550', 'EAC4', custom_x_axis_ds, fill_intervals_ds, degree=2)
create_histogram(ds3, 'TOTEXTTAU', 'MEERA2', custom_x_axis_ds, fill_intervals_ds, degree=2)
create_histogram(ds4, 'total_aod', 'NAAPS', custom_x_axis_ds, fill_intervals_ds, degree=2)
