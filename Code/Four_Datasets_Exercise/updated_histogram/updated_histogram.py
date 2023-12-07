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

fil_list4 = api.file_list( f'{dir_data4}/202006{{14..28}}*' )

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
# print(ds['aod'].min().compute())
aod_values_ds = ds['aod']
max_aod_value_ds = ds['aod'].max().compute().item()
print(f'The max AOD value in VIIRS is: {max_aod_value_ds}')
# Count occurrences of greatest AOD values in ds.
count_max_aod_ds = (aod_values_ds == max_aod_value_ds).sum().values
print(f"Number of occurrences of the greatest AOD value: {count_max_aod_ds}")
# Total AOD values in ds.
total_aod_sum_ds = ds['aod'].sum().values
print(f"Total sum of AOD values in VIIRS: {total_aod_sum_ds}")
probability_dist_ds = (count_max_aod_ds/total_aod_sum_ds)
print(f"Probability distribution: {probability_dist_ds:.10f}")

print('')
ds2 = xr.open_mfdataset(fil_list2)
ds2 = ds2.rename({'longitude': 'lon', 'latitude': 'lat'})
aod_values_ds2 = ds2['aod550']
max_aod_value_ds2 = ds2['aod550'].max().compute().item()
print(f'The max AOD value in EAC4 is: {max_aod_value_ds2}')
# Count occurrences of greatest AOD values in ds2.
count_max_aod_ds2 = (aod_values_ds2 == max_aod_value_ds2).sum().values
print(f"Number of occurrences of the greatest AOD value: {count_max_aod_ds2}")
# Total AOD values in ds.
total_aod_sum_ds2 = ds2['aod550'].sum().values
print(f"Total sum of AOD values in EAC4: {total_aod_sum_ds2}")
probability_dist_ds2 = (count_max_aod_ds2/total_aod_sum_ds2)
print(f"Probability distribution: {probability_dist_ds2:.10f}")

ds2 = xr.open_mfdataset(fil_list2)
ds2 = ds2.rename({'longitude': 'lon', 'latitude': 'lat'})

print('')
ds3 = xr.open_mfdataset(fil_list3)
aod_values_ds3 = ds3['TOTEXTTAU']
max_aod_value_ds3 = ds3['TOTEXTTAU'].max().compute().item()
print(f'The max AOD value in MEERA2 is: {max_aod_value_ds3}')
# Count occurrences of greatest AOD values in ds2.
count_max_aod_ds3 = (aod_values_ds3 == max_aod_value_ds3).sum().values
print(f"Number of occurrences of the greatest AOD value: {count_max_aod_ds3}")
# Total AOD values in ds.
total_aod_sum_ds3 = ds3['TOTEXTTAU'].sum().values
print(f"Total sum of AOD values in MEERA2: {total_aod_sum_ds3}")
probability_dist_ds3 = (count_max_aod_ds3/total_aod_sum_ds3)
print(f"Probability distribution: {probability_dist_ds3:.10f}")

print('')
ds4 = xr.open_mfdataset(fil_list4) #the file couldn't be accessed like in the above commands so we had to add parameters to aquire the proper days and time.
aod_values_ds4 = ds4['total_aod']
max_aod_value_ds4 = ds4['total_aod'].max().compute().item()
print(f'The max AOD value in NAAPS is: {max_aod_value_ds4}')
# Count occurrences of greatest AOD values in ds2.
count_max_aod_ds4 = (aod_values_ds4 == max_aod_value_ds4).sum().values
print(f"Number of occurrences of the greatest AOD value: {count_max_aod_ds4}")
# Total AOD values in ds.
total_aod_sum_ds4 = ds4['total_aod'].sum().values
print(f"Total sum of AOD values in NAAPS: {total_aod_sum_ds4}")
probability_dist_ds4 = (count_max_aod_ds4/total_aod_sum_ds4)
print(f"Probability distribution: {probability_dist_ds4:.10f}")
# date = [fileName.split("_")[0].split("/")[-1] for fileName in fil_list4] #split the file so that we could access the time.
# ds4['time'] = pd.to_datetime(date, format='%Y%m%d%H')
# ds4['lat']  = ds4['lat'] - 89 #changed the location that appears in the subplots.
# ds4['lon']  = ds4['lon'] - 179
#print(ds4)

# Cell 4. 
import seaborn as sns

# Function to create histograms with KDE curve
def create_histogram_with_kde(dataset, variable_name, dataset_name, custom_x_axis=None, fill_intervals=None):
    plt.figure(figsize=(10, 6))

    # Histogram with specified intervals for filling
    n, bins, patches = plt.hist(dataset[variable_name].values.flatten(), bins=50, density=True, histtype='step', range=(min(custom_x_axis), max(custom_x_axis)), color='skyblue', edgecolor='black', alpha=0.7)

    # Add a KDE curve
    sns.kdeplot(dataset[variable_name].values.flatten(), bw_method=0.1, label='Best Fit', linestyle='--', color='blue')

    plt.title(f'Histogram of {variable_name} in {dataset_name}')
    plt.xlabel(variable_name)
    plt.ylabel('Probability Distribution')

    if custom_x_axis is not None:
        plt.xticks(np.arange(0, 5.1, 0.2), fontsize=6)  # Set ticks from 0 to 5 with step 0.2, adjust as needed

    plt.legend()
    plt.grid(True)
    plt.show()

# Rest of your code...

# Create histograms with custom x-axis and KDE curve for each dataset
create_histogram_with_kde(ds, 'aod', 'VIIRS', custom_x_axis_ds, fill_intervals_ds)
create_histogram_with_kde(ds2, 'aod550', 'EAC4', custom_x_axis_ds, fill_intervals_ds)
create_histogram_with_kde(ds3, 'TOTEXTTAU', 'MEERA2', custom_x_axis_ds, fill_intervals_ds)
create_histogram_with_kde(ds4, 'total_aod', 'NAAPS', custom_x_axis_ds, fill_intervals_ds)

# Cell 5. 
# Import seaborn for KDE.
# Seaborn is a data visualization library based on Matplotlib.
# sns allows me to access functions provided by seaborn.
import seaborn as sns

# Function to create histograms
def create_histogram(dataset, variable_name, dataset_name, custom_x_axis=None, fill_intervals=None):
    # Histogram with specified intervals for filling
    n, bins, patches = plt.hist(dataset[variable_name].values.flatten(), bins=50, density=True, histtype='step', range=(min(custom_x_axis), max(custom_x_axis)), color='skyblue', edgecolor='black', alpha=0.7)

    # Add a curve of best fit using kernel density estimation (KDE). KDE is more advantageous to use over regression because it is useful for visualizing the underlying distribution of a dataset and obtaining a smooth curve that represents the likelihood of different values occurring.
    # Regression is used more for making predictions so it may over/under fit data.
    # sns.kdeplot is used to create a KDE plot.
    sns.kdeplot(dataset[variable_name].values.flatten(), bw_method=0.1, label=f'{dataset_name}', linestyle='--')

# Create a single plot for all datasets.
plt.figure(figsize=(10, 6))

# Create histograms with custom x-axis for each dataset.
create_histogram(ds, 'aod', 'VIIRS', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds2, 'aod550', 'EAC4', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds3, 'TOTEXTTAU', 'MEERA2', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds4, 'total_aod', 'NAAPS', custom_x_axis_ds, fill_intervals_ds)

# Customize the plot.
plt.title('Comparison of Probability Distributions')
plt.xlabel('Aerosol Optical Depth (AOD)')
plt.ylabel('Probability Distribution')

# Set the desired x-axis and y-axis limits.
plt.xlim(0, 5.0)
plt.ylim(0, 7)

# Adjust x-axis tick values
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0], fontsize=6)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Cell 6. 
# FIXED Zoomed towards extreme AOD values.
import seaborn as sns  # Import seaborn for KDE

# Function to create histograms
def create_histogram(dataset, variable_name, dataset_name, custom_x_axis=None, fill_intervals=None):
    # Histogram with specified intervals for filling
    n, bins, patches = plt.hist(dataset[variable_name].values.flatten(), bins=50, density=True, histtype='step', range=(min(custom_x_axis), max(custom_x_axis)), color='skyblue', edgecolor='black', alpha=0.7)

    # Add a curve of best fit using kernel density estimation
    sns.kdeplot(dataset[variable_name].values.flatten(), bw_method=0.1, label=f'{dataset_name}', linestyle='--')

# Create a single plot for all datasets
plt.figure(figsize=(10, 6))

# Create histograms with custom x-axis for each dataset
create_histogram(ds, 'aod', 'VIIRS', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds2, 'aod550', 'EAC4', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds3, 'TOTEXTTAU', 'MEERA2', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds4, 'total_aod', 'NAAPS', custom_x_axis_ds, fill_intervals_ds)

# Customize the plot
plt.title('Comparison of Probability Distributions (Zoomed Probability Distribution from 0 to 0.10)')
plt.xlabel('Aerosol Optical Depth (AOD)')
plt.ylabel('Probability Distribution')

# Set the desired x-axis and y-axis limits
plt.xlim(0, 5.0)
plt.ylim(0, .1)

# Adjust x-axis tick values
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0], fontsize=6)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Cell 7. 
# FIXED Zoomed towards extreme AOD values.
import seaborn as sns  # Import seaborn for KDE

# Function to create histograms
def create_histogram(dataset, variable_name, dataset_name, custom_x_axis=None, fill_intervals=None):
    # Histogram with specified intervals for filling
    n, bins, patches = plt.hist(dataset[variable_name].values.flatten(), bins=50, density=True, histtype='step', range=(min(custom_x_axis), max(custom_x_axis)), color='skyblue', edgecolor='black', alpha=0.7)

    # Add a curve of best fit using kernel density estimation
    sns.kdeplot(dataset[variable_name].values.flatten(), bw_method=0.1, label=f'{dataset_name}', linestyle='--')

# Create a single plot for all datasets
plt.figure(figsize=(10, 6))

# Create histograms with custom x-axis for each dataset
create_histogram(ds, 'aod', 'VIIRS', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds2, 'aod550', 'EAC4', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds3, 'TOTEXTTAU', 'MEERA2', custom_x_axis_ds, fill_intervals_ds)
create_histogram(ds4, 'total_aod', 'NAAPS', custom_x_axis_ds, fill_intervals_ds)

# Customize the plot
plt.title('Comparison of Probability Distributions (Zoomed AOD from 3 to 5 and Probability Distribution from 0 to 0.010)')
plt.xlabel('Aerosol Optical Depth (AOD)')
plt.ylabel('Probability Distribution')

# Set the desired x-axis and y-axis limits
plt.xlim(3, 5.0)
plt.ylim(0, .01)

# Adjust x-axis tick values
plt.xticks([3, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0], fontsize=6)
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

