#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This code reprojects the converted (netCDF) Haines Index files. 
This is written by Saurav Dey Shuvo (saurav.met@gmail.com). 
'''


# In[ ]:


import os
import xarray as xr
import rioxarray

# Define paths
haines_index_directory = "D:/2Courses/8902/data/hrrr/"  # Parent directory containing subfolders with Haines Index files
output_directory = "D:/2Courses/8902/reprojected_data/HainesIndex/"

# Target CRS (EPSG:32617 for the circuit shapefile)
TARGET_CRS = "EPSG:32617"
DEFAULT_CRS = "EPSG:4326"  # Assuming the original CRS is WGS84. But it should not be an issue if this is not the CRS as it is temporary. 

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def reproject_haines_index(file_path, output_dir, default_crs, target_crs):
    """
    Process a Haines Index netCDF file: set spatial dimensions, assign CRS, and reproject to target CRS.
    """
    try:
        # Open the netCDF file
        dataset = xr.open_dataset(file_path)

        # Process each time slice
        for time in dataset['time'].values:
            slice_data = dataset.sel(time=time)  # Select individual time slice
            
            # Ensure spatial dimensions are set
            slice_data = slice_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
            
            # Assign CRS if missing
            if not slice_data.rio.crs:
                slice_data = slice_data.rio.write_crs(default_crs)

            # Reproject to target CRS
            reprojected = slice_data.rio.reproject(target_crs)

            # Save the reprojected slice as GeoTIFF
            output_path = os.path.join(output_dir, f"reprojected_{os.path.basename(file_path).replace('.nc', f'_{time}.tif')}")
            reprojected.rio.to_raster(output_path)
            print(f"Reprojected: {file_path}, time={time} -> {output_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_haines_index_files(input_dir, output_dir, default_crs, target_crs):
    """
    Process all Haines Index files in the subdirectories.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".nc"):
                input_file = os.path.join(root, file)
                reproject_haines_index(input_file, output_dir, default_crs, target_crs)

# Process Haines Index files
print("Reprojecting Haines Index files...")
process_haines_index_files(haines_index_directory, output_directory, DEFAULT_CRS, TARGET_CRS)
print(f"All reprojected files saved to: {output_directory}")

