#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
This code reprojects HRRR data files to the CRS of the circuits (EPSG:43617). 
This is written by Saurav Dey Shuvo (saurav.met@gmail.com). 
'''


# In[ ]:


# Libraries 
import os
import xarray as xr
import rioxarray
import numpy as np
from affine import Affine

# Define paths
input_directory = r"D:\2Courses\8902\data\hrrr\20231026\hrrr"  # Folder containing HRRR netCDF files
output_directory = r"D:\2Courses\8902\reprojected_data"          # Base folder for saving reprojected files

# Create a new subfolder inside the output directory
new_output_folder = os.path.join(output_directory, "HRRR_Reprojected")
os.makedirs(new_output_folder, exist_ok=True)

# Define the source and target CRS.
# Even though the files have no defined CRS, we assume they are in geographic coordinates.
SOURCE_CRS = "EPSG:4326"  # Assumed for lat-lon in degrees; modify if needed for a different type of data file. 
TARGET_CRS = "EPSG:32617"  # Your desired target projection.

def compute_affine_transform(ds):
    """
    Compute an affine transform from the 2D 'longitude' and 'latitude' coordinates.
    Assumes that the grid is regular.
    """
    lon = ds['longitude'].values
    lat = ds['latitude'].values
    dx = float(lon[0, 1] - lon[0, 0])
    dy = float(lat[1, 0] - lat[0, 0])
    transform = Affine.translation(float(lon[0, 0]) - dx/2,
                                     float(lat[0, 0]) - dy/2) * Affine.scale(dx, dy)
    return transform

def assign_1d_coords(ds, transform):
    """
    Drop the existing 2D 'latitude' and 'longitude' variables and assign new 1D coordinate arrays
    computed from the affine transform.
    """
    # Get the number of rows and columns from the shape of a data variable.
    first_var = list(ds.data_vars.values())[0]
    nrows, ncols = first_var.shape

    dx = transform.a
    dy = transform.e
    x0 = transform.c + dx/2  # center of first cell
    y0 = transform.f + dy/2

    x_coords = x0 + dx * np.arange(ncols)
    y_coords = y0 + dy * np.arange(nrows)

    ds = ds.drop_vars(["latitude", "longitude"])
    ds = ds.assign_coords(longitude=("x", x_coords), latitude=("y", y_coords))
    return ds

def reproject_hrrr_file(file_path, output_dir, source_crs, target_crs):
    """
    Opens an HRRR netCDF file, computes an affine transform,
    attaches that transform and the supplied CRS to the chosen data variable (Temperature if available),
    reprojects that DataArray to the target CRS, and saves the result as a GeoTIFF.
    """
    try:
        ds = xr.open_dataset(file_path)
        print(f"Processing file: {file_path}")
        print(f"Variables: {list(ds.variables.keys())}")
        print(f"Dimensions (sizes): {ds.sizes}")
        print(f"Coordinates: {list(ds.coords.keys())}")
        
        if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
            raise ValueError(f"Missing 'latitude' and/or 'longitude' coordinates in {file_path}")
        
        # Compute the affine transform from the 2D coordinate arrays.
        transform = compute_affine_transform(ds)
        
        # Write the computed transform and the supplied source CRS
        ds = ds.rio.write_transform(transform, inplace=False)
        ds = ds.rio.write_crs(source_crs, inplace=False)
        
        # Optionally, replace 2D coordinates with new 1D coordinates if needed.
        ds = assign_1d_coords(ds, transform)
        
        # Set the spatial dimensions based on the new coordinate names.
        ds = ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)
        
        # Select the data variable to reproject (Temperature if available)
        if "Temperature" in ds.data_vars:
            da = ds["Temperature"]
        else:
            da = list(ds.data_vars.values())[0]
        
        # Attach transform and CRS to the selected DataArray.
        da = da.rio.write_transform(transform, inplace=False)
        da = da.rio.write_crs(source_crs, inplace=False)
        da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=False)
        
        # Reproject the data array to the target CRS.
        reprojected = da.rio.reproject(target_crs)
        
        # Save the reprojected result as a GeoTIFF in the new folder.
        out_file = os.path.join(output_dir, f"reprojected_{os.path.basename(file_path).replace('.nc', '.tif')}")
        reprojected.rio.to_raster(out_file)
        print(f"Reprojected: {file_path} -> {out_file}")
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_hrrr_files(input_dir, output_dir, source_crs, target_crs):
    """
    Walks through the input directory and processes HRRR netCDF files whose names start with '2023'
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("2023") and file.endswith(".nc"):
                file_path = os.path.join(root, file)
                reproject_hrrr_file(file_path, output_dir, source_crs, target_crs)

print("Reprojecting HRRR netCDF files...")
process_hrrr_files(input_directory, new_output_folder, SOURCE_CRS, TARGET_CRS)
print(f"All reprojected files saved to: {new_output_folder}")

