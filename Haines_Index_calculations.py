#!/usr/bin/env python
# coding: utf-8

# In[29]:


'''
This python code calculates Haines Index (https://www.spc.noaa.gov/exper/firecomp/INFO/hainesinfo.html) from HRRR data files. 
This code does two things - (1) Calculates the Haines Index and stores them in a separate netCDF file. The file has the calculated 
values for the four time periods of the day. (2) Plots the calculated Haines Index for each of the times from the file. 
This code is written by Saurav Dey Shuvo (saurav.met@gmail.com). 
'''


# In[30]:


import os
import pygrib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from cartopy import feature as cfeature

# Define input folder
data_folder = r"D:\2Courses\8902\data\hrrr\20231026\hrrr\20231115"

# Extract the date from the folder path (assuming folder name is the date)
folder_date = os.path.basename(data_folder)
formatted_date = f"{folder_date[:4]}-{folder_date[4:6]}-{folder_date[6:]}"

# Initialize data storage
time_steps = ["00Z", "06Z", "12Z", "18Z"]
haines_index_dict = {}

# Define a function to process a single GRIB2 file for pressure levels
def process_pressure_file(filepath):
    with pygrib.open(filepath) as grib:
        t850 = None
        t700 = None
        dp850 = None
        for msg in grib:
            if msg.name == "Temperature" and msg.level == 850:
                t850 = msg.values
            elif msg.name == "Temperature" and msg.level == 700:
                t700 = msg.values
            elif msg.name == "Dew point temperature" and msg.level == 850:
                dp850 = msg.values
        return t850, t700, dp850

# Define a function to calculate the corrected Haines Index
def calculate_haines_index(t850, t700, dp850):
    if t850 is not None and t700 is not None and dp850 is not None:
        # Normalize stability and moisture to match Haines Index scale (0 to 6)
        stability = np.clip((t850 - t700) / 2, 0, 3)  
        moisture = np.clip((t850 - dp850) / 5, 0, 3)  
        haines_index = stability + moisture
        return haines_index
    else:
        return None

# Process data files for each time step
for time in time_steps:
    haines_index_values = []
    for file in os.listdir(data_folder):
        if f"t{time[:2].lower()}z.wrfprsf" in file and file.endswith(".grib2"):  # Pressure level data files
            filepath = os.path.join(data_folder, file)
            t850, t700, dp850 = process_pressure_file(filepath)
            haines_index = calculate_haines_index(t850, t700, dp850)
            if haines_index is not None:
                haines_index_values.append(haines_index)
    if haines_index_values:
        haines_index_combined = np.mean(haines_index_values, axis=0) 
        haines_index_dict[time] = haines_index_combined

# Combine results into an xarray Dataset for NetCDF output
if haines_index_dict:
    time_dim = list(haines_index_dict.keys())
    lat_dim, lon_dim = haines_index_combined.shape
    haines_array = np.stack([haines_index_dict[time] for time in time_dim], axis=0)  # Shape: (time, lat, lon)

    ds = xr.Dataset(
        {"Haines_Index": (("time", "lat", "lon"), haines_array)},
        coords={
            "time": time_dim,
            "lat": np.linspace(-90, 90, lat_dim),  # Replace with latitude of interest, optional 
            "lon": np.linspace(-180, 180, lon_dim),  # Replace with longitude of interest, optional 
        },
    )
    # Save as NetCDF
    netcdf_output = os.path.join(data_folder, f"haines_index_{folder_date}.nc")
    ds.to_netcdf(netcdf_output)
    print(f"Haines Index saved to NetCDF: {netcdf_output}")

    # Generate PNG for each time step
    for i, time in enumerate(time_dim):
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-130, -60, 20, 55], crs=ccrs.PlateCarree())  # Set map boundaries for the US
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)  # International boundaries
        ax.add_feature(cfeature.STATES, edgecolor="gray", linewidth=0.5)    # US state boundaries
        mesh = ax.pcolormesh(ds.lon, ds.lat, ds.Haines_Index[i, :, :], shading="auto", cmap="hot_r", vmin=0, vmax=6)
        plt.colorbar(mesh, label="Haines Index", pad=0.05, orientation="horizontal")
        plt.title(f"Haines Index at {time} on {formatted_date}", fontsize=16)
        png_output = os.path.join(data_folder, f"haines_index_{time}_{folder_date}.png")
        plt.savefig(png_output, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Haines Index map saved as PNG for {time}: {png_output}")
else:
    print("No Haines Index values calculated. Please check your input data.")

