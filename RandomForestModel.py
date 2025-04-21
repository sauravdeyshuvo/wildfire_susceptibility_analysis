#!/usr/bin/env python
# coding: utf-8

# In[3]:


'''
This code calculates the wildfire susceptibility from a set of static (e.g. DEM, Landcover, Slope and Aspect, etc.) and dynamic (e.g. Temperature, 
Relative Humidity, NDVI, etc.) data using the Random Forest model. 
The wildfire susceptibility is predicted from results of Haines Index (which is calculated from HRRR data). 
This code is written to provide simulated forecast for the succeeding 7 days after the model is trained for 32 days (01 October to 01 November). 
But before all the calculations would begin, the code will resample all data (regardless of them being X or Y), will be resampled to the circuit 
shapefile. 
This code is written by Saurav Dey Shuvo (saurav.met@gmail.com). 
'''


# In[4]:


# Libraries 
import os
import numpy as np
import pandas as pd  
import rasterio
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xarray as xr

# Directories
base_dir = r"D:\2Courses\8902\reprojected_data"
haines_dir = os.path.join(base_dir, "HainesIndex")
hrrr_dir = os.path.join(base_dir, "HRRR_Reprojected")  
circuit_shapefile = r"D:\2Courses\8902\data\Filtered_circuits\Filtered_circuits\filtered_circuits.shp"
csv_file = os.path.join(base_dir, "features_dataset.csv")  
output_dir = os.path.join(base_dir, "SimulatedResults")  

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to read raster data
def read_raster(file_path):
    """Reads raster or NetCDF files and extracts the first band or variable."""
    if file_path.endswith('.nc'):  
        try:
            dataset = xr.open_dataset(file_path)
            data_vars = list(dataset.data_vars)  # Get available variables
            if not data_vars:
                raise RuntimeError(f"No data variables found in NetCDF file: {file_path}")
            data = dataset[data_vars[0]].values  
            if len(data.shape) == 3:  
                data = data[0, :, :]
            elif len(data.shape) != 2:
                raise RuntimeError(f"Unexpected shape in NetCDF file: {data.shape}")
            profile = {
                "crs": "EPSG:32617",  # Manually specify CRS
                "width": data.shape[1],
                "height": data.shape[0],
                "transform": None,  
            }
        except Exception as e:
            raise RuntimeError(f"Error reading NetCDF file: {file_path}. Details: {str(e)}")
    else:
        with rasterio.open(file_path) as src:  
            data = src.read(1)  
            profile = src.profile
    return data, profile

# Extract extent and resolution from Circuit Shapefile
def get_circuit_profile(circuit_shapefile):
    """Extracts spatial extent and resolution information from the circuit shapefile."""
    circuits = gpd.read_file(circuit_shapefile)
    bounds = circuits.total_bounds  
    resolution = 300  # Resolution in meters
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = rasterio.transform.from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
    profile = {
        "crs": circuits.crs.to_string(),
        "transform": transform,
        "width": width,
        "height": height,
    }
    return profile

# Get Circuit Shapefile profile
circuit_profile = get_circuit_profile(circuit_shapefile)

# Resampling function
def resample_raster(input_data, input_profile, reference_profile):
    """Resamples raster data to match the resolution and extent of the circuit shapefile."""
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(driver="GTiff",
                          width=input_profile["width"],
                          height=input_profile["height"],
                          count=1,
                          dtype=input_data.dtype,
                          crs=input_profile["crs"],
                          transform=input_profile["transform"]) as mem_dataset:
            mem_dataset.write(input_data, 1)
            with rasterio.vrt.WarpedVRT(mem_dataset,
                                        crs=reference_profile["crs"],
                                        transform=reference_profile["transform"],
                                        width=reference_profile["width"],
                                        height=reference_profile["height"]) as vrt:
                resampled_data = vrt.read(1)
    return resampled_data

# Feature extraction and CSV creation
def create_features_csv():
    """Extracts features from raster data and saves as a CSV file."""
    feature_data = []

    # Load static datasets
    landcover, lc_profile = read_raster(os.path.join(base_dir, "LandCover_reprojected.nc"))
    slope_aspect, sa_profile = read_raster(os.path.join(base_dir, "SlopeAspect_reprojected.nc"))
    dem, dem_profile = read_raster(os.path.join(base_dir, "DEM_reprojected.tif"))
    ndvi, ndvi_profile = read_raster(os.path.join(base_dir, "NDVI_reprojected.tif"))
    ndwi, ndwi_profile = read_raster(os.path.join(base_dir, "NDWI_reprojected.tif"))
    nbr, nbr_profile = read_raster(os.path.join(base_dir, "NBR_reprojected.tif"))  

    # Resample and add features to the CSV
    datasets = {
        "landcover": resample_raster(landcover, lc_profile, circuit_profile).flatten(),
        "dem": resample_raster(dem, dem_profile, circuit_profile).flatten(),
        "ndvi": resample_raster(ndvi, ndvi_profile, circuit_profile).flatten(),
        "ndwi": resample_raster(ndwi, ndwi_profile, circuit_profile).flatten(),
        "nbr": resample_raster(nbr, nbr_profile, circuit_profile).flatten(),
    }
    if "slope_aspect" in locals():
        datasets["slope_aspect"] = resample_raster(slope_aspect, sa_profile, circuit_profile).flatten()

    # Load HRRR data
    hrrr_files = [f for f in os.listdir(hrrr_dir) if f.endswith('.tif')]
    hrrr_files.sort()  
    for file in hrrr_files:
        file_date = file.split('_')[-2]
        data, profile = read_raster(os.path.join(hrrr_dir, file))
        resampled_data = resample_raster(data, profile, circuit_profile).flatten()
        datasets[f"hrrr_{file_date}"] = resampled_data

    # Save all features to a CSV
    df = pd.DataFrame(datasets)
    df.to_csv(csv_file, index=False)
    print(f"Feature data saved to: {csv_file}")

create_features_csv()  

# Load and process Haines Index files (Y variable)
haines_files = [f for f in os.listdir(haines_dir) if f.endswith('.tif')]
haines_files.sort()  

# Use the full temporal range of Haines Index data 
haines_data = []
for file in haines_files:
    file_date = file.split('_')[-2]
    if "20231001" <= file_date <= "20231101":  # Select the date range
        data, profile = read_raster(os.path.join(haines_dir, file))
        resampled_data = resample_raster(data, profile, circuit_profile)
        haines_data.append(resampled_data)

# Retain information of Haines data
full_haines_data = haines_data  
haines_data = haines_data[-7:]  

# Load feature data from CSV
df = pd.read_csv(csv_file)  

# Loop through each day 
for i, daily_haines in enumerate(haines_data):
    # Loop through the four simulations: 00Z, 06Z, 12Z, and 18Z
    for time in ["00Z", "06Z", "12Z", "18Z"]:
        # Identify relevant HRRR and Haines Index files for the current day and time
        date_str = f"202310{i+1:02d}"  
        hrrr_file = os.path.join(hrrr_dir, f"reprojected_{date_str}_{time}.tif")
        haines_file = os.path.join(haines_dir, f"reprojected_haines_index_{date_str}_{time}.tif")
        
        # Check if files exist
        if not (os.path.exists(hrrr_file) and os.path.exists(haines_file)):
            raise RuntimeError(f"Missing files for day {date_str} and time {time}. Please check file paths.")

        # Resample HRRR data
        hrrr_data, hrrr_profile = read_raster(hrrr_file)
        resampled_hrrr = resample_raster(hrrr_data, hrrr_profile, circuit_profile).flatten()
        
        # Resample Haines Index data
        haines_data_time, haines_profile = read_raster(haines_file)
        resampled_haines = resample_raster(haines_data_time, haines_profile, circuit_profile).flatten()

        # Categorize Haines Index values for the current simulation time
        y_categorized = np.digitize(resampled_haines, bins=[0, 1, 2, 3, 4, 5, 6])

        # Combine HRRR data with static features
        combined_df = df.copy()
        combined_df[f"hrrr_{date_str}_{time}"] = resampled_hrrr

        # Train-test split and Random Forest training
        X_train, X_test, y_train, y_test = train_test_split(combined_df.values, y_categorized, test_size=0.2, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        wildfire_susceptibility = rf_model.predict(combined_df.values).reshape(circuit_profile["height"], circuit_profile["width"])
        
        # Save map for the specific day and time
        output_file = os.path.join(output_dir, f"wildfire_susceptibility_forecast_day_{i+1}_{time}.tif")
        out_profile = circuit_profile
        out_profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_file, "w", **out_profile) as dest:
            dest.write(wildfire_susceptibility.astype(rasterio.uint8), 1)
        print(f"Wildfire susceptibility map for day {i+1} at {time} saved to: {output_file}")

