#!/usr/bin/env python
# coding: utf-8

# In[4]:


'''
This is a two part code. In the first part, the code assesses HRRR data files, and lists out all the necessary information. 
In the second part, the code assesses the netCDF-converted Haines Index files, and lists out the information. 
This code is written by Saurav Dey Shuvo (saurav.met@gmail.com).
'''


# In[5]:


# Assessing the HRRR data files 
import os
import xarray as xr

def list_hrrr_variables_with_filters(data_folder, sample_date):
    """
    List all variables in a sample HRRR GRIB2 file with filtering.
    """
    folder_path = os.path.join(data_folder, sample_date)
    grib_files = [f for f in os.listdir(folder_path) if f.endswith('.grib2')]
    
    filter_options = [
        {'typeOfLevel': 'surface'},
        {'typeOfLevel': 'heightAboveGround', 'level': 10},
        {'typeOfLevel': 'isobaricInhPa', 'level': 500},
        {'typeOfLevel': 'isobaricInhPa', 'level': 700},
    ]
    
    for grib_file in grib_files:
        grib_path = os.path.join(folder_path, grib_file)
        print(f"--- Variables in {grib_file} ---")
        for filter_key in filter_options:
            try:
                ds = xr.open_dataset(grib_path, engine='cfgrib', filter_by_keys=filter_key)
                print(f"Filter applied: {filter_key}")
                print(ds.variables.keys())
            except Exception as e:
                print(f"Could not read with filter {filter_key}: {e}")
            print()

# Usage
if __name__ == '__main__':
    data_folder = r'D:\2Courses\8902\data\hrrr\20231026\hrrr'
    sample_date = '20231001'  # Pick a date folder with GRIB2 files to inspect
    list_hrrr_variables_with_filters(data_folder, sample_date)


# In[6]:


# Assessing the Haines Index (netCDF) data files 
import os
import xarray as xr

def assess_haines_index(data_folder, sample_date):
    """
    Assess variables in a sample Haines Index netCDF file.
    """
    folder_path = os.path.join(data_folder, sample_date)
    haines_file = os.path.join(folder_path, f"haines_index_{sample_date}.nc")
    
    print(f"--- Assessing Haines Index file: {haines_file} ---")
    try:
        ds = xr.open_dataset(haines_file)
        print("Variables in the dataset:")
        print(ds.variables.keys())
        print("\nAttributes of the dataset:")
        print(ds.attrs)
    except Exception as e:
        print(f"Could not read the file {haines_file}: {e}")

# Usage
if __name__ == '__main__':
    data_folder = r'D:\2Courses\8902\data\hrrr\20231026\hrrr'
    sample_date = '20231001'  # Replace with the appropriate date folder
    assess_haines_index(data_folder, sample_date)


# In[ ]:




