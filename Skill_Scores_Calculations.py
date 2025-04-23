#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
This code calculates three skill scores - Heidke Skill Score (HSS), Brier Skill Score (BSS), and Cohen’s Kappa. 
It evaluates simulated wildfire susceptibility and results from Haines Index. 
This code is written by Saurav Dey Shuvo (saurav.met@gmail.com). 
'''


# In[2]:


# Libraries 
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from sklearn.metrics import cohen_kappa_score

def read_raster(file_path):
    """
    Reads a raster file and returns its data array, affine transform, and CRS.
    Assumes the data are in the first band.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs
    return data, transform, crs

def resample_to_target(source_data, source_transform, source_crs,
                       target_shape, target_transform, target_crs):
    """
    Reprojects/resamples the source raster data so its grid matches that of a target raster.
    """
    destination = np.empty(target_shape, dtype=source_data.dtype)
    reproject(
        source=source_data,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest
    )
    return destination

def calculate_hss(observed, predicted, num_classes=7):
    """
    Calculates the Heidke Skill Score (HSS) based on a confusion matrix.
    Both observed and predicted arrays are assumed to contain integer values.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for o, p in zip(observed, predicted):
        confusion_matrix[int(o)][int(p)] += 1

    total = np.sum(confusion_matrix)
    correct = np.trace(confusion_matrix)
    row_sum = np.sum(confusion_matrix, axis=1)
    col_sum = np.sum(confusion_matrix, axis=0)
    expected_correct = np.sum(row_sum * col_sum) / total if total > 0 else 0
    hss = (correct - expected_correct) / (total - expected_correct) if (total - expected_correct) != 0 else np.nan
    return hss

def calculate_scores(obs_dir, sim_dir):
    """
    Reads the raster files from the observation and simulation directories,
    aligns them if necessary, computes the skill scores for each file pair,
    and returns lists of HSS, BSS, and Cohen's Kappa values.
    """
    # Get sorted list of files in each directory ending with .tif.
    obs_files = sorted([os.path.join(obs_dir, f) for f in os.listdir(obs_dir) if f.endswith('.tif')])
    sim_files = sorted([os.path.join(sim_dir, f) for f in os.listdir(sim_dir) if f.endswith('.tif')])

    if len(obs_files) != len(sim_files):
        raise ValueError("Mismatch in the number of files between the observation and simulation directories.")

    hss_list = []
    bss_list = []
    kappa_list = []

    # Loop over matching file pairs.
    for obs_file, sim_file in zip(obs_files, sim_files):
        # Read observation raster.
        obs_data, obs_transform, obs_crs = read_raster(obs_file)
        # Read simulation raster.
        sim_data, sim_transform, sim_crs = read_raster(sim_file)

        # If needed, resample the simulation data to exactly match the observation grid.
        if obs_data.shape != sim_data.shape or (obs_crs != sim_crs) or (obs_transform != sim_transform):
            sim_data = resample_to_target(sim_data, sim_transform, sim_crs,
                                          obs_data.shape, obs_transform, obs_crs)

        # Flatten arrays for easier processing.
        observed = obs_data.flatten()
        simulated = sim_data.flatten()

        # Remove pixels with NaN in either array.
        mask = ~np.isnan(observed) & ~np.isnan(simulated)
        observed = observed[mask]
        simulated = simulated[mask]

        # Convert to integers (assumes categories from 0 to 6).
        observed_int = observed.astype(int)
        simulated_int = simulated.astype(int)

        # Calculate the Heidke Skill Score and Cohen's Kappa.
        hss = calculate_hss(observed_int, simulated_int, num_classes=7)
        kappa = cohen_kappa_score(observed_int, simulated_int)

        # --- Brier Skill Score (BSS) ---
        # Normalize values to the [0, 1] range by dividing by 6.
        observed_prob = observed_int / 6.0
        simulated_prob = simulated_int / 6.0
        brier = np.mean((observed_prob - simulated_prob) ** 2)
        # Use the average observed probability as the climatology.
        climatology = np.mean(observed_prob)
        ref_brier = np.mean((observed_prob - climatology) ** 2)
        bss = 1 - (brier / ref_brier) if ref_brier != 0 else np.nan

        hss_list.append(hss)
        kappa_list.append(kappa)
        bss_list.append(bss)

    return hss_list, bss_list, kappa_list

if __name__ == "__main__":
    obs_dir = r"D:\2Courses\8902\reprojected_data\ClippedHaines"
    sim_dir = r"D:\2Courses\8902\reprojected_data\ArtificialNeuralNetwork\November2023\ClippedResults"

    hss_list, bss_list, kappa_list = calculate_scores(obs_dir, sim_dir)
    
    # Build a list of strings for results.
    results = []
    for i, (hss, bss, kappa) in enumerate(zip(hss_list, bss_list, kappa_list)):
        results.append(f"File Pair {i + 1}:")
        results.append(f"  Heidke Skill Score: {hss:.4f}")
        results.append(f"  Brier Skill Score:  {bss:.4f}")
        results.append(f"  Cohen's Kappa:      {kappa:.4f}")
        results.append("-" * 40)

    # Calculate averages.
    avg_hss = np.mean(hss_list)
    avg_kappa = np.mean(kappa_list)
    # nanmean is used here to safely ignore any nan values for BSS.
    avg_bss = np.nanmean(bss_list)

    results.append("Averaged Scores across all file pairs:")
    results.append(f"  Average Heidke Skill Score: {avg_hss:.4f}")
    results.append(f"  Average Brier Skill Score:  {avg_bss:.4f}")
    results.append(f"  Average Cohen's Kappa:      {avg_kappa:.4f}")

    # Print the results to the console.
    for line in results:
        print(line)

    # Save the results into a text file.
    output_file = "skill_scores_results_ANN.txt"
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")
            
    print(f"\nResults have been saved to '{output_file}'.")

