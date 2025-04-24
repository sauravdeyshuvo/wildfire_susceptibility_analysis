import geopandas as gpd
import rioxarray as rxr
import pandas as pd
from shapely.geometry import box
import os
import time

# === Start timer ===
start = time.time()

# ==== Config ====
raster_path = r"C:\Users\15673\OneDrive - The Ohio State University\General - Wildfire Index GEOG 8902\Results\RandomForest\November2023\ClippedResults\wildfire_susceptibility_forecast_day_1_00Z.tif"
circuit_path = r"C:\Users\15673\OneDrive - The Ohio State University\General - Wildfire Index GEOG 8902\All_Data\Appalachian Power Data\Filtered_circuits\filtered_circuits.shp"
output_csv = r"C:\Users\15673\OneDrive - The Ohio State University\General - Wildfire Index GEOG 8902\All_Data\Cole_python_stuff\grid_circuit_lookup.csv"

# ==== Load Raster ====
raster = rxr.open_rasterio(raster_path, masked=True).squeeze()
res_x, res_y = raster.rio.resolution()
crs = raster.rio.crs

# ==== Generate Raster Grid Cells ====
grid_cells = []
for y in range(raster.shape[0]):
    for x in range(raster.shape[1]):
        x0, y0 = raster.x[x].item(), raster.y[y].item()
        geom = box(x0, y0 - abs(res_y), x0 + abs(res_x), y0)
        grid_cells.append({"row": y, "col": x, "geometry": geom})

grid_gdf = gpd.GeoDataFrame(grid_cells, crs=crs)

# ==== Load Circuits ====
circuits = gpd.read_file(circuit_path).to_crs(crs)

# ==== Spatial Join ====
joined = gpd.sjoin(grid_gdf, circuits, how="inner", predicate="intersects")

# ==== Drop Missing circuit_id Values ====
initial_len = len(joined)
joined = joined[~joined["circuit_id"].isna()]
dropped = initial_len - len(joined)
print(f"Dropped rows with missing circuit_id: {dropped}")

# ==== Group and Aggregate ====
grouped = joined.groupby(["row", "col"]).agg({
    "circuit_id": lambda ids: ",".join(sorted(set(ids)))
}).reset_index()

# ==== Save Output ====
grouped.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

# ==== Report Runtime ====
print(f"⏱️ Completed in {round(time.time() - start, 2)} seconds")

