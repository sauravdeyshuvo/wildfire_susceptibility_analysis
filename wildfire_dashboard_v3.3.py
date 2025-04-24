import panel as pn
pn.extension()

import matplotlib
matplotlib.use('agg')

import numpy as np
import geopandas as gpd
import rioxarray as rxr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
import pandas as pd
from shapely.geometry import box
from datetime import datetime, timedelta
from functools import lru_cache
import glob

# =====================================================
# =============== CONFIGURATION SECTION ===============
# =====================================================

# 🔁 Toggle this flag to scan for the most recent forecast files
realtime_mode = False # Set to False for static historical date labeling

# 🔢 File pattern used to match forecast rasters
RASTER_PATTERN = "wildfire_susceptibility_forecast_day_{day}_{hour}Z.tif"

# 📂 Forecast output directories for each model
MODEL_PATHS = {
    "Random Forest": r"C:\Users\15673\OneDrive - The Ohio State University\General - Wildfire Index GEOG 8902\Results\RandomForest\November2023\ClippedResults",
    "Artificial Neural Network": r"C:\Users\15673\OneDrive - The Ohio State University\General - Wildfire Index GEOG 8902\Results\ArtificialNeuralNetwork\November2023\ClippedResults"
}

# 📈 Lookup table linking grid cells to circuits
CIRCUIT_LOOKUP_PATH = r"C:\Users\15673\OneDrive - The Ohio State University\General - Wildfire Index GEOG 8902\All_Data\Wildfire_Grid_to_lines\grid_circuit_lookup.csv"
grid_circuit_df = pd.read_csv(CIRCUIT_LOOKUP_PATH)

# 📅 Base date for labeling — static if realtime_mode is off, dynamic otherwise
if realtime_mode:
    rf_path = MODEL_PATHS["Random Forest"]
    first_file = os.path.join(rf_path, RASTER_PATTERN.format(day=1, hour="00"))
    if os.path.exists(first_file):
        base_date = datetime.fromtimestamp(os.path.getmtime(first_file)).date()
    else:
        base_date = datetime.today().date()  # fallback
else:
    base_date = datetime(2023, 11, 2).date()

# ======================Section #1=====================
# ============== FILE SCANNING FUNCTIONS ==============
# =====================================================

def get_available_forecasts(model):
    """Scans for available forecast files for the most recent forecast cycle."""
    folder = MODEL_PATHS[model]
    tif_files = glob.glob(os.path.join(folder, "*.tif"))

    forecasts = []
    for f in tif_files:
        base = os.path.basename(f)
        parts = base.replace("wildfire_susceptibility_forecast_day_", "").replace(".tif", "").split("_")
        if len(parts) == 2:
            day, hourZ = parts
            hour = hourZ.replace("Z", "")
            forecasts.append((int(day), hour, os.path.getmtime(f)))

    if not forecasts:
        return []

    df = pd.DataFrame(forecasts, columns=["day", "hour", "mtime"])
    latest_time = df["mtime"].max()
    df_latest = df[df["mtime"] >= latest_time - 60]  # 1-minute buffer

    return sorted(df_latest[["day", "hour"]].drop_duplicates().values.tolist())

# ====================Section #2=======================
# ================= RASTER HANDLING ===================
# =====================================================

@lru_cache(maxsize=32)
def load_and_reclassify_raster(model, day, hour):
    raster_dir = MODEL_PATHS[model]
    raster_path = os.path.join(raster_dir, RASTER_PATTERN.format(day=day, hour=hour))
    if not os.path.exists(raster_path):
        return None

    raster = rxr.open_rasterio(raster_path, masked=True).squeeze()
    data = raster.values.copy()

    reclass = np.select(
        [data <= 3, data == 4, data == 5, data == 6],
        [0, 1, 2, 3],
        default=np.nan
    )
    raster.data = reclass
    return raster

def fast_extreme_cell_check(raster):
    return np.count_nonzero(raster.data == 3) > 0

# ====================Section #3=======================
# ================= PLOT RISK FUNCTION ================
# =====================================================

def plot_risk(model, day, hour):
    raster = load_and_reclassify_raster(model, day, hour)
    if raster is None:
        return pn.pane.Markdown(f"### ⚠️ No raster found for {model}, Day {day}, Hour {hour}Z")

    cmap = ListedColormap(['#4CAF50', '#FFEB3B', '#FF9800', '#F44336'])
    norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)
    risk_labels = ['No Risk', 'Low', 'Moderate', 'High']

    fig, ax = plt.subplots(figsize=(8, 6))
    im = raster.plot(ax=ax, cmap=cmap, norm=norm, add_colorbar=False)

    forecast_label = datetime.combine(base_date, datetime.min.time()) + timedelta(days=int(day) - 1)
    ax.set_title(f"Wildfire Susceptibility – {forecast_label.strftime('%B %d, %Y')} at {hour}Z")
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(risk_labels)
    cbar.set_label('Wildfire Risk Level')

    if not fast_extreme_cell_check(raster):
        plt.close(fig)
        return pn.Column(
            pn.pane.Matplotlib(fig, tight=True),
            pn.pane.Markdown("#### No circuits intersecting high risk cells.")
        )

    high_risk_coords = np.argwhere(raster.data == 3)
    coord_df = pd.DataFrame({'row': high_risk_coords[:, 0], 'col': high_risk_coords[:, 1]})
    merged = pd.merge(coord_df, grid_circuit_df, on=["row", "col"], how="inner")

    if merged.empty:
        plt.close(fig)
        return pn.Column(
            pn.pane.Matplotlib(fig, tight=True),
            pn.pane.Markdown("#### No circuits intersecting high risk cells.")
        )

    circuit_series = merged["circuit_id"].dropna().apply(lambda ids: ids.split(","))
    all_circuits = sorted(set(cid for sublist in circuit_series for cid in sublist))
    circuit_df = pd.DataFrame({"Circuit ID": all_circuits})

    plt.close(fig)
    return pn.Column(
        pn.pane.Matplotlib(fig, tight=True),
        pn.pane.Markdown("### Circuits in High Susceptibility Areas"),
        pn.pane.DataFrame(circuit_df, index=False, width=600)
    )

# ====================Section #4=======================
# ==================== UI SETUP =======================
# =====================================================

if realtime_mode:
    default_forecast = get_available_forecasts("Random Forest")
    days = sorted({str(d) for d, _ in default_forecast})
    hours = sorted({h for _, h in default_forecast})
else:
    days = [str(d) for d in range(1, 8)]
    hours = ["00", "06", "12", "18"]

view_mode = pn.widgets.RadioButtonGroup(name="View Mode", options=["Single", "Compare"], button_type='primary')
model_options = list(MODEL_PATHS.keys())

model_single = pn.widgets.Select(name="Model", options=model_options)
day_single = pn.widgets.Select(name="Day", options=days)
hour_single = pn.widgets.Select(name="Hour (UTC)", options=hours)

model_1 = pn.widgets.Select(name="Model A", options=model_options)
day_1 = pn.widgets.Select(name="Day A", options=days)
hour_1 = pn.widgets.Select(name="Hour A", options=hours)

model_2 = pn.widgets.Select(name="Model B", options=model_options)
day_2 = pn.widgets.Select(name="Day B", options=days)
hour_2 = pn.widgets.Select(name="Hour B", options=hours)

# ===================Section #5========================
# ================== VIEW LOGIC =======================
# =====================================================

def single_view():
    return pn.Column(
        pn.Row(model_single, day_single, hour_single),
        pn.bind(plot_risk, model=model_single, day=day_single, hour=hour_single)
    )

def compare_view():
    left = pn.bind(plot_risk, model=model_1, day=day_1, hour=hour_1)
    right = pn.bind(plot_risk, model=model_2, day=day_2, hour=hour_2)
    return pn.Column(
        pn.Row(
            pn.Column("### 🔍 Left View", model_1, day_1, hour_1),
            pn.Column("### 🔍 Right View", model_2, day_2, hour_2)
        ),
        pn.Row(left, right)
    )

@pn.depends(view_mode)
def dashboard_content(mode):
    return single_view() if mode == "Single" else compare_view()

# =====================Section #6======================
# ================== DASHBOARD WRAP ===================
# =====================================================

dashboard = pn.Column(
    pn.pane.Markdown("## 🔥 Appalachian Power: Fire Risk Viewer"),
    view_mode,
    dashboard_content
)

dashboard.servable()
