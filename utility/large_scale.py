"""
1. Perform circularity cleaning before merge tiles
2. Merge by chunk: every 100 files,
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd
import rasterio
from tqdm import tqdm

from pipeline import _merge_tile_files
from utility.zs_utility import clean_crowns, fill_holes, filter_by_shape


tiles_dir = Path("tiled_crowns")
chunk_dir = Path("chunk_crowns")
tif_dir = Path("/mnt/c/users/cs0330/Documents/ArcGIS/Projects/WildRice/clipped/") 
tif_path = tif_dir / "WildRice_2023_naip_Becker_clipped.tif"
output_gdb = "/mnt/c/users/cs0330/Documents/ArcGIS/Projects/WildRice/wildrice_zs_trees.gdb"

chunk_dir.mkdir(exist_ok=True)
name = tif_path.stem.split("_")[-1]

with rasterio.open(tif_path) as src:
    # H, W = src.height, src.width
    # transform = src.transform
    crs = src.crs

tile_files = sorted(tiles_dir.glob("tile_*.gpkg"))
tiles_num = len(tile_files)
print(f"Find {tiles_num} files in {tiles_dir}")

for start in tqdm(range(0, tiles_num, 100)):
    end = min(start+100, tiles_num)
    
    chunk = gpd.GeoDataFrame(
        pd.concat([gpd.read_file(f) for f in tile_files[start:end]], ignore_index=True),
        crs=crs,
    )

    chunk = filter_by_shape(chunk, use_both=True)
    chunk["geometry"] = chunk["geometry"].apply(fill_holes)
    chunk["geometry"] = chunk["geometry"].simplify(0.3)
    chunk = clean_crowns(chunk, area_threshold=2)
    chunk.to_file(chunk_dir / f"chunk_{start}.gpkg")

chunk_files = sorted(chunk_dir.glob("*.gpkg"))
print(f"Find {len(chunk_files)} chunks in {chunk_dir}.")
tree_crowns = _merge_tile_files(chunk_dir, crs)
clean = clean_crowns(tree_crowns, area_threshold=2)

clean.to_file(
    output_gdb,
    driver="OpenFileGDB",
    layer=f"{name}_zs_3_125__2",
    layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},
    )
