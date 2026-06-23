"""
This script combines the tree seg result from ZS and Detectree2
Assign the confidence score of ZS tree seg as 0.5
"""

import geopandas as gpd
from pathlib import Path
from cross_model.dt_utility import clean_crowns, post_clean, canopy_mask_filter
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon


def largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)

# ------ Config -------

gdb_path = Path("output/Golden_tree_segmentation.gdb")

zs_confidence = 0.5
iou_threshold = 0.4
confidence = 0.2
area_threshold = 20
containment_threshold = 0.55

crowns_zs = gpd.read_file("output/Golden_zs.gdb", layer="clip_250_2")
crowns_dt = gpd.read_file("/home/cisong/detectree2-implementation/data/Golden/Golden_prediction.gpkg", layer="Golden_260506_nearmap")

# --------------------

crowns_zs["Confidence_score"] = zs_confidence
crowns_zs["tree_frac_in_mask"] = 0.9

crowns = gpd.GeoDataFrame(
    pd.concat([crowns_zs, crowns_dt], ignore_index=True)
)

clean = clean_crowns(crowns, iou_threshold, confidence, area_threshold, containment_threshold=containment_threshold)

# clean["geometry"] = (clean.geometry.apply(largest_polygon))
# clean = clean[clean.geometry.notna()]

clean.to_file(gdb_path, 
              driver="OpenFileGDB",
              layer="Golden_trees_combined_0617",
              layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"}
              )