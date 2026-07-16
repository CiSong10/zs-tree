"""
This script combines the tree seg result from ZS and Detectree2
Assign the confidence score of ZS tree seg as 0.5
"""

from datetime import datetime
import geopandas as gpd
from pathlib import Path
from dt_utility import clean_crowns, post_clean, canopy_mask_filter
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from zs_utility import filter_by_shape

def largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)

# ------ Config -------

area_list = ["Greenport"]

for area in area_list:
    print(area)
        
    output_gdb = Path(f"output/Columbia_tree_segmentation.gdb")

    zs_confidence = 0.9
    iou_threshold = 0.4
    confidence = 0.2
    area_threshold = 20 
    containment_threshold = 0.5

    crowns_zs = gpd.read_file(f"output/Columbia_zs.gdb", layer=f"zs_{area}_d140_t1")
    crowns_dt = gpd.read_file(f"/home/cisong/detectree2-implementation/data/{area}/{area}_prediction.gdb", layer=f"{area}_flexi")
    # --------------------

    crowns_zs["Confidence_score"] = zs_confidence
    crowns_zs["tree_frac_in_mask"] = 1.0

    crowns = gpd.GeoDataFrame(
        pd.concat([crowns_zs, crowns_dt], ignore_index=True)
    )

    tree_polygons = clean_crowns(crowns, iou_threshold, confidence, area_threshold, containment_threshold=containment_threshold)

    # clean["geometry"] = (clean.geometry.apply(largest_polygon))
    # clean = clean[clean.geometry.notna()]

    # tree_points = tree_polygons[["Confidence_score", "geometry"]].copy()
    # tree_points['geometry'] = tree_polygons.geometry.representative_point()


    # today = datetime.today().strftime("%m%d")

    tree_polygons.to_file(output_gdb, 
                driver="OpenFileGDB",
                layer=f"{area}_tree_polygons",
                layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"}
                )

    # tree_points.to_file(output_gdb,
    #             driver="OpenFileGDB",
    #             layer=f"{area}_tree_points",
    #             layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"}
    #             )
