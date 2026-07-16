"""
This script combines the tree seg result from ZS and Detectree2
Assign the confidence score of ZS tree seg as 0.5
"""

from datetime import datetime
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon


def largest_polygon(geom):
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)

# ------ Config -------

area = "Greenport"

output_gdb = Path(f"output/Columbia_tree_segmentation.gdb")

# --------------------

tree_polygons = gpd.read_file(output_gdb, layer=f"{area}_tree_polygons")

tree_points = tree_polygons[["Confidence_score", "geometry"]].copy()
tree_points['geometry'] = tree_polygons.geometry.representative_point()

tree_points.to_file(output_gdb,
            driver="OpenFileGDB",
            layer=f"{area}_tree_points",
            layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"}
            )
