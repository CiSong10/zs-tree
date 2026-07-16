from datetime import datetime
import geopandas as gpd
from pathlib import Path
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

area = "Angola"
gdb_path = f"/home/cisong/detectree2-implementation/data/{area}/{area}_prediction.gdb"
crowns_dt = gpd.read_file(gdb_path, layer=f"{area}_flexi")

# crowns_dt["area"] = crowns_dt.geometry.area

crowns_dt.loc[crowns_dt.geometry.area > 3500, "Confidence_score"] = 0.2

crowns_dt.to_file(gdb_path, 
                driver="OpenFileGDB",
                layer=f"{area}_flexi",
                layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"}
                )