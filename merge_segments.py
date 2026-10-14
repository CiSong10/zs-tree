import geopandas as gpd
import pandas as pd
from pathlib import Path

from utility.dt_utility import clean_crowns, canopy_mask_filter, post_clean
from utility.zs_utility import fill_holes
from utility.clean_geo import clean_geometries


area = "Coshocton"
data_dir = Path("output")
gdb_path = data_dir / "Coshocton_tiles.gdb"

crowns1 = gpd.read_file("Coshocton_tree_segmentation.gdb", layer="Coschocton_tree_polygons")
crowns2 = gpd.read_file("output/Coshocton_zs.gdb")

merged = gpd.GeoDataFrame(pd.concat([crowns1, crowns2], ignore_index=True))

merged["circularity"] = merged["circularity"].fillna(1.0)
merged["tree_frac_in_mask"] = merged["tree_frac_in_mask"].fillna(1.0)

merged.loc[merged["Confidence_score"].isna(), "Confidence_score"] = 0.5 + 0.5 * merged["circularity"]

merged["area"] = merged.geometry.area
merged.loc[merged.geometry.area > 500, "Confidence_score"] = merged["Confidence_score"] * 0.1

clean = clean_crowns(merged, iou_threshold=0.5, containment_threshold=0.5)

tree_polygons = post_clean(merged, clean, max_iterations=1)

tree_polygons.to_file("Coshocton_tree.gpkg", driver="GPKG", layer="Coschocton_tree_polygons")

tree_polygons = clean_geometries(tree_polygons)

tree_polygons.to_file("Coshocton_tree_segmentation.gdb",    
            driver="OpenFileGDB",
            layer="Coschocton_tree_polygons_new",
            layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},
)

tree_points = tree_polygons[["Confidence_score", "geometry"]].copy()
tree_points['geometry'] = tree_polygons.geometry.representative_point()

tree_points.to_file("Coshocton_tree_segmentation.gdb",
            driver="OpenFileGDB",
            layer=f"{area}_tree_points",
            layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"}
            )
