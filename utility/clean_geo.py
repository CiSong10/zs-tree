import geopandas as gpd
from shapely.validation import make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

def clean_geometries(gdf):
    """
    Diagnose, fix, and filter geometries before writing to OpenFileGDB.
    Coerces all geometry to Polygon, keeping the largest polygon part
    from any MultiPolygon or GeometryCollection.
    """
    print(f"Total features: {len(gdf)}")
    print(f"Geometry types present:\n{gdf.geometry.geom_type.value_counts()}\n")

    # 1. Drop null geometries
    null_mask = gdf.geometry.isna()
    if null_mask.any():
        print(f"Dropping {null_mask.sum()} null geometries")
        gdf = gdf[~null_mask].copy()

    # 2. Drop empty geometries
    empty_mask = gdf.geometry.is_empty
    if empty_mask.any():
        print(f"Dropping {empty_mask.sum()} empty geometries")
        gdf = gdf[~empty_mask].copy()

    # 3. Repair invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        print(f"Repairing {invalid_mask.sum()} invalid geometries via make_valid()")
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)

    # 4. Coerce to single Polygon — keep largest part for multi/collection types
    def coerce_to_polygon(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Polygon":
            return geom
        if geom.geom_type == "MultiPolygon":
            return max(geom.geoms, key=lambda g: g.area)
        # GeometryCollection — extract polygon parts, keep largest
        if hasattr(geom, "geoms"):
            polys = [g for g in geom.geoms if g.geom_type == "Polygon"]
            if polys:
                return max(polys, key=lambda g: g.area)
        return None  # Points, Lines, etc. — discard

    gdf["geometry"] = gdf["geometry"].apply(coerce_to_polygon)

    # 5. Drop anything that couldn't be coerced
    bad_mask = gdf.geometry.isna() | gdf.geometry.is_empty
    if bad_mask.any():
        print(f"Dropping {bad_mask.sum()} features that couldn't be coerced to Polygon")
        gdf = gdf[~bad_mask].copy()

    print(f"\nCleaned: {len(gdf)} features remaining")
    print(f"Geometry types now:\n{gdf.geometry.geom_type.value_counts()}")
    return gdf