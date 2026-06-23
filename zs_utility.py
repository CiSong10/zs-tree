
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict, deque
from tqdm import tqdm
from pathlib import Path


def fill_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    return geom


def output_sem_mask(sem_mask, transform, crs, output_path="output_sem.tif"):
    if sem_mask.ndim == 2:
        sem_mask = sem_mask[np.newaxis, ...]  # (1, H, W)
    elif sem_mask.ndim == 3:
        sem_mask = sem_mask.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)

    bands, height, width = sem_mask.shape

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype=sem_mask.dtype,   # e.g. float16, uint8, etc.
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(sem_mask)


def calc_iou(shape1, shape2):
    """Calculate the IoU of two shapes."""
    iou = shape1.intersection(shape2).area / shape1.union(shape2).area
    return iou


def min_vertices(geom):
    if isinstance(geom, Polygon):
        return len(geom.exterior.coords) - 1
    elif isinstance(geom, MultiPolygon):
        return min(len(p.exterior.coords) - 1 for p in geom.geoms)
    return 0


def clean_crowns(
    crowns,
    iou_threshold=0.5,
    area_threshold=2,
    containment_threshold=0.5,
    verbose=True
):
    crowns = crowns[~crowns.is_empty & crowns.is_valid].copy()
    crowns = crowns[crowns.area > area_threshold].copy()

    if verbose:
        print("[clean_crowns] Performing spatial join...")
    join = gpd.sjoin(crowns, crowns, how="inner", predicate="intersects")
    join = join[join.index != join["index_right"]]

    # Build a conflict graph: high IoU OR containment
    conflicts = defaultdict(set) # crowns_idx -> set of conflicting crown_idxs

    for _, row in tqdm(
        join.iterrows(),
        total=len(join),
        desc="[clean_crowns] Building conflict graph",
        disable=not verbose
    ):
        i = row.name
        j = row["index_right"]
        if i >= j:
            continue

        geom_i = crowns.at[i, "geometry"]
        geom_j = crowns.at[j, "geometry"]
        intersection_area = geom_i.intersection(geom_j).area

        iou_val = calc_iou(geom_i, geom_j)
        is_conflict = iou_val > iou_threshold

        if not is_conflict and containment_threshold is not None:
            min_area = min(geom_i.area, geom_j.area)
            if min_area > 0 and (intersection_area / min_area) > containment_threshold:
                is_conflict = True

        if is_conflict:
            conflicts[i].add(j)
            conflicts[j].add(i)

    # Find connected components via BFS, keep largest crown per component
    all_indices = set(crowns.index)
    visited = set()
    keep = set()

    for i in all_indices:
        if i in visited:
            continue

        if i not in conflicts:
            # No conflicts at all — keep unconditionally
            keep.add(i)
            visited.add(i)
            continue

        # BFS to collect the full connected component
        component = []
        queue = deque([i])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in conflicts.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)

        # Keep the crown with the largest area in this component
        best = max(component, key=lambda idx: crowns.at[idx, "geometry"].area)
        keep.add(best)

    clean = crowns.loc[sorted(keep)].reset_index(drop=True)

    clean = clean[clean.geometry.apply(min_vertices) >= 5]

    if verbose:
        print(
            f"[clean_crowns] {len(crowns)} → {len(clean)} crowns "
            f"(removed {len(crowns) - len(clean)})"
        )

    return clean


def calculate_circularity(geometry):
    """
    Calculate the circularity (roundness) of a polygon.
    
    Circularity = 4 * pi * Area / Perimeter^2
    
    - Value ranges from 0 to 1
    - 1.0 = perfect circle
    - Lower values = more elongated / irregular shapes
    """
    if geometry is None or geometry.is_empty:
        return 0.0
    
    area = geometry.area
    perimeter = geometry.length
    
    if perimeter == 0:
        return 0.0
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def calculate_elongation(geometry):
    """
    Alternative metric: elongation ratio from the minimum rotated bounding box.
    
    Elongation = min_side / max_side
    
    - Value ranges from 0 to 1
    - 1.0 = square bounding box (compact shape)
    - Lower values = more elongated (long thin shapes)
    """
    if geometry is None or geometry.is_empty:
        return 0.0
    
    bbox = geometry.minimum_rotated_rectangle
    coords = list(bbox.exterior.coords)
    
    # Calculate side lengths of the bounding box
    sides = []
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        sides.append(np.sqrt(dx**2 + dy**2))
    
    if not sides or max(sides) == 0:
        return 0.0
    
    return min(sides) / max(sides)


def filter_by_shape(
    gdf: gpd.GeoDataFrame,
    circularity_threshold: float = 0.4,
    elongation_threshold: float = 0.3,
    use_both: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame to keep only sufficiently circular/compact polygons.

    Args:
        gdf: Input GeoDataFrame with polygon geometries.
        circularity_threshold: Minimum circularity score (0–1). Polygons below
            this value are removed. Default 0.4 works well for tree crowns.
        elongation_threshold: Minimum elongation ratio (0–1). Filters long thin
            shapes. Default 0.3.
        use_both: If True, a polygon must pass BOTH thresholds to be kept.
            If False (default), passing EITHER threshold is sufficient.

    Returns:
        Filtered GeoDataFrame with added 'circularity' and 'elongation' columns.
    """
    gdf = gdf.copy()
    
    # Calculate metrics
    gdf["circularity"] = gdf.geometry.apply(calculate_circularity)
    gdf["elongation"] = gdf.geometry.apply(calculate_elongation)

    # Build filter masks
    circ_mask = gdf["circularity"] >= circularity_threshold
    elon_mask = gdf["elongation"] >= elongation_threshold
    
    if use_both:
        mask = circ_mask & elon_mask
    else:
        mask = circ_mask | elon_mask
    
    filtered = gdf[mask].reset_index(drop=True)
    
    # Summary
    # removed = len(gdf) - len(filtered)
    # print(f"Original polygons : {len(gdf)}")
    # print(f"Kept              : {len(filtered)}")
    # print(f"Removed           : {removed} ({removed / len(gdf) * 100:.1f}%)")
    # print(f"\nCircularity stats (before filtering):")
    # print(gdf["circularity"].describe().round(3))
    # print(f"\nElongation stats (before filtering):")
    # print(gdf["elongation"].describe().round(3))
    
    return filtered


if __name__ == "__main__":
    output_gdb = "output/wildrice_zs_test.gdb"
    # Load your GeoDataFrame
    gdf = gpd.read_file(output_gdb, layer="test_3_125__2")

    filtered_gdf = filter_by_shape(
        gdf,
        circularity_threshold=0.4,
        elongation_threshold=0.35,
        use_both=True,
    )

    # # Inspect the scores to help tune your threshold
    # gdf["circularity"] = gdf.geometry.apply(calculate_circularity)
    # gdf["elongation"] = gdf.geometry.apply(calculate_elongation)
    # print(gdf[["circularity", "elongation"]].sort_values("circularity").head(20))

    # Save result
    filtered_gdf.to_file("filtered_trees.gpkg", driver="GPKG")
    filtered_gdf.to_file(
    output_gdb,
    driver="OpenFileGDB",
    layer=f"test_cir",
    layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},
)


def _merge_tile_files(tiles_dir: Path, crs) -> gpd.GeoDataFrame:
    """Concatenate all per-tile GeoPackages into one GeoDataFrame."""
    tile_files = sorted(tiles_dir.glob("*.gpkg"))
    if not tile_files:
        raise RuntimeError(f"No tile GeoPackages found in {tiles_dir}.")
    return gpd.GeoDataFrame(
        pd.concat([gpd.read_file(f) for f in tile_files], ignore_index=True),
        crs=crs,
    )
