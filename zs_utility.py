
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from collections import defaultdict, deque
from tqdm import tqdm


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


def clean_crowns(
    crowns,
    iou_threshold=0.5,
    area_threshold=2,
    containment_threshold=0.5,
):
    crowns = crowns[~crowns.is_empty & crowns.is_valid].copy()
    crowns = crowns[crowns.area > area_threshold].copy()

    print("[clean_crowns] Performing spatial join...")
    join = gpd.sjoin(crowns, crowns, how="inner", predicate="intersects")
    join = join[join.index != join["index_right"]]

    # Build a conflict graph: high IoU OR containment
    conflicts = defaultdict(set) # crowns_idx -> set of conflicting crown_idxs

    for _, row in tqdm(
        join.iterrows(),
        total=len(join),
        desc="[clean_crowns] Building conflict graph",
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

    crowns_clean = crowns.loc[sorted(keep)].reset_index(drop=True)
    print(
        f"[clean_crowns] {len(crowns)} → {len(crowns_clean)} crowns "
        f"(removed {len(crowns) - len(crowns_clean)})"
    )
    return crowns_clean