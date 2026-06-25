"""
A tile-wise tree-crown delineation pipeline 
for large scale (city-wide) tree segmentation
based on ZS-TreeSeg Framework (https://github.com/Pengyu-gis/ZS-TreeSeg/)
It takes an optional pre-calculated tree mask.
"""

import gc
import warnings
from contextlib import nullcontext
from itertools import product
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from cellpose import models
from PIL import Image
from rasterio.features import shapes
from rasterio.warp import Resampling, transform_bounds
from rasterio.windows import Window
from shapely.geometry import box, shape
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
 
from zs_utility import clean_crowns, fill_holes, filter_by_shape, _merge_tile_files

warnings.filterwarnings("ignore", category=UserWarning, module="cellpose")
torch.sparse.check_sparse_tensor_invariants.disable()

# --------------- CONFIGURATION --------------------
SEMANTIC_REPO      = "restor/tcd-segformer-mit-b5"

device         = "cuda" if torch.cuda.is_available() else "cpu"
processor      = SegformerImageProcessor.from_pretrained(SEMANTIC_REPO)
semantic_model = SegformerForSemanticSegmentation.from_pretrained(SEMANTIC_REPO).to(device)
cp_model       = models.CellposeModel(gpu=(device == "cuda"))

 
# --------------- HELPERS --------------------

def to_uint8(band: np.ndarray) -> np.ndarray:
    """Stretch a single-band array to [0, 255] uint8."""
    if band.dtype == np.uint8:
        return band
    band = band.astype(np.float32)
    band = (band - band.min()) / (band.max() - band.min() + 1e-6)
    return (band * 255).astype(np.uint8)


def load_geotiff(tif_path: str) -> tuple:
    """
    Load an RGB GeoTIFF
    
    Returns
    -------
    image_pil : PIL.Image        — for the SegFormer processor
    image_np  : ndarray (H×W×3)  — for Cellpose
    transform : rasterio Affine  — georeference of the raster
    crs       : rasterio CRS
    """
    with rasterio.open(tif_path) as src:
        r, g, b   = to_uint8(src.read(1)), to_uint8(src.read(2)), to_uint8(src.read(3))
        transform = src.transform
        crs = src.crs
    
    image_np  = np.stack([r, g, b], axis=-1)
    image_pil = Image.fromarray(image_np)
    return image_pil, image_np, transform, crs


def read_mask_for_window(
    mask_src: rasterio.DatasetReader,
    rgb_transform,
    rgb_crs,
    row0: int,
    col0: int,
    th: int,
    tw: int,
) -> np.ndarray:
    """
    Read the precomputed semantic mask for one RGB tile window.
 
    The mask file stays open for the lifetime of the run; only the pixels
    that correspond to the current tile are loaded — no full-raster read.
    Handles CRS mismatch, resolution mismatch, and pixel misalignment via
    rasterio's on-the-fly windowed resampling. Pixels outside the mask
    extent are filled with 0 (background).
 
    Returns
    -------
    mask_tile : ndarray (th×tw, uint8)
    """
    tile_transform = rasterio.windows.transform(
        Window(col0, row0, tw, th), rgb_transform
    )

    left   = tile_transform.c
    top    = tile_transform.f
    right  = left + tw * tile_transform.a
    bottom = top  + th * tile_transform.e

    if mask_src.crs != rgb_crs:
        left, bottom, right, top = transform_bounds(
            rgb_crs, mask_src.crs, left, bottom, right, top
        )
    
    mask_window = rasterio.windows.from_bounds(
        left, bottom, right, top, transform=mask_src.transform
    )

    # boundless=True fills any out-of-extent pixels with fill_value=0
    mask_tile = mask_src.read(
        1,
        window=mask_window,
        out_shape=(th, tw),
        resampling=Resampling.nearest,
        boundless=True,
        fill_value=0,
    )
    return mask_tile.astype(np.uint8)


# --------------- CORE --------------------

def get_semantic_mask(image_pil: Image.Image) -> np.ndarray:
    """
    Run SegFormer and return a binary tree mask (uint8, 0/1) at input resolution.
    Note: fp16 halves SegFormer VRAM usage
    """
    inputs = processor(images=image_pil, return_tensors="pt")
    inputs = {
        k: v.to(device, dtype=torch.float16 if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
    }

    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
        outputs = semantic_model(**inputs)

    logits    = outputs.logits.float()  # back to fp32 for interpolation    
    upsampled = torch.nn.functional.interpolate(
        logits, size=image_pil.size[::-1], mode="bilinear", align_corners=False
    )
    mask = (upsampled.argmax(dim=1)[0] == 1).cpu().numpy().astype(np.uint8)
    torch.cuda.empty_cache()
    return mask


def run_cellpose(
    image_rgb: np.ndarray,
    semantic_mask: np.ndarray,
    ksize: int = 3,
    diameter: int = 250,
    cellprob_threshold: float = -3.0,
) -> np.ndarray:
    """
    Run CellposeModel within the semantic canopy mask.
    
    Parameters
    -------
    diameter: approximate average crown diameter (pixels). 
      Default: 250 was used for OAM-TCD 10 cm resolution benchmark data
    cellprob_threshold: Default -3.0 to force segmentation inside mask

    Returns
    -------
    instance_mask : ndarray (H×W, int32) — per-crown integer labels
    """
    masked_img = image_rgb.copy()
    masked_img[semantic_mask == 0] = 0
    blurred_img = cv2.GaussianBlur(masked_img, (ksize, ksize), 0)

    masks, _, _ = cp_model.eval(
        blurred_img,
        diameter=diameter,
        flow_threshold=1,
        cellprob_threshold=cellprob_threshold, 
        progress=True
    )

    instance_mask = (masks * semantic_mask).astype(np.int32)
    return instance_mask


def instance_mask_to_gdf(instance_mask: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """
    Vectorise an integer instance mask into a GeoDataFrame.
    Each unique non-zero label becomes one polygon row.
    """
    records = [
        {"geometry": shape(geom)}
        for geom, val in shapes(instance_mask, mask=instance_mask>0, transform=transform)
    ]

    if not records: 
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    return gpd.GeoDataFrame(records, crs=crs)


def process_tiles(
    tif_path: str,
    mask_path: str | None = None,
    diameter: int = 250,
    cellprob_threshold: float = -3.0,
    ksize: int = 3,
    tile_size: int = 2048,
    overlap: int = 128,
    resume: bool = False,
    output_semantic: str | Path | None = None,
    verbose: bool = False
) -> gpd.GeoDataFrame:
    """
    Run the two-stage segmentation pipeline over a tiled GeoTIFF.

    For each tile:
      1. Read the RGB window.
      2. Run SegFormer for a semantic tree mask, optionally intersected with
         the precomputed canopy mask (read tile-by-tile — no full-raster load).
      3. Run CellposeModel for instance segmentation.
      4. Vectorise and write a per-tile GeoPackage.
    """
    run_segformer = output_semantic is not None
    if not run_segformer and mask_path is None:
        raise ValueError("Provide either mask_path (precomputed) or output_semantic (run SegFormer).")

    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        transform = src.transform
        crs = src.crs

    stride = tile_size - overlap  
    row_starts = list(range(0, H, stride))
    col_starts = list(range(0, W, stride))
    if verbose:
        print(
            f"Image: {W}×{H}px  |  "
            f"Tiles: {len(col_starts)}×{len(row_starts)} = {len(row_starts) * len(col_starts)}"
        )

    tiles_dir = Path("tiled_crowns")
    tiles_dir.mkdir(exist_ok=True)
    if not resume:
        for f in tiles_dir.iterdir():
            f.unlink()

    mask_ctx = rasterio.open(mask_path) if mask_path else nullcontext()

    if output_semantic:
        output_semantic = Path(output_semantic)
        output_semantic.parent.mkdir(parents=True, exist_ok=True)
        sem_ctx = rasterio.open(
            output_semantic,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress="lzw",
            nodata=255,
        )
    else:
        sem_ctx = nullcontext()

    with rasterio.open(tif_path) as src, mask_ctx as mask_src, sem_ctx as sem_dst:
        for row0, col0 in tqdm(list(product(row_starts, col_starts)), smoothing=0.1, disable=not verbose):
            tile_gpkg = tiles_dir / f"tile_{row0}_{col0}.gpkg"
            if resume and tile_gpkg.exists():
                continue
            
            row1   = min(row0 + tile_size, H)
            col1   = min(col0 + tile_size, W)
            th, tw = row1 - row0, col1 - col0

            window         = Window(col0, row0, tw, th)
            tile_transform = rasterio.windows.transform(window, transform)

            r = to_uint8(src.read(1, window=window))
            g = to_uint8(src.read(2, window=window))
            b = to_uint8(src.read(3, window=window))
            tile_np = np.stack([r, g, b], axis=-1)

            if tile_np.max() == 0:
                continue

            # Stage 1 — semantic segmentation
            if run_segformer:
                tile_pil = Image.fromarray(tile_np)
                segformer_mask = get_semantic_mask(tile_pil)

                if isinstance(mask_src, rasterio.DatasetReader):
                    precomp_tile = read_mask_for_window(mask_src, transform, crs, row0, col0, th, tw)
                    sem_tile = ((precomp_tile == 1) | (segformer_mask == 1)).astype(np.uint8)
                else:
                    sem_tile = segformer_mask
                if sem_dst is not None:
                    sem_dst.write(sem_tile, 1, window=window)
            else:
                sem_tile = read_mask_for_window(mask_src, transform, crs, row0, col0, th, tw)

            if sem_tile.sum() == 0:
                continue

            # Stage 2 — instance segmentation
            inst_mask = run_cellpose(tile_np, sem_tile, ksize, diameter, cellprob_threshold)
            if inst_mask.max() == 0:
                continue

            crowns_tile = instance_mask_to_gdf(inst_mask, tile_transform, crs)

            # Discard crowns that touch the tile edge
            x_min, y_max = tile_transform * (1, 1)
            x_max, y_min = tile_transform * (tw - 1, th - 1)
            interior_bbox = gpd.GeoDataFrame(
                {"geometry": [box(x_min, y_min, x_max, y_max)]}, crs=crs
            )
            crowns_tile = gpd.sjoin(crowns_tile, interior_bbox, "inner", "within").drop(columns=["index_right"])

            if not crowns_tile.empty:
                crowns_tile.to_file(tile_gpkg, driver="GPKG")

            gc.collect()

        if run_segformer and output_semantic and verbose:
            print(f"Semantic mosaic written → {output_semantic}")

    crowns = _merge_tile_files(tiles_dir, crs)
    crowns["geometry"] = crowns["geometry"].apply(fill_holes)
    crowns["geometry"] = crowns["geometry"].simplify(0.3)
    return crowns


if __name__ == "__main__":
    tif_path = "imagery/RGBN_TestTile.tif"
    mask_path = None
    output_gdb = "output/io_areo.gdb"
    output_semantic = "output/IO_semantic.tif"

    ksize              = 3
    diameter           = 250 # 250 --> 125 due to the 30 cm resolution
    cellprob_threshold = -3

    tree_crowns = process_tiles(
        tif_path,
        mask_path=mask_path,
        ksize=ksize,
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        output_semantic=output_semantic,
        skip_semantic=True
    )

    tree_crowns = filter_by_shape(tree_crowns, circularity_threshold=0.4, elongation_threshold=0.3, use_both=True)
    clean = clean_crowns(tree_crowns, area_threshold=2)
    clean.to_file(
        output_gdb,
        driver="OpenFileGDB",
        layer=f"zs_{ksize}_{diameter}_{abs(cellprob_threshold)}",
        layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},
    )
