import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np


davey_mask = "imagery/Elwood_DaveyCanopy.tif"
zs_mask    = "imagery/Elwood_CanopyMask.tif"
cmb_mask   = "output/Elwood_combined_mask.tif"

with rasterio.open(davey_mask) as src:
    H_d, W_d = src.height, src.width
    transform_d = src.transform
    crs_d = src.crs
    bounds_d = src.bounds

print("Davey Tree Mask\n"
      f"Height: {H_d}, Width: {W_d}\n"
      f"Transform: {transform_d}\n"
      f"CRS: {crs_d}\n"
      f"Bounds: {bounds_d}\n")

with rasterio.open(zs_mask) as src:
    H_z, W_z = src.height, src.width
    transform_z = src.transform
    crs_z = src.crs
    bounds_z = src.bounds

print("ZS Tree Mask\n"
      f"Height: {H_z}, Width: {W_z}\n"
      f"Transform: {transform_z}\n"
      f"CRS: {crs_z}\n"
      f"Bounds: {bounds_z}\n")

assert crs_d == crs_z, "CRS mismatch between the two masks."

# --- Build the union output grid ---
# Pixel size: use Davey's resolution as the reference grid resolution.
px_w = transform_d.a
px_h = transform_d.e  # negative for north-up rasters


if not np.isclose(px_w, transform_z.a) or not np.isclose(px_h, transform_z.e):
    print("Warning: pixel resolutions differ between the two rasters. "
          "Using Davey mask resolution for the output grid; the ZS mask "
          "will be resampled to match.")
    

union_left   = min(bounds_d.left,   bounds_z.left)
union_bottom = min(bounds_d.bottom, bounds_z.bottom)
union_right  = max(bounds_d.right,  bounds_z.right)
union_top    = max(bounds_d.top,    bounds_z.top)
 
union_width  = int(np.ceil((union_right - union_left) / px_w))
union_height = int(np.ceil((union_top - union_bottom) / (-px_h)))
 
union_transform = rasterio.transform.from_origin(union_left, union_top, px_w, -px_h)
 
print(f"Union grid: {union_height} rows x {union_width} cols\n"
      f"Union transform: {union_transform}\n")


def load_and_warp(path, dst_shape, dst_transform, dst_crs):
    """Read a single-band raster and resample/align it onto the destination grid.
    Areas outside the source raster's original extent come back as 0."""
    with rasterio.open(path) as src:
        dst_array = np.zeros(dst_shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,  # nearest keeps mask values binary (0/1)
            dst_nodata=0,
        )
    return dst_array

sem_d = load_and_warp(davey_mask, (union_height, union_width), union_transform, crs_d)
sem_z = load_and_warp(zs_mask,    (union_height, union_width), union_transform, crs_d)

semantic = ((sem_d == 1) | (sem_z == 1)).astype(np.uint8)

with rasterio.open(
    cmb_mask,
    "w",
    driver="GTiff",
    height=union_height,
    width=union_width,
    count=1,
    dtype=np.uint8,
    crs=crs_d,
    transform=union_transform,
    compress="lzw",
    nbit=1,
) as dst:
    dst.write(semantic, 1)

print(f"Combined mask written to {cmb_mask}")
