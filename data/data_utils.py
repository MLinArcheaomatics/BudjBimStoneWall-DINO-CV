import numpy as np
import rasterio as rio

from pathlib import Path
from typing import Union
from tifffile import imread

from rasterio.transform import Affine
from rasterio.crs import CRS


def dem_to_rgb(dem, cmap='gray'):
    # Normalize elev to 0 and 1
    dem = (dem - np.min(dem)) / (np.max(dem) - np.min(dem))
    # Get color map
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap)
    # Convert to RGB
    dem = cm(dem.squeeze())
    dem = (dem[:, :, :3] * 255).astype(np.uint8)
    return dem
  

def arr_to_tiff(
    arr: np.ndarray, 
    output: Union[str, Path], 
    crs: Union[str, CRS, None] = None, 
    transform: Union[Affine, None] = None
):
    """Save a NumPy array as a GeoTIFF file.

    Args:
        arr (np.ndarray): Image array with shape (H, W, C).
        output (Union[str, Path]): Output file path.
        crs (Union[str, CRS, None], optional): Coordinate reference system. Defaults to None.
        transform (Union[Affine, None], optional): Geotransform parameters. Defaults to None.
    """
    assert arr.ndim == 3, "Input array must have shape (H, W, C)"

    tiff_meta = {
        'driver': 'GTiff',
        'height': arr.shape[0], 
        'width': arr.shape[1],
        'count': arr.shape[2], 
        'dtype': str(arr.dtype),
        'crs': CRS.from_string(crs) if isinstance(crs, str) else crs,
        'transform': transform
    }

    with rio.open(output, 'w', **tiff_meta) as ds:
        for band_id in range(arr.shape[2]): 
            ds.write(arr[:, :, band_id], band_id + 1)
            

def tiff_to_arr(fname: Union[str, Path], data_type: str) -> np.ndarray:
    """Load a TIFF image and convert it to a properly formatted NumPy array.

    Args:
        fname (Union[str, Path]): Path to the TIFF file.
        data_type (str): Type of image, e.g., 'mask' (prevents channel stacking).

    Returns:
        np.ndarray: Processed image array in uint8 format.
    """
    img_arr = imread(fname).astype(np.float32)  # Load as float32 to handle scaling correctly
    img_arr = np.nan_to_num(img_arr, nan=0.0, copy=False)  # Replace NaNs efficiently

    # Convert binary image (0,1) to (0,255) range if needed
    if img_arr.min() >= 0.0 and img_arr.max() <= 1.0:
        img_arr *= 255

    img_arr = np.clip(img_arr, 0, 255)  # Ensure values are within uint8 range

    # Expand grayscale (H, W) -> (H, W, 1) and optionally replicate to RGB
    if img_arr.ndim == 2:
        img_arr = np.expand_dims(img_arr, axis=2)  # Shape: (H, W, 1)

        if data_type != "mask":  # Replicate channels only for non-mask images
            img_arr = np.repeat(img_arr, 3, axis=2)  # Shape: (H, W, 3)

    assert img_arr.ndim == 3, "Image array must have 3 dimensions (H, W, C)"
    return img_arr.astype(np.uint8)
