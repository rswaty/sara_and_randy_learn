#!/usr/bin/env python3
"""
Optional step: Smooth speckled cluster maps using a sieve filter.

Why:
- KMeans cluster maps can be noisy at pixel level.
- This removes tiny isolated patches and keeps larger coherent zones.

Input:
- clusters_kXX.tif

Output:
- clusters_kXX_sieved.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import sieve


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smooth cluster raster with sieve filtering.")
    p.add_argument("--cluster-raster", required=True, help="Path to clusters_kXX.tif")
    p.add_argument(
        "--min-pixels",
        type=int,
        default=9,
        help="Minimum connected pixels to keep (default: 9).",
    )
    p.add_argument(
        "--connectivity",
        type=int,
        default=8,
        choices=[4, 8],
        help="Pixel connectivity for regions (4 or 8; default: 8).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Optional output path. Default: <input>_sieved.tif",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.cluster_raster)
    if not in_path.exists():
        raise FileNotFoundError(f"Cluster raster not found: {in_path}")

    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_sieved.tif")

    with rasterio.open(in_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    # Treat 0 (or declared nodata) as no-data and keep unchanged.
    nodata_value = 0 if nodata is None else nodata
    valid_mask = arr != nodata_value
    if not np.any(valid_mask):
        raise ValueError("No valid cluster pixels found.")

    filtered = sieve(arr.astype(np.int32), size=args.min_pixels, connectivity=args.connectivity, mask=valid_mask)
    filtered = filtered.astype(arr.dtype)
    filtered[~valid_mask] = nodata_value

    profile.update({"count": 1, "compress": "LZW", "tiled": True})
    if nodata is None:
        profile["nodata"] = nodata_value

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)
        dst.set_band_description(1, "clusters_sieved")

    print("[DONE] Wrote:", out_path)
    print("Tip: increase --min-pixels (e.g., 16 or 25) for stronger smoothing.")


if __name__ == "__main__":
    main()
