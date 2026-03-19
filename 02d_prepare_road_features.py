#!/usr/bin/env python3
"""
Optional step: Rasterize roads and append to predictor stack.

Inputs:
- predictor_stack.tif
- roads_osm_clipped.gpkg (from 02c)

Outputs:
- road_centerline.tif
- road_buffer_<N>m.tif
- predictor_stack_plus_roads.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare road-derived raster features.")
    p.add_argument("--run-dir", required=True, help="Run directory (e.g., run_01).")
    p.add_argument(
        "--roads",
        default=None,
        help="Roads vector path. Default: <run-dir>/roads_osm_clipped.gpkg",
    )
    p.add_argument(
        "--road-buffer-m",
        type=float,
        default=30.0,
        help="Road buffer size in meters for second feature (default: 30).",
    )
    return p.parse_args()


def read_band_names(src: rasterio.io.DatasetReader) -> List[str]:
    names = []
    for i in range(1, src.count + 1):
        desc = src.descriptions[i - 1]
        names.append(desc if desc else f"band_{i}")
    return names


def write_single(path: Path, arr: np.ndarray, profile: dict, desc: str) -> None:
    p = profile.copy()
    p.update({"count": 1, "dtype": "uint8", "compress": "LZW", "tiled": True, "nodata": 0})
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(np.uint8), 1)
        dst.set_band_description(1, desc)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    stack_path = run_dir / "predictor_stack.tif"
    roads_path = Path(args.roads) if args.roads else (run_dir / "roads_osm_clipped.gpkg")

    if not stack_path.exists():
        raise FileNotFoundError(f"Missing stack: {stack_path}")
    if not roads_path.exists():
        raise FileNotFoundError(f"Missing roads layer: {roads_path}")

    roads = gpd.read_file(roads_path)
    roads = roads[roads.geometry.notna()].copy()
    if roads.empty:
        raise ValueError("Roads layer has no valid geometries.")

    with rasterio.open(stack_path) as src:
        stack = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        out_shape = (src.height, src.width)
        feature_names = read_band_names(src)

    roads = roads.to_crs(crs)

    # Binary centerline mask.
    centerline = rasterize(
        [(geom, 1) for geom in roads.geometry if geom is not None and not geom.is_empty],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    # Binary buffered-road mask.
    road_buffer = roads.geometry.buffer(args.road_buffer_m)
    buffered = rasterize(
        [(geom, 1) for geom in road_buffer if geom is not None and not geom.is_empty],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    center_path = run_dir / "road_centerline.tif"
    buffer_path = run_dir / f"road_buffer_{int(args.road_buffer_m)}m.tif"
    write_single(center_path, centerline, profile, "road_centerline")
    write_single(buffer_path, buffered, profile, f"road_buffer_{int(args.road_buffer_m)}m")
    print("[INFO] Wrote:", center_path)
    print("[INFO] Wrote:", buffer_path)

    # Append road layers to predictor stack.
    plus_path = run_dir / "predictor_stack_plus_roads.tif"
    plus = np.concatenate([stack, centerline[np.newaxis, :, :].astype(np.float32), buffered[np.newaxis, :, :].astype(np.float32)], axis=0)

    plus_profile = profile.copy()
    plus_profile.update({"count": plus.shape[0], "dtype": "float32", "nodata": np.nan, "compress": "LZW", "tiled": True})
    with rasterio.open(plus_path, "w", **plus_profile) as dst:
        for i in range(plus.shape[0]):
            dst.write(plus[i], i + 1)
        for i, name in enumerate(feature_names, start=1):
            dst.set_band_description(i, name)
        dst.set_band_description(len(feature_names) + 1, "road_centerline")
        dst.set_band_description(len(feature_names) + 2, f"road_buffer_{int(args.road_buffer_m)}m")

    print("[DONE] Wrote:", plus_path)
    print("Use this stack in step 4 with --stack predictor_stack_plus_roads.tif to test impact.")


if __name__ == "__main__":
    main()
