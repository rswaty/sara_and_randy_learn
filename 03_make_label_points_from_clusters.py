#!/usr/bin/env python3
"""
Step 3: Create candidate training points from cluster map.

Input:
- clusters_kXX.tif (from Step 2)

Output:
- label_points_unlabeled.gpkg

You label these points in QGIS by filling class_id (1..6).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stratified random points by cluster.")
    p.add_argument("--cluster-raster", required=True, help="Path to clusters_kXX.tif")
    p.add_argument("--out", required=True, help="Output gpkg path for unlabeled points.")
    p.add_argument(
        "--points-per-cluster",
        type=int,
        default=40,
        help="Target points per cluster (default: 40).",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    with rasterio.open(args.cluster_raster) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    clusters = sorted([c for c in np.unique(arr) if c > 0])
    if not clusters:
        raise ValueError("No cluster labels (>0) found.")

    points: List[Point] = []
    rows_out = []

    for cid in clusters:
        idx = np.argwhere(arr == cid)
        if idx.size == 0:
            continue
        n = min(args.points_per_cluster, idx.shape[0])
        chosen = idx[rng.choice(idx.shape[0], size=n, replace=False)]

        for r, c in chosen:
            x, y = rasterio.transform.xy(transform, int(r), int(c), offset="center")
            points.append(Point(x, y))
            rows_out.append(
                {
                    "cluster_id": int(cid),
                    "row": int(r),
                    "col": int(c),
                    "class_id": None,  # user fills this in QGIS
                }
            )

    gdf = gpd.GeoDataFrame(rows_out, geometry=points, crs=crs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    print(f"[DONE] Step 3 complete. Wrote {len(gdf)} points to {out_path}")
    print("Now open this layer in QGIS and fill class_id values (1..6).")


if __name__ == "__main__":
    main()
