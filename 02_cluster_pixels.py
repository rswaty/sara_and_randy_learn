#!/usr/bin/env python3
"""
Step 2: Build predictor stack and cluster similar pixels (unsupervised).

Input from Step 1:
- composites/pre_composite.tif
- composites/post_composite.tif
- composites/recovery_composite.tif

Outputs:
- predictor_stack.tif
- clusters_kXX.tif
- cluster_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stack and run KMeans clustering.")
    p.add_argument("--run-dir", required=True, help="Folder created by Step 1 (contains composites).")
    p.add_argument("--k", type=int, default=24, help="Number of clusters (default: 24).")
    p.add_argument("--sample-size", type=int, default=250000, help="Max pixels to fit KMeans.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def read_composite(path: Path) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # [bands, rows, cols]
        meta = src.profile.copy()
    return arr, meta


def indices(comp: np.ndarray) -> Dict[str, np.ndarray]:
    # comp order expected: blue, green, red, nir, swir16, swir22
    blue, green, red, nir, swir16, swir22 = comp
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    nbr = (nir - swir22) / (nir + swir22 + eps)
    ndmi = (nir - swir16) / (nir + swir16 + eps)
    return {"ndvi": ndvi.astype(np.float32), "nbr": nbr.astype(np.float32), "ndmi": ndmi.astype(np.float32)}


def write_multiband(path: Path, bands: List[np.ndarray], names: List[str], ref_meta: dict) -> None:
    profile = ref_meta.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": len(bands),
            "dtype": "float32",
            "compress": "LZW",
            "tiled": True,
            "nodata": np.nan,
        }
    )
    with rasterio.open(path, "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(bands, names), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, name)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    comp_dir = run_dir / "composites"

    pre, meta = read_composite(comp_dir / "pre_composite.tif")
    post, _ = read_composite(comp_dir / "post_composite.tif")
    rec, _ = read_composite(comp_dir / "recovery_composite.tif")

    pre_i = indices(pre)
    post_i = indices(post)
    rec_i = indices(rec)
    dnbr = (pre_i["nbr"] - post_i["nbr"]).astype(np.float32)
    dndvi = (pre_i["ndvi"] - post_i["ndvi"]).astype(np.float32)

    bands = []
    names = []

    def add(prefix: str, c: np.ndarray, idx: Dict[str, np.ndarray]) -> None:
        for bname, arr in zip(
            ["blue", "green", "red", "nir", "swir16", "swir22"],
            c,
        ):
            bands.append(arr)
            names.append(f"{prefix}_{bname}")
        for iname in ["ndvi", "nbr", "ndmi"]:
            bands.append(idx[iname])
            names.append(f"{prefix}_{iname}")

    add("pre", pre, pre_i)
    add("post", post, post_i)
    add("recovery", rec, rec_i)
    bands.extend([dnbr, dndvi])
    names.extend(["dnbr_pre_post", "dndvi_pre_post"])

    stack_path = run_dir / "predictor_stack.tif"
    write_multiband(stack_path, bands, names, meta)
    print("[INFO] Wrote predictor stack:", stack_path)

    X = np.stack(bands, axis=0)  # [features, h, w]
    valid = np.all(np.isfinite(X), axis=0)
    rows, cols = valid.shape
    xv = X[:, valid].T  # [n, f]

    if xv.shape[0] == 0:
        raise ValueError("No valid pixels found for clustering.")

    rng = np.random.default_rng(args.seed)
    n_fit = min(args.sample_size, xv.shape[0])
    fit_idx = rng.choice(xv.shape[0], size=n_fit, replace=False)
    x_fit = xv[fit_idx]

    scaler = StandardScaler()
    x_fit_s = scaler.fit_transform(x_fit)
    xv_s = scaler.transform(xv)

    print(f"[INFO] Fitting KMeans k={args.k} with {n_fit} sampled pixels...")
    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    km.fit(x_fit_s)
    labels = km.predict(xv_s).astype(np.int16) + 1  # 1..k

    cluster_map = np.zeros((rows, cols), dtype=np.int16)
    cluster_map[valid] = labels

    cl_path = run_dir / f"clusters_k{args.k:02d}.tif"
    profile = meta.copy()
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "int16",
            "compress": "LZW",
            "tiled": True,
            "nodata": 0,
        }
    )
    with rasterio.open(cl_path, "w", **profile) as dst:
        dst.write(cluster_map, 1)
    print("[INFO] Wrote cluster map:", cl_path)

    # Write simple summary table: cluster ID + pixel count.
    counts = np.bincount(labels, minlength=args.k + 1)
    csv_path = run_dir / "cluster_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "pixel_count"])
        for cid in range(1, args.k + 1):
            w.writerow([cid, int(counts[cid])])
    print("[INFO] Wrote cluster summary:", csv_path)
    print("[DONE] Step 2 complete.")


if __name__ == "__main__":
    main()
