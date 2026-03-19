#!/usr/bin/env python3
"""
Step 1: Download Sentinel-2 scenes and build cloud-masked composites.

Outputs:
- raw_s2/                       downloaded per-scene bands
- composites/pre_composite.tif
- composites/post_composite.tif
- composites/recovery_composite.tif
- quicklooks/*_rgb_quicklook.tif
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import requests
import rasterio
from pystac_client import Client
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

S2_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]
BAD_SCL_VALUES = {0, 1, 2, 3, 8, 9, 10}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Sentinel-2 and make composites.")
    p.add_argument("--aoi", required=True, help="AOI polygon path (gpkg/shp/geojson).")
    p.add_argument("--outdir", required=True, help="Output folder for step 1.")
    p.add_argument("--pre", required=True, help="Pre period (YYYY-MM-DD/YYYY-MM-DD).")
    p.add_argument("--post", required=True, help="Post period (YYYY-MM-DD/YYYY-MM-DD).")
    p.add_argument("--recovery", required=True, help="Recovery period (YYYY-MM-DD/YYYY-MM-DD).")
    p.add_argument("--max-cloud", type=float, default=35.0, help="Max eo:cloud_cover.")
    p.add_argument("--max-scenes", type=int, default=6, help="Max scenes per period.")
    return p.parse_args()


def search_items(catalog: Client, bbox: Sequence[float], dt: str, max_cloud: float) -> List:
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=dt,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


def dominant_group(items: Sequence) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick a grouping key to keep scenes spatially consistent.

    Preference order:
    1) s2:mgrs_tile
    2) proj:epsg
    3) no grouping (None, None)
    """
    tiles = [i.properties.get("s2:mgrs_tile") for i in items if i.properties.get("s2:mgrs_tile")]
    if tiles:
        return "s2:mgrs_tile", str(Counter(tiles).most_common(1)[0][0])

    epsgs = [i.properties.get("proj:epsg") for i in items if i.properties.get("proj:epsg")]
    if epsgs:
        return "proj:epsg", str(Counter(epsgs).most_common(1)[0][0])

    return None, None


def select_items(items: Sequence, n: int) -> List:
    return sorted(items, key=lambda x: x.properties.get("eo:cloud_cover", 100.0))[:n]


def download(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def fetch_assets(items: Sequence, raw_dir: Path) -> List[Dict[str, Path]]:
    out = []
    for item in items:
        rec: Dict[str, Path] = {"id": Path(item.id)}
        for b in S2_BANDS:
            href = item.assets[b].href
            path = raw_dir / f"{item.id}_{b}.tif"
            download(href, path)
            rec[b] = path
        out.append(rec)
    return out


def build_grid(reference_red: Path, aoi_target: gpd.GeoDataFrame) -> Dict:
    with rasterio.open(reference_red) as src:
        bounds = aoi_target.total_bounds
        win = src.window(*bounds).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        transform = src.window_transform(win)
        h = int(win.height)
        w = int(win.width)
        aoi_mask = ~geometry_mask(
            aoi_target.geometry,
            out_shape=(h, w),
            transform=transform,
            invert=False,
        )
        return {"crs": src.crs, "transform": transform, "height": h, "width": w, "aoi_mask": aoi_mask}


def regrid(path: Path, grid: Dict, resampling: Resampling) -> np.ndarray:
    out = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def composite(assets: List[Dict[str, Path]], grid: Dict) -> Dict[str, np.ndarray]:
    stacks = {b: [] for b in S2_BANDS if b != "scl"}
    for item in assets:
        scl = regrid(item["scl"], grid, Resampling.nearest)
        good = np.isfinite(scl) & (~np.isin(scl.astype(np.int16), list(BAD_SCL_VALUES))) & grid["aoi_mask"]
        for b in stacks:
            arr = regrid(item[b], grid, Resampling.bilinear) / 10000.0
            arr[~good] = np.nan
            stacks[b].append(arr)

    comp = {}
    for b, arrs in stacks.items():
        comp[b] = np.nanmedian(np.stack(arrs, axis=0), axis=0).astype(np.float32)
        comp[b][~grid["aoi_mask"]] = np.nan
    return comp


def write_multiband(path: Path, arrays: List[np.ndarray], names: List[str], grid: Dict) -> None:
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": len(arrays),
        "dtype": "float32",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "LZW",
        "tiled": True,
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(arrays, names), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, name)


def write_rgb(path: Path, comp: Dict[str, np.ndarray], grid: Dict) -> None:
    def stretch(a: np.ndarray, lo: float = 0.02, hi: float = 0.30) -> np.ndarray:
        x = (a - lo) / (hi - lo)
        x = np.clip(x, 0, 1) * 255.0
        x[~np.isfinite(a)] = 0
        return x.astype(np.uint8)

    r = stretch(comp["red"])
    g = stretch(comp["green"])
    b = stretch(comp["blue"])
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": 3,
        "dtype": "uint8",
        "crs": grid["crs"],
        "transform": grid["transform"],
        "compress": "LZW",
        "tiled": True,
        "nodata": 0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(r, 1)
        dst.write(g, 2)
        dst.write(b, 3)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    raw_dir = outdir / "raw_s2"
    comp_dir = outdir / "composites"
    ql_dir = outdir / "quicklooks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)
    ql_dir.mkdir(parents=True, exist_ok=True)

    aoi = gpd.read_file(args.aoi).dissolve()
    aoi4326 = aoi.to_crs("EPSG:4326")
    bbox = list(aoi4326.total_bounds)

    catalog = Client.open("https://earth-search.aws.element84.com/v1")
    periods = {"pre": args.pre, "post": args.post, "recovery": args.recovery}
    selected: Dict[str, List] = {}

    for name, dt in periods.items():
        items = search_items(catalog, bbox, dt, args.max_cloud)
        if not items:
            raise ValueError(f"No scenes found for {name} period ({dt}).")
        group_key, group_val = dominant_group(items)
        if group_key is None:
            # Fallback if STAC properties do not expose tile/epsg.
            candidate_items = items
            group_msg = "no tile/epsg grouping"
        else:
            candidate_items = [i for i in items if str(i.properties.get(group_key)) == group_val]
            group_msg = f"{group_key}={group_val}"

        chosen = select_items(candidate_items, args.max_scenes)
        if not chosen:
            raise ValueError(f"No scenes selected for {name} period.")
        selected[name] = chosen
        print(f"[INFO] {name}: {len(candidate_items)} candidate scenes ({group_msg}), selected {len(chosen)}")

    pre_assets = fetch_assets(selected["pre"], raw_dir)
    with rasterio.open(pre_assets[0]["red"]) as src:
        aoi_target = aoi.to_crs(src.crs)
    grid = build_grid(pre_assets[0]["red"], aoi_target)

    post_assets = fetch_assets(selected["post"], raw_dir)
    rec_assets = fetch_assets(selected["recovery"], raw_dir)

    for name, assets in [("pre", pre_assets), ("post", post_assets), ("recovery", rec_assets)]:
        print(f"[INFO] Building {name} composite...")
        comp = composite(assets, grid)
        bands = [comp[k] for k in ["blue", "green", "red", "nir", "swir16", "swir22"]]
        names = [f"{name}_{k}" for k in ["blue", "green", "red", "nir", "swir16", "swir22"]]
        write_multiband(comp_dir / f"{name}_composite.tif", bands, names, grid)
        write_rgb(ql_dir / f"{name}_rgb_quicklook.tif", comp, grid)

    print("[DONE] Step 1 complete.")
    print(" -", comp_dir / "pre_composite.tif")
    print(" -", comp_dir / "post_composite.tif")
    print(" -", comp_dir / "recovery_composite.tif")
    print(" -", ql_dir / "pre_rgb_quicklook.tif")


if __name__ == "__main__":
    main()
