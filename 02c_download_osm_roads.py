#!/usr/bin/env python3
"""
Optional step: Download OSM roads for AOI via Overpass API.

Outputs:
- roads_osm_raw.gpkg       (all roads from bbox query)
- roads_osm_clipped.gpkg   (roads clipped to AOI)

Notes:
- Uses AOI bounding box for query, then clips to AOI polygon.
- Includes all OSM ways with a "highway" tag.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import requests
from shapely.geometry import LineString


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download OSM road data for an AOI.")
    p.add_argument("--aoi", required=True, help="AOI polygon path (gpkg/shp/geojson).")
    p.add_argument("--outdir", required=True, help="Output folder, usually run_01.")
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=240,
        help="Overpass timeout in seconds (default: 240).",
    )
    p.add_argument(
        "--max-retries-per-endpoint",
        type=int,
        default=2,
        help="Retries per Overpass endpoint (default: 2).",
    )
    return p.parse_args()


def build_query(minx: float, miny: float, maxx: float, maxy: float, timeout_seconds: int) -> str:
    # Overpass bbox order is: south,west,north,east = miny,minx,maxy,maxx
    return f"""
[out:json][timeout:{timeout_seconds}];
(
  way["highway"]({miny},{minx},{maxy},{maxx});
);
out body;
>;
out skel qt;
""".strip()


def download_overpass(query: str, max_retries_per_endpoint: int) -> Dict:
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter",
    ]
    last_error: Exception | None = None

    for endpoint in endpoints:
        for attempt in range(1, max_retries_per_endpoint + 1):
            try:
                print(f"[INFO] Trying {endpoint} (attempt {attempt}/{max_retries_per_endpoint})")
                resp = requests.post(endpoint, data={"data": query}, timeout=300)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_error = exc
                # brief backoff before retrying
                time.sleep(2 * attempt)
                continue

    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_error}")


def parse_osm_roads(overpass_json: Dict) -> gpd.GeoDataFrame:
    elements = overpass_json.get("elements", [])
    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el.get("type") == "node"}

    rows: List[Dict] = []
    geoms = []
    for el in elements:
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {})
        if "highway" not in tags:
            continue
        node_ids = el.get("nodes", [])
        coords = [nodes[nid] for nid in node_ids if nid in nodes]
        if len(coords) < 2:
            continue
        try:
            geom = LineString(coords)
        except Exception:
            continue
        geoms.append(geom)
        rows.append(
            {
                "osm_id": int(el["id"]),
                "highway": str(tags.get("highway", "")),
                "name": str(tags.get("name", "")),
                "surface": str(tags.get("surface", "")),
                "tracktype": str(tags.get("tracktype", "")),
            }
        )

    if not rows:
        return gpd.GeoDataFrame(columns=["osm_id", "highway", "name", "surface", "tracktype", "geometry"], crs="EPSG:4326")

    return gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    aoi = gpd.read_file(args.aoi)
    if aoi.empty:
        raise ValueError("AOI is empty.")
    aoi = aoi[aoi.geometry.notna()].copy().dissolve()
    aoi4326 = aoi.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = aoi4326.total_bounds

    query = build_query(minx, miny, maxx, maxy, args.timeout_seconds)
    print("[INFO] Querying Overpass API for roads...")
    data = download_overpass(query, args.max_retries_per_endpoint)

    roads = parse_osm_roads(data)
    if roads.empty:
        raise ValueError("No OSM roads returned for AOI bbox.")

    raw_path = outdir / "roads_osm_raw.gpkg"
    roads.to_file(raw_path, driver="GPKG")
    print(f"[INFO] Wrote raw roads: {raw_path} ({len(roads)} features)")

    # Clip roads to AOI polygon.
    roads_clip = gpd.overlay(roads, aoi4326, how="intersection")
    clipped_path = outdir / "roads_osm_clipped.gpkg"
    roads_clip.to_file(clipped_path, driver="GPKG")
    print(f"[DONE] Wrote clipped roads: {clipped_path} ({len(roads_clip)} features)")


if __name__ == "__main__":
    main()
