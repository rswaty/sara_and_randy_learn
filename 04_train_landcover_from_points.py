#!/usr/bin/env python3
"""
Step 4: Train land cover model from labeled points and predict full map.

Inputs:
- predictor_stack.tif (from Step 2)
- label_points_labeled.gpkg (from Step 3 after filling class_id in QGIS)

Outputs:
- landcover_map.tif
- landcover_confidence.tif
- accuracy_report.txt
- confusion_matrix.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CLASS_NAMES = {
    1: "Urban",
    2: "Water",
    3: "Forest_Young",
    4: "Forest_Mid",
    5: "Forest_Mature",
    6: "Recently_Burned",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF from labeled points and map classes.")
    p.add_argument("--stack", required=True, help="Predictor stack raster path.")
    p.add_argument("--labels", required=True, help="Labeled points gpkg with class_id.")
    p.add_argument("--class-field", default="class_id", help="Class field name (default: class_id).")
    p.add_argument("--outdir", required=True, help="Output folder for final maps and reports.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def sample_points(stack_path: Path, labels_path: Path, class_field: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    gdf = gpd.read_file(labels_path)
    if class_field not in gdf.columns:
        raise ValueError(f"Field '{class_field}' not found in labels file.")

    # Keep only rows that have class labels.
    gdf = gdf[gdf[class_field].notna()].copy()
    if gdf.empty:
        raise ValueError("No labeled points found. Fill class_id in QGIS first.")
    gdf[class_field] = gdf[class_field].astype(int)

    with rasterio.open(stack_path) as src:
        gdf = gdf.to_crs(src.crs)
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        samples = list(src.sample(coords))
        X = np.array(samples, dtype=np.float32)
        y = gdf[class_field].values.astype(np.int16)
        meta = src.profile.copy()

    valid = np.all(np.isfinite(X), axis=1)
    X = X[valid]
    y = y[valid]
    if X.shape[0] < 30:
        raise ValueError("Too few valid labeled points after sampling.")

    return X, y, meta


def train_model(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[RandomForestClassifier, str, np.ndarray, List[int]]:
    labels = sorted(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
        oob_score=True,
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        labels=labels,
        target_names=[CLASS_NAMES.get(c, str(c)) for c in labels],
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, pred, labels=labels)
    return rf, report, cm, labels


def predict_map(rf: RandomForestClassifier, stack_path: Path, out_class: Path, out_conf: Path) -> None:
    with rasterio.open(stack_path) as src:
        arr = src.read().astype(np.float32)  # [features, h, w]
        meta = src.profile.copy()

    valid = np.all(np.isfinite(arr), axis=0)
    h, w = valid.shape
    xv = arr[:, valid].T

    pred = rf.predict(xv).astype(np.uint8)
    conf = rf.predict_proba(xv).max(axis=1).astype(np.float32)

    class_map = np.zeros((h, w), dtype=np.uint8)
    conf_map = np.zeros((h, w), dtype=np.float32)
    class_map[valid] = pred
    conf_map[valid] = conf

    class_meta = meta.copy()
    class_meta.update({"count": 1, "dtype": "uint8", "nodata": 0, "compress": "LZW", "tiled": True})
    conf_meta = meta.copy()
    conf_meta.update({"count": 1, "dtype": "float32", "nodata": 0.0, "compress": "LZW", "tiled": True})

    with rasterio.open(out_class, "w", **class_meta) as dst:
        dst.write(class_map, 1)
    with rasterio.open(out_conf, "w", **conf_meta) as dst:
        dst.write(conf_map, 1)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y, _ = sample_points(Path(args.stack), Path(args.labels), args.class_field)
    print(f"[INFO] Training samples: {X.shape[0]} points, {X.shape[1]} features")
    rf, report, cm, labels = train_model(X, y, args.seed)

    (outdir / "accuracy_report.txt").write_text(report + "\n", encoding="utf-8")
    np.savetxt(outdir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    print("[INFO] OOB score:", getattr(rf, "oob_score_", None))

    out_class = outdir / "landcover_map.tif"
    out_conf = outdir / "landcover_confidence.tif"
    predict_map(rf, Path(args.stack), out_class, out_conf)

    legend_path = outdir / "class_legend.txt"
    with legend_path.open("w", encoding="utf-8") as f:
        f.write("0=NoData\n")
        for cid in labels:
            f.write(f"{cid}={CLASS_NAMES.get(cid, 'Unknown')}\n")

    print("[DONE] Step 4 complete.")
    print(" -", out_class)
    print(" -", out_conf)
    print(" -", outdir / "accuracy_report.txt")


if __name__ == "__main__":
    main()
