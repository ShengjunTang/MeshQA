"""Point cloud classification utilities.

This module provides a small pipeline that takes 3D point cloud coordinates
and trains a Random Forest classifier.

It is intentionally lightweight and uses only NumPy + scikit-learn.

The workflow is:
- Generate (or load) XYZ point data + per-point labels
- Build local geometric features from k-nearest neighbors
- Train and evaluate a RandomForestClassifier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def compute_local_features(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Compute local geometric features for a point cloud.

    For each point, this returns a feature vector that includes:
    - the original (x, y, z)
    - mean distance to the k nearest neighbors
    - standard deviation of distances to the k nearest neighbors
    - local linearity / planarity / scattering (PCA eigenvalue ratios)

    This is a lightweight alternative to full point cloud descriptors.

    Args:
        points: (N, 3) array of XYZ coordinates.
        k: Number of neighbors (including the point itself).

    Returns:
        (N, F) feature array.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    # NearestNeighbors includes the point itself in the first neighbor.
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(points)
    distances, indices = nbrs.kneighbors(points, return_distance=True)

    # Drop self-distance (0.0) for statistics.
    # distances shape: (N, k)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    mean_dist = np.mean(distances, axis=1)
    std_dist = np.std(distances, axis=1)

    # Compute local PCA feature ratios (linearity / planarity / scattering)
    ratios = np.zeros((points.shape[0], 3), dtype=float)
    for i, neigh_idxs in enumerate(indices):
        neigh_pts = points[neigh_idxs]
        cov = np.cov(neigh_pts.T)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(eig)[::-1]

        # numerical stability
        if eig[0] <= 0:
            eig = np.maximum(eig, 1e-12)

        linearity = (eig[0] - eig[1]) / eig[0]
        planarity = (eig[1] - eig[2]) / eig[0]
        scattering = eig[2] / eig[0]
        ratios[i] = [linearity, planarity, scattering]

    return np.hstack([points, mean_dist[:, None], std_dist[:, None], ratios])


def train_random_forest(
    points: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 0,
    n_estimators: int = 100,
    k_neighbors: int = 16,
) -> TrainingResult:
    """Train a Random Forest classifier on a point cloud.

    Args:
        points: (N, 3) array of XYZ coordinates.
        labels: (N,) array of integer labels.
        test_size: Fraction of points reserved for evaluation.
        random_state: Seed for reproducibility.
        n_estimators: Number of trees in the forest.
        k_neighbors: Number of neighbors used for feature extraction.

    Returns:
        A TrainingResult containing the fitted model and train/test splits.
    """

    X = compute_local_features(points, k=k_neighbors)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    return TrainingResult(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> str:
    """Evaluate a trained classifier and return a text report."""

    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred)


def make_synthetic_dataset(
    n_points: int = 4000, n_classes: int = 3, random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple synthetic 3D point cloud dataset.

    This is meant for quick experimentation and demo purposes.

    Returns:
        points: (N, 3) XYX coordinates.
        labels: (N,) integer labels in [0, n_classes).
    """

    rng = np.random.RandomState(random_state)
    centers = rng.uniform(-5.0, 5.0, size=(n_classes, 3))

    points = []
    labels = []
    per_class = max(1, n_points // n_classes)
    for label in range(n_classes):
        pts = centers[label] + rng.normal(scale=0.5, size=(per_class, 3))
        points.append(pts)
        labels.append(np.full(per_class, label, dtype=int))

    points = np.vstack(points)[:n_points]
    labels = np.concatenate(labels)[:n_points]
    return points, labels


def load_pcd(file_path: str) -> np.ndarray:
    """Load a subset of fields from a PCD (Point Cloud Data) file.

    Currently supports `ascii` and `binary` PCD formats.

    The returned array is (N, 3) representing the first 3 coordinate fields (usually x,y,z).

    Args:
        file_path: Path to the .pcd file.

    Returns:
        (N, 3) NumPy array of point coordinates.
    """

    with open(file_path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file while reading PCD header")
            line_str = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(line_str)
            if line_str.startswith("DATA"):
                break

        header = { }
        for line in header_lines:
            if not line or line.startswith("#"):
                continue
            key, *vals = line.split()
            header[key] = " ".join(vals)

        if "FIELDS" not in header or "SIZE" not in header or "TYPE" not in header:
            raise ValueError("Unsupported or malformed PCD header")

        fields = header["FIELDS"].split()
        sizes = list(map(int, header["SIZE"].split()))
        types = header["TYPE"].split()

        count_vals = header.get("COUNT", "1").split()
        counts = [int(x) for x in count_vals]
        if len(counts) == 1 and len(fields) > 1:
            counts = [counts[0]] * len(fields)
        if len(counts) != len(fields):
            # Fallback: assume scalar fields if COUNT is malformed
            counts = [1] * len(fields)

        # Build dtype for numpy structured array
        dtype_fields = []
        for name, size, typ, c in zip(fields, sizes, types, counts):
            if typ == "F":
                np_type = {4: "<f4", 8: "<f8"}.get(size)
            elif typ == "I":
                np_type = {1: "<i1", 2: "<i2", 4: "<i4"}.get(size)
            elif typ == "U":
                np_type = {1: "<u1", 2: "<u2", 4: "<u4"}.get(size)
            else:
                np_type = None
            if np_type is None:
                raise ValueError(f"Unsupported PCD field type: {typ}{size}")

            # allow multi-element fields via shape in dtype
            if c != 1:
                dtype_fields.append((name, np_type, c))
            else:
                dtype_fields.append((name, np_type))

        data_format = header.get("DATA", "ascii").lower()

        if data_format == "ascii":
            # Remaining file is text lines for points
            text = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            values = []
            for line in text:
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < len(fields):
                    continue
                values.append(parts[: len(fields)])
            arr = np.array(values, dtype=float)
        elif data_format == "binary":
            # Binary data starts immediately after DATA line
            raw = f.read()
            arr_struct = np.frombuffer(raw, dtype=np.dtype(dtype_fields))

            # Prefer to return a clean (N,3) XYZ array when possible.
            if {"x", "y", "z"}.issubset(arr_struct.dtype.names):
                return np.vstack([arr_struct["x"], arr_struct["y"], arr_struct["z"]]).T

            # Fallback: try to interpret as flat float array
            try:
                arr = arr_struct.view(np.float32).reshape(-1, len(fields))
            except Exception:
                raise ValueError("Unable to interpret PCD binary payload as float matrix")
        else:
            raise ValueError(f"Unsupported PCD data format: {data_format}")

        # Attempt to return x,y,z fields
        if {"x", "y", "z"}.issubset(fields):
            xyz_indices = [fields.index(c) for c in ("x", "y", "z")]
            return arr[:, xyz_indices]

        # Fallback: take first three columns
        return np.asarray(arr)[:, :3]


def save_labeled_pcd(output_path: str, points: np.ndarray, labels: np.ndarray) -> None:
    """Write an ASCII PCD file containing XYZ and label for each point."""

    if points.shape[0] != labels.shape[0]:
        raise ValueError("points and labels must have the same length")

    header = [
        "# PCD generated by demo",
        "VERSION 0.7",
        "FIELDS x y z label",
        "SIZE 4 4 4 4",
        "TYPE F F F I",
        "COUNT 1 1 1 1",
        f"WIDTH {points.shape[0]}",
        "HEIGHT 1",
        f"POINTS {points.shape[0]}",
        "DATA ascii",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        for (x, y, z), lbl in zip(points, labels):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(lbl)}\n")


def run_demo(
    pcd_file: str | None = None,
    output_dir: str | None = None,
    max_points: int = 200_000,
) -> None:
    """Run a small demo that trains and evaluates a point cloud classifier."""

    if pcd_file:
        print(f"Loading point cloud from: {pcd_file}")
        points = load_pcd(pcd_file)
        # Create dummy labels by clustering or simple partitioning.
        # Here we do a very naive split by (x + y + z) sign.
        labels = (points.sum(axis=1) > np.median(points.sum(axis=1))).astype(int)

        print(f"Loaded {points.shape[0]} points from PCD.")

        if points.shape[0] > max_points:
            print(f"Subsampling to {max_points} points for demo speed...")
            choice = np.random.default_rng(0).choice(points.shape[0], size=max_points, replace=False)
            points = points[choice]
            labels = labels[choice]
    else:
        print("Generating synthetic point cloud...")
        points, labels = make_synthetic_dataset(n_points=4000, n_classes=3, random_state=42)

    if output_dir:
        import os

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "labeled_points.pcd")
        print(f"Saving labeled point cloud to: {out_path}")
        save_labeled_pcd(out_path, points, labels)

    print("Extracting features and training Random Forest...")
    result = train_random_forest(
        points,
        labels,
        test_size=0.25,
        random_state=42,
        n_estimators=150,
        k_neighbors=16,
    )

    report = evaluate_model(result.model, result.X_test, result.y_test)
    print("\nClassification report:\n")
    print(report)
