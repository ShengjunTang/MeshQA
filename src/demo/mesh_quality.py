"""Mesh quality classification tooling.

This module provides a small pipeline for training a per-face quality classifier
from a labeled OBJ mesh, and then using that classifier to predict per-face quality
on new meshes (e.g., GLB input).

The training input expects an OBJ where faces are grouped into quality labels,
for example:

    g high_quality
    f ...
    g low_quality
    f ...

The output prediction can be written as an OBJ with faces grouped by predicted
quality.

This implementation is intentionally minimal, using geometric face features and
a RandomForestClassifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass
class MeshQualityModel:
    classifier: Any
    feature_names: List[str]


def _parse_obj_face_labels(obj_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse an OBJ and return per-face vertex indices + labels.

    This will attempt to infer labels from (in priority order):
    1) group names (`g ...`) or material names (`usemtl ...`) containing "high"/"low".
    2) vertex colors (v x y z r g b).

    Returns:
        verts: (V, 3) vertex coordinates
        colors: (V, 3) vertex colors in [0, 1] (default 1.0 if missing)
        faces: (F, 3) int indices into the vertex list.
        labels: (F,) {0,1} where 1 is high quality.
    """

    verts: List[Tuple[float, float, float]] = []
    colors: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    labels: List[int] = []
    current_label = 0

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("g ") or line.startswith("o "):
                name = line.split(maxsplit=1)[1].strip().lower()
                if "high" in name:
                    current_label = 1
                elif "low" in name:
                    current_label = 0
                continue
            if line.startswith("usemtl "):
                name = line.split(maxsplit=1)[1].strip().lower()
                if "high" in name:
                    current_label = 1
                elif "low" in name:
                    current_label = 0
                continue
            if line.startswith("v "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                x, y, z = parts[0], parts[1], parts[2]
                verts.append((float(x), float(y), float(z)))
                if len(parts) >= 6:
                    # OBJ can include vertex colors after xyz
                    r, g, b = float(parts[3]), float(parts[4]), float(parts[5])
                    colors.append((r, g, b))
                else:
                    colors.append((1.0, 1.0, 1.0))
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                idx = [int(p.split("/")[0]) - 1 for p in parts[:3]]
                faces.append(tuple(idx))
                labels.append(current_label)

    verts_arr = np.array(verts, dtype=float)
    colors_arr = np.array(colors, dtype=float) if colors else np.ones((len(verts), 3), dtype=float)
    faces_arr = np.array(faces, dtype=int)
    labels_arr = np.array(labels, dtype=int)

    # If all labels are the same, attempt to infer from vertex colors.
    if len(np.unique(labels_arr)) <= 1 and colors_arr.size > 0:
        face_colors = colors_arr[faces_arr].mean(axis=1)
        # High quality assumed near white; low quality assumed colored.
        is_high = np.mean(face_colors, axis=1) > 0.9
        labels_arr = is_high.astype(int)

    return verts_arr, colors_arr, faces_arr, labels_arr


def _face_features(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-face geometric features."""

    # corners of each triangle
    tri = vertices[faces]

    # edge vectors
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 1]

    # edge lengths
    l0 = np.linalg.norm(e0, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)

    # area
    cross = np.cross(e0, e1)
    area = 0.5 * np.linalg.norm(cross, axis=1)

    # aspect ratios
    lengths = np.vstack([l0, l1, l2]).T
    longest = np.max(lengths, axis=1)
    shortest = np.min(lengths, axis=1)
    aspect = np.divide(longest, shortest, out=np.zeros_like(longest), where=shortest > 0)

    # normal vector (unit)
    normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

    # curvature proxy: deviation of normals within face (small triangle => low curvature)
    normal_deviation = np.linalg.norm(normals - normals.mean(axis=0), axis=1)

    features = np.stack(
        [area, aspect, longest, shortest, normal_deviation], axis=1
    )
    return features


def _face_texture_colors(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    """Compute per-face average RGB from mesh texture (if available)."""

    # If no texture, return zeros
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return np.zeros((faces.shape[0], 3), dtype=float)

    # Try to access UV coordinates
    uv = getattr(visual, "uv", None)
    uv_faces = getattr(visual, "uv_faces", None) or getattr(visual, "uv_indices", None)
    if uv is None or uv_faces is None:
        return np.zeros((faces.shape[0], 3), dtype=float)

    # Default texture as numpy array
    image = None
    material = getattr(visual, "material", None)
    if material is not None:
        image = getattr(material, "image", None)
    if image is None:
        # some versions store image directly in visual
        image = getattr(visual, "image", None)

    if image is None:
        return np.zeros((faces.shape[0], 3), dtype=float)

    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    h, w, _ = img.shape

    # uv coordinates are expected in range [0, 1]
    face_uv = uv[uv_faces[faces]]  # (F, 3, 2)
    uv_centroid = face_uv.mean(axis=1)

    # convert to pixel coordinates
    px = np.clip((uv_centroid[:, 0] * (w - 1)).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv_centroid[:, 1]) * (h - 1)).astype(int), 0, h - 1)

    colors = img[py, px] / 255.0
    return colors


class TextureTransformer(nn.Module):
    """Transformer encoder to embed texture patches.

    This transformer accepts either a token sequence `(B, T, 3)` or a patch image
    `(B, H, W, 3)` and flattens it into tokens before encoding.
    """

    def __init__(self, dim: int = 64, depth: int = 2, heads: int = 4):
        super().__init__()
        self.token_proj = nn.Linear(3, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pool = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W, 3) or (B, T, 3)."""
        if x.ndim == 4:
            # Flatten spatial dims into tokens
            b, h, w, c = x.shape
            x = x.view(b, h * w, c)
        x = self.token_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.pool(x)


def _sample_texture_patches(mesh: trimesh.Trimesh, faces: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """Sample a small texture patch around each face's UV centroid."""

    visual = getattr(mesh, "visual", None)
    if visual is None:
        return np.zeros((faces.shape[0], patch_size, patch_size, 3), dtype=np.float32)

    uv = getattr(visual, "uv", None)
    uv_faces = getattr(visual, "uv_faces", None) or getattr(visual, "uv_indices", None)
    if uv is None or uv_faces is None:
        return np.zeros((faces.shape[0], patch_size, patch_size, 3), dtype=np.float32)

    image = None
    material = getattr(visual, "material", None)
    if material is not None:
        image = getattr(material, "image", None)
    if image is None:
        image = getattr(visual, "image", None)

    if image is None:
        return np.zeros((faces.shape[0], patch_size, patch_size, 3), dtype=np.float32)

    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    h, w, _ = img.shape
    face_uv = uv[uv_faces[faces]]  # (F, 3, 2)
    centroid = face_uv.mean(axis=1)  # (F, 2)

    # Create a sampling grid around centroid
    patch_coords = np.linspace(-0.5, 0.5, patch_size)
    grid_uv = np.stack(np.meshgrid(patch_coords, patch_coords), axis=-1)  # (P,P,2)
    grid_uv = grid_uv[None, ...] + centroid[:, None, None, :]

    # Convert to pixel coords
    grid_x = np.clip((grid_uv[..., 0] * w).astype(int), 0, w - 1)
    grid_y = np.clip(((1.0 - grid_uv[..., 1]) * h).astype(int), 0, h - 1)

    patches = img[grid_y, grid_x] / 255.0
    return patches.astype(np.float32)


class TexturePatchEncoder(nn.Module):
    """A small CNN to encode sampled texture patches."""

    def __init__(self, patch_size: int = 16, emb_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MeshQualityDataset(Dataset):
    def __init__(
        self,
        geom_features: np.ndarray,
        texture_patches: np.ndarray,
        labels: np.ndarray,
    ):
        self.geom = torch.from_numpy(geom_features.astype(np.float32))
        # Keep texture patches as (B, H, W, 3) so the transformer can ingest raw pixels.
        self.tex = torch.from_numpy(texture_patches.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.geom[idx], self.tex[idx], self.labels[idx]


class GeometryTextureNet(nn.Module):
    def __init__(self, geom_dim: int, tex_emb: int = 64, hidden: int = 128):
        super().__init__()
        # Use a transformer to embed raw texture patches into a per-face vector.
        self.texture_encoder = TextureTransformer(dim=tex_emb)
        self.geom_mlp = nn.Sequential(
            nn.Linear(geom_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + tex_emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, geom: torch.Tensor, tex: torch.Tensor) -> torch.Tensor:
        g = self.geom_mlp(geom)
        t = self.texture_encoder(tex)
        x = torch.cat([g, t], dim=1)
        return self.head(x).squeeze(1)


def train_from_labeled_obj(
    obj_path: str,
    glb_path: str | None = None,
    random_state: int = 0,
    test_size: float = 0.2,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Tuple[MeshQualityModel, float]:
    """Train a per-face quality classifier from a labeled OBJ.

    If `glb_path` is provided, texture-based features will be extracted from the
    GLB mesh and used in an end-to-end neural network.
    """

    verts, colors, faces, labels = _parse_obj_face_labels(obj_path)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) <= 1:
        raise ValueError(
            "Training labels are all the same. Make sure the OBJ uses groups/materials or vertex colors to indicate quality."
        )
    print(f"Loaded OBJ with label distribution: {dict(zip(unique, counts))}")

    geom = _face_features(verts, faces)

    # texture patches (B, H, W, 3)
    tex_patches = np.zeros((faces.shape[0], 16, 16, 3), dtype=np.float32)
    if glb_path is not None:
        mesh = trimesh.load(glb_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Expected a single mesh in the GLB file")
        tex_patches = _sample_texture_patches(mesh, faces, patch_size=16)

    dataset = MeshQualityDataset(geom, tex_patches, labels)
    n_train = int(len(dataset) * (1 - test_size))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(random_state))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryTextureNet(geom_dim=geom.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for g, t, y in train_loader:
            g, t, y = g.to(device), t.to(device), y.to(device)
            logits = model(g, t)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * g.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            total = 0
            for g, t, y in val_loader:
                g, t, y = g.to(device), t.to(device), y.to(device)
                logits = model(g, t)
                val_loss += loss_fn(logits, y).item() * g.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y).sum().item()
                total += y.size(0)
            val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    # Save model and feature names
    trained = MeshQualityModel(classifier=model, feature_names=[])
    return trained, val_acc


def predict_mesh_quality(
    model: MeshQualityModel,
    mesh_path: str,
    output_obj: str,
    max_faces: Optional[int] = 0,
    scores_out: Optional[str] = None,
) -> None:
    """Predict per-face quality for a mesh and write a labeled OBJ.

    `max_faces <= 0` disables subsampling and uses all faces from the input mesh.

    If `scores_out` is provided, writes a CSV with per-face confidence scores.
    """

    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a single mesh in the input file")

    verts = mesh.vertices
    faces = mesh.faces

    total_faces = faces.shape[0]
    print(f"Loaded GLB with {total_faces:,} faces")

    if max_faces is not None and max_faces > 0 and total_faces > max_faces:
        idx = np.random.default_rng(0).choice(total_faces, size=max_faces, replace=False)
        faces = faces[idx]
        print(f"Subsampled to {faces.shape[0]:,} faces for prediction")
    else:
        print(f"Using all {faces.shape[0]:,} faces for prediction")

    features = _face_features(verts, faces)

    # Predict per-face quality scores (0..1) and binary labels.
    clf = model.classifier
    face_scores: np.ndarray
    if isinstance(clf, GeometryTextureNet):
        patches = _sample_texture_patches(mesh, faces, patch_size=16)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.eval()
        with torch.no_grad():
            g = torch.from_numpy(features.astype(np.float32)).to(device)
            t = torch.from_numpy(patches.astype(np.float32)).to(device)
            logits = clf(g, t)
            face_scores = torch.sigmoid(logits).cpu().numpy()
        labels = (face_scores > 0.5).astype(int)
    else:
        # Prefer probability outputs when available.
        if hasattr(clf, "predict_proba"):
            face_scores = clf.predict_proba(features)[:, 1]
            labels = (face_scores > 0.5).astype(int)
        elif hasattr(clf, "decision_function"):
            logits = clf.decision_function(features)
            face_scores = 1 / (1 + np.exp(-logits))
            labels = (face_scores > 0.5).astype(int)
        else:
            labels = clf.predict(features)
            face_scores = labels.astype(float)

    print(
        f"Predicted {faces.shape[0]:,} faces (mean confidence {float(face_scores.mean()):.3f})"
    )

    # Optionally write per-face confidence scores (and labels) to a CSV for analysis.
    if scores_out:
        import csv

        with open(scores_out, "w", newline="", encoding="utf-8") as f_scores:
            writer = csv.writer(f_scores)
            writer.writerow(["face_index", "score", "label"])
            for i, (s, l) in enumerate(zip(face_scores, labels)):
                writer.writerow([i, float(s), int(l)])
        print(f"Wrote per-face scores to {scores_out}")

    # Write OBJ with two groups: high_quality and low_quality
    output_dir = os.path.dirname(output_obj)
    os.makedirs(output_dir or ".", exist_ok=True)

    mtl_name = os.path.splitext(os.path.basename(output_obj))[0] + ".mtl"
    mtl_path = os.path.join(output_dir, mtl_name)

    # Create a simple MTL file with two materials
    with open(mtl_path, "w", encoding="utf-8") as mtl:
        mtl.write("# Materials for mesh_quality output\n")
        mtl.write("newmtl high_quality\n")
        mtl.write("Kd 0.0 1.0 0.0\n")
        mtl.write("newmtl low_quality\n")
        mtl.write("Kd 1.0 0.0 0.0\n")

    # Compute a per-vertex score so viewers that ignore mtls still show a heatmap.
    vertex_scores = np.zeros((verts.shape[0],), dtype=float)
    counts = np.zeros((verts.shape[0],), dtype=int)
    for face, score in zip(faces, face_scores):
        for vi in face:
            vertex_scores[vi] += score
            counts[vi] += 1
    counts = np.maximum(counts, 1)
    vertex_scores = vertex_scores / counts

    def _score_to_rgb(score: float) -> Tuple[float, float, float]:
        # Map score [0,1] to red->green heatmap.
        score = float(np.clip(score, 0.0, 1.0))
        return (1.0 - score, score, 0.0)

    with open(output_obj, "w", encoding="utf-8") as f:
        f.write("# OBJ generated by mesh_quality\n")
        f.write(f"mtllib {mtl_name}\n")
        for v_idx, v in enumerate(verts):
            r, g, b = _score_to_rgb(vertex_scores[v_idx])
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

        # Write faces, switching material when label changes.
        current_label = None
        for face, lbl in zip(faces, labels):
            if lbl != current_label:
                current_label = lbl
                mat_name = "high_quality" if lbl == 1 else "low_quality"
                f.write(f"usemtl {mat_name}\n")
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
