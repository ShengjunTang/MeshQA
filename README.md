# demo

A minimal Python project scaffolded by Copilot.

This project includes a small example of a 3D point cloud classifier that uses a Random Forest.

## Run

Run the built-in synthetic demo:

```bash
PYTHONPATH=src python -m demo
```

Run with a PCD point cloud file (ASCII or binary):

```bash
PYTHONPATH=src python -m demo pointcloud --pcd path/to/your.pcd
```

Save a labeled point cloud output (PCD) into a folder:

```bash
PYTHONPATH=src python -m demo pointcloud --pcd path/to/your.pcd --out output_folder
```

---

## Mesh quality classifier (OBJ/GLB)

Train a classifier from an OBJ where faces are grouped into quality labels (e.g. `g high_quality` / `g low_quality`):

```bash
PYTHONPATH=src python -m demo mesh-quality train --obj labeled.obj --out model.pkl
```

If you also have a GLB with texture/UV you want to include in training:

```bash
PYTHONPATH=src python -m demo mesh-quality train --obj labeled.obj --glb textured.glb --out model.pkl
```

Predict quality on a new mesh (OBJ or GLB) and write a labeled OBJ output (with material coloring and per-vertex heatmap based on confidence scores):

```bash
PYTHONPATH=src python -m demo mesh-quality predict --model model.pkl --in new_model.glb --out labeled.obj
```

You can also export per-face confidence scores (0..1) to a CSV file:

```bash
PYTHONPATH=src python -m demo mesh-quality predict --model model.pkl --in new_model.glb --out labeled.obj --scores scores.csv
```

Subsampling is disabled by default (the tool will use all faces from the input mesh). If you want to limit the number of faces for speed, set `--max-faces` to a positive number:

```bash
PYTHONPATH=src python -m demo mesh-quality predict --model model.pkl --in new_model.glb --out labeled.obj --max-faces 200000
```

The output will produce `labeled.obj` plus `labeled.mtl`, with high-quality faces in green and low-quality faces in red.

## Notes

- The demo generates a synthetic 3D point cloud and trains a Random Forest classifier if no PCD file is provided.
- It uses local geometric features computed from k-nearest neighbors.
- The PCD loader supports ASCII and binary formats (x/y/z fields required).
