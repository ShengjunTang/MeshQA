"""Entry point for `python -m demo`."""

import argparse

from . import __version__
from .pointcloud_rf import run_demo


def main() -> None:
    """Run the demo application."""

    parser = argparse.ArgumentParser(prog="python -m demo")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pcd = sub.add_parser("pointcloud", help="Run point cloud random forest demo")
    pcd.add_argument("--pcd", dest="pcd_file", help="Path to a PCD file.", default=None)
    pcd.add_argument(
        "--out",
        dest="out_dir",
        help="Directory to write the labeled point cloud (PCD).",
        default=None,
    )

    mq = sub.add_parser("mesh-quality", help="Train or run mesh quality classifier")
    mq_sub = mq.add_subparsers(dest="mode", required=True)

    train = mq_sub.add_parser("train", help="Train a mesh quality classifier")
    train.add_argument("--obj", required=True, help="Labeled OBJ file (high/low groups)")
    train.add_argument(
        "--glb",
        dest="glb_file",
        required=False,
        help="Optional GLB file with texture/uv to include texture features.",
    )
    train.add_argument("--out", required=True, help="Output path for trained model (pickle)")

    predict = mq_sub.add_parser("predict", help="Predict mesh quality on a new mesh")
    predict.add_argument("--model", required=True, help="Path to trained model (pickle)")
    predict.add_argument("--in", dest="in_mesh", required=True, help="Input mesh (OBJ/GLB)")
    predict.add_argument("--out", required=True, help="Output OBJ path (with quality groups)")
    predict.add_argument(
        "--max-faces",
        dest="max_faces",
        type=int,
        default=0,
        help="Maximum number of faces to process (<=0 disables subsampling).",
    )
    predict.add_argument(
        "--scores",
        dest="scores_out",
        required=False,
        help="Optional path to save per-face confidence scores (CSV).",
    )

    args = parser.parse_args()

    print(f"demo version: {__version__}")

    if args.cmd == "pointcloud":
        run_demo(pcd_file=args.pcd_file, output_dir=args.out_dir)
    elif args.cmd == "mesh-quality":
        from .mesh_quality import MeshQualityModel, predict_mesh_quality, train_from_labeled_obj

        if args.mode == "train":
            model, score = train_from_labeled_obj(args.obj, args.glb_file)
            import pickle

            with open(args.out, "wb") as f:
                pickle.dump(model, f)
            print(f"Trained model saved to {args.out}. validation accuracy: {score:.3f}")
        elif args.mode == "predict":
            import pickle

            with open(args.model, "rb") as f:
                model: MeshQualityModel = pickle.load(f)
            predict_mesh_quality(
                model,
                args.in_mesh,
                args.out,
                max_faces=args.max_faces,
                scores_out=args.scores_out,
            )
            print(f"Predicted mesh quality written to {args.out}")


if __name__ == "__main__":
    main()
